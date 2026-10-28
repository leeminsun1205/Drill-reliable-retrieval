"""Leakage-safe hard-negative mining for cross-encoder fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Sequence

from drill_ir.data import Article, Query, RetrievalResult, iter_articles, load_queries
from drill_ir.retrieval import build_retriever

from .factory import build_reranker


@dataclass(frozen=True, slots=True)
class MinedPassage:
    aid: int
    chunk_id: str
    text: str
    score: float


def select_fpt_semi_hard_negatives(
    ranked_aids: Sequence[int],
    relevant_aids: Sequence[int],
    *,
    eligible_aids: set[int],
    top_n: int = 30,
    skip_top: int = 5,
    large_pool_threshold: int = 15,
    large_sample_size: int = 10,
    small_sample_size: int = 5,
    rng: random.Random,
) -> tuple[int, ...]:
    """Select FPT-style semi-hard negatives from a reranked article list.

    Gold and over-length articles are removed first. The top ``top_n`` eligible
    non-gold hits form the hard pool, its first ``skip_top`` hits are discarded,
    and the remainder is sampled without replacement. This follows the published
    VLSP 2025 FPT.AI procedure and deliberately avoids the very hardest hits,
    which are the most likely false negatives.
    """
    if top_n <= 0 or skip_top < 0 or skip_top >= top_n:
        raise ValueError("Require top_n > skip_top >= 0")
    if min(large_pool_threshold, large_sample_size, small_sample_size) <= 0:
        raise ValueError("FPT negative sampling sizes must be positive")
    relevant = set(relevant_aids)
    unique_candidates = tuple(
        dict.fromkeys(
            aid
            for aid in ranked_aids
            if aid not in relevant and aid in eligible_aids
        )
    )[:top_n]
    pool = unique_candidates[skip_top:]
    requested = (
        large_sample_size if len(pool) > large_pool_threshold else small_sample_size
    )
    if len(pool) < requested:
        return ()
    return tuple(rng.sample(list(pool), requested))


def _training_split(
    queries: Sequence[Query], *, dev_fraction: float, seed: int
) -> tuple[list[Query], list[Query]]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    shuffled = list(queries)
    random.Random(seed).shuffle(shuffled)
    dev_size = max(1, round(len(shuffled) * dev_fraction))
    dev_qids = {query.qid for query in shuffled[:dev_size]}
    return (
        [query for query in queries if query.qid not in dev_qids],
        [query for query in queries if query.qid in dev_qids],
    )


def select_best_passages(
    aids: Sequence[int],
    chunks_by_aid: dict[int, Sequence[Any]],
    scores: Sequence[float],
) -> dict[int, MinedPassage]:
    """Select the highest-scoring representation per aid with stable ties."""
    flattened = [chunk for aid in aids for chunk in chunks_by_aid[aid]]
    if len(flattened) != len(scores):
        raise ValueError("Chunk/score length mismatch during reranker mining")
    selected: dict[int, MinedPassage] = {}
    for chunk, score in zip(flattened, scores):
        candidate = MinedPassage(chunk.aid, chunk.chunk_id, chunk.text, float(score))
        current = selected.get(chunk.aid)
        if current is None or candidate.score > current.score:
            selected[chunk.aid] = candidate
    return selected


def prefilter_representations(
    query: str, chunks: Sequence[Any], *, maximum: int
) -> tuple[Any, ...]:
    """Bound pathological articles using deterministic lexical-overlap prefiltering."""
    if maximum <= 0:
        raise ValueError("maximum representations per aid must be positive")
    if len(chunks) <= maximum:
        return tuple(chunks)
    query_terms = set(re.findall(r"\w+", query.casefold(), flags=re.UNICODE))

    def key(chunk: Any) -> tuple[float, int, str]:
        chunk_terms = set(re.findall(r"\w+", chunk.text.casefold(), flags=re.UNICODE))
        overlap = len(query_terms & chunk_terms) / max(1, len(query_terms))
        return (-overlap, int(chunk.chunk_index), str(chunk.chunk_id))

    return tuple(sorted(chunks, key=key)[:maximum])


def prepare_reranker_training_data(
    config: dict[str, Any],
    destination: str | Path,
    *,
    negatives_per_query: int = 5,
    negative_pool_size: int = 50,
    max_representations_per_aid: int = 8,
    max_positive_representations_per_aid: int = 32,
) -> Path:
    """Mine one best Điểm representation per gold/candidate aid on train only."""
    if negatives_per_query <= 0 or negative_pool_size < negatives_per_query:
        raise ValueError("negative_pool_size must be >= positive negatives_per_query")
    if max_representations_per_aid <= 0:
        raise ValueError("max_representations_per_aid must be positive")
    if max_positive_representations_per_aid <= 0:
        raise ValueError("max_positive_representations_per_aid must be positive")
    started = time.perf_counter()

    def stage(message: str) -> None:
        elapsed = time.perf_counter() - started
        print(f"[reranker-mining {elapsed:8.1f}s] {message}", flush=True)

    try:
        from tqdm.auto import tqdm

        article_iterator = tqdm(
            iter_articles(config["data"]["corpus_path"], include_blank=False),
            desc="Loading corpus articles",
            unit="article",
        )
    except ImportError:
        article_iterator = iter_articles(
            config["data"]["corpus_path"], include_blank=False
        )
    stage("Loading and flattening corpus")
    articles = list(article_iterator)
    by_aid = {article.aid: article for article in articles}
    stage(f"Loaded {len(articles):,} non-blank articles")
    all_queries = load_queries(config["data"]["train_path"])
    train_queries, dev_queries = _training_split(
        all_queries,
        dev_fraction=float(config["data"]["dev_fraction"]),
        seed=int(config["seed"]),
    )
    stage(
        f"Split queries: train={len(train_queries):,}, dev excluded={len(dev_queries):,}"
    )

    stage("Building and fitting frozen hybrid retriever")
    retriever = build_retriever("hybrid", config["retrieval"], config.get("runtime", {}))
    retriever.fit(articles)
    stage("Hybrid retriever fitted; retrieving train-query candidates")
    rankings = retriever.retrieve(train_queries, top_k=int(config["retrieval"]["top_k"]))
    ranking_by_qid = {ranking.qid: ranking for ranking in rankings}
    stage(f"Retrieved candidates for {len(rankings):,} train queries")

    selected_aids = {
        aid
        for query in train_queries
        for aid in query.relevant_aids
    }
    try:
        from tqdm.auto import tqdm

        candidate_iterator = tqdm(
            train_queries, desc="Collecting candidate aids", unit="query"
        )
    except ImportError:
        candidate_iterator = train_queries
    for query in candidate_iterator:
        gold = set(query.relevant_aids)
        ranking = ranking_by_qid[query.qid]
        selected_aids.update(
            aid for aid in ranking.aids if aid not in gold
        )
    missing = selected_aids - set(by_aid)
    if missing:
        raise ValueError(f"Training candidates missing from corpus: {sorted(missing)[:10]}")

    stage(f"Preparing reranker representations for {len(selected_aids):,} unique aids")
    reranker = build_reranker(config["reranking"], config.get("runtime", {}))
    reranker.fit([by_aid[aid] for aid in sorted(selected_aids)])
    stage(
        "Reranker representations ready; loading teacher and mining hard negatives"
    )

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    audit_path = destination.with_name(destination.stem + "_audit.jsonl")
    temporary_destination = destination.with_suffix(destination.suffix + ".tmp")
    temporary_audit = audit_path.with_suffix(audit_path.suffix + ".tmp")
    written = 0
    skipped = []
    total_scored_pairs = 0
    maximum_pairs_per_query = 0
    prefiltered_gold_aid_instances = 0
    prefiltered_negative_aid_instances = 0
    representations_before_prefilter = 0
    representations_after_prefilter = 0
    try:
        from tqdm.auto import tqdm

        query_iterator = tqdm(train_queries, desc="Mining reranker train queries")
    except ImportError:
        query_iterator = train_queries
    with temporary_destination.open("w", encoding="utf-8") as train_stream, temporary_audit.open(
        "w", encoding="utf-8"
    ) as audit_stream:
        for query in query_iterator:
            gold_aids = tuple(dict.fromkeys(query.relevant_aids))
            gold = set(gold_aids)
            candidate_aids = tuple(
                aid
                for aid in ranking_by_qid[query.qid].aids
                if aid not in gold
            )[:negative_pool_size]
            if len(candidate_aids) < negatives_per_query:
                skipped.append(query.qid)
                continue
            scored_aids = gold_aids + candidate_aids
            chunks_by_aid = {}
            for aid in scored_aids:
                original_chunks = reranker.representations_for_aid(aid)
                maximum = (
                    max_positive_representations_per_aid
                    if aid in gold
                    else max_representations_per_aid
                )
                retained_chunks = prefilter_representations(
                    query.question,
                    original_chunks,
                    maximum=maximum,
                )
                chunks_by_aid[aid] = retained_chunks
                representations_before_prefilter += len(original_chunks)
                representations_after_prefilter += len(retained_chunks)
                if len(original_chunks) > len(retained_chunks):
                    if aid in gold:
                        prefiltered_gold_aid_instances += 1
                    else:
                        prefiltered_negative_aid_instances += 1
            empty = [aid for aid, chunks in chunks_by_aid.items() if not chunks]
            if empty:
                raise ValueError(f"No reranker representation for aids: {empty[:10]}")
            pairs = [
                (query.question, chunk.text)
                for aid in scored_aids
                for chunk in chunks_by_aid[aid]
            ]
            scores = reranker.scorer.score(pairs)
            total_scored_pairs += len(pairs)
            maximum_pairs_per_query = max(maximum_pairs_per_query, len(pairs))
            best = select_best_passages(scored_aids, chunks_by_aid, scores)
            negatives = sorted(
                (best[aid] for aid in candidate_aids),
                key=lambda item: (-item.score, candidate_aids.index(item.aid), item.aid),
            )[:negatives_per_query]
            positives = [best[aid] for aid in gold_aids]
            train_stream.write(
                json.dumps(
                    {
                        "query": query.question,
                        "pos": [item.text for item in positives],
                        "neg": [item.text for item in negatives],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            audit_stream.write(
                json.dumps(
                    {
                        "qid": query.qid,
                        "positive": [asdict(item) for item in positives],
                        "negative": [asdict(item) for item in negatives],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    temporary_destination.replace(destination)
    temporary_audit.replace(audit_path)
    stage(
        f"Finished mining {written:,} queries and scoring {total_scored_pairs:,} pairs"
    )

    metadata = {
        "selection": "zero_shot_max_score_per_aid",
        "seed": int(config["seed"]),
        "train_queries": len(train_queries),
        "train_qids": [query.qid for query in train_queries],
        "dev_queries_excluded": len(dev_queries),
        "dev_qids": [query.qid for query in dev_queries],
        "written_queries": written,
        "skipped_qids": skipped,
        "negatives_per_query": negatives_per_query,
        "negative_pool_size": negative_pool_size,
        "max_representations_per_aid": max_representations_per_aid,
        "max_positive_representations_per_aid": max_positive_representations_per_aid,
        "representation_prefilter": "query_lexical_overlap_v1",
        "prefiltered_gold_aid_instances": prefiltered_gold_aid_instances,
        "prefiltered_negative_aid_instances": prefiltered_negative_aid_instances,
        "representations_before_prefilter": representations_before_prefilter,
        "representations_after_prefilter": representations_after_prefilter,
        "total_scored_pairs": total_scored_pairs,
        "maximum_pairs_per_query": maximum_pairs_per_query,
        "candidate_source": "hybrid_article_ranking",
        "runtime_seconds": time.perf_counter() - started,
        "reranker": reranker.run_metadata(),
        "resolved_config": config,
    }
    destination.with_name(destination.stem + "_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stage(f"Saved training data and provenance under {destination.parent}")
    return destination


def prepare_fpt_reranker_training_data(
    config: dict[str, Any],
    destination: str | Path,
    *,
    maximum_article_tokens: int = 1024,
    reranked_pool_size: int = 30,
    skip_top_negatives: int = 5,
    large_pool_threshold: int = 15,
    large_sample_size: int = 10,
    small_sample_size: int = 5,
) -> Path:
    """Create article-level labelled pairs using the published FPT.AI recipe.

    Only the persisted train side of the repository's deterministic split is
    mined. Positives are complete gold articles, not teacher-selected chunks.
    Negatives are sampled after zero-shot article reranking and exclusion of the
    five highest non-gold hits. The result is pointwise ``query``/``passage``/
    ``label`` JSONL suitable for SentenceTransformers BCE training.
    """
    if maximum_article_tokens <= 0:
        raise ValueError("maximum_article_tokens must be positive")
    started = time.perf_counter()

    def stage(message: str) -> None:
        print(
            f"[fpt-reranker-mining {time.perf_counter() - started:8.1f}s] {message}",
            flush=True,
        )

    stage("Loading and flattening non-blank corpus articles")
    articles = list(iter_articles(config["data"]["corpus_path"], include_blank=False))
    by_aid = {article.aid: article for article in articles}
    all_queries = load_queries(config["data"]["train_path"])
    train_queries, dev_queries = _training_split(
        all_queries,
        dev_fraction=float(config["data"]["dev_fraction"]),
        seed=int(config["seed"]),
    )
    stage(
        f"Loaded {len(articles):,} articles; train={len(train_queries):,}, "
        f"dev excluded={len(dev_queries):,}"
    )

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "FPT-style mining requires dense dependencies. "
            "Install with: python -m pip install -e '.[dense]'"
        ) from error
    reranker_config = config["reranking"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(reranker_config["model_name"]),
        revision=reranker_config.get("revision"),
        trust_remote_code=bool(reranker_config.get("trust_remote_code", False)),
        use_fast=True,
    )
    stage(f"Measuring article lengths with limit={maximum_article_tokens:,} tokens")
    eligible_aids: set[int] = set()
    article_token_lengths: dict[int, int] = {}
    batch_size = 256
    try:
        from tqdm.auto import tqdm

        starts = tqdm(
            range(0, len(articles), batch_size),
            desc="Tokenizing full articles",
            unit="batch",
        )
    except ImportError:
        starts = range(0, len(articles), batch_size)
    for start in starts:
        batch = articles[start : start + batch_size]
        encoded = tokenizer(
            [article.text for article in batch],
            add_special_tokens=False,
            truncation=False,
            return_length=True,
        )
        lengths = encoded["length"]
        for article, length in zip(batch, lengths):
            length = int(length)
            article_token_lengths[article.aid] = length
            if length <= maximum_article_tokens:
                eligible_aids.add(article.aid)
    stage(
        f"Eligible full articles: {len(eligible_aids):,}/{len(articles):,}; "
        "over-length articles will be excluded from positives and negatives"
    )

    stage("Fitting frozen hybrid retriever and retrieving train candidates")
    retriever = build_retriever("hybrid", config["retrieval"], config.get("runtime", {}))
    retriever.fit(articles)
    first_stage = retriever.retrieve(
        train_queries, top_k=int(config["retrieval"]["top_k"])
    )
    stage("Zero-shot reranking full-article candidates")
    reranker = build_reranker(reranker_config, config.get("runtime", {}))
    reranker.fit(articles)
    reranked = reranker.rerank(train_queries, first_stage)
    reranked_by_qid = {result.qid: result for result in reranked}

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    audit_path = destination.with_name(destination.stem + "_audit.jsonl")
    temporary_destination = destination.with_suffix(destination.suffix + ".tmp")
    temporary_audit = audit_path.with_suffix(audit_path.suffix + ".tmp")
    rng = random.Random(int(config["seed"]))
    written_queries = 0
    positive_pairs = 0
    negative_pairs = 0
    skipped_no_positive: list[int] = []
    skipped_no_negative: list[int] = []
    stage("Writing deterministic BCE pairs and audit records")
    try:
        from tqdm.auto import tqdm

        query_iterator = tqdm(train_queries, desc="Writing FPT-style pairs", unit="query")
    except ImportError:
        query_iterator = train_queries
    with temporary_destination.open("w", encoding="utf-8") as train_stream, temporary_audit.open(
        "w", encoding="utf-8"
    ) as audit_stream:
        for query in query_iterator:
            gold_aids = tuple(
                aid
                for aid in dict.fromkeys(query.relevant_aids)
                if aid in eligible_aids
            )
            if not gold_aids:
                skipped_no_positive.append(query.qid)
                continue
            ranking = reranked_by_qid[query.qid]
            negative_aids = select_fpt_semi_hard_negatives(
                ranking.aids,
                query.relevant_aids,
                eligible_aids=eligible_aids,
                top_n=reranked_pool_size,
                skip_top=skip_top_negatives,
                large_pool_threshold=large_pool_threshold,
                large_sample_size=large_sample_size,
                small_sample_size=small_sample_size,
                rng=rng,
            )
            if not negative_aids:
                skipped_no_negative.append(query.qid)
                continue
            for aid in gold_aids:
                train_stream.write(
                    json.dumps(
                        {"query": query.question, "passage": by_aid[aid].text, "label": 1.0},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                positive_pairs += 1
            for aid in negative_aids:
                train_stream.write(
                    json.dumps(
                        {"query": query.question, "passage": by_aid[aid].text, "label": 0.0},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                negative_pairs += 1
            audit_stream.write(
                json.dumps(
                    {
                        "qid": query.qid,
                        "positive_aids": list(gold_aids),
                        "negative_aids": list(negative_aids),
                        "positive_token_lengths": [article_token_lengths[aid] for aid in gold_aids],
                        "negative_token_lengths": [article_token_lengths[aid] for aid in negative_aids],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written_queries += 1
    temporary_destination.replace(destination)
    temporary_audit.replace(audit_path)
    metadata = {
        "selection": "fpt_article_bce_semi_hard_v1",
        "published_recipe": {
            "positive": "complete ground-truth article",
            "maximum_article_tokens": maximum_article_tokens,
            "reranked_pool_size": reranked_pool_size,
            "skip_top_negatives": skip_top_negatives,
            "large_pool_threshold": large_pool_threshold,
            "large_sample_size": large_sample_size,
            "small_sample_size": small_sample_size,
        },
        "seed": int(config["seed"]),
        "train_queries": len(train_queries),
        "train_qids": [query.qid for query in train_queries],
        "dev_queries_excluded": len(dev_queries),
        "dev_qids": [query.qid for query in dev_queries],
        "eligible_articles": len(eligible_aids),
        "over_length_articles": len(articles) - len(eligible_aids),
        "written_queries": written_queries,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "skipped_no_eligible_positive_qids": skipped_no_positive,
        "skipped_insufficient_negative_qids": skipped_no_negative,
        "candidate_source": "hybrid_then_zero_shot_full_article_reranker",
        "runtime_seconds": time.perf_counter() - started,
        "reranker": reranker.run_metadata(),
        "resolved_config": config,
    }
    destination.with_name(destination.stem + "_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stage(
        f"Saved {positive_pairs:,} positive and {negative_pairs:,} negative pairs "
        f"from {written_queries:,} queries"
    )
    return destination


def prepare_direct_chunk_bce_training_data(
    config: dict[str, Any],
    destination: str | Path,
    *,
    candidate_pool_size: int | None = None,
    retrieved_chunk_depth: int = 100,
    reranked_pool_size: int = 30,
    skip_top_negatives: int = 5,
    negatives_per_query: int = 10,
    max_positive_representations_per_aid: int = 32,
) -> Path:
    """Mine BCE pairs aligned with direct retrieved-chunk inference.

    Negatives are the exact fused chunks exposed to the inference reranker.
    Because supervision is article-level, the zero-shot teacher selects one
    same-chunker representation inside each gold article as a weakly supervised
    positive. The most suspicious five non-gold reranker hits are excluded
    before deterministic sampling, following the false-negative safeguard from
    FPT.AI while retaining the project's 448/64 chunk unit.
    """
    if candidate_pool_size is None:
        candidate_pool_size = retrieved_chunk_depth
    if candidate_pool_size <= 0 or retrieved_chunk_depth < candidate_pool_size:
        raise ValueError(
            "Require retrieved_chunk_depth >= candidate_pool_size > 0"
        )
    if candidate_pool_size < reranked_pool_size:
        raise ValueError("candidate_pool_size must be >= reranked_pool_size")
    if not 0 <= skip_top_negatives < reranked_pool_size:
        raise ValueError("Require reranked_pool_size > skip_top_negatives >= 0")
    if negatives_per_query <= 0:
        raise ValueError("negatives_per_query must be positive")
    started = time.perf_counter()

    def stage(message: str) -> None:
        print(f"[direct-chunk-mining {time.perf_counter() - started:8.1f}s] {message}", flush=True)

    articles = list(iter_articles(config["data"]["corpus_path"], include_blank=False))
    by_aid = {article.aid: article for article in articles}
    all_queries = load_queries(config["data"]["train_path"])
    train_queries, dev_queries = _training_split(
        all_queries,
        dev_fraction=float(config["data"]["dev_fraction"]),
        seed=int(config["seed"]),
    )
    stage(f"Loaded {len(articles):,} articles; train={len(train_queries):,}, dev excluded={len(dev_queries):,}")

    retriever = build_retriever("hybrid", config["retrieval"], config.get("runtime", {}))
    retriever.fit(articles)
    stage("Retrieving article rankings required for sparse/dense chunk fusion")
    retriever.retrieve(train_queries, top_k=int(config["retrieval"]["top_k"]))
    retrieve_chunks = getattr(retriever, "retrieve_chunks", None)
    if not callable(retrieve_chunks):
        raise TypeError("Direct-chunk mining requires a retriever with chunk evidence")
    chunk_candidates = retrieve_chunks(
        train_queries,
        top_k=candidate_pool_size,
        dense_depth=retrieved_chunk_depth,
    )
    chunks_by_qid = {result.qid: result.chunks for result in chunk_candidates}
    stage(
        f"Retrieved {candidate_pool_size} fused chunks at dense depth "
        f"{retrieved_chunk_depth} for {len(chunk_candidates):,} train queries"
    )

    gold_aids = {aid for query in train_queries for aid in query.relevant_aids}
    missing = gold_aids - set(by_aid)
    if missing:
        raise ValueError(f"Gold aids missing from corpus: {sorted(missing)[:10]}")
    reranker = build_reranker(config["reranking"], config.get("runtime", {}))
    reranker.fit([by_aid[aid] for aid in sorted(gold_aids)])
    rng = random.Random(int(config["seed"]))

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    audit_path = destination.with_name(destination.stem + "_audit.jsonl")
    train_tmp = destination.with_suffix(destination.suffix + ".tmp")
    audit_tmp = audit_path.with_suffix(audit_path.suffix + ".tmp")
    written_queries = positive_pairs = negative_pairs = total_scored_pairs = 0
    skipped: list[int] = []
    try:
        from tqdm.auto import tqdm
        query_iterator = tqdm(train_queries, desc="Mining direct-chunk BCE pairs", unit="query")
    except ImportError:
        query_iterator = train_queries
    with train_tmp.open("w", encoding="utf-8") as train_stream, audit_tmp.open(
        "w", encoding="utf-8"
    ) as audit_stream:
        for query in query_iterator:
            query_gold_aids = tuple(dict.fromkeys(query.relevant_aids))
            positive_chunks_by_aid = {
                aid: prefilter_representations(
                    query.question,
                    reranker.representations_for_aid(aid),
                    maximum=max_positive_representations_per_aid,
                )
                for aid in query_gold_aids
            }
            negative_chunks = tuple(
                chunk for chunk in chunks_by_qid[query.qid] if chunk.aid not in set(query_gold_aids)
            )
            positive_flat = [
                chunk for aid in query_gold_aids for chunk in positive_chunks_by_aid[aid]
            ]
            pairs = [(query.question, chunk.text) for chunk in positive_flat + list(negative_chunks)]
            scores = reranker.scorer.score(pairs)
            total_scored_pairs += len(pairs)
            positive_scores = scores[: len(positive_flat)]
            negative_scores = scores[len(positive_flat) :]
            positives = select_best_passages(
                query_gold_aids,
                positive_chunks_by_aid,
                positive_scores,
            )
            ranked_negatives = sorted(
                zip(negative_chunks, negative_scores),
                key=lambda item: (-float(item[1]), item[0].chunk_id),
            )[:reranked_pool_size]
            semi_hard_pool = ranked_negatives[skip_top_negatives:]
            if len(semi_hard_pool) < negatives_per_query:
                skipped.append(query.qid)
                continue
            selected_negatives = rng.sample(semi_hard_pool, negatives_per_query)
            for aid in query_gold_aids:
                train_stream.write(json.dumps({
                    "query": query.question,
                    "passage": positives[aid].text,
                    "label": 1.0,
                }, ensure_ascii=False) + "\n")
                positive_pairs += 1
            for chunk, score in selected_negatives:
                train_stream.write(json.dumps({
                    "query": query.question,
                    "passage": chunk.text,
                    "label": 0.0,
                }, ensure_ascii=False) + "\n")
                negative_pairs += 1
            audit_stream.write(json.dumps({
                "qid": query.qid,
                "positive": [asdict(positives[aid]) for aid in query_gold_aids],
                "negative": [
                    {"aid": chunk.aid, "chunk_id": chunk.chunk_id, "text": chunk.text, "score": float(score)}
                    for chunk, score in selected_negatives
                ],
            }, ensure_ascii=False) + "\n")
            written_queries += 1
    train_tmp.replace(destination)
    audit_tmp.replace(audit_path)
    metadata = {
        "selection": "direct_retrieved_chunks_bce_semi_hard_v1",
        "seed": int(config["seed"]),
        "train_queries": len(train_queries),
        "train_qids": [query.qid for query in train_queries],
        "dev_queries_excluded": len(dev_queries),
        "dev_qids": [query.qid for query in dev_queries],
        "written_queries": written_queries,
        "skipped_qids": skipped,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "candidate_pool_size": candidate_pool_size,
        "retrieved_chunk_depth": retrieved_chunk_depth,
        "reranked_pool_size": reranked_pool_size,
        "skip_top_negatives": skip_top_negatives,
        "negatives_per_query": negatives_per_query,
        "positive_selection": "zero_shot_max_same_448_64_chunker",
        "negative_source": "exact_fused_retrieved_chunks_used_at_inference",
        "total_scored_pairs": total_scored_pairs,
        "runtime_seconds": time.perf_counter() - started,
        "reranker": reranker.run_metadata(),
        "resolved_config": config,
    }
    destination.with_name(destination.stem + "_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stage(f"Saved {positive_pairs:,} positive and {negative_pairs:,} negative pairs")
    return destination


def prepare_pairwise_training_data_from_audit(
    audit_path: str | Path,
    query_path: str | Path,
    destination: str | Path,
) -> Path:
    """Expand audited direct-chunk positives/negatives into ranking pairs.

    Each selected gold passage is paired with every audited semi-hard negative
    for the same train query. The source audit remains the provenance record;
    this derivative contains only text required by the pairwise trainer.
    """
    queries = {query.qid: query for query in load_queries(query_path)}
    audit_path = Path(audit_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    rows = 0
    qids: list[int] = []
    with audit_path.open(encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as output:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            qid = int(record["qid"])
            query = queries.get(qid)
            if query is None:
                raise ValueError(f"Audit line {line_number}: unknown qid={qid}")
            positives = [str(item["text"]) for item in record["positive"]]
            negatives = [str(item["text"]) for item in record["negative"]]
            if not positives or not negatives:
                raise ValueError(
                    f"Audit line {line_number}: positives and negatives are required"
                )
            for positive in positives:
                for negative in negatives:
                    output.write(
                        json.dumps(
                            {
                                "query": query.question,
                                "positive": positive,
                                "negative": negative,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    rows += 1
            qids.append(qid)
    if not rows:
        raise ValueError(f"No pairwise rows were derived from {audit_path}")
    temporary.replace(destination)
    metadata = {
        "selection": "audited_direct_chunk_positive_x_semihard_negative_v1",
        "source_audit": str(audit_path),
        "queries": len(qids),
        "unique_queries": len(set(qids)),
        "rows": rows,
    }
    destination.with_name(destination.stem + "_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination
