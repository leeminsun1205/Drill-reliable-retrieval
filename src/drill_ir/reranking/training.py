"""Leakage-safe hard-negative mining for cross-encoder fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
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


def prepare_reranker_training_data(
    config: dict[str, Any],
    destination: str | Path,
    *,
    negatives_per_query: int = 5,
    negative_pool_size: int = 50,
) -> Path:
    """Mine one best Điểm representation per gold/candidate aid on train only."""
    if negatives_per_query <= 0 or negative_pool_size < negatives_per_query:
        raise ValueError("negative_pool_size must be >= positive negatives_per_query")
    started = time.perf_counter()
    articles = list(iter_articles(config["data"]["corpus_path"], include_blank=False))
    by_aid = {article.aid: article for article in articles}
    all_queries = load_queries(config["data"]["train_path"])
    train_queries, dev_queries = _training_split(
        all_queries,
        dev_fraction=float(config["data"]["dev_fraction"]),
        seed=int(config["seed"]),
    )

    retriever = build_retriever("hybrid", config["retrieval"], config.get("runtime", {}))
    retriever.fit(articles)
    rankings = retriever.retrieve(train_queries, top_k=int(config["retrieval"]["top_k"]))
    ranking_by_qid = {ranking.qid: ranking for ranking in rankings}

    selected_aids = {
        aid
        for query in train_queries
        for aid in query.relevant_aids
    }
    for query in train_queries:
        gold = set(query.relevant_aids)
        ranking = ranking_by_qid[query.qid]
        selected_aids.update(
            aid for aid in ranking.aids if aid not in gold
        )
    missing = selected_aids - set(by_aid)
    if missing:
        raise ValueError(f"Training candidates missing from corpus: {sorted(missing)[:10]}")

    reranker = build_reranker(config["reranking"], config.get("runtime", {}))
    reranker.fit([by_aid[aid] for aid in sorted(selected_aids)])

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    audit_path = destination.with_name(destination.stem + "_audit.jsonl")
    written = 0
    skipped = []
    with destination.open("w", encoding="utf-8") as train_stream, audit_path.open(
        "w", encoding="utf-8"
    ) as audit_stream:
        for query in train_queries:
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
            chunks_by_aid = {
                aid: reranker.representations_for_aid(aid) for aid in scored_aids
            }
            empty = [aid for aid, chunks in chunks_by_aid.items() if not chunks]
            if empty:
                raise ValueError(f"No reranker representation for aids: {empty[:10]}")
            pairs = [
                (query.question, chunk.text)
                for aid in scored_aids
                for chunk in chunks_by_aid[aid]
            ]
            scores = reranker.scorer.score(pairs)
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
        "candidate_source": "hybrid_article_ranking",
        "runtime_seconds": time.perf_counter() - started,
        "reranker": reranker.run_metadata(),
        "resolved_config": config,
    }
    destination.with_name(destination.stem + "_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination
