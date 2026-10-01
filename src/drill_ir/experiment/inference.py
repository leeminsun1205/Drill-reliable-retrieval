"""Inference over official test sets with reproducible submission artifacts."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from typing import Any, Sequence

import yaml

from drill_ir.data import iter_articles, load_queries
from drill_ir.reranking import build_reranker
from drill_ir.retrieval import build_retriever

from .submission import package_submission, package_threshold_submission


def _threshold_slug(value: float) -> str:
    """Return a stable filename-safe decimal without losing supplied precision."""
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def _write_rankings(path: Path, rankings: Sequence[Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for result in rankings:
            stream.write(
                json.dumps(
                    {
                        "qid": result.qid,
                        "retrieved_aids": list(result.aids),
                        "scores": list(result.scores),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def run_submission_inference(
    config: dict[str, Any],
    test_paths: dict[str, str | Path],
    output_dir: str | Path,
    *,
    score_threshold: float | None = None,
    score_thresholds: Sequence[float] | None = None,
    threshold_fallback_k: int = 2,
) -> Path:
    """Fit once, infer each named test set, and create one ZIP per test set."""
    started = time.perf_counter()
    if score_threshold is not None and score_thresholds is not None:
        raise ValueError("Use either score_threshold or score_thresholds, not both")
    threshold_values = (
        [float(score_threshold)]
        if score_threshold is not None
        else list(dict.fromkeys(map(float, score_thresholds or ())))
    )
    if not test_paths:
        raise ValueError("At least one test set is required")
    method = str(config["method"])
    if method == "organizer_two_stage":
        raise NotImplementedError("organizer_two_stage is not implemented for inference")

    named_queries = {name: load_queries(path) for name, path in test_paths.items()}
    if any(not queries for queries in named_queries.values()):
        raise ValueError("Test query files must not be empty")
    articles = list(iter_articles(config["data"]["corpus_path"], include_blank=False))
    if not articles:
        raise ValueError("Corpus contains no non-blank articles")

    retrieval_method = "hybrid" if method == "hybrid_reranker" else method
    retriever = build_retriever(
        retrieval_method, config["retrieval"], config.get("runtime", {})
    )
    retriever.fit(articles)
    retrieve_k = int(config["retrieval"]["top_k"])
    output_k = int(config["retrieval"]["output_k"])
    if not 0 < output_k <= retrieve_k:
        raise ValueError("retrieval.output_k must be positive and cannot exceed retrieval.top_k")

    first_stage = {
        name: retriever.retrieve(queries, top_k=retrieve_k)
        for name, queries in named_queries.items()
    }
    final_rankings = first_stage
    reranker = None
    if method == "hybrid_reranker":
        reranking_config = config.get("reranking")
        if not isinstance(reranking_config, dict):
            raise ValueError("hybrid_reranker requires a reranking mapping")
        candidate_pool_size = int(reranking_config.get("candidate_pool_size", 50))
        if not output_k <= candidate_pool_size <= retrieve_k:
            raise ValueError(
                "reranking.candidate_pool_size must be between output_k and retrieval.top_k"
            )
        reranker = build_reranker(reranking_config, config.get("runtime", {}))
        candidate_source = str(
            reranking_config.get("candidate_source", "article_rechunk")
        )
        if candidate_source == "article_rechunk":
            candidate_aids = {
                aid
                for rankings in first_stage.values()
                for result in rankings
                for aid in result.aids[:candidate_pool_size]
            }
            reranker.fit([article for article in articles if article.aid in candidate_aids])
            final_rankings = {
                name: reranker.rerank(named_queries[name], rankings)
                for name, rankings in first_stage.items()
            }
        elif candidate_source == "retrieved_chunks":
            retrieve_chunks = getattr(retriever, "retrieve_chunks", None)
            if not callable(retrieve_chunks):
                raise TypeError("retrieved_chunks requires a retriever with chunk evidence")
            dense_depth = int(
                reranking_config.get("retrieved_chunk_depth", candidate_pool_size)
            )
            final_rankings = {}
            for name, queries in named_queries.items():
                chunks = retrieve_chunks(
                    queries, top_k=candidate_pool_size, dense_depth=dense_depth
                )
                final_rankings[name] = reranker.rerank_chunks(queries, chunks)
        else:
            raise ValueError(
                "reranking.candidate_source must be 'article_rechunk' or 'retrieved_chunks'"
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    for name, rankings in final_rankings.items():
        predictions_path = output_dir / f"{name}_predictions.jsonl"
        _write_rankings(predictions_path, rankings)
        package_submission(
            predictions_path,
            {query.qid for query in named_queries[name]},
            output_dir / f"{name}_results.zip",
            output_k=output_k,
        )
        if score_threshold is not None:
            package_threshold_submission(
                predictions_path,
                {query.qid for query in named_queries[name]},
                output_dir / f"{name}_threshold_results.zip",
                score_threshold=score_threshold,
                fallback_k=threshold_fallback_k,
            )
        elif threshold_values:
            for threshold in threshold_values:
                package_threshold_submission(
                    predictions_path,
                    {query.qid for query in named_queries[name]},
                    output_dir
                    / f"{name}_threshold_{_threshold_slug(threshold)}_fallback{threshold_fallback_k}_results.zip",
                    score_threshold=threshold,
                    fallback_k=threshold_fallback_k,
                )
        if method == "hybrid_reranker":
            _write_rankings(
                output_dir / f"{name}_first_stage_predictions.jsonl",
                first_stage[name],
            )

    run_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "seed": int(config["seed"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runtime_seconds": time.perf_counter() - started,
        "corpus_path": str(config["data"]["corpus_path"]),
        "indexed_articles": len(articles),
        "test_sets": {
            name: {"path": str(test_paths[name]), "queries": len(queries)}
            for name, queries in named_queries.items()
        },
        "output_k": output_k,
        "threshold_policy": (
            {
                "selection": "score_threshold_with_empty_only_fallback",
                "threshold": score_threshold,
                "threshold_operator": ">=",
                "fallback_k": threshold_fallback_k,
            }
            if score_threshold is not None
            else None
        ),
        "threshold_sweep": (
            {
                "selection": "score_threshold_with_empty_only_fallback",
                "thresholds": threshold_values,
                "threshold_operator": ">=",
                "fallback_k": threshold_fallback_k,
                "note": "All ZIPs reuse one inference ranking pass",
            }
            if score_threshold is None and threshold_values
            else None
        ),
        "retriever": (
            retriever.run_metadata() if hasattr(retriever, "run_metadata") else {}
        ),
        "reranker": reranker.run_metadata() if reranker is not None else None,
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_dir
