"""Inference over official test sets with reproducible submission artifacts."""

from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import yaml

from drill_ir.data import iter_articles, load_queries
from drill_ir.reranking import build_reranker
from drill_ir.retrieval import build_retriever
from drill_ir.runtime_resources import cuda_snapshot, release_stage_resources
from drill_ir.text_features import chunk_pair_features

from .run import _read_stage_handoff, _write_stage_handoff
from .submission import package_submission, package_threshold_submission


_SUBMISSION_HANDOFF_ENV = "DRILL_SUBMISSION_RETRIEVAL_HANDOFF"


def _write_submission_handoff(
    directory: Path,
    *,
    first_stage: dict[str, Any],
    chunks: dict[str, Any],
    components: dict[str, dict[str, Any]],
    feedback: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, str] = {}
    for name, rankings in first_stage.items():
        path = directory / f"retrieval_stage_handoff_{name}.json"
        _write_stage_handoff(
            path,
            first_stage=list(rankings),
            chunks=list(chunks[name]) if name in chunks else None,
            components={
                component: list(values)
                for component, values in components.get(name, {}).items()
            },
            feedback=feedback.get(name, {}),
            metadata=metadata,
        )
        split_paths[name] = str(path)
    manifest = directory / "retrieval_stage_handoff_manifest.json"
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"schema_version": 1, "splits": split_paths}, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(manifest)
    return manifest


def _read_submission_handoff(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("splits"):
        raise ValueError(f"Invalid submission retrieval handoff: {manifest_path}")
    first_stage: dict[str, Any] = {}
    chunks: dict[str, Any] = {}
    components: dict[str, Any] = {}
    feedback: dict[str, Any] = {}
    metadata: dict[str, Any] | None = None
    for name, value in manifest["splits"].items():
        ranked, split_chunks, split_components, split_feedback, split_metadata = (
            _read_stage_handoff(Path(value))
        )
        first_stage[name] = ranked
        if split_chunks is not None:
            chunks[name] = split_chunks
        components[name] = split_components
        feedback[name] = split_feedback
        if metadata is None:
            metadata = split_metadata
        elif metadata != split_metadata:
            raise ValueError("Submission retrieval handoff metadata differs by split")
    return first_stage, chunks, components, feedback, metadata or {}


def _remove_submission_handoff(manifest_path: Path) -> None:
    """Remove a fully consumed same-directory submission handoff."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("splits"):
        raise ValueError(f"Invalid submission retrieval handoff: {manifest_path}")
    split_paths = [Path(value).resolve() for value in manifest["splits"].values()]
    if any(path.parent != manifest_path.parent for path in split_paths):
        raise ValueError("Submission handoff cleanup refuses paths outside its directory")
    for path in split_paths:
        path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    print(
        f"[inference] Consumed and removed retrieval handoff: {manifest_path}",
        flush=True,
    )


def _exec_submission_reranking_stage(manifest_path: Path) -> None:
    environment = dict(os.environ)
    environment[_SUBMISSION_HANDOFF_ENV] = str(manifest_path)
    print(
        "[inference] Retrieval artifacts valid; replacing process before reranking: "
        f"{manifest_path}",
        flush=True,
    )
    os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)


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
    cuda_snapshot("submission:process_start")
    if score_threshold is not None and score_thresholds is not None:
        raise ValueError("Use either score_threshold or score_thresholds, not both")
    threshold_values = (
        [float(score_threshold)]
        if score_threshold is not None
        else list(dict.fromkeys(map(float, score_thresholds or ())))
    )
    if any(not math.isfinite(value) for value in threshold_values):
        raise ValueError("Every score threshold must be finite")
    if (
        isinstance(threshold_fallback_k, bool)
        or not isinstance(threshold_fallback_k, int)
        or threshold_fallback_k <= 0
    ):
        raise ValueError("threshold_fallback_k must be a positive integer")
    threshold_slugs = [_threshold_slug(value) for value in threshold_values]
    if len(set(threshold_slugs)) != len(threshold_slugs):
        raise ValueError("Score thresholds collide after filename formatting")
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

    output_dir = Path(output_dir)
    handoff_value = os.environ.get(_SUBMISSION_HANDOFF_ENV)
    resumed = bool(handoff_value and Path(handoff_value).is_file())
    use_process_handoff = bool(
        config.get("runtime", {}).get("process_handoff_before_reranking", False)
    )

    retrieval_method = "hybrid" if method == "hybrid_reranker" else method
    retrieve_k = int(config["retrieval"]["top_k"])
    output_k = int(config["retrieval"]["output_k"])
    if not 0 < output_k <= retrieve_k:
        raise ValueError("retrieval.output_k must be positive and cannot exceed retrieval.top_k")

    reranking_config = config.get("reranking") if method == "hybrid_reranker" else None
    candidate_source = (
        str(reranking_config.get("candidate_source", "article_rechunk"))
        if isinstance(reranking_config, dict)
        else None
    )
    retriever = None
    retriever_metadata = None
    if resumed:
        manifest_path = Path(str(handoff_value))
        (
            first_stage,
            chunk_candidates,
            named_components,
            named_feedback,
            retriever_metadata,
        ) = _read_submission_handoff(manifest_path)
        if set(first_stage) != set(named_queries):
            raise ValueError("Submission retrieval handoff split coverage is incomplete")
        _remove_submission_handoff(manifest_path)
        os.environ.pop(_SUBMISSION_HANDOFF_ENV, None)
        cuda_snapshot("submission:reranking_process_start")
    else:
        retriever = build_retriever(
            retrieval_method, config["retrieval"], config.get("runtime", {})
        )
        cuda_snapshot("submission:after_build_retriever")
        retriever.fit(articles)
        cuda_snapshot("submission:after_retriever_fit")
        # WeightedRRFRetriever.retrieve_chunks() intentionally consumes the sparse
        # component ranking from its immediately preceding retrieve() call. For the
        # direct-chunk flow, process each named split sequentially below so public
        # chunks can never be fused against private sparse qids (or vice versa).
        first_stage = (
            {}
            if candidate_source == "retrieved_chunks"
            else {
                name: retriever.retrieve(queries, top_k=retrieve_k)
                for name, queries in named_queries.items()
            }
        )
        chunk_candidates = {}
        named_components = {}
        named_feedback = {}
    final_rankings = first_stage
    reranker = None
    named_chunk_scores: dict[str, dict[int, tuple[float, ...]]] = {}
    if method == "hybrid_reranker":
        if not isinstance(reranking_config, dict):
            raise ValueError("hybrid_reranker requires a reranking mapping")
        candidate_pool_size = int(reranking_config.get("candidate_pool_size", 50))
        if not output_k <= candidate_pool_size <= retrieve_k:
            raise ValueError(
                "reranking.candidate_pool_size must be between output_k and retrieval.top_k"
            )
        if candidate_source == "article_rechunk":
            candidate_aids = {
                aid
                for rankings in first_stage.values()
                for result in rankings
                for aid in result.aids[:candidate_pool_size]
            }
            if not resumed:
                retriever_metadata = retriever.run_metadata()
                if use_process_handoff:
                    manifest = _write_submission_handoff(
                        output_dir,
                        first_stage=first_stage,
                        chunks=chunk_candidates,
                        components=named_components,
                        feedback=named_feedback,
                        metadata=retriever_metadata,
                    )
                    _exec_submission_reranking_stage(manifest)
                print("[inference] Releasing retrieval resources before reranker load", flush=True)
                release_stage_resources(retriever, stage="submission:retrieval")
                retriever = None
                gc.collect()
            reranker = build_reranker(reranking_config, config.get("runtime", {}))
            reranker.fit([article for article in articles if article.aid in candidate_aids])
            final_rankings = {
                name: reranker.rerank(named_queries[name], rankings)
                for name, rankings in first_stage.items()
            }
        elif candidate_source == "retrieved_chunks":
            if not resumed:
                retrieve_chunks = getattr(retriever, "retrieve_chunks", None)
                if not callable(retrieve_chunks):
                    raise TypeError("retrieved_chunks requires a retriever with chunk evidence")
                dense_depth = int(
                    reranking_config.get("retrieved_chunk_depth", candidate_pool_size)
                )
                for name, queries in named_queries.items():
                    # Keep this call adjacent to retrieve_chunks(): the hybrid
                    # retriever stores sparse component ranks for the latest query
                    # batch as state used by chunk-level WRRF fusion.
                    first_stage[name] = retriever.retrieve(queries, top_k=retrieve_k)
                    component_results = getattr(retriever, "component_results", None)
                    if callable(component_results):
                        named_components[name] = {
                            component: list(values)
                            for component, values in component_results().items()
                        }
                    chunk_candidates[name] = retrieve_chunks(
                        queries, top_k=candidate_pool_size, dense_depth=dense_depth
                    )
                    feedback_records = getattr(retriever, "feedback_records", None)
                    if callable(feedback_records):
                        named_feedback[name] = feedback_records()
                # Complete dense/sparse candidate generation for every split before
                # the cross-encoder lazily loads onto GPU. This avoids returning to
                # dense inference after the larger reranker has occupied GPU memory.
                retriever_metadata = retriever.run_metadata()
                if use_process_handoff:
                    manifest = _write_submission_handoff(
                        output_dir,
                        first_stage=first_stage,
                        chunks=chunk_candidates,
                        components=named_components,
                        feedback=named_feedback,
                        metadata=retriever_metadata,
                    )
                    _exec_submission_reranking_stage(manifest)
                print("[inference] Releasing retrieval resources before reranker load", flush=True)
                release_stage_resources(retriever, stage="submission:retrieval")
                retriever = None
                gc.collect()
            reranker = build_reranker(reranking_config, config.get("runtime", {}))
            final_rankings = {}
            for name in named_queries:
                final_rankings[name] = reranker.rerank_chunks(
                    named_queries[name], chunk_candidates[name]
                )
                score_getter = getattr(reranker, "chunk_scores_by_qid", None)
                all_scores = score_getter() if callable(score_getter) else {}
                named_chunk_scores[name] = {
                    query.qid: all_scores[query.qid]
                    for query in named_queries[name]
                    if query.qid in all_scores
                }
        else:
            raise ValueError(
                "reranking.candidate_source must be 'article_rechunk' or 'retrieved_chunks'"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    submission_manifest: list[dict[str, Any]] = []
    for name, rankings in final_rankings.items():
        predictions_path = output_dir / f"{name}_predictions.jsonl"
        _write_rankings(predictions_path, rankings)
        package_submission(
            predictions_path,
            {query.qid for query in named_queries[name]},
            output_dir / f"{name}_results.zip",
            output_k=output_k,
        )
        submission_manifest.append(
            {
                "test_set": name,
                "policy": "fixed_top_k",
                "output_k": output_k,
                "file": f"{name}_results.zip",
            }
        )
        if score_threshold is not None:
            package_threshold_submission(
                predictions_path,
                {query.qid for query in named_queries[name]},
                output_dir / f"{name}_threshold_results.zip",
                score_threshold=score_threshold,
                fallback_k=threshold_fallback_k,
            )
            submission_manifest.append(
                {
                    "test_set": name,
                    "policy": "score_threshold_with_empty_only_fallback",
                    "threshold": score_threshold,
                    "fallback_k": threshold_fallback_k,
                    "file": f"{name}_threshold_results.zip",
                }
            )
        elif threshold_values:
            for threshold, slug in zip(threshold_values, threshold_slugs):
                filename = (
                    f"{name}_threshold_{slug}_fallback{threshold_fallback_k}_results.zip"
                )
                package_threshold_submission(
                    predictions_path,
                    {query.qid for query in named_queries[name]},
                    output_dir
                    / filename,
                    score_threshold=threshold,
                    fallback_k=threshold_fallback_k,
                )
                submission_manifest.append(
                    {
                        "test_set": name,
                        "policy": "score_threshold_with_empty_only_fallback",
                        "threshold": threshold,
                        "fallback_k": threshold_fallback_k,
                        "file": filename,
                    }
                )
        if method == "hybrid_reranker":
            _write_rankings(
                output_dir / f"{name}_first_stage_predictions.jsonl",
                first_stage[name],
            )
            if candidate_source == "retrieved_chunks":
                query_by_qid = {query.qid: query for query in named_queries[name]}
                reranker_chunk_scores = named_chunk_scores.get(name, {})
                with (output_dir / f"{name}_reranker_chunk_candidates.jsonl").open(
                    "w", encoding="utf-8"
                ) as stream:
                    for result in chunk_candidates[name]:
                        scores = reranker_chunk_scores.get(result.qid)
                        if scores is None or not scores or len(scores) > len(result.chunks):
                            raise RuntimeError(
                                "Missing aligned per-chunk reranker scores for "
                                f"split={name} qid={result.qid}"
                            )
                        query = query_by_qid[result.qid]
                        scored_chunks = result.chunks[: len(scores)]
                        fusion_scores = result.scores[: len(scores)]
                        stream.write(
                            json.dumps(
                                {
                                    "qid": result.qid,
                                    "chunks": [
                                        {
                                            "chunk_id": chunk.chunk_id,
                                            "aid": chunk.aid,
                                            "chunk_index": chunk.chunk_index,
                                            "chunk_type": chunk.chunk_type,
                                            "char_start": chunk.char_start,
                                            "char_end": chunk.char_end,
                                            "token_count": chunk.token_count,
                                            "fusion_score": score,
                                            "reranker_score": reranker_score,
                                            **chunk_pair_features(
                                                query.question, chunk.text
                                            ),
                                        }
                                        for chunk, score, reranker_score in zip(
                                            scored_chunks, fusion_scores, scores
                                        )
                                    ],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

        components = named_components.get(name)
        if components is None:
            component_results = getattr(retriever, "component_results", None)
            components = component_results() if callable(component_results) else None
        if components is not None:
            indexed = {
                component: {result.qid: result for result in values}
                for component, values in components.items()
            }
            with (output_dir / f"{name}_component_rankings.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:
                for query in named_queries[name]:
                    record: dict[str, Any] = {"qid": query.qid, "components": {}}
                    for component, by_qid in indexed.items():
                        result = by_qid[query.qid]
                        record["components"][component] = {
                            "retrieved_aids": list(result.aids),
                            "scores": list(result.scores),
                        }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if named_feedback.get(name):
            (output_dir / f"{name}_retrieval_feedback.json").write_text(
                json.dumps(named_feedback[name], ensure_ascii=False, indent=2),
                encoding="utf-8",
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
            retriever_metadata
            if retriever_metadata is not None
            else (retriever.run_metadata() if hasattr(retriever, "run_metadata") else {})
        ),
        "reranker": reranker.run_metadata() if reranker is not None else None,
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "submission_manifest.json").write_text(
        json.dumps(submission_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if reranker is not None:
        print("[inference] Releasing reranker resources after artifact write", flush=True)
        release_stage_resources(reranker, stage="submission:reranker")
        reranker = None
        gc.collect()
    return output_dir
