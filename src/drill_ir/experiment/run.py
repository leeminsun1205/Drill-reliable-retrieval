"""Config-driven experiment orchestration."""

from datetime import datetime, timezone
import json
import gc
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any

import yaml

from drill_ir.data import iter_articles, load_queries, validate_dataset
from drill_ir.evaluation import (
    evaluate_fixed_cutoffs,
    evaluate_predictions,
    macro_recall_at_k,
    tune_score_thresholds,
)
from drill_ir.reranking import build_reranker
from drill_ir.retrieval import build_retriever
from drill_ir.runtime_resources import cuda_snapshot, release_stage_resources


def _git_info() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"git_revision": revision, "git_dirty": dirty}


def deterministic_split(items: list[Any], *, dev_fraction: float, seed: int) -> tuple[list[Any], list[Any]]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    dev_size = max(1, round(len(shuffled) * dev_fraction))
    dev_qids = {item.qid for item in shuffled[:dev_size]}
    train = [item for item in items if item.qid not in dev_qids]
    dev = [item for item in items if item.qid in dev_qids]
    return train, dev


def _retrieval_artifacts(retriever: Any) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    component_getter = getattr(retriever, "component_results", None)
    components = (
        {name: list(values) for name, values in component_getter().items()}
        if callable(component_getter)
        else {}
    )
    feedback_getter = getattr(retriever, "feedback_records", None)
    feedback = feedback_getter() if callable(feedback_getter) else {}
    return components, feedback


def run_experiment(config: dict[str, Any]) -> Path:
    started = time.perf_counter()
    cuda_snapshot("experiment:process_start")
    seed = int(config["seed"])
    data = config["data"]
    report = validate_dataset(data["corpus_path"], data["train_path"])
    if report.duplicate_aids or report.duplicate_qids or report.missing_gold_aids:
        raise ValueError(f"Fatal dataset validation failure: {report.to_dict()}")

    articles = list(iter_articles(data["corpus_path"], include_blank=False))
    queries = load_queries(data["train_path"])
    _, dev_queries = deterministic_split(
        queries, dev_fraction=float(data["dev_fraction"]), seed=seed
    )

    method = config["method"]
    if method == "organizer_two_stage":
        raise NotImplementedError(
            "The organizer papers identify final output_k=2 but not the cross-encoder, "
            "BM25 candidate depth, training hyperparameters, or any score threshold. "
            "Set these explicitly and implement the reranker before running this config."
        )
    retrieval_method = "hybrid" if method == "hybrid_reranker" else method
    retriever = build_retriever(
        retrieval_method,
        config["retrieval"],
        config.get("runtime", {}),
    )
    cuda_snapshot("experiment:after_build_retriever")

    cuda_snapshot("experiment:before_retriever_fit")
    retriever.fit(articles)
    cuda_snapshot("experiment:after_retriever_fit")
    retrieve_k = int(config["retrieval"]["top_k"])
    output_k = int(config["retrieval"]["output_k"])
    if output_k > retrieve_k:
        raise ValueError("retrieval.output_k cannot exceed retrieval.top_k")
    first_stage_ranked = retriever.retrieve(dev_queries, top_k=retrieve_k)
    cuda_snapshot("experiment:after_first_stage_retrieve")
    component_results_data, feedback_data = _retrieval_artifacts(retriever)
    ranked = first_stage_ranked
    reranker = None
    retrieved_chunk_candidates = None
    retriever_metadata = None
    if method == "hybrid_reranker":
        reranking_config = config.get("reranking")
        if not isinstance(reranking_config, dict):
            raise ValueError("hybrid_reranker requires a reranking mapping")
        candidate_pool_size = int(reranking_config.get("candidate_pool_size", 50))
        if not output_k <= candidate_pool_size <= retrieve_k:
            raise ValueError(
                "reranking.candidate_pool_size must be between output_k and retrieval.top_k"
            )
        candidate_source = str(reranking_config.get("candidate_source", "article_rechunk"))
        if candidate_source == "article_rechunk":
            candidate_aids = {
                aid
                for result in first_stage_ranked
                for aid in result.aids[:candidate_pool_size]
            }
            candidate_articles = [article for article in articles if article.aid in candidate_aids]
            retriever_metadata = retriever.run_metadata()
            component_results_data, feedback_data = _retrieval_artifacts(retriever)
            print("[experiment] Releasing retrieval resources before reranker load", flush=True)
            release_stage_resources(retriever, stage="experiment:retrieval")
            retriever = None
            gc.collect()
            cuda_snapshot("experiment:before_reranker_build")
            reranker = build_reranker(reranking_config, config.get("runtime", {}))
            reranker.fit(candidate_articles)
            ranked = reranker.rerank(dev_queries, first_stage_ranked)
            cuda_snapshot("experiment:after_reranking")
        elif candidate_source == "retrieved_chunks":
            dense_depth = int(reranking_config.get("retrieved_chunk_depth", candidate_pool_size))
            retrieve_chunks = getattr(retriever, "retrieve_chunks", None)
            if not callable(retrieve_chunks):
                raise TypeError("retrieved_chunks requires a retriever with chunk evidence")
            retrieved_chunk_candidates = retrieve_chunks(
                dev_queries,
                top_k=candidate_pool_size,
                dense_depth=dense_depth,
            )
            cuda_snapshot("experiment:after_chunk_candidate_retrieve")
            retriever_metadata = retriever.run_metadata()
            component_results_data, feedback_data = _retrieval_artifacts(retriever)
            print("[experiment] Releasing retrieval resources before reranker load", flush=True)
            release_stage_resources(retriever, stage="experiment:retrieval")
            retriever = None
            gc.collect()
            cuda_snapshot("experiment:before_reranker_build")
            reranker = build_reranker(reranking_config, config.get("runtime", {}))
            ranked = reranker.rerank_chunks(dev_queries, retrieved_chunk_candidates)
            cuda_snapshot("experiment:after_reranking")
        else:
            raise ValueError(
                "reranking.candidate_source must be 'article_rechunk' or 'retrieved_chunks'"
            )
    predictions = {result.qid: result.aids[:output_k] for result in ranked}
    final_rankings = {result.qid: result.aids for result in ranked}
    if retrieved_chunk_candidates is None:
        candidate_rankings = {result.qid: result.aids for result in first_stage_ranked}
    else:
        candidate_rankings = {
            result.qid: tuple(dict.fromkeys(chunk.aid for chunk in result.chunks))
            for result in retrieved_chunk_candidates
        }
    gold = {query.qid: query.relevant_aids for query in dev_queries}
    metrics = evaluate_predictions(gold, predictions)
    evaluation_config = config.get("evaluation", {})
    recall_cutoffs = list(evaluation_config.get("recall_at_k", []))
    if any(cutoff > retrieve_k for cutoff in recall_cutoffs):
        raise ValueError("evaluation.recall_at_k cutoffs cannot exceed retrieval.top_k")
    recall_at_k = (
        macro_recall_at_k(gold, candidate_rankings, recall_cutoffs)
        if recall_cutoffs
        else {}
    )
    output_cutoffs = list(evaluation_config.get("output_k_values", [output_k]))
    if output_k not in output_cutoffs:
        output_cutoffs.append(output_k)
    maximum_output_cutoff = (
        int(config.get("reranking", {}).get("candidate_pool_size", retrieve_k))
        if method == "hybrid_reranker"
        else retrieve_k
    )
    if any(cutoff > maximum_output_cutoff for cutoff in output_cutoffs):
        raise ValueError(
            "evaluation.output_k_values cannot exceed the available final ranking depth"
        )
    fixed_cutoff_metrics = evaluate_fixed_cutoffs(gold, final_rankings, output_cutoffs)
    threshold_metrics = None
    threshold_config = evaluation_config.get("threshold_sweep", {})
    if reranker is not None and bool(threshold_config.get("enabled", False)):
        scored_rankings = {
            result.qid: (result.aids, result.scores)
            for result in ranked
        }
        threshold_metrics = tune_score_thresholds(
            gold,
            scored_rankings,
            threshold_config.get("fallback_k_values", [1, 2]),
        )

    run_dir = Path(config["output"]["root_dir"]) / str(config["experiment_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        metrics_payload = {
            **metrics.to_dict(),
            "recall_at_k": {str(k): value for k, value in recall_at_k.items()},
        }
        json.dump(metrics_payload, stream, ensure_ascii=False, indent=2)
    with (run_dir / "metrics_by_output_k.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "selection": "fixed_top_k",
                "primary_output_k": output_k,
                "results": {
                    str(cutoff): cutoff_metrics.to_dict()
                    for cutoff, cutoff_metrics in fixed_cutoff_metrics.items()
                },
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    if threshold_metrics is not None:
        with (run_dir / "metrics_by_score_threshold.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(threshold_metrics, stream, ensure_ascii=False, indent=2)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for result in ranked:
            record = {
                "qid": result.qid,
                "retrieved_aids": list(result.aids),
                "scores": list(result.scores),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if reranker is not None:
        with (run_dir / "first_stage_predictions.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for result in first_stage_ranked:
                record = {
                    "qid": result.qid,
                    "retrieved_aids": list(result.aids),
                    "scores": list(result.scores),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if retrieved_chunk_candidates is not None:
        with (run_dir / "reranker_chunk_candidates.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for result in retrieved_chunk_candidates:
                record = {
                    "qid": result.qid,
                    "chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "aid": chunk.aid,
                            "chunk_index": chunk.chunk_index,
                            "chunk_type": chunk.chunk_type,
                            "fusion_score": score,
                        }
                        for chunk, score in zip(result.chunks, result.scores)
                    ],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if component_results_data:
        components = component_results_data
        indexed_components = {
            name: {result.qid: result for result in results}
            for name, results in components.items()
        }
        with (run_dir / "component_rankings.jsonl").open("w", encoding="utf-8") as stream:
            for query in dev_queries:
                record: dict[str, Any] = {"qid": query.qid, "components": {}}
                for name, by_qid in indexed_components.items():
                    result = by_qid[query.qid]
                    record["components"][name] = {
                        "retrieved_aids": list(result.aids),
                        "scores": list(result.scores),
                    }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if feedback_data:
        (run_dir / "retrieval_feedback.json").write_text(
            json.dumps(feedback_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (run_dir / "split_dev_qids.json").write_text(
        json.dumps([query.qid for query in dev_queries], indent=2), encoding="utf-8"
    )
    run_info = {
        "experiment_name": config["experiment_name"],
        "method": method,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runtime_seconds": time.perf_counter() - started,
        "dataset_validation": report.to_dict(),
        "indexed_articles": len(articles),
        "dev_queries": len(dev_queries),
        "retriever": (
            retriever_metadata
            if retriever_metadata is not None
            else (retriever.run_metadata() if hasattr(retriever, "run_metadata") else {})
        ),
        "reranker": (
            reranker.run_metadata() if reranker is not None else None
        ),
        **_git_info(),
    }
    with (run_dir / "run_info.json").open("w", encoding="utf-8") as stream:
        json.dump(run_info, stream, ensure_ascii=False, indent=2)
    return run_dir
