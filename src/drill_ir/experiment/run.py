"""Config-driven experiment orchestration."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any

import yaml

from drill_ir.data import iter_articles, load_queries, validate_dataset
from drill_ir.evaluation import evaluate_predictions
from drill_ir.retrieval import BM25Retriever


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


def run_experiment(config: dict[str, Any]) -> Path:
    started = time.perf_counter()
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
    if method == "bm25":
        params = config["retrieval"].get("bm25", {})
        retriever = BM25Retriever(k1=float(params.get("k1", 1.2)), b=float(params.get("b", 0.75)))
    elif method == "organizer_two_stage":
        raise NotImplementedError(
            "The organizer paper does not identify the cross-encoder, candidate depth, or threshold. "
            "Set these explicitly and implement the reranker before running this config."
        )
    else:  # protected by strict config validation
        raise ValueError(f"Unsupported method: {method}")

    retriever.fit(articles)
    retrieve_k = int(config["retrieval"]["top_k"])
    output_k = int(config["retrieval"]["output_k"])
    if output_k > retrieve_k:
        raise ValueError("retrieval.output_k cannot exceed retrieval.top_k")
    ranked = retriever.retrieve(dev_queries, top_k=retrieve_k)
    predictions = {result.qid: result.aids[:output_k] for result in ranked}
    gold = {query.qid: query.relevant_aids for query in dev_queries}
    metrics = evaluate_predictions(gold, predictions)

    run_dir = Path(config["output"]["root_dir"]) / str(config["experiment_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics.to_dict(), stream, ensure_ascii=False, indent=2)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for result in ranked:
            record = {
                "qid": result.qid,
                "retrieved_aids": list(result.aids),
                "scores": list(result.scores),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
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
        **_git_info(),
    }
    with (run_dir / "run_info.json").open("w", encoding="utf-8") as stream:
        json.dump(run_info, stream, ensure_ascii=False, indent=2)
    return run_dir
