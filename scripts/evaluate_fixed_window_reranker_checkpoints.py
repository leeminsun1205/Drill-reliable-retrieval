#!/usr/bin/env python3
"""Evaluate a warm start and three fixed-window pointwise-MIL checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


RESULT_FILES = (
    "metrics.json",
    "metrics_by_output_k.json",
    "metrics_by_score_threshold.json",
    "predictions.jsonl",
    "first_stage_predictions.jsonl",
    "reranker_chunk_candidates.jsonl",
    "split_dev_qids.json",
)


def _validate_config(config: dict[str, Any]) -> None:
    retrieval = config["retrieval"]
    hybrid = retrieval["hybrid"]
    chunking = retrieval["dense"]["chunking"]
    reranking = config["reranking"]
    references = hybrid["reference_expansion"]
    contract = {
        "method": config["method"],
        "seed": int(config["seed"]),
        "top_k": int(retrieval["top_k"]),
        "rrf_k": int(hybrid["rrf_k"]),
        "sparse_weight": float(hybrid["sparse_weight"]),
        "dense_weights": [
            (item["name"], float(item["weight"]), int(item["top_k"]))
            for item in hybrid["dense_components"]
        ],
        "chunking": (
            chunking["method"],
            int(chunking["max_tokens"]),
            int(chunking["overlap_tokens"]),
            chunking["tokenizer_model_name"],
        ),
        "reference_enabled": bool(references["enabled"]),
        "same_law_neighbor_expansion": bool(
            hybrid["same_law_neighbor_expansion"]["enabled"]
        ),
        "candidate_source": reranking["candidate_source"],
        "candidate_pool_size": int(reranking["candidate_pool_size"]),
        "retrieved_chunk_depth": int(reranking["retrieved_chunk_depth"]),
        "max_length": int(reranking["max_length"]),
    }
    expected = {
        "method": "hybrid_reranker",
        "seed": 42,
        "top_k": 100,
        "rrf_k": 10,
        "sparse_weight": 0.15,
        "dense_weights": [
            ("vietlegal_harrier", 0.65, 100),
            ("multilingual_e5_large", 0.20, 100),
        ],
        "chunking": ("token_window", 448, 64, "BAAI/bge-m3"),
        "reference_enabled": False,
        "same_law_neighbor_expansion": False,
        "candidate_source": "retrieved_chunks",
        "candidate_pool_size": 50,
        "retrieved_chunk_depth": 100,
        "max_length": 512,
    }
    if contract != expected:
        raise ValueError(f"Config is not the frozen fixed-window treatment: {contract}")


def _validate_checkpoint(path: Path, epoch: int, step: int) -> None:
    manifest_path = path / "checkpoint_manifest.json"
    if not path.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint/manifest: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "objective": "pointwise_mil_bce",
        "mil_passage_sampling": "seeded_cycle",
        "mil_max_passages_per_epoch": 4,
        "epoch_completed": epoch,
        "global_step": step,
        "scheduler_total_epochs": 5,
    }
    mismatch = {key: (value, manifest.get(key)) for key, value in required.items() if manifest.get(key) != value}
    if mismatch:
        raise ValueError(f"Checkpoint {path} does not match epoch {epoch}: {mismatch}")
    if not (path / "config.json").is_file() or not any(path.glob("*.safetensors")):
        raise FileNotFoundError(f"Checkpoint weights/config are incomplete: {path}")
    if not any((path / name).is_file() for name in ("tokenizer.json", "sentencepiece.bpe.model")):
        raise FileNotFoundError(f"Checkpoint tokenizer is missing: {path}")


def _run(command: list[str], env: dict[str, str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    result = subprocess.run(command, env=env, check=False)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_scoreboard(run_dirs: dict[str, Path], output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for label, run_dir in run_dirs.items():
        fixed = json.loads((run_dir / "metrics_by_output_k.json").read_text())[
            "results"
        ]
        threshold = json.loads(
            (run_dir / "metrics_by_score_threshold.json").read_text()
        )["best_by_fallback_k"]["2"]
        for cutoff in ("1", "2", "3", "4", "5"):
            rows.append({"model": label, "policy": f"top_{cutoff}", **fixed[cutoff]})
        rows.append({"model": label, "policy": "dev_threshold_fallback_2", **threshold})
    rows.sort(key=lambda row: (row["policy"], -float(row["f2"])))
    (output_root / "checkpoint_scoreboard.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_root / "checkpoint_scoreboard.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["model", "policy", "precision", "recall", "f2"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    print("\nFrozen-development scoreboard", flush=True)
    for row in rows:
        print(
            f"{row['model']:<24} {row['policy']:<27} "
            f"P={row['precision']:.4f} R={row['recall']:.4f} F2={row['f2']:.4f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-steps", required=True)
    parser.add_argument("--warm-start-model", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--devices", type=int, default=2)
    args = parser.parse_args()

    steps = [int(item.strip()) for item in args.checkpoint_steps.split(",") if item.strip()]
    if len(steps) != 3 or len(set(steps)) != 3 or any(step <= 0 for step in steps):
        raise ValueError("Provide exactly three distinct positive checkpoint steps")
    if args.devices <= 0:
        raise ValueError("devices must be positive")
    for path in (args.corpus_path, args.train_path, args.warm_start_model / "config.json"):
        if not path.exists():
            raise FileNotFoundError(path)

    base_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    _validate_config(base_config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    handoff_dir = args.output_root / "fixed448_pool50_refoff_dev_retrieval"
    handoff_path = handoff_dir / "retrieval_stage_handoff.json"
    handoff_manifest_path = handoff_dir / "retrieval_handoff_manifest.json"

    retrieval_config = json.loads(json.dumps(base_config))
    retrieval_config["experiment_name"] = handoff_dir.name
    retrieval_config["reranking"]["model_name"] = str(args.warm_start_model)
    retrieval_config.setdefault("runtime", {})["stop_after_retrieval_handoff"] = True
    retrieval_config["runtime"]["process_handoff_before_reranking"] = False
    retrieval_config_path = args.output_root / "fixed_dev_retrieval.yaml"
    retrieval_config_path.write_text(
        yaml.safe_dump(retrieval_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    if not handoff_path.is_file() or not handoff_manifest_path.is_file():
        started = time.perf_counter()
        _run(
            [
                sys.executable,
                "scripts/run_experiment.py",
                "--config",
                str(retrieval_config_path),
                "--corpus-path",
                str(args.corpus_path),
                "--train-path",
                str(args.train_path),
                "--output-root",
                str(args.output_root),
                "--devices",
                str(args.devices),
            ],
            dict(os.environ),
        )
        print(f"Retrieval-only stage completed in {time.perf_counter() - started:.1f}s")
    if not handoff_path.is_file() or not handoff_manifest_path.is_file():
        raise RuntimeError(f"Retrieval-only stage did not produce a frozen handoff in {handoff_dir}")
    handoff_manifest = json.loads(handoff_manifest_path.read_text(encoding="utf-8"))
    actual_hash = _sha256(handoff_path)
    if handoff_manifest.get("handoff_sha256") != actual_hash:
        raise RuntimeError("Frozen development handoff checksum validation failed")
    if handoff_manifest.get("dev_queries") != 438:
        raise RuntimeError(f"Expected 438 dev queries, got {handoff_manifest.get('dev_queries')}")
    if handoff_manifest.get("retrieval_contract", {}).get("weights") != {
        "bm25": 0.15,
        "vietlegal_harrier": 0.65,
        "multilingual_e5_large": 0.2,
    }:
        raise RuntimeError("Frozen handoff weights do not match the selected R3 treatment")

    cases: list[tuple[str, Path, int | None, int | None]] = [
        ("warm_start_vietnamese", args.warm_start_model, None, None)
    ]
    for epoch, step in enumerate(steps, start=1):
        checkpoint = args.checkpoint_root / f"checkpoint-{step}"
        _validate_checkpoint(checkpoint, epoch, step)
        cases.append((f"adaptation_epoch_{epoch}", checkpoint, epoch, step))

    run_dirs: dict[str, Path] = {}
    provenance = {
        "selection": "fixed_window448_pool50_refoff_checkpoint_evaluation",
        "seed": 42,
        "dev_queries": 438,
        "retrieval_handoff": str(handoff_path),
        "retrieval_handoff_sha256": actual_hash,
        "retrieval_contract": handoff_manifest["retrieval_contract"],
        "checkpoint_root": str(args.checkpoint_root),
        "checkpoint_steps": steps,
        "warm_start_model": str(args.warm_start_model),
        "corpus_path": str(args.corpus_path),
        "train_path": str(args.train_path),
        "devices": args.devices,
        "cases": {},
        "threshold_note": "Dev-set threshold metrics are descriptive and tuned on dev; do not interpret as held-out estimates.",
    }
    run_root = args.output_root / "checkpoint_runs"
    config_root = args.output_root / "configs"
    config_root.mkdir(exist_ok=True)
    for position, (label, model_path, epoch, step) in enumerate(cases, start=1):
        run_dir = run_root / f"fixed448_pool50_{label}"
        run_dirs[label] = run_dir
        provenance["cases"][label] = {
            "model_path": str(model_path),
            "epoch": epoch,
            "global_step": step,
        }
        if all((run_dir / name).is_file() for name in RESULT_FILES):
            print(f"[checkpoint-eval] reusing complete {label}: {run_dir}", flush=True)
            continue
        case_config = json.loads(json.dumps(base_config))
        case_config["experiment_name"] = run_dir.name
        case_config["reranking"]["model_name"] = str(model_path)
        case_config["reranking"]["batching"] = "packed_stream"
        case_config["reranking"]["batch_size"] = 64
        case_config.setdefault("runtime", {})["stop_after_retrieval_handoff"] = False
        case_config["runtime"]["process_handoff_before_reranking"] = False
        case_config_path = config_root / f"{label}.yaml"
        case_config_path.write_text(
            yaml.safe_dump(case_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "scripts/run_experiment.py",
            "--config",
            str(case_config_path),
            "--corpus-path",
            str(args.corpus_path),
            "--train-path",
            str(args.train_path),
            "--output-root",
            str(run_root),
            "--devices",
            str(args.devices),
        ]
        environment = dict(os.environ)
        environment["DRILL_RETRIEVAL_HANDOFF"] = str(handoff_path)
        started = time.perf_counter()
        print(f"\n[checkpoint-eval] case_start {position}/{len(cases)} {label}", flush=True)
        _run(command, environment)
        missing = [name for name in RESULT_FILES if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Incomplete evaluation for {label}: {missing}")
        provenance["cases"][label]["evaluation_seconds"] = round(time.perf_counter() - started, 3)
        print(f"[checkpoint-eval] case_complete {label}", flush=True)

    (args.output_root / "evaluation_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_scoreboard(run_dirs, args.output_root)


if __name__ == "__main__":
    main()
