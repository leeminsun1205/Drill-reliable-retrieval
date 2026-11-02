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
import shutil
from pathlib import Path
from typing import Any

import yaml

from drill_ir.data import load_queries
from drill_ir.experiment.run import deterministic_split


RESULT_FILES = (
    "metrics.json",
    "metrics_by_output_k.json",
    "metrics_by_score_threshold.json",
    "predictions.jsonl",
    "first_stage_predictions.jsonl",
    "reranker_chunk_candidates.jsonl",
    "component_rankings.jsonl",
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


def _validate_existing_handoff(
    handoff_path: Path, *, config: dict[str, Any], train_path: Path
) -> tuple[dict[str, Any], str]:
    source_config_path = handoff_path.parent / "config.yaml"
    if not source_config_path.is_file():
        raise FileNotFoundError(
            f"An attached dev handoff must include sibling config.yaml: {handoff_path}"
        )
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    _validate_config(source_config)
    if int(source_config["seed"]) != int(config["seed"]):
        raise ValueError("Attached handoff uses a different split seed")
    if float(source_config["data"]["dev_fraction"]) != float(config["data"]["dev_fraction"]):
        raise ValueError("Attached handoff uses a different dev fraction")

    payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("chunks") is None:
        raise ValueError(f"Attached file is not a chunk-bearing retrieval-stage handoff: {handoff_path}")
    first_qids = [int(row["qid"]) for row in payload.get("first_stage", [])]
    chunk_qids = [int(row["qid"]) for row in payload["chunks"]]
    queries = load_queries(train_path)
    _, dev_queries = deterministic_split(
        queries,
        dev_fraction=float(config["data"]["dev_fraction"]),
        seed=int(config["seed"]),
    )
    expected_qids = {query.qid for query in dev_queries}
    if len(first_qids) != len(set(first_qids)) or set(first_qids) != expected_qids:
        raise ValueError("Attached handoff first-stage qids do not match the frozen dev split")
    if len(chunk_qids) != len(set(chunk_qids)) or set(chunk_qids) != expected_qids:
        raise ValueError("Attached handoff chunk qids do not match the frozen dev split")

    digest = _sha256(handoff_path)
    source_manifest_path = handoff_path.parent / "retrieval_handoff_manifest.json"
    source_manifest: dict[str, Any] = {}
    if source_manifest_path.is_file():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest.get("handoff_sha256") != digest:
            raise ValueError("Attached handoff checksum does not match its manifest")
        if source_manifest.get("retrieval_contract") != {
            "rrf_k": 10,
            "weights": {
                "bm25": 0.15,
                "vietlegal_harrier": 0.65,
                "multilingual_e5_large": 0.2,
            },
            "candidate_budget": 50,
            "retrieved_chunk_depth": 100,
            "reference_enabled": False,
            "reference_scopes": ["same_document"],
            "reference_seed_top_k": 5,
            "reference_max_expansions": 4,
            "same_law_neighbor_expansion": False,
            "candidate_source": "retrieved_chunks",
            "chunk_candidate_selection": "top_chunks",
        }:
            raise ValueError("Attached handoff manifest has an incompatible retrieval contract")
    else:
        source_manifest = {
            "experiment_name": source_config.get("experiment_name"),
            "dev_queries": len(expected_qids),
            "retrieval_contract": {
                "rrf_k": 10,
                "weights": {
                    "bm25": 0.15,
                    "vietlegal_harrier": 0.65,
                    "multilingual_e5_large": 0.2,
                },
                "candidate_budget": 50,
                "retrieved_chunk_depth": 100,
                "reference_enabled": False,
                "reference_scopes": ["same_document"],
                "reference_seed_top_k": 5,
                "reference_max_expansions": 4,
                "same_law_neighbor_expansion": False,
                "candidate_source": "retrieved_chunks",
                "chunk_candidate_selection": "top_chunks",
            },
        }
    return source_manifest, digest


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
    parser.add_argument("--handoff-path", type=Path)
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
    handoff_origin = "rebuilt_once"
    if args.handoff_path is not None:
        source_manifest, actual_hash = _validate_existing_handoff(
            args.handoff_path, config=base_config, train_path=args.train_path
        )
        handoff_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.handoff_path, handoff_path)
        shutil.copy2(args.handoff_path.parent / "config.yaml", handoff_dir / "config.yaml")
        handoff_manifest = {
            "schema_version": 2,
            "selection": "reused_compatible_development_retrieval_handoff",
            "handoff_file": handoff_path.name,
            "handoff_sha256": actual_hash,
            "experiment_name": source_manifest.get("experiment_name"),
            "seed": int(base_config["seed"]),
            "dev_queries": 438,
            "first_stage_rows": 438,
            "chunk_rows": 438,
            "retrieval_contract": source_manifest["retrieval_contract"],
            "config_file": "config.yaml",
            "source_handoff": str(args.handoff_path),
        }
        handoff_manifest_path.write_text(
            json.dumps(handoff_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        handoff_origin = "attached_compatible_dev_handoff"
    else:
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
        _validate_config(yaml.safe_load((handoff_dir / "config.yaml").read_text(encoding="utf-8")))
        expected_contract = {
            "rrf_k": 10,
            "weights": {
                "bm25": 0.15,
                "vietlegal_harrier": 0.65,
                "multilingual_e5_large": 0.2,
            },
            "candidate_budget": 50,
            "retrieved_chunk_depth": 100,
            "reference_enabled": False,
            "reference_scopes": ["same_document"],
            "reference_seed_top_k": 5,
            "reference_max_expansions": 4,
            "same_law_neighbor_expansion": False,
            "candidate_source": "retrieved_chunks",
            "chunk_candidate_selection": "top_chunks",
        }
        if handoff_manifest.get("retrieval_contract") != expected_contract:
            raise RuntimeError("Frozen handoff retrieval contract does not match the selected R3 treatment")

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
        "retrieval_handoff_origin": handoff_origin,
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
        if epoch is not None:
            checkpoint_manifest = json.loads(
                (model_path / "checkpoint_manifest.json").read_text(encoding="utf-8")
            )
            provenance["cases"][label]["checkpoint_manifest"] = checkpoint_manifest
            manifest_dir = args.output_root / "checkpoint_manifests"
            manifest_dir.mkdir(exist_ok=True)
            (manifest_dir / f"{label}.json").write_text(
                json.dumps(checkpoint_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if all((run_dir / name).is_file() for name in RESULT_FILES):
            print(f"[checkpoint-eval] reusing complete {label}: {run_dir}", flush=True)
        else:
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

        ltr_dir = args.output_root / "lightgbm_dev" / label
        ltr_required = (
            "enhanced_v2_dev_feature_rows.npz",
            "feature_schema.json",
            "query_fold_assignment.json",
            "oof_predictions.jsonl",
            "oof_metrics.json",
            "development_selection_diagnostics.json",
            "lightgbm_final_model.txt",
            "provenance.json",
        )
        if not all((ltr_dir / name).is_file() for name in ltr_required):
            ltr_dir.mkdir(parents=True, exist_ok=True)
            ltr_command = [
                sys.executable,
                "scripts/evaluate_lightgbm_dev_artifacts.py",
                "--run-dir",
                str(run_dir),
                "--queries",
                str(args.train_path),
                "--output-dir",
                str(ltr_dir),
                "--folds",
                "5",
                "--seed",
                "42",
                "--score-gap-maximum-cutoffs",
                "3",
                "4",
                "5",
            ]
            ltr_started = time.perf_counter()
            print(f"[checkpoint-eval] lightgbm_start label={label}", flush=True)
            _run(ltr_command, dict(os.environ))
            missing_ltr = [name for name in ltr_required if not (ltr_dir / name).is_file()]
            if missing_ltr:
                raise RuntimeError(f"Incomplete LightGBM dev artifacts for {label}: {missing_ltr}")
            provenance["cases"][label]["lightgbm_seconds"] = round(
                time.perf_counter() - ltr_started, 3
            )
            print(f"[checkpoint-eval] lightgbm_complete label={label}", flush=True)
        else:
            print(f"[checkpoint-eval] reusing LightGBM artifacts label={label}", flush=True)

    (args.output_root / "evaluation_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_scoreboard(run_dirs, args.output_root)
    _write_lightgbm_scoreboard(
        {label: args.output_root / "lightgbm_dev" / label for label, _, _, _ in cases},
        args.output_root,
    )


def _write_lightgbm_scoreboard(ltr_dirs: dict[str, Path], output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for label, ltr_dir in ltr_dirs.items():
        metrics = json.loads((ltr_dir / "oof_metrics.json").read_text(encoding="utf-8"))
        rows.extend(
            {"model": label, "policy": f"lightgbm_top_{cutoff}", **value}
            for cutoff, value in metrics["fixed_output_k"].items()
            if cutoff in {"1", "2", "3", "4", "5"}
        )
        threshold = metrics["threshold"]["best_by_fallback_k"]["2"]
        rows.append({"model": label, "policy": "lightgbm_dev_threshold_fallback_2", **threshold})
        score_gap = metrics["score_gap"]
        rows.append(
            {
                "model": label,
                "policy": "lightgbm_pooled_oof_score_gap_dev_tuned",
                **score_gap["pooled_oof_tuning_for_deployment"]["best"],
            }
        )
        rows.append(
            {
                "model": label,
                "policy": "lightgbm_nested_oof_score_gap",
                **{
                    key: score_gap["nested_oof_estimate"][key]
                    for key in ("precision", "recall", "f2")
                },
            }
        )
    rows.sort(key=lambda row: (row["policy"], -float(row["f2"])))
    (output_root / "lightgbm_checkpoint_scoreboard.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_root / "lightgbm_checkpoint_scoreboard.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["model", "policy", "precision", "recall", "f2"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
