"""Sequential evaluation of reranker checkpoints on one frozen retrieval handoff."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml


REQUIRED_RESULTS = (
    "metrics.json",
    "metrics_by_output_k.json",
    "metrics_by_score_threshold.json",
    "predictions.jsonl",
    "reranker_chunk_candidates.jsonl",
    "split_dev_qids.json",
)


@dataclass(frozen=True)
class EvaluationCase:
    label: str
    model_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contract_from_config(config: dict[str, Any]) -> dict[str, Any]:
    hybrid = config["retrieval"]["hybrid"]
    reference = hybrid["reference_expansion"]
    reranking = config["reranking"]
    return {
        "rrf_k": hybrid["rrf_k"],
        "weights": {
            "bm25": hybrid["sparse_weight"],
            **{item["name"]: item["weight"] for item in hybrid["dense_components"]},
        },
        "candidate_budget": reranking["candidate_pool_size"],
        "retrieved_chunk_depth": reranking["retrieved_chunk_depth"],
        "reference_enabled": reference["enabled"],
        "reference_scopes": reference["scopes"],
        "reference_seed_top_k": reference["seed_top_k"],
        "reference_max_expansions": reference["max_expansions"],
        "same_law_neighbor_expansion": hybrid["same_law_neighbor_expansion"]["enabled"],
        "candidate_source": reranking["candidate_source"],
        "chunk_candidate_selection": hybrid["chunk_candidate_selection"],
    }


def expected_contract(pool_size: int) -> dict[str, Any]:
    if pool_size not in (50, 100):
        raise ValueError(f"Supported candidate pools are 50 and 100, found {pool_size}")
    return {
        "rrf_k": 10,
        "weights": {
            "bm25": 0.15,
            "vietlegal_harrier": 0.55,
            "multilingual_e5_large": 0.3,
        },
        "candidate_budget": pool_size,
        "retrieved_chunk_depth": 100,
        "reference_enabled": True,
        "reference_scopes": ["same_document"],
        "reference_seed_top_k": 5,
        "reference_max_expansions": 4,
        "same_law_neighbor_expansion": False,
        "candidate_source": "retrieved_chunks",
        "chunk_candidate_selection": "top_chunks",
    }


def _validate_model_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing model directory for {label}: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json for {label}: {path}")
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"Missing model weights for {label}: {path}")
    tokenizer_markers = (
        "tokenizer_config.json",
        "tokenizer.json",
        "sentencepiece.bpe.model",
        "spiece.model",
    )
    if not any((path / name).is_file() for name in tokenizer_markers):
        raise FileNotFoundError(
            f"Checkpoint {label} has no tokenizer files. Training must save the tokenizer "
            f"beside the weights: {path}"
        )


def load_and_validate_handoff(
    handoff_root: Path, pool_size: int
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest_path = handoff_root / "retrieval_handoff_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_experiment = f"frozen_dev_handoff_pool{pool_size}_refon_articlemilv1"
    if manifest.get("experiment_name") != expected_experiment:
        raise RuntimeError(
            f"Handoff experiment mismatch: expected {expected_experiment!r}, "
            f"found {manifest.get('experiment_name')!r}"
        )
    if manifest.get("schema_version") not in (1, 2):
        raise RuntimeError(f"Unsupported handoff schema: {manifest.get('schema_version')}")
    if manifest.get("dev_queries") != 438:
        raise RuntimeError(f"Expected 438 dev queries, found {manifest.get('dev_queries')}")

    config_path = handoff_root / manifest.get("config_file", "config.yaml")
    handoff_path = handoff_root / manifest["handoff_file"]
    if not config_path.is_file() or not handoff_path.is_file():
        raise FileNotFoundError(
            f"Handoff dataset is incomplete: config={config_path}, handoff={handoff_path}"
        )
    actual_digest = _sha256(handoff_path)
    if actual_digest != manifest.get("handoff_sha256"):
        raise RuntimeError(
            f"Handoff SHA-256 mismatch: expected {manifest.get('handoff_sha256')}, "
            f"found {actual_digest}"
        )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = expected_contract(pool_size)
    actual = _contract_from_config(config)
    if actual != expected:
        diff = {
            key: {"expected": expected.get(key), "actual": actual.get(key)}
            for key in sorted(set(expected) | set(actual))
            if expected.get(key) != actual.get(key)
        }
        raise RuntimeError(f"Frozen retrieval contract mismatch: {diff}")
    manifest_contract = manifest.get("retrieval_contract")
    if manifest_contract is not None and manifest_contract != expected:
        raise RuntimeError(
            "Manifest retrieval_contract disagrees with its verified config.yaml: "
            f"{manifest_contract}"
        )
    return config, handoff_path, manifest


def build_cases(
    warm_start_model: Path,
    checkpoint_root: Path,
    checkpoint_steps: Sequence[int],
) -> list[EvaluationCase]:
    if len(checkpoint_steps) != 5 or len(set(checkpoint_steps)) != 5:
        raise ValueError("Exactly five distinct checkpoint steps are required")
    cases = [EvaluationCase("warm_start_vietnamese", warm_start_model)]
    cases.extend(
        EvaluationCase(f"adaptation_epoch_{epoch}", checkpoint_root / f"checkpoint-{step}")
        for epoch, step in enumerate(checkpoint_steps, start=1)
    )
    for case in cases:
        _validate_model_directory(case.model_path, case.label)
    return cases


def _complete(run_dir: Path) -> bool:
    return all((run_dir / name).is_file() for name in REQUIRED_RESULTS)


def _stream(command: list[str], env: dict[str, str]) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _write_scoreboard(run_dirs: dict[str, Path], output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for label, run_dir in run_dirs.items():
        fixed = json.loads(
            (run_dir / "metrics_by_output_k.json").read_text(encoding="utf-8")
        )["results"]
        threshold = json.loads(
            (run_dir / "metrics_by_score_threshold.json").read_text(encoding="utf-8")
        )["best_by_fallback_k"]["2"]
        for cutoff in ("1", "2", "3"):
            rows.append({"model": label, "policy": f"top_{cutoff}", **fixed[cutoff]})
        rows.append({"model": label, "policy": "threshold_fallback_2", **threshold})
    rows.sort(key=lambda row: (row["policy"], -float(row["f2"])))
    (output_root / "checkpoint_scoreboard.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = ["model", "policy", "precision", "recall", "f2"]
    with (output_root / "checkpoint_scoreboard.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("\n[checkpoint-eval] scoreboard", flush=True)
    for row in rows:
        print(
            f"  {row['model']:<24} {row['policy']:<22} "
            f"P={row['precision']:.6f} R={row['recall']:.6f} F2={row['f2']:.6f}",
            flush=True,
        )


def evaluate_checkpoints(
    *,
    pool_size: int,
    checkpoint_root: Path,
    checkpoint_steps: Sequence[int],
    handoff_root: Path,
    warm_start_model: Path,
    corpus_path: Path,
    train_path: Path,
    output_root: Path,
    devices: int = 2,
) -> None:
    if devices <= 0:
        raise ValueError(f"devices must be positive, found {devices}")
    config, handoff_path, manifest = load_and_validate_handoff(handoff_root, pool_size)
    cases = build_cases(warm_start_model, checkpoint_root, checkpoint_steps)
    output_root.mkdir(parents=True, exist_ok=True)
    config_root = output_root / "configs"
    config_root.mkdir(exist_ok=True)
    run_dirs: dict[str, Path] = {}
    provenance = {
        "pool_size": pool_size,
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_steps": list(checkpoint_steps),
        "handoff_root": str(handoff_root),
        "handoff_sha256": manifest["handoff_sha256"],
        "warm_start_model": str(warm_start_model),
        "corpus_path": str(corpus_path),
        "train_path": str(train_path),
        "devices": devices,
        "cases": {case.label: str(case.model_path) for case in cases},
    }
    (output_root / "evaluation_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for position, case in enumerate(cases, start=1):
        experiment_name = f"pointwise_mil_pool{pool_size}_{case.label}"
        run_dir = output_root / experiment_name
        run_dirs[case.label] = run_dir
        print(
            f"\n[checkpoint-eval] case_start {position}/{len(cases)} "
            f"label={case.label} model={case.model_path}",
            flush=True,
        )
        if _complete(run_dir):
            print(f"[checkpoint-eval] case_reused label={case.label} dir={run_dir}", flush=True)
            continue
        case_config = json.loads(json.dumps(config))
        case_config["method"] = "hybrid_reranker"
        case_config["experiment_name"] = experiment_name
        case_config["reranking"]["model_name"] = str(case.model_path)
        case_config["reranking"]["batching"] = "packed_stream"
        case_config["reranking"]["batch_size"] = 256
        case_config.setdefault("runtime", {})["process_handoff_before_reranking"] = False
        case_config["runtime"]["stop_after_retrieval_handoff"] = False
        config_path = config_root / f"{case.label}.yaml"
        config_path.write_text(
            yaml.safe_dump(case_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "scripts/run_experiment.py",
            "--config",
            str(config_path),
            "--corpus-path",
            str(corpus_path),
            "--train-path",
            str(train_path),
            "--output-root",
            str(output_root),
            "--devices",
            str(devices),
        ]
        env = dict(os.environ)
        env["DRILL_RETRIEVAL_HANDOFF"] = str(handoff_path)
        started = time.perf_counter()
        _stream(command, env)
        elapsed = time.perf_counter() - started
        missing = [name for name in REQUIRED_RESULTS if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Incomplete evaluation for {case.label}: {missing}")
        print(
            f"[checkpoint-eval] case_complete label={case.label} seconds={elapsed:.1f}; "
            "subprocess exited, model and CUDA state released",
            flush=True,
        )
    _write_scoreboard(run_dirs, output_root)

