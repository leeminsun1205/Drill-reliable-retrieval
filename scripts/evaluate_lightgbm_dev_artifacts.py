#!/usr/bin/env python3
"""Materialize reusable enhanced-v2 LightGBM development artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm
from lightgbm import LGBMClassifier

from drill_ir.analysis import build_ltr_examples, evaluate_oof_lightgbm
from drill_ir.analysis.ltr import LIGHTGBM_PARAMETERS
from drill_ir.data import load_queries


REQUIRED_RUN_FILES = (
    "predictions.jsonl",
    "first_stage_predictions.jsonl",
    "component_rankings.jsonl",
    "reranker_chunk_candidates.jsonl",
    "split_dev_qids.json",
    "config.yaml",
    "run_info.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_rankings(path: Path, rankings: dict[int, tuple[list[int], list[float]]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for qid in sorted(rankings):
            aids, scores = rankings[qid]
            stream.write(
                json.dumps(
                    {"qid": qid, "retrieved_aids": aids, "scores": scores},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-gap-maximum-cutoffs", type=int, nargs="+", default=(3, 4, 5))
    args = parser.parse_args()

    missing = [name for name in REQUIRED_RUN_FILES if not (args.run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete reranker run {args.run_dir}: {missing}")
    if args.folds < 2:
        raise ValueError("folds must be at least 2")

    expected_qids = set(
        map(int, json.loads((args.run_dir / "split_dev_qids.json").read_text()))
    )
    queries = [query for query in load_queries(args.queries) if query.qid in expected_qids]
    if {query.qid for query in queries} != expected_qids:
        raise ValueError("Saved dev qids do not match labeled training queries")

    examples = build_ltr_examples(
        args.run_dir,
        queries,
        feature_schema="enhanced_v2",
    )
    if len(examples["feature_names"]) != 61:
        raise ValueError(
            f"Expected the frozen enhanced_v2 61-feature schema; got {len(examples['feature_names'])}"
        )
    result = evaluate_oof_lightgbm(
        examples,
        queries,
        folds=args.folds,
        seed=args.seed,
        include_score_gap=True,
        score_gap_maximum_cutoffs=args.score_gap_maximum_cutoffs,
        progress_label=args.run_dir.name,
    )
    rankings = result.pop("rankings")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    row_score_by_qid_aid: dict[int, dict[int, float]] = {}
    for qid, (aids, scores) in rankings.items():
        row_score_by_qid_aid[int(qid)] = {
            int(aid): float(score) for aid, score in zip(aids, scores)
        }
    oof_probabilities = np.asarray(
        [row_score_by_qid_aid[int(qid)][int(aid)] for qid, aid in zip(examples["qids"], examples["aids"])],
        dtype=np.float32,
    )

    feature_archive = args.output_dir / "enhanced_v2_dev_feature_rows.npz"
    np.savez_compressed(
        feature_archive,
        features=examples["features"],
        labels=examples["labels"],
        qids=examples["qids"],
        aids=examples["aids"],
        original_ranks=examples["original_ranks"],
        oof_probabilities=oof_probabilities,
    )
    (args.output_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "feature_schema": examples["feature_schema"],
                "feature_count": len(examples["feature_names"]),
                "feature_names": list(examples["feature_names"]),
                "rows": len(examples["labels"]),
                "positive_rows": int(examples["labels"].sum()),
                "queries": len(expected_qids),
                "archive": feature_archive.name,
                "array_contract": {
                    "features": "float64 [rows, features] (same precision used by OOF fitting)",
                    "labels": "int8 [rows]",
                    "qids": "int64 [rows]",
                    "aids": "int64 [rows]",
                    "original_ranks": "int64 [rows]",
                    "oof_probabilities": "float32 [rows], aligned to feature rows",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    ordered_qids = sorted(expected_qids)
    shuffled_qids = ordered_qids.copy()
    random.Random(args.seed).shuffle(shuffled_qids)
    fold_by_qid = {qid: index % args.folds for index, qid in enumerate(shuffled_qids)}
    (args.output_dir / "query_fold_assignment.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "folds": args.folds,
                "assignment": {str(qid): fold_by_qid[qid] for qid in ordered_qids},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_rankings(args.output_dir / "oof_predictions.jsonl", rankings)
    (args.output_dir / "oof_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    threshold = result["threshold"]["best_by_fallback_k"]["2"]
    score_gap = result["score_gap"]["pooled_oof_tuning_for_deployment"]["best"]
    (args.output_dir / "development_selection_diagnostics.json").write_text(
        json.dumps(
            {
                "top2": result["fixed_output_k"]["2"],
                "pooled_oof_threshold_fallback2": threshold,
                "pooled_oof_score_gap": score_gap,
                "nested_oof_score_gap": result["score_gap"]["nested_oof_estimate"],
                "note": "Threshold and pooled score-gap values are dev-tuned diagnostics; nested score-gap is the fold-safe selector estimate.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    final_model = LGBMClassifier(**LIGHTGBM_PARAMETERS, random_state=args.seed)
    final_model.fit(examples["features"], examples["labels"])
    final_model.booster_.save_model(str(args.output_dir / "lightgbm_final_model.txt"))

    input_hashes = {
        name: _sha256(args.run_dir / name)
        for name in REQUIRED_RUN_FILES
        if name.endswith(".jsonl") or name == "split_dev_qids.json"
    }
    provenance: dict[str, Any] = {
        "stage": "development_lightgbm_oof_and_refit",
        "source_run_dir": str(args.run_dir),
        "feature_schema": examples["feature_schema"],
        "feature_count": len(examples["feature_names"]),
        "queries": len(expected_qids),
        "candidate_rows": len(examples["labels"]),
        "positive_rows": int(examples["labels"].sum()),
        "folds": args.folds,
        "seed": args.seed,
        "parameters": {**LIGHTGBM_PARAMETERS, "random_state": "seed + fold for OOF; seed for final fit"},
        "lightgbm_version": lightgbm.__version__,
        "python_version": platform.python_version(),
        "input_sha256": input_hashes,
        "feature_rows_sha256": _sha256(feature_archive),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir),
                "queries": len(expected_qids),
                "candidate_rows": len(examples["labels"]),
                "feature_count": len(examples["feature_names"]),
                "top2": result["fixed_output_k"]["2"],
                "nested_oof_score_gap": {
                    key: result["score_gap"]["nested_oof_estimate"][key]
                    for key in ("precision", "recall", "f2")
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
