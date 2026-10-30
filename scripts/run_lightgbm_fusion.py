#!/usr/bin/env python3
"""Evaluate OOF LightGBM fusion, fit dev labels, and package test submissions."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm

from drill_ir.analysis import (
    build_ltr_examples,
    evaluate_oof_lightgbm,
    fit_lightgbm_and_rank,
)
from drill_ir.data import load_queries
from drill_ir.experiment import (
    package_score_gap_submission,
    package_submission,
    package_threshold_submission,
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
            stream.write(json.dumps({"qid": qid, "retrieved_aids": aids, "scores": scores}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-run-dir", required=True, type=Path)
    parser.add_argument("--test-run-dir", required=True, type=Path)
    parser.add_argument("--train-queries", required=True, type=Path)
    parser.add_argument("--public-test", required=True, type=Path)
    parser.add_argument("--private-test", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-schema",
        choices=("base_v1", "enhanced_v1", "enhanced_v2", "enhanced_v3"),
        default="enhanced_v3",
        help=(
            "base_v1 is the historical base (15 features with two dense models); "
            "v2 adds persisted chunk evidence; v3 also requires static article features"
        ),
    )
    parser.add_argument(
        "--static-article-features",
        type=Path,
        help="JSONL made by scripts/build_ltr_static_features.py; required for enhanced_v3",
    )
    parser.add_argument(
        "--adaptive-selection",
        choices=("disabled", "score_gap"),
        default="disabled",
        help="Optionally package a score-gap submission in addition to unchanged baselines",
    )
    args = parser.parse_args()

    dev_qids_path = args.dev_run_dir / "split_dev_qids.json"
    expected_qids = set(json.loads(dev_qids_path.read_text(encoding="utf-8")))
    dev_queries = [q for q in load_queries(args.train_queries) if q.qid in expected_qids]
    if {q.qid for q in dev_queries} != expected_qids:
        raise ValueError("Saved dev qids do not match the labeled training dataset")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    dev_examples = build_ltr_examples(
        args.dev_run_dir,
        dev_queries,
        feature_schema=args.feature_schema,
        static_article_features=args.static_article_features,
    )
    oof = evaluate_oof_lightgbm(
        dev_examples,
        dev_queries,
        folds=args.folds,
        seed=args.seed,
        include_score_gap=args.adaptive_selection == "score_gap",
    )
    oof_rankings = oof.pop("rankings")
    (output / "oof_metrics.json").write_text(
        json.dumps(oof, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_rankings(output / "oof_predictions.jsonl", oof_rankings)

    selected = oof["threshold"]["best_by_fallback_k"]["2"]
    threshold = selected["threshold"]
    if threshold is None:
        raise ValueError("OOF tuning selected fallback-only output; no threshold to transfer")
    score_gap_policy = (
        oof["score_gap"]["pooled_oof_tuning_for_deployment"]["best"]
        if args.adaptive_selection == "score_gap"
        else None
    )

    test_sets = {
        "public": (args.public_test, "public_"),
        "private": (args.private_test, "private_"),
    }
    fitted_model = None
    summaries = {}
    for name, (query_path, prefix) in test_sets.items():
        queries = load_queries(query_path)
        examples = build_ltr_examples(
            args.test_run_dir,
            queries,
            prefix=prefix,
            feature_schema=args.feature_schema,
            static_article_features=args.static_article_features,
        )
        rankings, fitted_model = fit_lightgbm_and_rank(
            dev_examples, examples, seed=args.seed
        )
        predictions = output / f"{name}_lightgbm_predictions.jsonl"
        _write_rankings(predictions, rankings)
        expected = {query.qid for query in queries}
        package_submission(
            predictions, expected, output / f"{name}_lightgbm_top2_results.zip", output_k=2
        )
        package_threshold_submission(
            predictions,
            expected,
            output / f"{name}_lightgbm_threshold_results.zip",
            score_threshold=float(threshold),
            fallback_k=2,
        )
        if args.adaptive_selection == "score_gap":
            assert score_gap_policy is not None
            package_score_gap_submission(
                predictions,
                expected,
                output / f"{name}_lightgbm_score_gap_results.zip",
                minimum_gap=float(score_gap_policy["minimum_gap"]),
                fallback_k=int(score_gap_policy["fallback_k"]),
                maximum_k=int(score_gap_policy["maximum_k"]),
            )
        summaries[name] = {"queries": len(queries), "candidate_rows": len(examples["labels"])}

    assert fitted_model is not None
    fitted_model.booster_.save_model(str(output / "lightgbm_final_model.txt"))
    inputs = {}
    for path in sorted(args.dev_run_dir.glob("*.jsonl")):
        inputs[str(path)] = _sha256(path)
    if args.static_article_features is not None:
        inputs[str(args.static_article_features)] = _sha256(
            args.static_article_features
        )
    provenance = {
        "protocol": "query-grouped OOF evaluation; final fit on all labeled dev candidates",
        "seed": args.seed,
        "folds": args.folds,
        "python": platform.python_version(),
        "lightgbm": lightgbm.__version__,
        "dev_run_dir": str(args.dev_run_dir),
        "test_run_dir": str(args.test_run_dir),
        "feature_names": list(dev_examples["feature_names"]),
        "feature_schema": args.feature_schema,
        "development": {
            "queries": len(dev_queries),
            "candidate_rows": len(dev_examples["labels"]),
            "positive_rows": int(dev_examples["labels"].sum()),
            "negative_rows": int((dev_examples["labels"] == 0).sum()),
        },
        "test_sets": summaries,
        "selected_oof_policy": selected,
        "adaptive_selection": {
            "mode": args.adaptive_selection,
            "selected_oof_policy": (
                score_gap_policy if args.adaptive_selection == "score_gap" else None
            ),
            "baseline_outputs_preserved": True,
        },
        "input_sha256": inputs,
        "note": "The pooled OOF threshold is a development-selected diagnostic.",
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"oof": oof, "provenance": provenance}, ensure_ascii=False, indent=2))
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
