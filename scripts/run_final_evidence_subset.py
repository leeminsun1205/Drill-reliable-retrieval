#!/usr/bin/env python3
"""Evaluate final non-prefix evidence selectors over frozen top-five rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.analysis.evidence_subset import (  # noqa: E402
    apply_subset_case,
    evaluate_subset_case,
    oracle_subset_metrics,
)
from drill_ir.data import load_queries  # noqa: E402
from drill_ir.experiment import package_submission  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rankings(path: Path) -> dict[int, tuple[list[int], list[float]]]:
    rankings = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rankings[int(row["qid"])] = (
            list(map(int, row["retrieved_aids"])),
            list(map(float, row["scores"])),
        )
    return rankings


def _write_predictions(
    path: Path,
    selected: dict[int, list[int]],
    diagnostics: dict[int, dict[str, Any]],
) -> None:
    with path.open("w") as stream:
        for qid in sorted(selected):
            row = {"qid": qid, "retrieved_aids": selected[qid], **diagnostics[qid]}
            stream.write(json.dumps(row) + "\n")


def _model_payload(case: str, fitted: dict[str, Any]) -> dict[str, Any]:
    def classifier_payload(model: Any) -> dict[str, Any]:
        payload = {
            "class": type(model).__name__,
            "classes": model.classes_.tolist(),
        }
        if hasattr(model, "coef_"):
            payload["coefficient"] = model.coef_.tolist()
            payload["intercept"] = model.intercept_.tolist()
        if hasattr(model, "constant"):
            payload["constant"] = np.asarray(model.constant).tolist()
        return payload

    if case == "cardinality_binary_subset":
        return {
            "query_scaler": {
                "mean": fitted["query_scaler"].mean_.tolist(),
                "scale": fitted["query_scaler"].scale_.tolist(),
            },
            "multiclass": classifier_payload(fitted["multiclass"]),
            "candidate_scaler": {
                "mean": fitted["candidate_scaler"].mean_.tolist(),
                "scale": fitted["candidate_scaler"].scale_.tolist(),
            },
            "binary_selector": classifier_payload(fitted["binary"]),
        }
    return {
        "feature_scaler": {
            "mean": fitted["scaler"].mean_.tolist(),
            "scale": fitted["scaler"].scale_.tolist(),
        },
        "ridge": {
            "coefficient": np.asarray(fitted["model"].coef_).tolist(),
            "intercept": float(fitted["model"].intercept_),
        },
        "candidate_masks": [mask.astype(int).tolist() for mask in fitted["masks"]],
    }


def _validate_alignment(
    name: str,
    rankings: dict[int, tuple[list[int], list[float]]],
    expected_qids: set[int],
) -> None:
    if set(rankings) != expected_qids:
        missing = sorted(expected_qids - set(rankings))[:5]
        extra = sorted(set(rankings) - expected_qids)[:5]
        raise ValueError(f"{name} alignment failed: missing={missing}, extra={extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config", "dev-oof-predictions", "train-queries", "dev-qids",
        "public-predictions", "private-predictions", "public-test",
        "private-test", "output-dir",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    dev_qids = set(map(int, json.loads(args.dev_qids.read_text())))
    gold = {
        query.qid: query.relevant_aids
        for query in load_queries(args.train_queries)
        if query.qid in dev_qids
    }
    dev_rankings = _load_rankings(args.dev_oof_predictions)
    _validate_alignment("development gold", {qid: ([], []) for qid in gold}, dev_qids)
    _validate_alignment("development rankings", dev_rankings, dev_qids)

    test_inputs = {
        "public": (_load_rankings(args.public_predictions), args.public_test),
        "private": (_load_rankings(args.private_predictions), args.private_test),
    }
    for split, (rankings, query_path) in test_inputs.items():
        _validate_alignment(split, rankings, {query.qid for query in load_queries(query_path)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = [
        args.config, args.dev_oof_predictions, args.train_queries, args.dev_qids,
        args.public_predictions, args.private_predictions, args.public_test,
        args.private_test,
    ]
    maximum_k = int(config["maximum_k"])
    oracle_metrics, oracle_prefix, oracle_subset = oracle_subset_metrics(
        dev_rankings, gold, maximum_k
    )
    (args.output_dir / "oracle_metrics.json").write_text(
        json.dumps(oracle_metrics, indent=2)
    )
    for name, predictions in (("prefix", oracle_prefix), ("subset", oracle_subset)):
        with (args.output_dir / f"oracle_{name}_predictions.jsonl").open("w") as stream:
            for qid in sorted(predictions):
                stream.write(json.dumps({"qid": qid, "retrieved_aids": predictions[qid]}) + "\n")

    summary = []
    for case in config["cases"]:
        grid = (
            list(map(float, config["binary_selector"]["C"]))
            if case == "cardinality_binary_subset"
            else list(map(float, config["subset_utility"]["alpha"]))
        )
        print(f"[evidence-subset] start {case}", flush=True)
        metrics, deployment, fitted, nested_predictions, nested_diagnostics = (
            evaluate_subset_case(
                case,
                dev_rankings,
                gold,
                outer_folds=int(config["outer_folds"]),
                inner_folds=int(config["inner_folds"]),
                seed=int(config["seed"]),
                maximum_k=maximum_k,
                C_multi=float(config["multiclass"]["C"]),
                grid=grid,
            )
        )
        directory = args.output_dir / case
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "nested_oof_metrics.json").write_text(json.dumps(metrics, indent=2))
        (directory / "deployment_selection.json").write_text(json.dumps(deployment, indent=2))
        (directory / "model.json").write_text(json.dumps(_model_payload(case, fitted), indent=2))
        _write_predictions(
            directory / "nested_oof_predictions.jsonl",
            nested_predictions,
            nested_diagnostics,
        )
        for split, (rankings, _) in test_inputs.items():
            selected, diagnostics = apply_subset_case(case, fitted, rankings, maximum_k)
            prediction_path = directory / f"{split}_predictions.jsonl"
            _write_predictions(prediction_path, selected, diagnostics)
            package_submission(
                prediction_path,
                set(rankings),
                directory / f"{split}_{case}_results.zip",
                output_k=maximum_k,
            )
        provenance = {
            "case": case,
            "config": config,
            "nested_oof": metrics,
            "deployment": deployment,
            "input_sha256": {str(path): _sha256(path) for path in input_paths},
            "private_labels_used": False,
        }
        (directory / "provenance.json").write_text(json.dumps(provenance, indent=2))
        summary.append({
            "case": case,
            **{key: metrics[key] for key in ("precision", "recall", "f2")},
            "mean_output_size": metrics["output_size"]["mean"],
            "non_prefix_rate": metrics["non_prefix_rate"],
            "deployment": deployment,
        })
        print(f"[evidence-subset] complete {case} f2={metrics['f2']:.6f}", flush=True)

    summary.sort(key=lambda row: (row["f2"], row["precision"], row["recall"]), reverse=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "selected_by_nested_oof.json").write_text(json.dumps(summary[0], indent=2))
    consolidated = {
        "prior_measured_codabench": config["reported_codabench"],
        "new_nested_oof_cases": summary,
        "oracle_development_upper_bounds": oracle_metrics,
        "selection_rule": "New cases are selected only by nested development OOF; public/private are reporting only.",
    }
    (args.output_dir / "consolidated_policy_results.json").write_text(
        json.dumps(consolidated, indent=2)
    )
    print(json.dumps(consolidated, indent=2))


if __name__ == "__main__":
    main()
