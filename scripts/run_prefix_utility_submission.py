#!/usr/bin/env python3
"""Fit fold-safe listwise prefix utility and package public/private outputs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.analysis import (  # noqa: E402
    apply_prefix_utility_policy,
    evaluate_nested_prefix_utility,
    fit_prefix_utility_model,
)
from drill_ir.data import load_queries  # noqa: E402
from drill_ir.experiment import package_submission  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rankings(path: Path) -> dict[int, tuple[list[int], list[float]]]:
    rankings = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = int(row["qid"])
            if qid in rankings:
                raise ValueError(f"Duplicate qid in rankings: {qid}")
            rankings[qid] = (
                list(map(int, row["retrieved_aids"])),
                list(map(float, row["scores"])),
            )
    if not rankings:
        raise ValueError(f"Empty ranking file: {path}")
    return rankings


def _write_selected(
    path: Path,
    selected: dict[int, list[int]],
    diagnostics: dict[int, dict],
) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for qid in sorted(selected):
            stream.write(
                json.dumps(
                    {
                        "qid": qid,
                        "retrieved_aids": selected[qid],
                        **diagnostics[qid],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", required=True, type=Path)
    parser.add_argument("--dev-oof-predictions", required=True, type=Path)
    parser.add_argument("--train-queries", required=True, type=Path)
    parser.add_argument("--dev-qids", required=True, type=Path)
    parser.add_argument("--public-predictions", required=True, type=Path)
    parser.add_argument("--private-predictions", required=True, type=Path)
    parser.add_argument("--public-test", required=True, type=Path)
    parser.add_argument("--private-test", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.policy_config.read_text(encoding="utf-8"))
    expected = {
        "policy": "fold_safe_listwise_prefix_utility",
        "target": "per_query_prefix_f2",
        "tie_break": "smaller_k",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Policy config mismatch for {key}: {config.get(key)!r}")
    if config.get("estimator", {}).get("type") != "ridge":
        raise ValueError("Only the predeclared Ridge estimator is supported")
    folds = int(config["folds"])
    seed = int(config["seed"])
    maximum_k = int(config["maximum_k"])
    alpha = float(config["estimator"]["alpha"])

    expected_dev = set(map(int, json.loads(args.dev_qids.read_text(encoding="utf-8"))))
    dev_queries = [
        query for query in load_queries(args.train_queries) if query.qid in expected_dev
    ]
    gold = {query.qid: query.relevant_aids for query in dev_queries}
    dev_rankings = _read_rankings(args.dev_oof_predictions)
    if set(gold) != expected_dev or set(dev_rankings) != expected_dev:
        raise ValueError("Development queries, qids, and OOF rankings must match")

    nested = evaluate_nested_prefix_utility(
        gold,
        dev_rankings,
        folds=folds,
        seed=seed,
        maximum_k=maximum_k,
        alpha=alpha,
    )
    model = fit_prefix_utility_model(
        dev_rankings, gold, expected_dev, maximum_k=maximum_k, alpha=alpha
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "nested_oof_metrics.json").write_text(
        json.dumps(nested, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "prefix_utility_model.json").write_text(
        json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    inputs = {
        "policy_config": args.policy_config,
        "dev_oof_predictions": args.dev_oof_predictions,
        "train_queries": args.train_queries,
        "dev_qids": args.dev_qids,
        "public_predictions": args.public_predictions,
        "private_predictions": args.private_predictions,
        "public_test": args.public_test,
        "private_test": args.private_test,
    }
    for split, prediction_path, query_path in (
        ("public", args.public_predictions, args.public_test),
        ("private", args.private_predictions, args.private_test),
    ):
        rankings = _read_rankings(prediction_path)
        expected_qids = {query.qid for query in load_queries(query_path)}
        if set(rankings) != expected_qids:
            raise ValueError(f"{split} ranking qids do not match its query file")
        selected, diagnostics = apply_prefix_utility_policy(model, rankings)
        selected_path = args.output_dir / f"{split}_prefix_utility_predictions.jsonl"
        _write_selected(selected_path, selected, diagnostics)
        package_submission(
            selected_path,
            expected_qids,
            args.output_dir / f"{split}_lightgbm_prefix_utility_results.zip",
            output_k=maximum_k,
        )

    provenance = {
        "protocol": "query-grouped outer-fold prefix-utility evaluation; fixed Ridge alpha; final model fit on all OOF development rankings",
        "config": config,
        "nested_oof": nested,
        "input_sha256": {str(path): _sha256(path) for path in inputs.values()},
        "private_labels_used": False,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"nested_oof": nested, "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

