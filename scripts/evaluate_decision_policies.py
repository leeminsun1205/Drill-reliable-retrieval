#!/usr/bin/env python3
"""Screen leakage-aware final-set policies from saved LightGBM OOF rankings."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.analysis import evaluate_nested_decision_policies
from drill_ir.data import load_queries
from drill_ir.evaluation import evaluate_fixed_cutoffs, evaluate_oracle_prefix


def _read_rankings(path: Path) -> dict[int, tuple[list[int], list[float]]]:
    rankings = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = int(row["qid"])
            if qid in rankings:
                raise ValueError(f"Duplicate qid in OOF predictions: {qid}")
            rankings[qid] = (
                list(map(int, row["retrieved_aids"])),
                list(map(float, row["scores"])),
            )
    if not rankings:
        raise ValueError("OOF prediction artifact is empty")
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-predictions", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--dev-qids", required=True, type=Path)
    parser.add_argument(
        "--existing-lightgbm-metrics",
        type=Path,
        help="Optional metrics.json supplying the already nested score-gap baseline",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fallback-k", type=int, default=2)
    parser.add_argument("--maximum-k", type=int, default=5)
    args = parser.parse_args()

    expected_qids = set(map(int, json.loads(args.dev_qids.read_text(encoding="utf-8"))))
    queries = [query for query in load_queries(args.queries) if query.qid in expected_qids]
    gold = {query.qid: query.relevant_aids for query in queries}
    rankings = _read_rankings(args.oof_predictions)
    if set(gold) != expected_qids or set(rankings) != expected_qids:
        raise ValueError("Queries, saved dev qids, and OOF predictions must match exactly")

    result = evaluate_nested_decision_policies(
        gold,
        rankings,
        folds=args.folds,
        seed=args.seed,
        fallback_k=args.fallback_k,
        maximum_k=args.maximum_k,
    )
    fixed = evaluate_fixed_cutoffs(
        gold, {qid: values[0] for qid, values in rankings.items()}, range(1, 6)
    )
    result["fixed_output_k"] = {
        str(cutoff): metrics.to_dict() for cutoff, metrics in fixed.items()
    }
    result["oracle_prefix"] = evaluate_oracle_prefix(
        gold, rankings, maximum_k=args.maximum_k
    )
    result["inputs"] = {
        "oof_predictions": str(args.oof_predictions),
        "queries": str(args.queries),
        "dev_qids": str(args.dev_qids),
    }
    if args.existing_lightgbm_metrics:
        existing = json.loads(args.existing_lightgbm_metrics.read_text(encoding="utf-8"))
        result["existing_nested_score_gap"] = existing["score_gap"][
            "nested_oof_estimate"
        ]
        result["inputs"]["existing_lightgbm_metrics"] = str(
            args.existing_lightgbm_metrics
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = result.pop("predictions")
    (args.output_dir / "decision_policy_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "decision_policy_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for policy in sorted(predictions):
            for qid in sorted(predictions[policy]):
                stream.write(
                    json.dumps(
                        {
                            "policy": policy,
                            "qid": qid,
                            "retrieved_aids": predictions[policy][qid],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    rows = []
    for cutoff, metrics in result["fixed_output_k"].items():
        rows.append({"policy": f"fixed_top_{cutoff}", **metrics})
    for policy, metrics in result["policies"].items():
        rows.append(
            {
                "policy": policy,
                **{key: metrics[key] for key in ("precision", "recall", "f2", "queries")},
                "mean_output_size": metrics["output_size"]["mean"],
                "max_output_size": metrics["output_size"]["maximum"],
            }
        )
    if "existing_nested_score_gap" in result:
        metrics = result["existing_nested_score_gap"]
        rows.append(
            {
                "policy": "score_gap_nested_existing",
                **{key: metrics[key] for key in ("precision", "recall", "f2", "queries")},
            }
        )
    oracle = result["oracle_prefix"]
    rows.append(
        {
            "policy": "oracle_prefix_upper_bound",
            **{key: oracle[key] for key in ("precision", "recall", "f2", "queries")},
            "mean_output_size": oracle["output_size"]["mean"],
            "max_output_size": oracle["output_size"]["maximum"],
        }
    )
    with (args.output_dir / "decision_policy_scoreboard.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "policy",
                "precision",
                "recall",
                "f2",
                "queries",
                "mean_output_size",
                "max_output_size",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(sorted(rows, key=lambda row: row["f2"], reverse=True), indent=2))
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
