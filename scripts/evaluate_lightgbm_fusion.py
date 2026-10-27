#!/usr/bin/env python3
"""Evaluate out-of-fold LightGBM fusion from a saved development run."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.analysis import build_ltr_examples, evaluate_oof_lightgbm
from drill_ir.data import load_queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--adaptive-selection",
        choices=("disabled", "score_gap"),
        default="disabled",
    )
    args = parser.parse_args()

    expected_qids = set(
        json.loads((args.run_dir / "split_dev_qids.json").read_text(encoding="utf-8"))
    )
    queries = [query for query in load_queries(args.queries) if query.qid in expected_qids]
    if {query.qid for query in queries} != expected_qids:
        raise ValueError("Saved dev qids do not match the labeled query dataset")

    examples = build_ltr_examples(args.run_dir, queries)
    result = evaluate_oof_lightgbm(
        examples,
        queries,
        folds=args.folds,
        seed=args.seed,
        include_score_gap=args.adaptive_selection == "score_gap",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rankings = result.pop("rankings")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "oof_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for qid in sorted(rankings):
            aids, scores = rankings[qid]
            stream.write(
                json.dumps(
                    {"qid": qid, "retrieved_aids": aids, "scores": scores},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
