#!/usr/bin/env python3
"""Evaluate out-of-fold LightGBM fusion from a saved development run."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.analysis import (
    build_ltr_examples,
    evaluate_oof_lightgbm,
    limit_ltr_examples,
)
from drill_ir.data import load_queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-depths",
        nargs="+",
        default=("10", "20", "30", "all"),
        help="Reranker-row depths to compare before LightGBM; use 'all' for no cutoff",
    )
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
        progress_label="all",
    )
    depth_specs: list[tuple[str, int | None]] = []
    for raw_depth in args.input_depths:
        if raw_depth == "all":
            depth_specs.append(("all", None))
        else:
            depth = int(raw_depth)
            if depth <= 0:
                raise ValueError("--input-depths values must be positive or 'all'")
            depth_specs.append((str(depth), depth))
    if len({name for name, _ in depth_specs}) != len(depth_specs):
        raise ValueError("--input-depths contains duplicate depths")

    ablation = {}
    for name, depth in depth_specs:
        limited = limit_ltr_examples(examples, depth)
        if depth is None:
            depth_result = result
        else:
            depth_result = evaluate_oof_lightgbm(
                limited,
                queries,
                folds=args.folds,
                seed=args.seed,
                include_score_gap=args.adaptive_selection == "score_gap",
                progress_label=f"depth_{name}",
            )
        ablation[name] = {
            "maximum_reranker_rank": depth,
            "candidate_rows": len(limited["labels"]),
            "positive_rows": int(limited["labels"].sum()),
            "fixed_output_k": depth_result["fixed_output_k"],
            "threshold": depth_result["threshold"],
            "score_gap": depth_result.get("score_gap"),
        }
    result["lightgbm_input_depth_ablation"] = ablation
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rankings = result.pop("rankings")
    (args.output_dir / "lightgbm_input_depth_ablation.json").write_text(
        json.dumps(ablation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
