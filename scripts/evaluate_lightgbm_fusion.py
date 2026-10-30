#!/usr/bin/env python3
"""Evaluate out-of-fold LightGBM fusion from a saved development run."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.analysis import (
    build_ltr_examples,
    drop_ltr_features,
    evaluate_oof_lightgbm,
    limit_ltr_examples,
    ltr_feature_groups,
)
from drill_ir.data import load_queries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
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
        "--input-depths",
        nargs="+",
        default=("10", "20", "30", "all"),
        help="Reranker-row depths to compare before LightGBM; use 'all' for no cutoff",
    )
    parser.add_argument(
        "--score-gap-maximum-cutoffs",
        nargs="+",
        type=int,
        default=(3, 4, 5, 6, 7, 8, 9, 10),
        help="Maximum output-prefix sizes considered by score-gap tuning",
    )
    parser.add_argument(
        "--adaptive-selection",
        choices=("disabled", "score_gap"),
        default="disabled",
    )
    parser.add_argument(
        "--feature-group-ablation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run query-grouped OOF after dropping each interpretable feature group",
    )
    args = parser.parse_args()

    expected_qids = set(
        json.loads((args.run_dir / "split_dev_qids.json").read_text(encoding="utf-8"))
    )
    queries = [query for query in load_queries(args.queries) if query.qid in expected_qids]
    if {query.qid for query in queries} != expected_qids:
        raise ValueError("Saved dev qids do not match the labeled query dataset")

    examples = build_ltr_examples(
        args.run_dir,
        queries,
        feature_schema=args.feature_schema,
        static_article_features=args.static_article_features,
    )
    result = evaluate_oof_lightgbm(
        examples,
        queries,
        folds=args.folds,
        seed=args.seed,
        include_score_gap=args.adaptive_selection == "score_gap",
        score_gap_maximum_cutoffs=tuple(args.score_gap_maximum_cutoffs),
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
                score_gap_maximum_cutoffs=tuple(args.score_gap_maximum_cutoffs),
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
    result["feature_schema"] = args.feature_schema
    result["static_article_features"] = (
        {
            "path": str(args.static_article_features),
            "sha256": _sha256(args.static_article_features),
        }
        if args.static_article_features is not None
        else None
    )
    feature_group_ablation = {}
    if args.feature_group_ablation:
        for group, names in ltr_feature_groups(examples["feature_names"]).items():
            reduced = drop_ltr_features(examples, names)
            group_result = evaluate_oof_lightgbm(
                reduced,
                queries,
                folds=args.folds,
                seed=args.seed,
                include_score_gap=args.adaptive_selection == "score_gap",
                score_gap_maximum_cutoffs=tuple(args.score_gap_maximum_cutoffs),
                progress_label=f"drop_{group}",
            )
            group_result.pop("rankings")
            feature_group_ablation[group] = {
                "dropped_features": list(names),
                "remaining_feature_count": len(reduced["feature_names"]),
                "fixed_output_k": group_result["fixed_output_k"],
                "threshold": group_result["threshold"],
                "score_gap": group_result.get("score_gap"),
            }
    result["feature_group_ablation"] = feature_group_ablation
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rankings = result.pop("rankings")
    (args.output_dir / "lightgbm_input_depth_ablation.json").write_text(
        json.dumps(ablation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "feature_group_ablation.json").write_text(
        json.dumps(feature_group_ablation, ensure_ascii=False, indent=2),
        encoding="utf-8",
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
