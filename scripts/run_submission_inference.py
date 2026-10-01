#!/usr/bin/env python3
"""Infer the public/private DRiLL test sets and create two submission ZIPs."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.experiment import (
    apply_config_overrides,
    load_config,
    run_submission_inference,
    set_config_value,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--corpus-path", required=True, type=Path)
    parser.add_argument("--public-test", required=True, type=Path)
    parser.add_argument("--private-test", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("submission_results"), type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--encode-batch-size", type=int)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--output-k", type=int)
    parser.add_argument(
        "--score-threshold",
        type=float,
        help="Also package variable-size results using score >= threshold",
    )
    parser.add_argument(
        "--score-thresholds",
        type=float,
        nargs="+",
        help=(
            "Package one variable-size submission per threshold from the same "
            "inference pass; mutually exclusive with --score-threshold"
        ),
    )
    parser.add_argument(
        "--threshold-fallback-k",
        type=int,
        default=2,
        help="Use top-k only when no score reaches --score-threshold (default: 2)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override an existing config leaf with a YAML-typed value; repeatable",
    )
    args = parser.parse_args()

    config = apply_config_overrides(load_config(args.config), args.set)
    overrides = {
        "retrieval.dense.encoder.model_name": args.model_name,
        "retrieval.dense.encoder.batch_size": args.encode_batch_size,
        "runtime.devices": args.devices,
        "retrieval.output_k": args.output_k,
    }
    for path, value in overrides.items():
        if value is not None:
            set_config_value(config, path, value)
    config["data"]["corpus_path"] = str(args.corpus_path)
    output_dir = run_submission_inference(
        config,
        {"public": args.public_test, "private": args.private_test},
        args.output_dir,
        score_threshold=args.score_threshold,
        score_thresholds=args.score_thresholds,
        threshold_fallback_k=args.threshold_fallback_k,
    )
    print(f"Artifacts: {output_dir}")
    print(f"Public submission: {output_dir / 'public_results.zip'}")
    print(f"Private submission: {output_dir / 'private_results.zip'}")
    if args.score_threshold is not None:
        print(f"Public threshold submission: {output_dir / 'public_threshold_results.zip'}")
        print(f"Private threshold submission: {output_dir / 'private_threshold_results.zip'}")
    elif args.score_thresholds:
        print("Threshold sweep submissions:")
        for path in sorted(output_dir.glob("*_threshold_*_results.zip")):
            print(path)


if __name__ == "__main__":
    main()
