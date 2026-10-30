#!/usr/bin/env python3
"""Evaluate five reranker checkpoints on one frozen development handoff."""

import argparse
from pathlib import Path

from drill_ir.experiment.checkpoint_evaluation import evaluate_checkpoints


def _steps(value: str) -> list[int]:
    try:
        steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoint steps must be comma-separated integers") from error
    if len(steps) != 5 or len(set(steps)) != 5 or any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("provide exactly five distinct positive checkpoint steps")
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-size", required=True, type=int, choices=(50, 100))
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--checkpoint-steps", required=True, type=_steps)
    parser.add_argument("--handoff-root", required=True, type=Path)
    parser.add_argument("--warm-start-model", required=True, type=Path)
    parser.add_argument("--corpus-path", required=True, type=Path)
    parser.add_argument("--train-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--devices", default=2, type=int)
    args = parser.parse_args()
    evaluate_checkpoints(
        pool_size=args.pool_size,
        checkpoint_root=args.checkpoint_root,
        checkpoint_steps=args.checkpoint_steps,
        handoff_root=args.handoff_root,
        warm_start_model=args.warm_start_model,
        corpus_path=args.corpus_path,
        train_path=args.train_path,
        output_root=args.output_root,
        devices=args.devices,
    )


if __name__ == "__main__":
    main()
