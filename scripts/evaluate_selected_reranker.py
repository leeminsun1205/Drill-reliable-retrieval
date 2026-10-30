#!/usr/bin/env python3
"""Evaluate one selected reranker checkpoint on a verified frozen handoff."""

import argparse
from pathlib import Path

from drill_ir.experiment.checkpoint_evaluation import evaluate_selected_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-size", required=True, type=int, choices=(50, 100))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--handoff-root", required=True, type=Path)
    parser.add_argument("--corpus-path", required=True, type=Path)
    parser.add_argument("--train-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--devices", default=2, type=int)
    args = parser.parse_args()
    run_dir = evaluate_selected_checkpoint(
        pool_size=args.pool_size,
        model_path=args.model_path,
        label=args.label,
        handoff_root=args.handoff_root,
        corpus_path=args.corpus_path,
        train_path=args.train_path,
        output_root=args.output_root,
        devices=args.devices,
    )
    print(f"Selected reranker artifacts: {run_dir}")


if __name__ == "__main__":
    main()
