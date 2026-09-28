#!/usr/bin/env python3
"""Run one configured retrieval experiment."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.experiment import load_config, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--corpus-path", type=Path)
    parser.add_argument("--train-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.corpus_path is not None:
        config["data"]["corpus_path"] = str(args.corpus_path)
    if args.train_path is not None:
        config["data"]["train_path"] = str(args.train_path)
    if args.output_root is not None:
        config["output"]["root_dir"] = str(args.output_root)
    output = run_experiment(config)
    print(output)


if __name__ == "__main__":
    main()
