#!/usr/bin/env python3
"""Mine train-only positive passages and hybrid hard negatives for a reranker."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.experiment import apply_config_overrides, load_config
from drill_ir.reranking import prepare_reranker_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--corpus-path", required=True, type=Path)
    parser.add_argument("--train-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--negatives-per-query", type=int, default=5)
    parser.add_argument("--negative-pool-size", type=int, default=50)
    parser.add_argument("--max-representations-per-aid", type=int, default=8)
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    args = parser.parse_args()
    config = apply_config_overrides(load_config(args.config), args.set)
    config["data"]["corpus_path"] = str(args.corpus_path)
    config["data"]["train_path"] = str(args.train_path)
    path = prepare_reranker_training_data(
        config,
        args.output,
        negatives_per_query=args.negatives_per_query,
        negative_pool_size=args.negative_pool_size,
        max_representations_per_aid=args.max_representations_per_aid,
    )
    print(path)


if __name__ == "__main__":
    main()
