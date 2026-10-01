#!/usr/bin/env python3
"""Mine FPT-style full-article BCE pairs on the training split only."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.experiment import apply_config_overrides, load_config
from drill_ir.reranking import prepare_fpt_reranker_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--corpus-path", required=True, type=Path)
    parser.add_argument("--train-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-article-tokens", type=int, default=1024)
    parser.add_argument("--reranked-pool-size", type=int, default=30)
    parser.add_argument("--skip-top-negatives", type=int, default=5)
    parser.add_argument("--large-pool-threshold", type=int, default=15)
    parser.add_argument("--large-sample-size", type=int, default=10)
    parser.add_argument("--small-sample-size", type=int, default=5)
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    args = parser.parse_args()
    config = apply_config_overrides(load_config(args.config), args.set)
    config["data"]["corpus_path"] = str(args.corpus_path)
    config["data"]["train_path"] = str(args.train_path)
    path = prepare_fpt_reranker_training_data(
        config,
        args.output,
        maximum_article_tokens=args.maximum_article_tokens,
        reranked_pool_size=args.reranked_pool_size,
        skip_top_negatives=args.skip_top_negatives,
        large_pool_threshold=args.large_pool_threshold,
        large_sample_size=args.large_sample_size,
        small_sample_size=args.small_sample_size,
    )
    print(path)


if __name__ == "__main__":
    main()
