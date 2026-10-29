#!/usr/bin/env python3
"""Mine BCE pairs aligned with direct retrieved-chunk reranking."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.experiment import apply_config_overrides, load_config
from drill_ir.reranking import prepare_direct_chunk_bce_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--corpus-path", required=True, type=Path)
    parser.add_argument("--train-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-pool-size", type=int)
    parser.add_argument("--retrieved-chunk-depth", type=int, default=100)
    parser.add_argument("--reranked-pool-size", type=int, default=30)
    parser.add_argument("--skip-top-negatives", type=int, default=5)
    parser.add_argument("--negatives-per-query", type=int, default=10)
    parser.add_argument("--max-positive-representations-per-aid", type=int, default=32)
    parser.add_argument(
        "--reranker-device",
        help="Mining-only cross-encoder device, for example cuda:1",
    )
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    args = parser.parse_args()
    config = apply_config_overrides(load_config(args.config), args.set)
    config["data"]["corpus_path"] = str(args.corpus_path)
    config["data"]["train_path"] = str(args.train_path)
    if args.reranker_device:
        config["reranking"]["device"] = args.reranker_device
    print(prepare_direct_chunk_bce_training_data(
        config,
        args.output,
        candidate_pool_size=args.candidate_pool_size,
        retrieved_chunk_depth=args.retrieved_chunk_depth,
        reranked_pool_size=args.reranked_pool_size,
        skip_top_negatives=args.skip_top_negatives,
        negatives_per_query=args.negatives_per_query,
        max_positive_representations_per_aid=args.max_positive_representations_per_aid,
    ))


if __name__ == "__main__":
    main()
