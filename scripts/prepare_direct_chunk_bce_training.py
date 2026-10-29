#!/usr/bin/env python3
"""Mine BCE pairs aligned with direct retrieved-chunk reranking."""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.experiment import apply_config_overrides, load_config
from drill_ir.reranking import prepare_direct_chunk_bce_training_data
from drill_ir.reranking import prepare_direct_chunk_candidate_handoff


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
        "--retrieval-handoff",
        type=Path,
        help="Reusable CPU JSONL handoff between retrieval and teacher processes.",
    )
    parser.add_argument(
        "--phase",
        choices=("orchestrate", "retrieval", "teacher"),
        default="orchestrate",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reranker-device",
        help="Mining-only cross-encoder device, for example cuda:1",
    )
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    args = parser.parse_args()
    handoff = args.retrieval_handoff or args.output.with_name(
        args.output.stem + "_retrieval_handoff.jsonl"
    )
    if args.phase == "orchestrate":
        base_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ]
        if args.retrieval_handoff is None:
            base_command.extend(["--retrieval-handoff", str(handoff)])
        print(
            f"[direct-chunk-orchestrator] Starting isolated retrieval process; "
            f"handoff={handoff}",
            flush=True,
        )
        subprocess.run([*base_command, "--phase", "retrieval"], check=True)
        print(
            "[direct-chunk-orchestrator] Retrieval process exited; starting a fresh "
            "teacher-only process",
            flush=True,
        )
        subprocess.run([*base_command, "--phase", "teacher"], check=True)
        print(args.output)
        return

    config = apply_config_overrides(load_config(args.config), args.set)
    config["data"]["corpus_path"] = str(args.corpus_path)
    config["data"]["train_path"] = str(args.train_path)
    if args.reranker_device:
        config["reranking"]["device"] = args.reranker_device
    candidate_pool_size = args.candidate_pool_size or args.retrieved_chunk_depth
    if args.phase == "retrieval":
        print(
            prepare_direct_chunk_candidate_handoff(
                config,
                handoff,
                candidate_pool_size=candidate_pool_size,
                retrieved_chunk_depth=args.retrieved_chunk_depth,
            )
        )
        return
    print(prepare_direct_chunk_bce_training_data(
        config,
        args.output,
        retrieval_handoff=handoff,
        candidate_pool_size=args.candidate_pool_size,
        retrieved_chunk_depth=args.retrieved_chunk_depth,
        reranked_pool_size=args.reranked_pool_size,
        skip_top_negatives=args.skip_top_negatives,
        negatives_per_query=args.negatives_per_query,
        max_positive_representations_per_aid=args.max_positive_representations_per_aid,
    ))


if __name__ == "__main__":
    main()
