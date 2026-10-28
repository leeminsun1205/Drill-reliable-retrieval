#!/usr/bin/env python3
"""Build pairwise query/positive/negative rows from an audited chunk miner."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.reranking import prepare_pairwise_training_data_from_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        prepare_pairwise_training_data_from_audit(
            args.audit,
            args.queries,
            args.output,
        )
    )


if __name__ == "__main__":
    main()
