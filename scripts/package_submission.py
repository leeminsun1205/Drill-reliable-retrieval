#!/usr/bin/env python3
"""Convert complete internal predictions to the official results.zip."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.data import load_queries
from drill_ir.experiment import package_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", default=Path("results.zip"), type=Path)
    parser.add_argument("--output-k", required=True, type=int)
    args = parser.parse_args()
    expected_qids = {query.qid for query in load_queries(args.queries)}
    result = package_submission(
        args.predictions, expected_qids, args.output, output_k=args.output_k
    )
    print(result)


if __name__ == "__main__":
    main()
