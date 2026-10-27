#!/usr/bin/env python3
"""Explain retrieval recall deficits by query and component."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.analysis import analyze_query_retrieval, write_query_diagnostics
from drill_ir.data import load_queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--query-ids", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--ranking", action="append", default=[], metavar="NAME=JSONL",
        help="Optional additional ranking such as lightgbm_oof=.../oof_predictions.jsonl",
    )
    args = parser.parse_args()
    queries = load_queries(args.queries)
    if args.query_ids is not None:
        qids = {int(value) for value in json.loads(args.query_ids.read_text())}
        queries = [query for query in queries if query.qid in qids]
        if len(queries) != len(qids):
            raise ValueError("query-ids contain values absent from the query dataset")
    extra = {}
    for value in args.ranking:
        if "=" not in value:
            parser.error("--ranking must use NAME=JSONL")
        name, path = value.split("=", 1)
        extra[name] = Path(path)
    report = analyze_query_retrieval(args.result_dir, queries, extra_rankings=extra)
    write_query_diagnostics(report, args.output_dir)
    fused50 = next(
        row for row in report["stage_metrics"]
        if row["stage"] == "fused_first_stage" and row["cutoff"] == 50
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "fused_recall_at_50": fused50["macro_recall"],
        "complete_coverage_queries_at_50": fused50["complete_coverage_queries"],
        "zero_coverage_queries_at_50": fused50["zero_coverage_queries"],
        "component_union_oracle_recall_at_50": report[
            "component_union_oracle_macro_recall_at_50"
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
