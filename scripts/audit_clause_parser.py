#!/usr/bin/env python3
"""Audit conservative Khoản extraction and create a manual-review sample."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.data import iter_articles, resolve_corpus_path
from drill_ir.structure import audit_clause_parser


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-status", type=int, default=30)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    corpus_path = resolve_corpus_path(args.corpus)
    articles = list(iter_articles(corpus_path, include_blank=False))
    summary, records = audit_clause_parser(
        articles,
        seed=args.seed,
        samples_per_status=args.samples_per_status,
    )
    summary.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "corpus_path": str(corpus_path),
            "parser": "conservative_line_sequence_v1",
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "review_sample.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
