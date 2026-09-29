#!/usr/bin/env python3
"""Audit conservative Khoản extraction and create a manual-review sample."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.data import iter_articles, resolve_corpus_path
from drill_ir.structure import audit_clause_parser, sample_articles_by_law


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-status", type=int, default=30)
    parser.add_argument("--sample-law-count", type=int)
    parser.add_argument(
        "--exclude-summary",
        type=Path,
        help="Prior audit summary whose sampled_law_ids must be excluded",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    corpus_path = resolve_corpus_path(args.corpus)
    articles = list(iter_articles(corpus_path, include_blank=False))
    selected_laws = None
    if args.sample_law_count is not None:
        excluded_law_ids = []
        if args.exclude_summary is not None:
            prior_summary = json.loads(args.exclude_summary.read_text(encoding="utf-8"))
            excluded_law_ids = prior_summary.get("sampled_law_ids") or []
            if not isinstance(excluded_law_ids, list) or not all(
                isinstance(law_id, str) for law_id in excluded_law_ids
            ):
                raise ValueError("exclude-summary sampled_law_ids must be a list of strings")
        articles, selected_laws = sample_articles_by_law(
            articles,
            law_count=args.sample_law_count,
            seed=args.seed,
            excluded_law_ids=excluded_law_ids,
        )
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
            "sampled_law_count": len(selected_laws) if selected_laws is not None else None,
            "sampled_law_ids": selected_laws,
            "excluded_law_count": len(excluded_law_ids) if selected_laws is not None else 0,
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
