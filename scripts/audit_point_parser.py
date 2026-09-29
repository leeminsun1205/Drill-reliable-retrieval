#!/usr/bin/env python3
"""Audit conservative Điểm extraction inside accepted Khoản spans."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.data import iter_articles, resolve_corpus_path
from drill_ir.structure import parse_clause_spans, parse_point_spans, sample_articles_by_law


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-status", type=int, default=30)
    parser.add_argument("--sample-law-count", type=int)
    parser.add_argument("--exclude-summary", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    corpus_path = resolve_corpus_path(args.corpus)
    articles = list(iter_articles(corpus_path, include_blank=False))
    selected_laws = None
    excluded_law_ids: list[str] = []
    if args.sample_law_count is not None:
        if args.exclude_summary is not None:
            prior = json.loads(args.exclude_summary.read_text(encoding="utf-8"))
            excluded_law_ids = prior.get("sampled_law_ids") or []
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

    article_statuses: Counter[str] = Counter()
    point_statuses: Counter[str] = Counter()
    grouped = defaultdict(list)
    point_representations = 0
    for article in articles:
        clauses = parse_clause_spans(article.text)
        article_statuses[clauses.status] += 1
        if clauses.status != "parsed":
            continue
        for clause in clauses.spans:
            clause_text = article.text[clause.char_start:clause.char_end]
            points = parse_point_spans(clause_text)
            point_statuses[points.status] += 1
            point_representations += len(points.spans)
            grouped[points.status].append((article, clause, clause_text, points))

    rng = random.Random(args.seed)
    records = []
    sampled_counts = {}
    for status in sorted(grouped):
        population = grouped[status]
        selected = rng.sample(population, min(args.samples_per_status, len(population)))
        sampled_counts[status] = len(selected)
        for article, clause, clause_text, points in selected:
            records.append(
                {
                    "aid": article.aid,
                    "law_id": article.law_id,
                    "clause_number": clause.number,
                    "point_parse_status": status,
                    "detected_letters": list(points.detected_letters),
                    "clause_text": clause_text,
                    "lead_in": clause_text[: points.lead_in_end],
                    "point_spans": [
                        {
                            "letter": span.letter,
                            "char_start": span.char_start,
                            "char_end": span.char_end,
                            "text": clause_text[span.char_start:span.char_end],
                        }
                        for span in points.spans
                    ],
                }
            )

    summary = {
        "seed": args.seed,
        "articles": len(articles),
        "article_parse_status_counts": dict(sorted(article_statuses.items())),
        "parsed_clauses": sum(point_statuses.values()),
        "point_parse_status_counts": dict(sorted(point_statuses.items())),
        "point_representations": point_representations,
        "sampled_counts": sampled_counts,
        "sampled_law_count": len(selected_laws) if selected_laws is not None else None,
        "sampled_law_ids": selected_laws,
        "excluded_law_count": len(excluded_law_ids),
        "clause_parser": "conservative_line_sequence_v1",
        "point_parser": "conservative_vietnamese_letter_sequence_v1",
        "corpus_path": str(corpus_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
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
