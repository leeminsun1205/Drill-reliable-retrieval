"""Deterministic corpus audit for conservative Khoản parsing."""

from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Any, Sequence

from drill_ir.data.models import Article

from .clauses import parse_clause_spans


def audit_clause_parser(
    articles: Sequence[Article],
    *,
    seed: int,
    samples_per_status: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if samples_per_status <= 0:
        raise ValueError("samples_per_status must be positive")
    counts: Counter[str] = Counter()
    grouped: defaultdict[str, list[tuple[Article, Any]]] = defaultdict(list)
    for article in articles:
        parsed = parse_clause_spans(article.text)
        counts[parsed.status] += 1
        grouped[parsed.status].append((article, parsed))

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    sampled_counts: dict[str, int] = {}
    for status in sorted(grouped):
        population = sorted(grouped[status], key=lambda item: item[0].aid)
        selected = rng.sample(population, min(samples_per_status, len(population)))
        selected.sort(key=lambda item: item[0].aid)
        sampled_counts[status] = len(selected)
        for article, parsed in selected:
            records.append(
                {
                    "aid": article.aid,
                    "law_id": article.law_id,
                    "parse_status": parsed.status,
                    "detected_numbers": list(parsed.detected_numbers),
                    "article_text": article.text,
                    "spans": [
                        {
                            "number": span.number,
                            "char_start": span.char_start,
                            "char_end": span.char_end,
                            "text": article.text[span.char_start : span.char_end],
                        }
                        for span in parsed.spans
                    ],
                    "review": {
                        "boundary_correct": None,
                        "failure_type": None,
                        "notes": None,
                    },
                }
            )
    total = len(articles)
    summary = {
        "seed": seed,
        "articles": total,
        "samples_per_status": samples_per_status,
        "status_counts": dict(sorted(counts.items())),
        "status_rates": {
            status: count / total if total else 0.0
            for status, count in sorted(counts.items())
        },
        "sampled_counts": sampled_counts,
    }
    return summary, records
