#!/usr/bin/env python3
"""Create a deterministic stratified sample for reference-edge review."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random


DEFAULT_COUNTS = {
    "eligible_same_document": 25,
    "eligible_cross_document": 25,
    "named_target_without_law_id": 15,
    "amendment_target_without_law_id": 10,
    "quoted_article_heading": 10,
    "target_not_in_corpus": 10,
    "other_unresolved": 10,
}


def review_stratum(record: dict) -> str:
    if record["candidate_eligible"]:
        return f"eligible_{record['scope']}"
    status = record["status"]
    if status in {
        "named_target_without_law_id",
        "amendment_target_without_law_id",
        "quoted_article_heading",
        "target_not_in_corpus",
    }:
        return status
    return "other_unresolved"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    mentions_path = args.graph_dir / "reference_mentions.jsonl"
    records = [json.loads(line) for line in mentions_path.open(encoding="utf-8")]
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_stratum[review_stratum(record)].append(record)

    randomizer = random.Random(args.seed)
    sampled: list[dict] = []
    for stratum, requested_count in DEFAULT_COUNTS.items():
        population = sorted(
            by_stratum.get(stratum, []),
            key=lambda record: (
                record["source_aid"],
                record["source_char_start"],
                record["article_number"],
            ),
        )
        selected = randomizer.sample(population, min(requested_count, len(population)))
        for record in selected:
            sampled.append({
                "review_id": (
                    f"{record['source_aid']}:{record['source_char_start']}:"
                    f"{record['article_number']}"
                ),
                "stratum": stratum,
                **record,
                "review_label": None,
                "expected_target_law_id": None,
                "expected_target_aid": None,
                "reviewer_notes": "",
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "review_sample.jsonl"
    with sample_path.open("w", encoding="utf-8") as stream:
        for record in sampled:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "source_graph": str(args.graph_dir),
        "seed": args.seed,
        "population_mentions": len(records),
        "population_by_stratum": dict(sorted(Counter(
            review_stratum(record) for record in records
        ).items())),
        "requested_by_stratum": DEFAULT_COUNTS,
        "sample_size": len(sampled),
        "sample_by_stratum": dict(sorted(Counter(
            record["stratum"] for record in sampled
        ).items())),
        "allowed_review_labels": [
            "correct", "incorrect_target", "not_a_reference", "ambiguous"
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"sample": str(sample_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
