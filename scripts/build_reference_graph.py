#!/usr/bin/env python3
"""Build an auditable explicit-reference graph from the raw corpus."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.structure import build_article_number_index, extract_reference_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--overrides", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    documents = pq.read_table(args.corpus).to_pylist()
    overrides = yaml.safe_load(args.overrides.read_text(encoding="utf-8")) or {}
    index, mappings = build_article_number_index(documents, **overrides)
    references = extract_reference_records(documents, index)
    unique_edges = {}
    self_loop_mentions = 0
    for record in references:
        if record.target_aid is None or not record.candidate_eligible:
            continue
        if record.source_aid == record.target_aid:
            self_loop_mentions += 1
            continue
        key = (record.source_aid, record.target_aid, record.scope)
        unique_edges.setdefault(key, record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "article_number_mapping.jsonl").open("w", encoding="utf-8") as stream:
        for record in mappings:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    with (args.output_dir / "reference_mentions.jsonl").open("w", encoding="utf-8") as stream:
        for record in references:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    with (args.output_dir / "reference_edges.jsonl").open("w", encoding="utf-8") as stream:
        for record in unique_edges.values():
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    summary = {
        "documents": len(documents),
        "article_mappings": len(mappings),
        "reference_mentions": len(references),
        "unique_resolved_edges": len(unique_edges),
        "status_counts": dict(sorted(Counter(record.status for record in references).items())),
        "scope_counts": dict(sorted(Counter(record.scope for record in references).items())),
        "candidate_eligible_mentions": sum(record.candidate_eligible for record in references),
        "candidate_eligible_edges": sum(
            record.candidate_eligible for record in unique_edges.values()
        ),
        "excluded_self_loop_mentions": self_loop_mentions,
        "overrides": overrides,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
