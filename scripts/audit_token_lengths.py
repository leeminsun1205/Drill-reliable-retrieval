#!/usr/bin/env python3
"""Audit corpus and query lengths with an exact model tokenizer."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.data import iter_articles, load_queries, resolve_corpus_path, resolve_train_path
from drill_ir.preprocessing.token_audit import audit_token_lengths, load_fast_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--thresholds", default="512,1024")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    thresholds = [int(value) for value in args.thresholds.split(",") if value.strip()]
    corpus_path = resolve_corpus_path(args.corpus)
    query_path = resolve_train_path(args.queries)
    tokenizer = load_fast_tokenizer(
        args.model_name,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
    )
    articles = list(iter_articles(corpus_path, include_blank=False))
    queries = load_queries(query_path)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "corpus_path": str(corpus_path),
        "query_path": str(query_path),
        "thresholds": thresholds,
        "articles": audit_token_lengths(
            (article.text for article in articles),
            tokenizer=tokenizer,
            prefix=args.document_prefix,
            thresholds=thresholds,
            batch_size=args.batch_size,
        ),
        "queries": audit_token_lengths(
            (query.question for query in queries),
            tokenizer=tokenizer,
            prefix=args.query_prefix,
            thresholds=thresholds,
            batch_size=args.batch_size,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
