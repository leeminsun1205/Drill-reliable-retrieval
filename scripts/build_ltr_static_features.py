#!/usr/bin/env python3
"""Build deterministic corpus-only article features for LightGBM."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.data import iter_articles
from drill_ir.text_features import article_static_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    rows = 0
    aids: set[int] = set()
    with temporary.open("w", encoding="utf-8") as stream:
        for article in iter_articles(args.corpus, include_blank=False):
            if article.aid in aids:
                raise ValueError(f"Duplicate aid in corpus: {article.aid}")
            aids.add(article.aid)
            record = {
                "aid": article.aid,
                "law_id": article.law_id,
                **article_static_features(article.text),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows += 1
    temporary.replace(args.output)
    manifest = {
        "schema_version": 1,
        "selection": "ltr_static_article_features_v1",
        "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "output_file": args.output.name,
        "output_sha256": _sha256(args.output),
        "rows": rows,
        "unique_aids": len(aids),
        "python": platform.python_version(),
    }
    manifest_path = args.output.with_name(args.output.stem + "_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
