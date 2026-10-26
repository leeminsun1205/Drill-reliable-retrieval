#!/usr/bin/env python3
"""Build one sparse or dense retrieval cache in an isolated process."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.data import iter_articles, resolve_corpus_path
from drill_ir.experiment import apply_config_overrides, load_config
from drill_ir.retrieval.cache_preparation import (
    configured_cache_components,
    prepare_retrieval_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exactly one validated retrieval cache and then exit."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--corpus-path", type=Path)
    parser.add_argument(
        "--component",
        required=True,
        help="'sparse', 'dense', or a configured multi-dense component name",
    )
    parser.add_argument("--devices", type=int)
    parser.add_argument(
        "--require-existing",
        action="store_true",
        help="Validate and load an existing cache; fail instead of building on a miss",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override an existing config leaf; repeatable",
    )
    args = parser.parse_args()
    if args.devices is not None and args.devices <= 0:
        parser.error("--devices must be positive")

    config = apply_config_overrides(load_config(args.config), args.set)
    corpus_path = resolve_corpus_path(args.corpus_path or config["data"].get("corpus_path"))
    retrieval_config = config["retrieval"]
    available = configured_cache_components(retrieval_config)
    if args.component not in available:
        parser.error(f"--component must be one of: {', '.join(available)}")
    runtime = dict(config.get("runtime", {}))
    if args.devices is not None:
        runtime["accelerator"] = "cuda"
        runtime["devices"] = args.devices

    articles = list(iter_articles(corpus_path, include_blank=False))
    result = prepare_retrieval_cache(
        retrieval_config=retrieval_config,
        runtime_config=runtime,
        articles=articles,
        component=args.component,
        require_existing=args.require_existing,
    )
    print(json.dumps({"corpus_path": str(corpus_path), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
