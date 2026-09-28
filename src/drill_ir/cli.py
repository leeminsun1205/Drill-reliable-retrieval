"""Installed command-line entry points for local and Kaggle execution."""

import argparse
import json
from pathlib import Path

from drill_ir.data import (
    load_queries,
    resolve_corpus_path,
    resolve_train_path,
    validate_dataset,
)
from drill_ir.experiment import load_config, package_submission, run_experiment


def validate_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--queries", type=Path)
    args = parser.parse_args()
    corpus_path = resolve_corpus_path(args.corpus)
    query_path = resolve_train_path(args.queries)
    print(f"corpus_path={corpus_path}")
    print(f"query_path={query_path}")
    report = validate_dataset(corpus_path, query_path)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


def run_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--corpus-path", type=Path)
    parser.add_argument("--train-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.corpus_path is not None:
        config["data"]["corpus_path"] = str(args.corpus_path)
    if args.train_path is not None:
        config["data"]["train_path"] = str(args.train_path)
    if args.output_root is not None:
        config["output"]["root_dir"] = str(args.output_root)
    config["data"]["corpus_path"] = str(
        resolve_corpus_path(config["data"].get("corpus_path"))
    )
    config["data"]["train_path"] = str(
        resolve_train_path(config["data"].get("train_path"))
    )
    print(run_experiment(config))


def submission_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", default=Path("results.zip"), type=Path)
    parser.add_argument("--output-k", required=True, type=int)
    args = parser.parse_args()
    expected_qids = {query.qid for query in load_queries(args.queries)}
    print(
        package_submission(
            args.predictions,
            expected_qids,
            args.output,
            output_k=args.output_k,
        )
    )
