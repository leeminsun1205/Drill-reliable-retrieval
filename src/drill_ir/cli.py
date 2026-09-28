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
from drill_ir.experiment import (
    apply_config_overrides,
    load_config,
    package_submission,
    run_experiment,
    set_config_value,
)


def _devices(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        devices = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "devices must be 'auto' or a positive integer"
        ) from error
    if devices <= 0:
        raise argparse.ArgumentTypeError("devices must be 'auto' or a positive integer")
    return devices


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
    parser = argparse.ArgumentParser(
        description="Run a config-driven retrieval experiment with optional CLI overrides."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--corpus-path", type=Path)
    parser.add_argument("--train-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--experiment-name")
    parser.add_argument("--model-name", help="Dense encoder model identifier")
    parser.add_argument("--model-revision", help="Dense model revision or commit hash")
    parser.add_argument("--query-prefix", help="Dense query prefix; use an empty string to clear")
    parser.add_argument(
        "--document-prefix", help="Dense document prefix; use an empty string to clear"
    )
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--encode-batch-size", type=int)
    parser.add_argument("--retrieval-top-k", type=int)
    parser.add_argument("--output-k", type=int)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--devices", type=_devices)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override an existing config leaf with a YAML-typed value; repeatable",
    )
    args = parser.parse_args()
    config = apply_config_overrides(load_config(args.config), args.set)
    convenience_overrides = {
        "experiment_name": args.experiment_name,
        "retrieval.dense.encoder.model_name": args.model_name,
        "retrieval.dense.encoder.revision": args.model_revision,
        "retrieval.dense.encoder.query_prefix": args.query_prefix,
        "retrieval.dense.encoder.document_prefix": args.document_prefix,
        "retrieval.dense.encoder.max_seq_length": args.max_seq_length,
        "retrieval.dense.encoder.batch_size": args.encode_batch_size,
        "retrieval.top_k": args.retrieval_top_k,
        "retrieval.output_k": args.output_k,
        "runtime.accelerator": args.accelerator,
        "runtime.devices": args.devices,
    }
    for path, value in convenience_overrides.items():
        if value is not None:
            set_config_value(config, path, value)
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
