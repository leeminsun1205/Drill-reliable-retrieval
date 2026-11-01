#!/usr/bin/env python3
"""Nested-screen cardinality policies and package one submission per case."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.analysis.cardinality_screen import apply_fitted_case, evaluate_nested_case  # noqa: E402
from drill_ir.data import load_queries  # noqa: E402
from drill_ir.experiment import package_submission  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rankings(path: Path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line); qid = int(row["qid"])
        if qid in result: raise ValueError(f"Duplicate qid: {qid}")
        result[qid] = (list(map(int, row["retrieved_aids"])), list(map(float, row["scores"])))
    if not result: raise ValueError(f"Empty ranking: {path}")
    return result


def _write(path: Path, selected, diagnostics):
    with path.open("w", encoding="utf-8") as stream:
        for qid in sorted(selected):
            stream.write(json.dumps({"qid": qid, "retrieved_aids": selected[qid], **diagnostics[qid]}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-config", required=True, type=Path)
    parser.add_argument("--dev-oof-predictions", required=True, type=Path)
    parser.add_argument("--train-queries", required=True, type=Path)
    parser.add_argument("--dev-qids", required=True, type=Path)
    parser.add_argument("--public-predictions", required=True, type=Path)
    parser.add_argument("--private-predictions", required=True, type=Path)
    parser.add_argument("--public-test", required=True, type=Path)
    parser.add_argument("--private-test", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.screen_config.read_text())
    if config.get("selection_metric") != "macro_f2" or config.get("tie_break") != "smaller_k":
        raise ValueError("Unexpected screen selection contract")
    seed, outer_folds, inner_folds, maximum_k = map(int, [config["seed"], config["outer_folds"], config["inner_folds"], config["maximum_k"]])
    dev_qids = set(map(int, json.loads(args.dev_qids.read_text())))
    gold = {query.qid: query.relevant_aids for query in load_queries(args.train_queries) if query.qid in dev_qids}
    dev_rankings = _read_rankings(args.dev_oof_predictions)
    if set(gold) != dev_qids or set(dev_rankings) != dev_qids: raise ValueError("Development inputs do not align")
    test_rankings = {"public": _read_rankings(args.public_predictions), "private": _read_rankings(args.private_predictions)}
    test_paths = {"public": args.public_test, "private": args.private_test}
    for split in test_rankings:
        if set(test_rankings[split]) != {q.qid for q in load_queries(test_paths[split])}: raise ValueError(f"{split} qids do not align")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for case, spec in config["cases"].items():
        print(f"[cardinality-screen] start case={case}", flush=True)
        metrics, params, tuning, fitted = evaluate_nested_case(case, spec, dev_rankings, gold, outer_folds=outer_folds, inner_folds=inner_folds, seed=seed, maximum_k=maximum_k)
        case_dir = args.output_dir / case; case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "nested_oof_metrics.json").write_text(json.dumps(metrics, indent=2))
        (case_dir / "deployment_selection.json").write_text(json.dumps({"selected_params": params, "cross_fitted_candidates": tuning}, indent=2))
        for split in ("public", "private"):
            selected, diagnostics = apply_fitted_case(case, fitted, test_rankings[split], maximum_k)
            prediction_path = case_dir / f"{split}_predictions.jsonl"; _write(prediction_path, selected, diagnostics)
            package_submission(prediction_path, set(test_rankings[split]), case_dir / f"{split}_{case}_results.zip", output_k=maximum_k)
        provenance = {"case": case, "screen_config": config, "selected_params": params, "nested_oof": metrics, "private_labels_used": False, "input_sha256": {str(path): _sha256(path) for path in [args.screen_config,args.dev_oof_predictions,args.train_queries,args.dev_qids,args.public_predictions,args.private_predictions,args.public_test,args.private_test]}}
        (case_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
        summary.append({"case": case, "selected_params": params, **{key: metrics[key] for key in ("precision","recall","f2")}, "mean_output_size": metrics["output_size"]["mean"], **metrics["oracle_k_diagnostic"]})
        print(f"[cardinality-screen] complete case={case} nested_f2={metrics['f2']:.6f}", flush=True)
    summary.sort(key=lambda row: (row["f2"], row["precision"], row["recall"]), reverse=True)
    (args.output_dir / "screen_summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "selected_by_nested_oof.json").write_text(json.dumps(summary[0], indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()

