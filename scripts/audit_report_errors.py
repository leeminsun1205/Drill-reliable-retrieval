#!/usr/bin/env python3
"""Audit a saved submission bundle without rerunning retrieval or neural scoring."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drill_ir.analysis.error_audit import audit_gold_flow, replay_saved_ensemble
from drill_ir.data import load_queries
from drill_ir.evaluation import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    with tarfile.open(args.archive) as archive:
        names = [m.name for m in archive if m.isfile()]

        def read(suffix, jsonl=False):
            matching = [name for name in names if name.endswith(suffix)]
            if len(matching) != 1:
                raise ValueError(f"Expected one member for {suffix}: {matching}")
            text = archive.extractfile(matching[0]).read().decode("utf-8")
            return [json.loads(line) for line in text.splitlines() if line.strip()] if jsonl else json.loads(text)

        provenance = read("multiclass_ridge_ensemble/provenance.json")
        metrics = read("multiclass_ridge_ensemble/nested_oof_metrics.json")
        expected_hashes = {Path(name).name: value for name, value in provenance["input_sha256"].items()}
        if hashlib.sha256(args.queries.read_bytes()).hexdigest() != expected_hashes["train.parquet"]:
            raise ValueError("Local labeled queries differ from the archived input")
        dev_qids = set(map(int, read("split_dev_qids.json")))
        queries = {q.qid: q for q in load_queries(args.queries) if q.qid in dev_qids}
        gold = {qid: query.relevant_aids for qid, query in queries.items()}
        if set(gold) != dev_qids:
            raise ValueError("Incomplete local gold mapping")
        ranking_rows = read("lightgbm_enhanced_v2/oof_predictions.jsonl", True)
        rankings = {int(row["qid"]): (row["retrieved_aids"], row["scores"]) for row in ranking_rows}
        selected, diagnostics = replay_saved_ensemble(rankings, gold, provenance["config"], metrics)
        dev_prefix = next(n.rsplit("/", 1)[0] for n in names if n.startswith("dev_run/") and n.endswith("run_info.json"))
        pool_rows = read(dev_prefix + "/reranker_chunk_candidates.jsonl", True)
        pool = {int(row["qid"]): [int(c["aid"]) for c in row["chunks"]] for row in pool_rows}
        rerank = {int(row["qid"]): row["retrieved_aids"] for row in read(dev_prefix + "/predictions.jsonl", True)}
        summary, rows = audit_gold_flow(gold, pool, rerank, {qid: value[0] for qid, value in rankings.items()}, selected)
        summary["source"] = {
            "archive": str(args.archive),
            "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
            "train_sha256": expected_hashes["train.parquet"],
            "checkpoint": read("checkpoint_provenance.json")["checkpoint_manifest"],
            "seed": provenance["config"]["seed"],
            "policy": "multiclass_ridge_ensemble",
            "reconstruction": "five outer-fold linear heads using archived inner-selected weights; fold and aggregate metrics verified to 1e-12",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "processing_seconds": time.perf_counter() - start,
            "audit_module_sha256": hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "src/drill_ir/analysis/error_audit.py").read_bytes()
            ).hexdigest(),
        }
        summary["artifact_lgbm_top2"] = evaluate_predictions(gold, {qid: value[0][:2] for qid, value in rankings.items()}).to_dict()
        for row in rows:
            row.update(question=queries[row["qid"]].question, **diagnostics[row["qid"]])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (args.output_dir / "per_query.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
