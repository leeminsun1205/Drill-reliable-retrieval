#!/usr/bin/env python3
"""Run a retrospective recall gate for an explicit-reference graph."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import pyarrow.parquet as pq


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def stable_unique(values: list[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


def macro_recall(rankings: dict[int, list[int]], gold: dict[int, set[int]], cutoff: int) -> float:
    recalls = []
    for qid, relevant in gold.items():
        retrieved = set(rankings[qid][:cutoff])
        recalls.append(len(retrieved & relevant) / len(relevant))
    return sum(recalls) / len(recalls)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev-qids", required=True, type=Path)
    parser.add_argument("--first-stage", required=True, type=Path)
    parser.add_argument("--reference-edges", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cutoff", type=int, default=50)
    parser.add_argument("--seed-depth", type=int, default=5)
    parser.add_argument("--max-expansions", type=int, default=4)
    args = parser.parse_args()

    dev_qids = {int(qid) for qid in json.loads(args.dev_qids.read_text(encoding="utf-8"))}
    query_rows = pq.read_table(args.train).to_pylist()
    gold = {
        int(row.get("qid", row.get("id"))): set(stable_unique(row["relevant_laws"]))
        for row in query_rows
        if int(row.get("qid", row.get("id"))) in dev_qids
    }
    questions = {
        int(row.get("qid", row.get("id"))): str(row["question"])
        for row in query_rows
        if int(row.get("qid", row.get("id"))) in dev_qids
    }
    rankings = {
        int(row["qid"]): stable_unique(row["retrieved_aids"])
        for row in load_jsonl(args.first_stage)
        if int(row["qid"]) in dev_qids
    }
    if set(rankings) != set(gold):
        raise RuntimeError("First-stage qids do not exactly match the requested development split")

    adjacency: dict[str, dict[int, list[dict]]] = {
        "same_document": defaultdict(list),
        "cross_document": defaultdict(list),
        "all": defaultdict(list),
    }
    for edge in load_jsonl(args.reference_edges):
        if not edge["candidate_eligible"] or edge["target_aid"] is None:
            continue
        source = int(edge["source_aid"])
        target = int(edge["target_aid"])
        if source == target:
            continue
        scope = edge["scope"]
        adjacency[scope][source].append(edge)
        adjacency["all"][source].append(edge)

    baseline_recall = macro_recall(rankings, gold, args.cutoff)
    policies = {}
    detail_rows = []
    for scope in ("same_document", "cross_document", "all"):
        expanded_unbudgeted = {}
        expanded_fixed = {}
        additions_by_qid = {}
        recovered_gold = set()
        dropped_gold = set()
        expanded_queries = 0
        for qid, ranking in rankings.items():
            baseline = ranking[:args.cutoff]
            baseline_set = set(baseline)
            additions = []
            expansion_edges = []
            seen = set(baseline)
            for seed in baseline[:args.seed_depth]:
                for edge in adjacency[scope].get(seed, []):
                    target = int(edge["target_aid"])
                    if target not in seen:
                        additions.append(target)
                        expansion_edges.append({
                            "seed_aid": seed,
                            "target_aid": target,
                            "target_law_id": edge["target_law_id"],
                            "source_char_start": edge["source_char_start"],
                            "context": edge["context"],
                        })
                        seen.add(target)
                    if len(additions) >= args.max_expansions:
                        break
                if len(additions) >= args.max_expansions:
                    break
            additions_by_qid[qid] = additions
            if additions:
                expanded_queries += 1
            expanded_unbudgeted[qid] = baseline + additions
            keep = max(0, args.cutoff - len(additions))
            expanded_fixed[qid] = baseline[:keep] + additions
            new_set = set(expanded_fixed[qid])
            for aid in gold[qid]:
                key = (qid, aid)
                if aid not in baseline_set and aid in new_set:
                    recovered_gold.add(key)
                if aid in baseline_set and aid not in new_set:
                    dropped_gold.add(key)
            if additions:
                recovered_for_query = {
                    aid for aid in gold[qid] if aid not in baseline_set and aid in new_set
                }
                for edge in expansion_edges:
                    edge["recovers_gold"] = edge["target_aid"] in recovered_for_query
                detail_rows.append({
                    "policy": scope,
                    "qid": qid,
                    "question": questions[qid],
                    "expansion_edges": expansion_edges,
                    "recovered_gold_aids": sorted(recovered_for_query),
                    "dropped_gold_aids": sorted(
                        aid for aid in gold[qid] if aid in baseline_set and aid not in new_set
                    ),
                })
        policies[scope] = {
            "expanded_queries": expanded_queries,
            "mean_additions": sum(map(len, additions_by_qid.values())) / len(gold),
            "recovered_gold_pairs_fixed_budget": len(recovered_gold),
            "dropped_gold_pairs_fixed_budget": len(dropped_gold),
            "recall_unbudgeted": macro_recall(expanded_unbudgeted, gold, args.cutoff + args.max_expansions),
            "recall_fixed_budget": macro_recall(expanded_fixed, gold, args.cutoff),
        }

    report = {
        "diagnostic_only": True,
        "queries": len(gold),
        "cutoff": args.cutoff,
        "seed_depth": args.seed_depth,
        "max_expansions": args.max_expansions,
        "baseline_recall": baseline_recall,
        "policies": policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    details_path = args.output.with_suffix(".details.jsonl")
    with details_path.open("w", encoding="utf-8") as stream:
        for row in detail_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output), "details": str(details_path), **report
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
