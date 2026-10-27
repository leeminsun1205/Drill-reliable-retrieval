"""Per-query retrieval coverage and stage-attribution diagnostics."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from drill_ir.data import Query


DEFAULT_CUTOFFS = (1, 5, 10, 20, 50, 100)
REFERENCE_PATTERN = re.compile(
    r"\b(?:điều|khoản|điểm|nghị\s+định|thông\s+tư|luật|bộ\s+luật)\s+\d+",
    flags=re.IGNORECASE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _ranking_rows(path: Path) -> dict[int, tuple[int, ...]]:
    rows: dict[int, tuple[int, ...]] = {}
    for record in _read_jsonl(path):
        qid = int(record["qid"])
        if qid in rows:
            raise ValueError(f"Duplicate qid={qid} in {path}")
        rows[qid] = tuple(int(aid) for aid in record["retrieved_aids"])
    return rows


def _chunk_candidate_rows(path: Path) -> dict[int, tuple[int, ...]]:
    rows: dict[int, tuple[int, ...]] = {}
    for record in _read_jsonl(path):
        qid = int(record["qid"])
        if qid in rows:
            raise ValueError(f"Duplicate qid={qid} in {path}")
        rows[qid] = tuple(
            dict.fromkeys(int(chunk["aid"]) for chunk in record["chunks"])
        )
    return rows


def _component_rows(path: Path) -> dict[str, dict[int, tuple[int, ...]]]:
    by_component: defaultdict[str, dict[int, tuple[int, ...]]] = defaultdict(dict)
    expected_names: set[str] | None = None
    for record in _read_jsonl(path):
        qid = int(record["qid"])
        components = record["components"]
        names = set(components)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError("Component names differ across query records")
        for name, payload in components.items():
            if qid in by_component[name]:
                raise ValueError(f"Duplicate qid={qid} for component={name}")
            by_component[name][qid] = tuple(
                int(aid) for aid in payload["retrieved_aids"]
            )
    return dict(by_component)


def _rank_of(aid: int, ranking: Sequence[int]) -> int | None:
    return next((rank for rank, value in enumerate(ranking, start=1) if value == aid), None)


def analyze_query_retrieval(
    result_dir: str | Path,
    queries: Sequence[Query],
    *,
    extra_rankings: Mapping[str, str | Path] | None = None,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    """Explain macro-recall deficits by query, component, and pipeline stage."""
    result_dir = Path(result_dir)
    component_path = result_dir / "component_rankings.jsonl"
    first_stage_path = result_dir / "first_stage_predictions.jsonl"
    if not component_path.is_file() or not first_stage_path.is_file():
        raise FileNotFoundError(
            "Query diagnostics require component_rankings.jsonl and "
            "first_stage_predictions.jsonl"
        )
    query_by_qid = {query.qid: query for query in queries}
    if len(query_by_qid) != len(queries):
        raise ValueError("Supplied queries contain duplicate qids")
    expected_qids = set(query_by_qid)

    components = _component_rows(component_path)
    stages: dict[str, dict[int, tuple[int, ...]]] = {
        f"component:{name}": rankings for name, rankings in components.items()
    }
    stages["fused_first_stage"] = _ranking_rows(first_stage_path)
    chunks_path = result_dir / "reranker_chunk_candidates.jsonl"
    if chunks_path.is_file():
        stages["direct_chunk_candidates"] = _chunk_candidate_rows(chunks_path)
    reranker_path = result_dir / "predictions.jsonl"
    if reranker_path.is_file():
        stages["epoch3_reranker"] = _ranking_rows(reranker_path)
    for name, path in (extra_rankings or {}).items():
        if not name or name in stages:
            raise ValueError(f"Invalid or duplicate extra ranking name: {name!r}")
        stages[name] = _ranking_rows(Path(path))
    for name, rankings in stages.items():
        if set(rankings) != expected_qids:
            raise ValueError(f"Stage {name!r} qids do not match supplied queries")

    normalized_cutoffs = tuple(sorted({int(value) for value in cutoffs}))
    if not normalized_cutoffs or normalized_cutoffs[0] <= 0:
        raise ValueError("cutoffs must contain positive integers")
    if 50 not in normalized_cutoffs:
        raise ValueError("cutoffs must include 50 for the primary deficit analysis")
    query_records: list[dict[str, Any]] = []
    metric_accumulator: defaultdict[tuple[str, int], list[float]] = defaultdict(list)
    complete_accumulator: defaultdict[tuple[str, int], int] = defaultdict(int)
    zero_accumulator: defaultdict[tuple[str, int], int] = defaultdict(int)

    component_names = sorted(components)
    attribution = {
        name: {"unique_gold_at_50": 0, "rescued_sparse_misses_at_50": 0}
        for name in component_names
    }
    union_recalls: list[float] = []
    fused_ranking = stages["fused_first_stage"]
    sparse_name = "sparse" if "sparse" in components else None

    for query in sorted(queries, key=lambda item: item.qid):
        gold = tuple(dict.fromkeys(query.relevant_aids))
        if not gold:
            raise ValueError(f"Query diagnostics require labels for qid={query.qid}")
        gold_set = set(gold)
        stage_values: dict[str, Any] = {}
        for stage_name, rankings in stages.items():
            ranking = rankings[query.qid]
            by_cutoff: dict[str, Any] = {}
            for cutoff in normalized_cutoffs:
                found = gold_set & set(ranking[:cutoff])
                recall = len(found) / len(gold_set)
                metric_accumulator[(stage_name, cutoff)].append(recall)
                complete_accumulator[(stage_name, cutoff)] += int(recall == 1.0)
                zero_accumulator[(stage_name, cutoff)] += int(recall == 0.0)
                by_cutoff[str(cutoff)] = {
                    "recall": recall,
                    "found_gold": sorted(found),
                    "missing_gold": sorted(gold_set - found),
                }
            stage_values[stage_name] = {
                "cutoffs": by_cutoff,
                "gold_ranks": {
                    str(aid): _rank_of(aid, ranking) for aid in gold
                },
            }

        component_top50 = {
            name: set(components[name][query.qid][:50]) for name in component_names
        }
        union = set().union(*component_top50.values()) if component_top50 else set()
        union_recalls.append(len(gold_set & union) / len(gold_set))
        gold_attribution: dict[str, Any] = {}
        sparse_top50 = component_top50.get(sparse_name, set())
        for aid in gold:
            finders = [name for name in component_names if aid in component_top50[name]]
            fused_rank = _rank_of(aid, fused_ranking[query.qid])
            if len(finders) == 1:
                attribution[finders[0]]["unique_gold_at_50"] += 1
            for name in finders:
                if sparse_name is not None and name != sparse_name and aid not in sparse_top50:
                    attribution[name]["rescued_sparse_misses_at_50"] += 1
            gold_attribution[str(aid)] = {
                "components_finding_at_50": finders,
                "missed_by_all_components_at_50": not finders,
                "fused_rank": fused_rank,
                "found_by_component_but_missing_fused_at_50": bool(finders) and (
                    fused_rank is None or fused_rank > 50
                ),
            }
        fused_recall50 = stage_values["fused_first_stage"]["cutoffs"]["50"]["recall"]
        query_records.append(
            {
                "qid": query.qid,
                "question": query.question,
                "query_whitespace_tokens": len(query.question.split()),
                "has_explicit_legal_reference": bool(REFERENCE_PATTERN.search(query.question)),
                "gold_aids": list(gold),
                "gold_count": len(gold),
                "fused_recall_at_50": fused_recall50,
                "macro_recall50_gap_contribution": (1.0 - fused_recall50) / len(queries),
                "gold_attribution": gold_attribution,
                "stages": stage_values,
            }
        )

    stage_metrics: list[dict[str, Any]] = []
    for stage_name in stages:
        for cutoff in normalized_cutoffs:
            values = metric_accumulator[(stage_name, cutoff)]
            stage_metrics.append(
                {
                    "stage": stage_name,
                    "cutoff": cutoff,
                    "macro_recall": sum(values) / len(values),
                    "complete_coverage_queries": complete_accumulator[(stage_name, cutoff)],
                    "zero_coverage_queries": zero_accumulator[(stage_name, cutoff)],
                    "queries": len(values),
                }
            )
    ordered_queries = sorted(
        query_records,
        key=lambda row: (
            -float(row["macro_recall50_gap_contribution"]),
            -int(row["gold_count"]),
            int(row["qid"]),
        ),
    )
    hard_queries = [
        {
            key: row[key]
            for key in (
                "qid", "question", "gold_count", "gold_aids",
                "fused_recall_at_50", "macro_recall50_gap_contribution",
                "has_explicit_legal_reference",
            )
        }
        for row in ordered_queries
    ]
    return {
        "protocol": {
            "queries": len(queries),
            "cutoffs": list(normalized_cutoffs),
            "component_names": component_names,
            "primary_deficit": "1 - per-query fused Recall@50",
            "note": (
                "Macro-recall gap contribution is divided by query count, so its "
                "sum equals 1 - fused macro Recall@50. Component uniqueness is "
                "descriptive and is not a causal leave-one-component-out estimate."
            ),
        },
        "stage_metrics": stage_metrics,
        "component_attribution_at_50": attribution,
        "component_union_oracle_macro_recall_at_50": sum(union_recalls) / len(union_recalls),
        "hard_queries": hard_queries,
        "queries": query_records,
    }


def write_query_diagnostics(report: Mapping[str, Any], output_dir: str | Path) -> None:
    """Write machine-readable detail plus compact review tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in report.items() if key != "queries"}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "query_diagnostics.jsonl").open("w", encoding="utf-8") as stream:
        for record in report["queries"]:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "stage_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        rows = report["stage_metrics"]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "hard_queries.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "qid", "question", "gold_count", "gold_aids", "fused_recall_at_50",
            "macro_recall50_gap_contribution", "has_explicit_legal_reference",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in report["hard_queries"]:
            writer.writerow({key: record[key] for key in fieldnames})
