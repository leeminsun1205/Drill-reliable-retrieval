"""Reconstruct saved outer-fold decisions and attribute missing legal evidence."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from drill_ir.evaluation import evaluate_predictions
from .cardinality_confirmatory import _fit, _folds, _raw, _select


def replay_saved_ensemble(
    rankings: Mapping[int, tuple[Sequence[int], Sequence[float]]],
    gold: Mapping[int, Sequence[int]],
    config: Mapping[str, Any],
    saved_metrics: Mapping[str, Any],
) -> tuple[dict[int, list[int]], dict[int, dict[str, Any]]]:
    """Refit five small outer-fold heads using already selected inner-fold weights.

    No neural or LightGBM model is refitted. Held-out labels do not enter a
    fold's fit. Verify every fold and the aggregate against the archived scores.
    """
    qids = set(gold)
    if set(rankings) != qids:
        raise ValueError("Ranking and gold query sets differ")
    seed = int(config["seed"])
    assignment = _folds(qids, int(config["outer_folds"]), seed)
    maximum_k = int(config["maximum_k"])
    predictions: dict[int, list[int]] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    for record in saved_metrics["folds"]:
        fold = int(record["fold"])
        train = {qid for qid in qids if assignment[qid] != fold}
        validation = qids - train
        fitted = _fit(
            rankings, gold, train, maximum_k, seed + fold,
            float(config["multiclass"]["C"]), float(config["ridge"]["alpha"]),
        )
        selected, details = _select(
            "multiclass_ridge_ensemble",
            _raw(fitted, {qid: rankings[qid] for qid in validation}, maximum_k),
            float(record["lambda_multiclass"]),
        )
        observed = evaluate_predictions(
            {qid: gold[qid] for qid in validation}, selected,
        ).to_dict()
        for key in ("precision", "recall", "f2"):
            if not np.isclose(observed[key], record["validation_metrics"][key], atol=1e-12, rtol=0):
                raise ValueError(f"Outer fold {fold}: archived {key} did not reproduce")
        predictions.update(selected)
        diagnostics.update({qid: {"fold": fold, **value} for qid, value in details.items()})
    if set(predictions) != qids:
        raise ValueError("Outer-fold replay did not cover all queries")
    observed = evaluate_predictions(gold, predictions).to_dict()
    for key in ("precision", "recall", "f2"):
        if not np.isclose(observed[key], saved_metrics[key], atol=1e-12, rtol=0):
            raise ValueError(f"Aggregate archived {key} did not reproduce")
    return predictions, diagnostics


def audit_gold_flow(
    gold: Mapping[int, Sequence[int]],
    pool: Mapping[int, Sequence[int]],
    reranker: Mapping[int, Sequence[int]],
    lgbm: Mapping[int, Sequence[int]],
    selected: Mapping[int, Sequence[int]],
    *,
    maximum_k: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Partition gold assignments by their observed position, not causal effects.

    Each (qid, gold aid) is counted exactly once. A gold outside LightGBM top-5
    is separated by whether it was also outside reranker top-5 or was demoted
    across that boundary. The latter is an observed transition, not a causal
    counterfactual about refitting LightGBM.
    """
    qids = set(gold)
    for values in (pool, reranker, lgbm, selected):
        if set(values) != qids:
            raise ValueError("Stage query sets differ")
    counts: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for qid in sorted(qids):
        relevant = set(map(int, gold[qid]))
        candidates = set(map(int, pool[qid]))
        rerank_aids = list(map(int, reranker[qid]))
        lgbm_aids = list(map(int, lgbm[qid]))
        prediction = list(map(int, selected[qid]))
        if len(rerank_aids) != len(set(rerank_aids)) or len(lgbm_aids) != len(set(lgbm_aids)):
            raise ValueError(f"Duplicate article rank at qid {qid}")
        if candidates != set(rerank_aids) or candidates != set(lgbm_aids):
            raise ValueError(f"Candidate sets differ at qid {qid}")
        if not prediction or prediction != lgbm_aids[:len(prediction)]:
            raise ValueError(f"Final prediction is not a nonempty prefix at qid {qid}")
        rrank = {aid: index + 1 for index, aid in enumerate(rerank_aids)}
        lrank = {aid: index + 1 for index, aid in enumerate(lgbm_aids)}
        true_positive = relevant & set(prediction)
        missing = relevant - set(prediction)
        extra = set(prediction) - relevant
        outcome = "mixed" if missing and extra else "missing_only" if missing else "extra_only" if extra else "exact"
        outcomes[outcome] += 1
        article_details = []
        for aid in sorted(relevant):
            if aid in true_positive:
                status = "recovered"
            elif aid not in candidates:
                status = "outside_pool"
            elif lrank[aid] > maximum_k:
                status = "low_in_both_rankings" if rrank[aid] > maximum_k else "demoted_by_lgbm"
            else:
                status = "cut_by_prefix"
            counts[status] += 1
            article_details.append({"aid": aid, "status": status, "reranker_rank": rrank.get(aid), "lgbm_rank": lrank.get(aid)})
        rows.append({
            "qid": qid, "gold": sorted(relevant), "selected": prediction,
            "missing": sorted(missing), "extra": sorted(extra), "outcome": outcome,
            "pool_unique_articles": len(candidates), "gold_details": article_details,
            "reranker_top5": rerank_aids[:maximum_k], "lgbm_top5": lgbm_aids[:maximum_k],
        })
    gold_count = sum(len(set(values)) for values in gold.values())
    if sum(counts.values()) != gold_count:
        raise AssertionError("Gold partition is not exhaustive")
    return {
        "queries": len(qids), "gold_assignments": gold_count,
        "gold_partition": dict(counts), "query_outcomes": dict(outcomes),
        "false_positive_assignments": sum(len(row["extra"]) for row in rows),
        "false_negative_assignments": sum(len(row["missing"]) for row in rows),
        "final_metrics": evaluate_predictions(gold, selected).to_dict(),
        "pool_macro_recall": float(np.mean([
            len(set(gold[qid]) & set(pool[qid])) / len(set(gold[qid])) for qid in sorted(qids)
        ])),
        "reranker_top5_macro_recall": evaluate_predictions(
            gold, {qid: list(reranker[qid])[:maximum_k] for qid in qids},
        ).to_dict()["recall"],
        "lgbm_top5_macro_recall": evaluate_predictions(
            gold, {qid: list(lgbm[qid])[:maximum_k] for qid in qids},
        ).to_dict()["recall"],
        "maximum_k": maximum_k,
    }, rows
