"""Confirmatory policies built on the selected multiclass cardinality model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import random
from typing import Any

import numpy as np

from drill_ir.evaluation import evaluate_predictions
from .prefix_utility import _prefix_f2, _prefix_features, _validate_ranking


Ranking = tuple[Sequence[int], Sequence[float]]


def _folds(qids: Iterable[int], folds: int, seed: int) -> dict[int, int]:
    values = sorted(set(map(int, qids)))
    random.Random(seed).shuffle(values)
    return {qid: index % folds for index, qid in enumerate(values)}


def _query(scores: np.ndarray, maximum_k: int) -> np.ndarray:
    return _prefix_features(scores, 1, maximum_k)


def _rows(rankings, gold, qids, maximum_k):
    query_x, labels, utility_x, utility_y, utility_by_qid = [], [], [], [], {}
    for qid in sorted(qids):
        aids, scores = _validate_ranking(qid, rankings[qid])
        utilities = np.asarray([_prefix_f2(aids, gold[qid], k) for k in range(1, min(maximum_k, len(aids)) + 1)])
        features = np.vstack([_prefix_features(scores, k, maximum_k) for k in range(1, len(utilities) + 1)])
        query_x.append(_query(scores, maximum_k)); labels.append(int(np.argmax(utilities)))
        utility_x.extend(features); utility_y.extend(utilities); utility_by_qid[qid] = utilities
    return np.vstack(query_x), np.asarray(labels), np.vstack(utility_x), np.asarray(utility_y), utility_by_qid


def _fit(rankings, gold, qids, maximum_k, seed, C, alpha):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    query_x, labels, utility_x, utility_y, utilities = _rows(rankings, gold, qids, maximum_k)
    query_scaler = StandardScaler().fit(query_x)
    multiclass = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=seed).fit(query_scaler.transform(query_x), labels)
    utility_scaler = StandardScaler().fit(utility_x)
    ridge = Ridge(alpha=alpha).fit(utility_scaler.transform(utility_x), utility_y)
    matrix = np.zeros((maximum_k, maximum_k), dtype=np.float64)
    counts = np.zeros(maximum_k, dtype=np.int64)
    for qid, true_class in zip(sorted(qids), labels, strict=True):
        values = utilities[qid]
        matrix[true_class, : len(values)] += values; counts[true_class] += 1
    overall = np.mean(np.vstack(list(utilities.values())), axis=0)
    for label in range(maximum_k):
        matrix[label] = matrix[label] / counts[label] if counts[label] else overall
    return {"query_scaler": query_scaler, "multiclass": multiclass, "utility_scaler": utility_scaler, "ridge": ridge, "utility_matrix": matrix}


def _raw(fitted, rankings, maximum_k):
    result = {}
    classes = list(map(int, fitted["multiclass"].classes_))
    for qid in sorted(rankings):
        aids, scores = _validate_ranking(qid, rankings[qid]); limit = min(maximum_k, len(aids))
        probability_raw = fitted["multiclass"].predict_proba(fitted["query_scaler"].transform(_query(scores, maximum_k)[None, :]))[0]
        probabilities = np.asarray([probability_raw[classes.index(k)] if k in classes else 0.0 for k in range(limit)])
        features = np.vstack([_prefix_features(scores, k, maximum_k) for k in range(1, limit + 1)])
        ridge = np.asarray(fitted["ridge"].predict(fitted["utility_scaler"].transform(features)))
        expected = probabilities @ fitted["utility_matrix"][:limit, :limit]
        result[qid] = {"aids": aids, "probabilities": probabilities, "ridge": ridge, "expected": expected}
    return result


def _standardize(values: np.ndarray) -> np.ndarray:
    scale = float(values.std())
    return (values - values.mean()) / scale if scale else np.zeros_like(values)


def _select(case: str, raw, lambda_multiclass: float | None = None):
    selected, diagnostics = {}, {}
    for qid, row in raw.items():
        if case == "baseline_multiclass": values = row["probabilities"]
        elif case == "cost_sensitive_multiclass": values = row["expected"]
        elif case == "multiclass_ridge_ensemble":
            if lambda_multiclass is None: raise ValueError("Ensemble lambda is required")
            values = lambda_multiclass * _standardize(np.log(row["probabilities"] + 1e-12)) + (1 - lambda_multiclass) * _standardize(row["ridge"])
        else: raise ValueError(case)
        cutoff = int(np.argmax(values)) + 1
        selected[qid] = list(dict.fromkeys(row["aids"][:cutoff]))
        diagnostics[qid] = {"selected_k": cutoff, "decision_scores": list(map(float, values))}
    return selected, diagnostics


def _crossfit(case, rankings, gold, qids, folds, seed, maximum_k, C, alpha, lambda_value=None):
    assignment = _folds(qids, folds, seed); selected, diagnostics = {}, {}
    for fold in range(folds):
        train = {qid for qid in qids if assignment[qid] != fold}; validation = qids - train
        fitted = _fit(rankings, gold, train, maximum_k, seed + fold, C, alpha)
        predictions, details = _select(case, _raw(fitted, {qid: rankings[qid] for qid in validation}, maximum_k), lambda_value)
        selected.update(predictions); diagnostics.update(details)
    return selected, diagnostics


def _choose_lambda(rankings, gold, qids, folds, seed, maximum_k, C, alpha, values):
    records, best = [], None
    for index, value in enumerate(values):
        predictions, _ = _crossfit("multiclass_ridge_ensemble", rankings, gold, qids, folds, seed + 1000, maximum_k, C, alpha, value)
        metrics = evaluate_predictions({qid: gold[qid] for qid in qids}, predictions).to_dict()
        records.append({"lambda_multiclass": value, "metrics": metrics})
        key = (metrics["f2"], metrics["precision"], metrics["recall"], -index)
        if best is None or key > best[0]: best = (key, value)
    return float(best[1]), records


def evaluate_confirmatory_case(case, rankings, gold, *, outer_folds, inner_folds, seed, maximum_k, C, alpha, lambda_values):
    qids = set(map(int, gold)); outer = _folds(qids, outer_folds, seed)
    predictions, diagnostics, records = {}, {}, []
    for fold in range(outer_folds):
        train = {qid for qid in qids if outer[qid] != fold}; validation = qids - train
        selected_lambda, tuning = (None, [])
        if case == "multiclass_ridge_ensemble": selected_lambda, tuning = _choose_lambda(rankings, gold, train, inner_folds, seed + fold, maximum_k, C, alpha, lambda_values)
        fitted = _fit(rankings, gold, train, maximum_k, seed + fold, C, alpha)
        selected, details = _select(case, _raw(fitted, {qid: rankings[qid] for qid in validation}, maximum_k), selected_lambda)
        predictions.update(selected); diagnostics.update(details)
        records.append({"fold": fold, "lambda_multiclass": selected_lambda, "inner_tuning": tuning, "validation_metrics": evaluate_predictions({qid: gold[qid] for qid in validation}, selected).to_dict()})
    metrics = evaluate_predictions(gold, predictions).to_dict(); sizes = np.asarray([len(v) for v in predictions.values()])
    metrics.update(output_size={"minimum":int(sizes.min()),"mean":float(sizes.mean()),"median":float(np.median(sizes)),"maximum":int(sizes.max())},folds=records)
    deployment_lambda, deployment_tuning = (None, [])
    if case == "multiclass_ridge_ensemble": deployment_lambda, deployment_tuning = _choose_lambda(rankings, gold, qids, outer_folds, seed, maximum_k, C, alpha, lambda_values)
    fitted = _fit(rankings, gold, qids, maximum_k, seed, C, alpha)
    return metrics, {"lambda_multiclass": deployment_lambda, "cross_fitted_tuning": deployment_tuning}, fitted


def apply_confirmatory_case(case: str, fitted: Any, rankings: Mapping[int, Ranking], maximum_k: int, deployment: Mapping[str, Any]):
    return _select(case, _raw(fitted, rankings, maximum_k), deployment.get("lambda_multiclass"))

