"""Nested screening of output-cardinality policies on frozen rankings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import product
import json
import random
from typing import Any

import numpy as np

from drill_ir.evaluation import evaluate_predictions
from .prefix_utility import _prefix_f2, _prefix_features, _validate_ranking


Ranking = tuple[Sequence[int], Sequence[float]]


def _folds(qids: Iterable[int], folds: int, seed: int) -> dict[int, int]:
    values = sorted(set(map(int, qids)))
    if folds < 2 or folds > len(values):
        raise ValueError("Invalid fold count")
    random.Random(seed).shuffle(values)
    return {qid: index % folds for index, qid in enumerate(values)}


def _utilities(aids: Sequence[int], gold: Iterable[int], maximum_k: int) -> np.ndarray:
    return np.asarray(
        [_prefix_f2(aids, gold, k) for k in range(1, min(maximum_k, len(aids)) + 1)],
        dtype=np.float64,
    )


def _query_features(scores: np.ndarray, maximum_k: int) -> np.ndarray:
    return _prefix_features(scores, 1, maximum_k)


def parameter_grid(case: str, spec: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = list(spec)
    if case == "utility_ridge" or case == "pairwise_logistic":
        keys = ["alpha"] if case == "utility_ridge" else ["C"]
    values = [list(spec[key]) for key in keys]
    return [dict(zip(keys, row, strict=True)) for row in product(*values)]


def _fit(case: str, params: Mapping[str, Any], rankings, gold, qids, maximum_k, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    query_x, query_y = [], []
    utility_x, utility_y = [], []
    pair_x, pair_y = [], []
    for qid in sorted(qids):
        aids, scores = _validate_ranking(qid, rankings[qid])
        features = [_prefix_features(scores, k, maximum_k) for k in range(1, min(maximum_k, len(aids)) + 1)]
        utilities = _utilities(aids, gold[qid], maximum_k)
        query_x.append(_query_features(scores, maximum_k))
        query_y.append(int(np.argmax(utilities)))
        utility_x.extend(features)
        utility_y.extend(utilities)
        for left in range(len(features)):
            for right in range(left + 1, len(features)):
                if utilities[left] == utilities[right]:
                    continue
                difference = features[left] - features[right]
                label = int(utilities[left] > utilities[right])
                pair_x.extend([difference, -difference])
                pair_y.extend([label, 1 - label])
    scaler = StandardScaler()
    if case.startswith("multiclass"):
        x = scaler.fit_transform(np.vstack(query_x)); y = np.asarray(query_y)
        if case == "multiclass_logistic":
            model = LogisticRegression(C=float(params["C"]), class_weight=params["class_weight"], max_iter=3000, random_state=seed)
        else:
            model = HistGradientBoostingClassifier(**params, random_state=seed)
    elif case.startswith("utility"):
        x = scaler.fit_transform(np.vstack(utility_x)); y = np.asarray(utility_y)
        model = Ridge(alpha=float(params["alpha"])) if case == "utility_ridge" else HistGradientBoostingRegressor(**params, random_state=seed)
    elif case == "pairwise_logistic":
        x = scaler.fit_transform(np.vstack(pair_x)); y = np.asarray(pair_y)
        model = LogisticRegression(C=float(params["C"]), class_weight="balanced", max_iter=3000, random_state=seed)
    else:
        raise ValueError(f"Unknown policy case: {case}")
    model.fit(x, y)
    return scaler, model


def _predict(case: str, fitted, rankings, maximum_k):
    scaler, model = fitted
    selected, diagnostics = {}, {}
    for qid in sorted(rankings):
        aids, scores = _validate_ranking(qid, rankings[qid])
        features = np.vstack([_prefix_features(scores, k, maximum_k) for k in range(1, min(maximum_k, len(aids)) + 1)])
        if case.startswith("multiclass"):
            query = scaler.transform(_query_features(scores, maximum_k)[None, :])
            probabilities = model.predict_proba(query)[0]
            classes = list(map(int, model.classes_))
            values = [float(probabilities[classes.index(k)]) if k in classes else 0.0 for k in range(len(features))]
        elif case.startswith("utility"):
            values = list(map(float, model.predict(scaler.transform(features))))
        else:
            coefficient = model.coef_[0]
            values = list(map(float, scaler.transform(features) @ coefficient))
        cutoff = int(np.argmax(values)) + 1
        selected[qid] = list(dict.fromkeys(aids[:cutoff]))
        diagnostics[qid] = {"selected_k": cutoff, "policy_scores": values}
    return selected, diagnostics


def _crossfit(case, params, rankings, gold, qids, folds, seed, maximum_k):
    fold_by_qid = _folds(qids, folds, seed)
    predictions, diagnostics = {}, {}
    for fold in range(folds):
        train = {qid for qid in qids if fold_by_qid[qid] != fold}
        validation = qids - train
        fitted = _fit(case, params, rankings, gold, train, maximum_k, seed + fold)
        selected, details = _predict(case, fitted, {qid: rankings[qid] for qid in validation}, maximum_k)
        predictions.update(selected); diagnostics.update(details)
    return predictions, diagnostics


def _select_params(case, grid, rankings, gold, qids, folds, seed, maximum_k):
    best = None
    records = []
    for index, params in enumerate(grid):
        predictions, _ = _crossfit(case, params, rankings, gold, qids, folds, seed + 1000, maximum_k)
        metrics = evaluate_predictions({qid: gold[qid] for qid in qids}, predictions).to_dict()
        record = {"params": params, "metrics": metrics}; records.append(record)
        key = (metrics["f2"], metrics["precision"], metrics["recall"], -index)
        if best is None or key > best[0]: best = (key, params)
    assert best is not None
    return dict(best[1]), records


def evaluate_nested_case(case, spec, rankings, gold, *, outer_folds, inner_folds, seed, maximum_k):
    qids = set(map(int, gold))
    if qids != set(rankings): raise ValueError("Gold and ranking qids must match")
    grid = parameter_grid(case, spec)
    outer = _folds(qids, outer_folds, seed)
    predictions, diagnostics, fold_records = {}, {}, []
    for fold in range(outer_folds):
        train = {qid for qid in qids if outer[qid] != fold}; validation = qids - train
        params, _ = _select_params(case, grid, rankings, gold, train, inner_folds, seed + fold, maximum_k)
        fitted = _fit(case, params, rankings, gold, train, maximum_k, seed + fold)
        selected, details = _predict(case, fitted, {qid: rankings[qid] for qid in validation}, maximum_k)
        predictions.update(selected); diagnostics.update(details)
        fold_records.append({"fold": fold, "selected_params": params, "validation_metrics": evaluate_predictions({qid: gold[qid] for qid in validation}, selected).to_dict()})
    metrics = evaluate_predictions(gold, predictions).to_dict()
    oracle = {}
    for qid in qids:
        aids, _ = _validate_ranking(qid, rankings[qid]); oracle[qid] = int(np.argmax(_utilities(aids, gold[qid], maximum_k))) + 1
    chosen = {qid: int(diagnostics[qid]["selected_k"]) for qid in qids}
    sizes = np.asarray(list(chosen.values()))
    metrics.update(output_size={"minimum": int(sizes.min()), "mean": float(sizes.mean()), "median": float(np.median(sizes)), "maximum": int(sizes.max())}, oracle_k_diagnostic={"exact_rate": sum(chosen[q]==oracle[q] for q in qids)/len(qids), "under_selection_rate": sum(chosen[q]<oracle[q] for q in qids)/len(qids), "over_selection_rate": sum(chosen[q]>oracle[q] for q in qids)/len(qids)}, folds=fold_records)
    deployment_params, tuning = _select_params(case, grid, rankings, gold, qids, outer_folds, seed, maximum_k)
    fitted = _fit(case, deployment_params, rankings, gold, qids, maximum_k, seed)
    return metrics, deployment_params, tuning, fitted


def apply_fitted_case(case: str, fitted: Any, rankings: Mapping[int, Ranking], maximum_k: int):
    return _predict(case, fitted, rankings, maximum_k)

