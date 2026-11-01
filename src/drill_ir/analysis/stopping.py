"""Fold-safe learned prefix stopping for ranked legal-article candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
import random
from typing import Any

import numpy as np

from drill_ir.evaluation import evaluate_predictions


Ranking = tuple[Sequence[int], Sequence[float]]


def _folds(qids: Iterable[int], folds: int, seed: int) -> dict[int, int]:
    values = sorted(set(map(int, qids)))
    if folds < 2 or folds > len(values):
        raise ValueError("folds must be between two and the number of queries")
    random.Random(seed).shuffle(values)
    return {qid: index % folds for index, qid in enumerate(values)}


def _validate_ranking(qid: int, ranking: Ranking) -> tuple[list[int], np.ndarray]:
    aids, raw_scores = ranking
    aids = list(map(int, aids))
    scores = np.asarray(raw_scores, dtype=np.float64)
    if not aids or len(aids) != len(scores):
        raise ValueError(f"Invalid aids/scores for qid={qid}")
    if not np.isfinite(scores).all():
        raise ValueError(f"Non-finite ranking score for qid={qid}")
    if np.any(scores[:-1] < scores[1:]):
        raise ValueError(f"Ranking scores must be descending for qid={qid}")
    return aids, scores


def _oracle_prefix(aids: Sequence[int], gold: Iterable[int], maximum_k: int) -> int:
    relevant = set(map(int, gold))
    if not relevant:
        raise ValueError("Stopping-policy training requires non-empty gold labels")
    best_key = (-1.0, -1.0, -1.0, 0)
    best_k = 1
    for cutoff in range(1, min(maximum_k, len(aids)) + 1):
        selected = set(map(int, aids[:cutoff]))
        true_positive = len(selected & relevant)
        precision = true_positive / cutoff
        recall = true_positive / len(relevant)
        denominator = 4 * precision + recall
        f2 = 5 * precision * recall / denominator if denominator else 0.0
        key = (f2, precision, recall, -cutoff)
        if key > best_key:
            best_key = key
            best_k = cutoff
    return best_k


def _query_state(scores: np.ndarray, maximum_k: int) -> tuple[np.ndarray, np.ndarray]:
    values = scores[:maximum_k]
    padded = np.pad(values, (0, maximum_k - len(values)), constant_values=values[-1])
    gaps = padded[:-1] - padded[1:]
    shifted = padded - padded.max()
    weights = np.exp(np.clip(shifted, -50.0, 0.0))
    probabilities = weights / weights.sum()
    return padded, np.concatenate(
        [
            gaps,
            [
                padded.mean(),
                padded.std(),
                padded.max() - padded.min(),
                -float(np.sum(probabilities * np.log(probabilities + 1e-12))),
            ],
        ]
    )


def _boundary_features(scores: np.ndarray, cutoff: int, maximum_k: int) -> np.ndarray:
    if not 1 <= cutoff < maximum_k:
        raise ValueError("cutoff must be in [1, maximum_k)")
    padded, summary = _query_state(scores, maximum_k)
    shifted = padded - padded.max()
    probabilities = np.exp(np.clip(shifted, -50.0, 0.0))
    probabilities /= probabilities.sum()
    prefix = padded[:cutoff]
    tail = padded[cutoff:]
    boundary = np.asarray(
        [
            cutoff / maximum_k,
            padded[cutoff - 1],
            padded[cutoff],
            padded[cutoff - 1] - padded[cutoff],
            prefix.mean(),
            prefix.std(),
            tail.mean(),
            tail.std(),
            probabilities[:cutoff].sum(),
        ],
        dtype=np.float64,
    )
    return np.concatenate([padded, summary, boundary])


def _training_rows(
    rankings: Mapping[int, Ranking],
    gold: Mapping[int, Iterable[int]],
    qids: set[int],
    maximum_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for qid in sorted(qids):
        aids, scores = _validate_ranking(qid, rankings[qid])
        available_k = min(maximum_k, len(aids))
        target = _oracle_prefix(aids, gold[qid], available_k)
        for cutoff in range(1, available_k):
            rows.append(_boundary_features(scores, cutoff, maximum_k))
            labels.append(int(target > cutoff))
    if not rows:
        raise ValueError("No stopping boundaries were available for training")
    return np.vstack(rows), np.asarray(labels, dtype=np.int8)


def fit_stopping_model(
    rankings: Mapping[int, Ranking],
    gold: Mapping[int, Iterable[int]],
    qids: set[int],
    *,
    maximum_k: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit a serializable binary continue/stop model on complete query groups."""
    from sklearn.linear_model import LogisticRegression

    if maximum_k < 2:
        raise ValueError("maximum_k must be at least two")
    x_train, y_train = _training_rows(rankings, gold, qids, maximum_k)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0.0] = 1.0
    normalized = (x_train - mean) / scale
    labels = np.unique(y_train)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "policy": "fold_safe_prefix_stopping_logistic",
        "maximum_k": maximum_k,
        "feature_count": int(x_train.shape[1]),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "training_queries": len(qids),
        "training_boundaries": len(y_train),
    }
    if len(labels) == 1:
        payload.update(mode="constant", constant_continue_probability=float(labels[0]))
        return payload
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    model.fit(normalized, y_train)
    payload.update(
        mode="logistic",
        coefficient=model.coef_[0].tolist(),
        intercept=float(model.intercept_[0]),
    )
    return payload


def _continue_probabilities(
    model: Mapping[str, Any], scores: np.ndarray
) -> list[float]:
    maximum_k = int(model["maximum_k"])
    boundaries = min(maximum_k, len(scores)) - 1
    if boundaries <= 0:
        return []
    if model["mode"] == "constant":
        return [float(model["constant_continue_probability"])] * boundaries
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    probabilities = []
    for cutoff in range(1, boundaries + 1):
        features = (_boundary_features(scores, cutoff, maximum_k) - mean) / scale
        logit = float(features @ coefficient + float(model["intercept"]))
        probabilities.append(1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, logit)))))
    return probabilities


def predict_continue_probabilities(
    model: Mapping[str, Any], rankings: Mapping[int, Ranking]
) -> dict[int, list[float]]:
    result = {}
    for qid in sorted(rankings):
        _, scores = _validate_ranking(qid, rankings[qid])
        result[int(qid)] = _continue_probabilities(model, scores)
    return result


def select_with_stopping_probabilities(
    rankings: Mapping[int, Ranking],
    probabilities: Mapping[int, Sequence[float]],
    *,
    continue_threshold: float,
    maximum_k: int,
) -> dict[int, list[int]]:
    if not 0.0 <= continue_threshold <= 1.0:
        raise ValueError("continue_threshold must be in [0, 1]")
    selected = {}
    for qid in sorted(rankings):
        aids, _ = _validate_ranking(qid, rankings[qid])
        limit = min(maximum_k, len(aids))
        count = limit
        values = list(map(float, probabilities[qid]))
        if len(values) != max(0, limit - 1):
            raise ValueError(f"Stopping probability count mismatch for qid={qid}")
        for cutoff, probability in enumerate(values, start=1):
            if probability < continue_threshold:
                count = cutoff
                break
        selected[qid] = list(dict.fromkeys(aids[:count]))
    return selected


def tune_stopping_threshold(
    gold: Mapping[int, Iterable[int]],
    rankings: Mapping[int, Ranking],
    probabilities: Mapping[int, Sequence[float]],
    *,
    maximum_k: int,
) -> dict[str, Any]:
    candidates = sorted(
        {0.0, 1.0, *[float(value) for values in probabilities.values() for value in values]}
    )
    best = None
    for threshold in candidates:
        predictions = select_with_stopping_probabilities(
            rankings,
            probabilities,
            continue_threshold=threshold,
            maximum_k=maximum_k,
        )
        metrics = evaluate_predictions(gold, predictions).to_dict()
        sizes = [len(values) for values in predictions.values()]
        candidate = {
            "continue_threshold": threshold,
            **metrics,
            "mean_output_size": float(np.mean(sizes)),
        }
        key = (metrics["f2"], metrics["precision"], metrics["recall"], threshold)
        if best is None or key > best[0]:
            best = (key, candidate)
    assert best is not None
    return best[1]


def _cross_fitted_probabilities(
    rankings: Mapping[int, Ranking],
    gold: Mapping[int, Iterable[int]],
    qids: set[int],
    *,
    folds: int,
    seed: int,
    maximum_k: int,
) -> dict[int, list[float]]:
    fold_by_qid = _folds(qids, folds, seed)
    probabilities: dict[int, list[float]] = {}
    for fold in range(folds):
        train_qids = {qid for qid in qids if fold_by_qid[qid] != fold}
        validation_qids = qids - train_qids
        model = fit_stopping_model(
            rankings, gold, train_qids, maximum_k=maximum_k, seed=seed + fold
        )
        probabilities.update(
            predict_continue_probabilities(
                model, {qid: rankings[qid] for qid in validation_qids}
            )
        )
    if set(probabilities) != qids:
        raise RuntimeError("Cross-fitted stopping probabilities are incomplete")
    return probabilities


def evaluate_nested_stopping_policy(
    gold: Mapping[int, Iterable[int]],
    rankings: Mapping[int, Ranking],
    *,
    folds: int = 5,
    seed: int = 42,
    maximum_k: int = 5,
) -> dict[str, Any]:
    """Estimate stopping quality with threshold tuning nested inside each fold."""
    if set(gold) != set(rankings):
        raise ValueError("Gold and ranking qids must match exactly")
    for qid, ranking in rankings.items():
        _validate_ranking(qid, ranking)
    qids = set(map(int, gold))
    outer = _folds(qids, folds, seed)
    predictions: dict[int, list[int]] = {}
    fold_records = []
    for fold in range(folds):
        train_qids = {qid for qid in qids if outer[qid] != fold}
        validation_qids = qids - train_qids
        inner_probabilities = _cross_fitted_probabilities(
            rankings,
            gold,
            train_qids,
            folds=folds,
            seed=seed + 100 + fold,
            maximum_k=maximum_k,
        )
        tuned = tune_stopping_threshold(
            {qid: gold[qid] for qid in train_qids},
            {qid: rankings[qid] for qid in train_qids},
            inner_probabilities,
            maximum_k=maximum_k,
        )
        model = fit_stopping_model(
            rankings, gold, train_qids, maximum_k=maximum_k, seed=seed + fold
        )
        validation_rankings = {qid: rankings[qid] for qid in validation_qids}
        probabilities = predict_continue_probabilities(model, validation_rankings)
        selected = select_with_stopping_probabilities(
            validation_rankings,
            probabilities,
            continue_threshold=float(tuned["continue_threshold"]),
            maximum_k=maximum_k,
        )
        predictions.update(selected)
        fold_records.append(
            {
                "fold": fold,
                "train_queries": len(train_qids),
                "validation_queries": len(validation_qids),
                "continue_threshold": tuned["continue_threshold"],
                "validation_metrics": evaluate_predictions(
                    {qid: gold[qid] for qid in validation_qids}, selected
                ).to_dict(),
            }
        )
    metrics = evaluate_predictions(gold, predictions).to_dict()
    sizes = [len(values) for values in predictions.values()]
    return {
        **metrics,
        "output_size": {
            "minimum": min(sizes),
            "mean": float(np.mean(sizes)),
            "median": float(np.median(sizes)),
            "maximum": max(sizes),
        },
        "folds": fold_records,
        "protocol": "outer query folds; inner cross-fitted threshold tuning",
    }


def fit_deployment_stopping_policy(
    gold: Mapping[int, Iterable[int]],
    rankings: Mapping[int, Ranking],
    *,
    folds: int = 5,
    seed: int = 42,
    maximum_k: int = 5,
) -> dict[str, Any]:
    """Tune from cross-fitted dev probabilities, then fit one deployable model."""
    if set(gold) != set(rankings):
        raise ValueError("Gold and ranking qids must match exactly")
    for qid, ranking in rankings.items():
        _validate_ranking(qid, ranking)
    qids = set(map(int, gold))
    probabilities = _cross_fitted_probabilities(
        rankings,
        gold,
        qids,
        folds=folds,
        seed=seed,
        maximum_k=maximum_k,
    )
    tuned = tune_stopping_threshold(
        gold, rankings, probabilities, maximum_k=maximum_k
    )
    model = fit_stopping_model(
        rankings, gold, qids, maximum_k=maximum_k, seed=seed
    )
    return {"model": model, "cross_fitted_tuning": tuned}


def apply_stopping_policy(
    policy: Mapping[str, Any], rankings: Mapping[int, Ranking]
) -> dict[int, list[int]]:
    model = policy["model"]
    probabilities = predict_continue_probabilities(model, rankings)
    return select_with_stopping_probabilities(
        rankings,
        probabilities,
        continue_threshold=float(policy["cross_fitted_tuning"]["continue_threshold"]),
        maximum_k=int(model["maximum_k"]),
    )
