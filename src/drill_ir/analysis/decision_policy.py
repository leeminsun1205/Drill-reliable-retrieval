"""Leakage-aware final-set policy evaluation over saved OOF rankings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
import random
from typing import Any

import numpy as np

from drill_ir.evaluation import evaluate_predictions, tune_score_thresholds


Ranking = tuple[Sequence[int], Sequence[float]]


def _folds(qids: Iterable[int], *, folds: int, seed: int) -> dict[int, int]:
    values = sorted(set(map(int, qids)))
    if folds < 2 or folds > len(values):
        raise ValueError("folds must be between two and the number of queries")
    random.Random(seed).shuffle(values)
    return {qid: index % folds for index, qid in enumerate(values)}


def _labels(
    rankings: Mapping[int, Ranking], gold: Mapping[int, Iterable[int]], qids: set[int]
) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    labels: list[int] = []
    for qid in sorted(qids):
        aids, values = rankings[qid]
        if len(aids) != len(values):
            raise ValueError(f"Ranking aids/scores length mismatch for qid={qid}")
        relevant = set(gold[qid])
        scores.extend(map(float, values))
        labels.extend(int(int(aid) in relevant) for aid in aids)
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int8)


def _apply_threshold(
    rankings: Mapping[int, Ranking], *, threshold: float, fallback_k: int
) -> dict[int, list[int]]:
    predictions: dict[int, list[int]] = {}
    for qid, (aids, scores) in rankings.items():
        count = sum(float(score) >= threshold for score in scores)
        count = count or min(fallback_k, len(aids))
        predictions[int(qid)] = list(dict.fromkeys(map(int, aids[:count])))
    return predictions


def _fit_calibrator(
    method: str,
    train_scores: np.ndarray,
    train_labels: np.ndarray,
) -> Any:
    if method == "raw":
        return None
    if len(np.unique(train_labels)) != 2:
        raise ValueError("Calibration requires both binary labels")
    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(train_scores.reshape(-1, 1), train_labels)
        if float(model.coef_[0, 0]) <= 0:
            raise ValueError("Platt calibration reversed the ranking score direction")
        return model
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(train_scores, train_labels)
        return model
    raise ValueError(f"Unknown calibration method: {method}")


def _apply_calibrator(method: str, calibrator: Any, values: np.ndarray) -> np.ndarray:
    if method == "raw":
        return values
    if method == "platt":
        return calibrator.predict_proba(values.reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return np.asarray(calibrator.predict(values), dtype=np.float64)
    raise ValueError(f"Unknown calibration method: {method}")


def _calibrated_rankings(
    rankings: Mapping[int, Ranking],
    qids: set[int],
    *,
    method: str,
    calibrator: Any,
) -> dict[int, tuple[list[int], list[float]]]:
    result = {}
    for qid in sorted(qids):
        aids, scores = rankings[qid]
        calibrated = _apply_calibrator(
            method, calibrator, np.asarray(scores, dtype=np.float64)
        )
        result[qid] = (list(map(int, aids)), calibrated.tolist())
    return result


def _query_features(scores: Sequence[float], width: int = 5) -> np.ndarray:
    values = np.asarray(scores[:width], dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cardinality policy requires non-empty rankings")
    padded = np.pad(values, (0, width - len(values)), constant_values=values[-1])
    gaps = padded[:-1] - padded[1:]
    shifted = padded - padded.max()
    weights = np.exp(shifted)
    probabilities = weights / weights.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return np.concatenate(
        [padded, gaps, [padded.mean(), padded.std(), padded.max() - padded.min(), entropy]]
    )


def _cardinality_predictions(
    rankings: Mapping[int, Ranking],
    gold: Mapping[int, Iterable[int]],
    train_qids: set[int],
    validation_qids: set[int],
    *,
    maximum_k: int,
    seed: int,
) -> dict[int, list[int]]:
    from sklearn.linear_model import LogisticRegression

    x_train = np.vstack([_query_features(rankings[qid][1]) for qid in sorted(train_qids)])
    y_train = np.asarray(
        [min(maximum_k, len(set(gold[qid]))) for qid in sorted(train_qids)], dtype=int
    )
    if len(np.unique(y_train)) < 2:
        fixed = int(y_train[0])
        return {
            qid: list(map(int, rankings[qid][0][:fixed])) for qid in validation_qids
        }
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    x_validation = np.vstack(
        [_query_features(rankings[qid][1]) for qid in sorted(validation_qids)]
    )
    predicted = model.predict(x_validation)
    return {
        qid: list(map(int, rankings[qid][0][: max(1, min(maximum_k, int(count)))]))
        for qid, count in zip(sorted(validation_qids), predicted)
    }


def evaluate_nested_decision_policies(
    gold: Mapping[int, Iterable[int]],
    rankings: Mapping[int, Ranking],
    *,
    folds: int = 5,
    seed: int = 42,
    fallback_k: int = 2,
    maximum_k: int = 5,
) -> dict[str, Any]:
    """Compare fold-safe threshold calibration and cardinality prediction."""
    if set(gold) != set(rankings):
        raise ValueError("Gold and ranking qids must match exactly")
    if fallback_k <= 0 or maximum_k < fallback_k:
        raise ValueError("Require 0 < fallback_k <= maximum_k")
    fold_by_qid = _folds(gold, folds=folds, seed=seed)
    methods = ("raw", "platt", "isotonic")
    predictions = {method: {} for method in methods}
    predictions["cardinality_logistic"] = {}
    fold_records: dict[str, list[dict[str, Any]]] = {name: [] for name in predictions}

    for fold in range(folds):
        train_qids = {qid for qid, value in fold_by_qid.items() if value != fold}
        validation_qids = {qid for qid, value in fold_by_qid.items() if value == fold}
        train_scores, train_labels = _labels(rankings, gold, train_qids)
        for method in methods:
            calibrator = _fit_calibrator(method, train_scores, train_labels)
            tuned_rankings = _calibrated_rankings(
                rankings,
                train_qids,
                method=method,
                calibrator=calibrator,
            )
            tuned = tune_score_thresholds(
                {qid: gold[qid] for qid in train_qids}, tuned_rankings, [fallback_k]
            )["best_by_fallback_k"][str(fallback_k)]
            threshold = tuned["threshold"]
            if threshold is None:
                threshold = math.inf
            validation_rankings = _calibrated_rankings(
                rankings,
                validation_qids,
                method=method,
                calibrator=calibrator,
            )
            selected = _apply_threshold(
                validation_rankings,
                threshold=float(threshold),
                fallback_k=fallback_k,
            )
            predictions[method].update(selected)
            fold_records[method].append(
                {
                    "fold": fold,
                    "train_queries": len(train_qids),
                    "validation_queries": len(validation_qids),
                    "threshold": None if math.isinf(threshold) else float(threshold),
                    "validation_metrics": evaluate_predictions(
                        {qid: gold[qid] for qid in validation_qids}, selected
                    ).to_dict(),
                }
            )

        selected_cardinality = _cardinality_predictions(
            rankings,
            gold,
            train_qids,
            validation_qids,
            maximum_k=maximum_k,
            seed=seed + fold,
        )
        predictions["cardinality_logistic"].update(selected_cardinality)
        fold_records["cardinality_logistic"].append(
            {
                "fold": fold,
                "train_queries": len(train_qids),
                "validation_queries": len(validation_qids),
                "validation_metrics": evaluate_predictions(
                    {qid: gold[qid] for qid in validation_qids}, selected_cardinality
                ).to_dict(),
            }
        )

    policies = {}
    for name, selected in predictions.items():
        sizes = [len(values) for values in selected.values()]
        policies[name] = {
            **evaluate_predictions(gold, selected).to_dict(),
            "output_size": {
                "minimum": min(sizes),
                "mean": float(np.mean(sizes)),
                "median": float(np.median(sizes)),
                "maximum": max(sizes),
            },
            "folds": fold_records[name],
        }
    return {
        "protocol": {
            "folds": folds,
            "seed": seed,
            "fallback_k": fallback_k,
            "maximum_k": maximum_k,
            "note": "Every learned policy is fit/tuned without its outer validation qids.",
        },
        "policies": policies,
        "predictions": predictions,
    }
