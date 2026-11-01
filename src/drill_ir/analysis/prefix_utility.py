"""Fold-safe listwise prefix-utility selection for ranked legal articles."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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


def _prefix_f2(aids: Sequence[int], gold: Iterable[int], cutoff: int) -> float:
    relevant = set(map(int, gold))
    if not relevant:
        raise ValueError("Prefix-utility training requires non-empty gold labels")
    selected = set(map(int, aids[:cutoff]))
    true_positive = len(selected & relevant)
    precision = true_positive / cutoff
    recall = true_positive / len(relevant)
    denominator = 4 * precision + recall
    return 5 * precision * recall / denominator if denominator else 0.0


def _prefix_features(scores: np.ndarray, cutoff: int, maximum_k: int) -> np.ndarray:
    limit = min(maximum_k, len(scores))
    if not 1 <= cutoff <= limit:
        raise ValueError("cutoff is outside the available prefix range")
    values = scores[:limit]
    padded = np.pad(values, (0, maximum_k - limit), constant_values=values[-1])
    gaps = padded[:-1] - padded[1:]
    shifted = padded - padded.max()
    weights = np.exp(np.clip(shifted, -50.0, 0.0))
    probabilities = weights / weights.sum()
    prefix = padded[:cutoff]
    tail = padded[cutoff:] if cutoff < maximum_k else padded[-1:]
    boundary_gap = padded[cutoff - 1] - padded[cutoff] if cutoff < maximum_k else 0.0
    return np.asarray(
        [
            *padded,
            *gaps,
            padded.mean(),
            padded.std(),
            padded.max() - padded.min(),
            -float(np.sum(probabilities * np.log(probabilities + 1e-12))),
            cutoff / maximum_k,
            padded[cutoff - 1],
            boundary_gap,
            prefix.mean(),
            prefix.std(),
            tail.mean(),
            tail.std(),
            probabilities[:cutoff].sum(),
        ],
        dtype=np.float64,
    )


def _training_rows(
    rankings: Mapping[int, Ranking],
    gold: Mapping[int, Iterable[int]],
    qids: set[int],
    maximum_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for qid in sorted(qids):
        aids, scores = _validate_ranking(qid, rankings[qid])
        for cutoff in range(1, min(maximum_k, len(aids)) + 1):
            rows.append(_prefix_features(scores, cutoff, maximum_k))
            targets.append(_prefix_f2(aids, gold[qid], cutoff))
    if not rows:
        raise ValueError("No prefix-utility training rows were available")
    return np.vstack(rows), np.asarray(targets, dtype=np.float64)


def fit_prefix_utility_model(
    rankings: Mapping[int, Ranking],
    gold: Mapping[int, Iterable[int]],
    qids: set[int],
    *,
    maximum_k: int = 5,
    alpha: float = 10.0,
) -> dict[str, Any]:
    """Fit a serializable Ridge regressor to per-query prefix F2 utilities."""
    from sklearn.linear_model import Ridge

    if maximum_k < 2 or alpha <= 0:
        raise ValueError("maximum_k must be >=2 and alpha must be positive")
    x_train, y_train = _training_rows(rankings, gold, qids, maximum_k)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0.0] = 1.0
    model = Ridge(alpha=alpha)
    model.fit((x_train - mean) / scale, y_train)
    return {
        "schema_version": 1,
        "policy": "fold_safe_listwise_prefix_utility_ridge",
        "maximum_k": maximum_k,
        "alpha": alpha,
        "feature_count": int(x_train.shape[1]),
        "training_queries": len(qids),
        "training_prefixes": len(y_train),
        "target": "per_query_prefix_f2",
        "tie_break": "smaller_k",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": np.asarray(model.coef_).tolist(),
        "intercept": float(model.intercept_),
    }


def _predict_k(model: Mapping[str, Any], scores: np.ndarray) -> tuple[int, list[float]]:
    maximum_k = int(model["maximum_k"])
    limit = min(maximum_k, len(scores))
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    utilities = []
    for cutoff in range(1, limit + 1):
        features = (_prefix_features(scores, cutoff, maximum_k) - mean) / scale
        utilities.append(float(features @ coefficient + float(model["intercept"])))
    # np.argmax returns the first maximum, implementing the declared smaller-k tie break.
    return int(np.argmax(utilities)) + 1, utilities


def apply_prefix_utility_policy(
    model: Mapping[str, Any], rankings: Mapping[int, Ranking]
) -> tuple[dict[int, list[int]], dict[int, dict[str, Any]]]:
    selected: dict[int, list[int]] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    for qid in sorted(rankings):
        aids, scores = _validate_ranking(qid, rankings[qid])
        cutoff, utilities = _predict_k(model, scores)
        selected[qid] = list(dict.fromkeys(aids[:cutoff]))
        diagnostics[qid] = {"selected_k": cutoff, "predicted_utilities": utilities}
    return selected, diagnostics


def evaluate_nested_prefix_utility(
    gold: Mapping[int, Iterable[int]],
    rankings: Mapping[int, Ranking],
    *,
    folds: int = 5,
    seed: int = 42,
    maximum_k: int = 5,
    alpha: float = 10.0,
) -> dict[str, Any]:
    """Evaluate with query-grouped outer folds; no held-out labels fit the model."""
    if set(gold) != set(rankings):
        raise ValueError("Gold and ranking qids must match exactly")
    qids = set(map(int, gold))
    fold_by_qid = _folds(qids, folds, seed)
    predictions: dict[int, list[int]] = {}
    selected_k: dict[int, int] = {}
    oracle_k: dict[int, int] = {}
    fold_records = []
    for fold in range(folds):
        train_qids = {qid for qid in qids if fold_by_qid[qid] != fold}
        validation_qids = qids - train_qids
        model = fit_prefix_utility_model(
            rankings, gold, train_qids, maximum_k=maximum_k, alpha=alpha
        )
        selected, diagnostics = apply_prefix_utility_policy(
            model, {qid: rankings[qid] for qid in validation_qids}
        )
        predictions.update(selected)
        for qid in validation_qids:
            selected_k[qid] = int(diagnostics[qid]["selected_k"])
            aids, _ = _validate_ranking(qid, rankings[qid])
            utilities = [
                _prefix_f2(aids, gold[qid], cutoff)
                for cutoff in range(1, min(maximum_k, len(aids)) + 1)
            ]
            oracle_k[qid] = int(np.argmax(utilities)) + 1
        fold_records.append(
            {
                "fold": fold,
                "train_queries": len(train_qids),
                "validation_queries": len(validation_qids),
                "validation_metrics": evaluate_predictions(
                    {qid: gold[qid] for qid in validation_qids}, selected
                ).to_dict(),
            }
        )
    metrics = evaluate_predictions(gold, predictions).to_dict()
    sizes = np.asarray(list(selected_k.values()), dtype=np.int64)
    exact = sum(selected_k[qid] == oracle_k[qid] for qid in qids)
    under = sum(selected_k[qid] < oracle_k[qid] for qid in qids)
    over = sum(selected_k[qid] > oracle_k[qid] for qid in qids)
    return {
        **metrics,
        "output_size": {
            "minimum": int(sizes.min()),
            "mean": float(sizes.mean()),
            "median": float(np.median(sizes)),
            "maximum": int(sizes.max()),
        },
        "oracle_k_diagnostic": {
            "exact_rate": exact / len(qids),
            "under_selection_rate": under / len(qids),
            "over_selection_rate": over / len(qids),
        },
        "folds": fold_records,
        "protocol": "outer query folds; fixed Ridge alpha; simultaneous prefix utility scoring",
    }

