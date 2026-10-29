"""Leakage-aware offline fusion of saved retrieval and reranker features."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence
import warnings

import numpy as np

from drill_ir.data import Query
from drill_ir.evaluation import (
    evaluate_fixed_cutoffs,
    evaluate_oracle_prefix,
    evaluate_predictions,
    select_by_score_gap,
    tune_score_gap,
    tune_score_thresholds,
)


FEATURE_NAMES = (
    "reranker_score",
    "reranker_reciprocal_rank",
    "reranker_margin_from_top",
    "first_stage_score",
    "first_stage_reciprocal_rank",
    "sparse_score",
    "sparse_reciprocal_rank",
    "dense_score",
    "dense_reciprocal_rank",
    "chunk_max_fusion_score",
    "chunk_reciprocal_best_rank",
    "chunk_count",
    "query_whitespace_tokens",
)

LIGHTGBM_PARAMETERS = {
    "objective": "binary",
    "n_estimators": 200,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 20,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbosity": -1,
}


def _predict_positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    """Predict class-one probability while hiding one known wrapper warning.

    LightGBM's sklearn wrapper records generated feature names even when it is
    fit from a NumPy matrix. Recent sklearn releases can then warn that another
    NumPy matrix has no valid feature names. Training and inference remain
    positionally aligned here, so suppress only that exact harmless warning.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "X does not have valid feature names, but LGBMClassifier was "
                "fitted with feature names"
            ),
            category=UserWarning,
            module=r"sklearn\.utils\.validation",
        )
        return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _rank_features(record: dict[str, Any]) -> dict[int, tuple[float, float]]:
    aids = record["retrieved_aids"]
    scores = record["scores"]
    if len(aids) != len(scores):
        raise ValueError("Ranking aids/scores length mismatch")
    return {
        int(aid): (float(score), 1.0 / rank)
        for rank, (aid, score) in enumerate(zip(aids, scores), start=1)
    }


def _dense_feature_schema(
    components: dict[int, dict[str, dict[int, tuple[float, float]]]],
) -> tuple[list[str], list[str]]:
    """Return a stable dense-component order and its score/rank feature names."""
    component_sets = [
        {name for name in values if name == "dense" or name.startswith("dense:")}
        for values in components.values()
    ]
    if not component_sets or not component_sets[0]:
        raise ValueError("Component rankings contain no dense retriever")
    expected = component_sets[0]
    if any(names != expected for names in component_sets[1:]):
        raise ValueError("Dense component names differ across queries")
    ordered = sorted(expected)
    feature_names: list[str] = []
    for name in ordered:
        slug = name.replace(":", "_").replace("-", "_")
        feature_names.extend((f"{slug}_score", f"{slug}_reciprocal_rank"))
    return ordered, feature_names


def build_ltr_examples(
    result_dir: str | Path,
    queries: Sequence[Query],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Build one feature row per reranked candidate from saved run artifacts."""
    result_dir = Path(result_dir)
    required = {
        "predictions": result_dir / f"{prefix}predictions.jsonl",
        "first_stage": result_dir / f"{prefix}first_stage_predictions.jsonl",
        "components": result_dir / f"{prefix}component_rankings.jsonl",
        "chunks": result_dir / f"{prefix}reranker_chunk_candidates.jsonl",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing LTR artifacts: {missing}")

    query_by_qid = {query.qid: query for query in queries}
    predictions = {int(row["qid"]): row for row in _read_jsonl(required["predictions"])}
    first_stage = {
        int(row["qid"]): _rank_features(row)
        for row in _read_jsonl(required["first_stage"])
    }
    components: dict[int, dict[str, dict[int, tuple[float, float]]]] = {}
    for row in _read_jsonl(required["components"]):
        components[int(row["qid"])] = {
            name: _rank_features(payload)
            for name, payload in row["components"].items()
        }
    chunks: dict[int, dict[int, tuple[float, float, float]]] = {}
    for row in _read_jsonl(required["chunks"]):
        by_aid: defaultdict[int, list[tuple[int, float]]] = defaultdict(list)
        for rank, chunk in enumerate(row["chunks"], start=1):
            by_aid[int(chunk["aid"])].append((rank, float(chunk["fusion_score"])))
        chunks[int(row["qid"])] = {
            aid: (
                max(score for _, score in values),
                1.0 / min(rank for rank, _ in values),
                float(len(values)),
            )
            for aid, values in by_aid.items()
        }

    expected_qids = set(query_by_qid)
    for name, mapping in (
        ("predictions", predictions),
        ("first_stage", first_stage),
        ("components", components),
        ("chunks", chunks),
    ):
        if set(mapping) != expected_qids:
            raise ValueError(f"{name} qids do not match the supplied query set")

    dense_components, dense_feature_names = _dense_feature_schema(components)
    feature_names = (
        FEATURE_NAMES[:7]
        + tuple(dense_feature_names)
        + FEATURE_NAMES[9:]
    )

    features: list[list[float]] = []
    labels: list[int] = []
    qids: list[int] = []
    aids: list[int] = []
    original_ranks: list[int] = []
    for query in sorted(queries, key=lambda item: item.qid):
        qid = query.qid
        prediction = predictions[qid]
        ranked_aids = [int(aid) for aid in prediction["retrieved_aids"]]
        reranker_scores = [float(score) for score in prediction["scores"]]
        if len(ranked_aids) != len(reranker_scores):
            raise ValueError(f"Prediction aids/scores mismatch for qid={qid}")
        top_score = reranker_scores[0]
        gold = set(query.relevant_aids)
        sparse = components[qid].get("sparse", {})
        for rank, (aid, reranker_score) in enumerate(
            zip(ranked_aids, reranker_scores), start=1
        ):
            first_score, first_rr = first_stage[qid].get(aid, (0.0, 0.0))
            sparse_score, sparse_rr = sparse.get(aid, (0.0, 0.0))
            dense_values: list[float] = []
            for component in dense_components:
                dense_score, dense_rr = components[qid][component].get(
                    aid, (0.0, 0.0)
                )
                dense_values.extend((dense_score, dense_rr))
            chunk_score, chunk_rr, chunk_count = chunks[qid].get(
                aid, (0.0, 0.0, 0.0)
            )
            features.append(
                [
                    reranker_score,
                    1.0 / rank,
                    top_score - reranker_score,
                    first_score,
                    first_rr,
                    sparse_score,
                    sparse_rr,
                    *dense_values,
                    chunk_score,
                    chunk_rr,
                    chunk_count,
                    float(len(query.question.split())),
                ]
            )
            labels.append(int(aid in gold))
            qids.append(qid)
            aids.append(aid)
            original_ranks.append(rank)

    return {
        "feature_names": feature_names,
        "features": np.asarray(features, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int8),
        "qids": np.asarray(qids, dtype=np.int64),
        "aids": np.asarray(aids, dtype=np.int64),
        "original_ranks": np.asarray(original_ranks, dtype=np.int64),
    }


def limit_ltr_examples(
    examples: dict[str, Any], maximum_reranker_rank: int | None
) -> dict[str, Any]:
    """Return an LTR-row view restricted to an upstream reranker depth.

    ``None`` preserves every reranked article. The operation is deliberately
    applied after feature construction so every depth ablation uses identical
    saved retrieval/reranker artifacts and changes only which rows LightGBM may
    inspect.
    """
    if maximum_reranker_rank is None:
        return examples
    if (
        isinstance(maximum_reranker_rank, bool)
        or not isinstance(maximum_reranker_rank, int)
        or maximum_reranker_rank <= 0
    ):
        raise ValueError("maximum_reranker_rank must be a positive integer or None")
    mask = examples["original_ranks"] <= maximum_reranker_rank
    limited = {
        key: value[mask]
        for key, value in examples.items()
        if key in {"features", "labels", "qids", "aids", "original_ranks"}
    }
    limited["feature_names"] = examples["feature_names"]
    return limited


def evaluate_oof_lightgbm(
    examples: dict[str, Any],
    queries: Sequence[Query],
    *,
    folds: int = 5,
    seed: int = 42,
    include_score_gap: bool = False,
    fixed_output_cutoffs: Sequence[int] = (1, 2, 3, 4, 5, 10, 20, 30, 50),
    score_gap_maximum_cutoffs: Sequence[int] = (3, 4, 5, 6, 7, 8, 9, 10),
    progress_label: str = "all",
) -> dict[str, Any]:
    """Produce query-grouped out-of-fold pointwise LightGBM predictions."""
    if folds < 2:
        raise ValueError("folds must be at least two")
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise RuntimeError("Install the project 'ltr' extra to use LightGBM") from error

    features = examples["features"]
    labels = examples["labels"]
    row_qids = examples["qids"]
    aids = examples["aids"]
    original_ranks = examples["original_ranks"]
    unique_qids = sorted({int(qid) for qid in row_qids})
    if folds > len(unique_qids):
        raise ValueError("fold count exceeds the number of queries")
    shuffled = list(unique_qids)
    random.Random(seed).shuffle(shuffled)
    fold_by_qid = {qid: index % folds for index, qid in enumerate(shuffled)}
    probabilities = np.zeros(len(labels), dtype=np.float64)
    importance = np.zeros(features.shape[1], dtype=np.float64)

    print(
        f"[lightgbm:{progress_label}] start rows={len(labels)} "
        f"positive_rows={int(labels.sum())} folds={folds} "
        f"nested={include_score_gap}",
        flush=True,
    )

    for fold in range(folds):
        validation = np.asarray(
            [fold_by_qid[int(qid)] == fold for qid in row_qids], dtype=bool
        )
        training = ~validation
        if len(set(labels[training])) != 2:
            raise ValueError(f"Fold {fold} training rows do not contain both labels")
        started = time.perf_counter()
        print(
            f"[lightgbm:{progress_label}] outer_oof_fit "
            f"{fold + 1}/{folds} train_rows={int(training.sum())} "
            f"validation_rows={int(validation.sum())}",
            flush=True,
        )
        model = LGBMClassifier(**LIGHTGBM_PARAMETERS, random_state=seed + fold)
        model.fit(features[training], labels[training])
        probabilities[validation] = _predict_positive_probability(
            model, features[validation]
        )
        importance += model.feature_importances_
        print(
            f"[lightgbm:{progress_label}] outer_oof_fit_complete "
            f"{fold + 1}/{folds} seconds={time.perf_counter() - started:.2f}",
            flush=True,
        )

    rankings: dict[int, tuple[list[int], list[float]]] = {}
    for qid in unique_qids:
        indices = np.flatnonzero(row_qids == qid)
        ordered = sorted(
            indices,
            key=lambda index: (
                -probabilities[index],
                original_ranks[index],
                aids[index],
            ),
        )
        rankings[qid] = (
            [int(aids[index]) for index in ordered],
            [float(probabilities[index]) for index in ordered],
        )

    gold = {query.qid: query.relevant_aids for query in queries}
    fixed = evaluate_fixed_cutoffs(
        gold,
        {qid: values[0] for qid, values in rankings.items()},
        fixed_output_cutoffs,
    )
    started = time.perf_counter()
    print(f"[lightgbm:{progress_label}] threshold_tuning start", flush=True)
    thresholds = tune_score_thresholds(gold, rankings, [1, 2])
    print(
        f"[lightgbm:{progress_label}] threshold_tuning complete "
        f"seconds={time.perf_counter() - started:.2f}",
        flush=True,
    )
    result = {
        "protocol": {
            "model": "LightGBM pointwise binary classifier",
            "prediction": f"{folds}-fold out-of-fold by qid",
            "seed": seed,
            "parameters": {**LIGHTGBM_PARAMETERS, "random_state": "seed + fold"},
            "note": (
                "Threshold selection on the pooled OOF predictions remains a "
                "development diagnostic, not an unbiased test estimate."
            ),
        },
        "feature_names": list(examples["feature_names"]),
        "mean_feature_importance": {
            name: float(value / folds)
            for name, value in zip(examples["feature_names"], importance)
        },
        "fixed_output_k": {
            str(cutoff): metrics.to_dict() for cutoff, metrics in fixed.items()
        },
        "threshold": thresholds,
        "rankings": rankings,
    }
    if include_score_gap:
        started = time.perf_counter()
        print(
            f"[lightgbm:{progress_label}] pooled_score_gap_tuning start "
            f"maximum_k={list(score_gap_maximum_cutoffs)}",
            flush=True,
        )
        score_gap = tune_score_gap(
            gold, rankings, maximum_cutoffs=score_gap_maximum_cutoffs
        )
        print(
            f"[lightgbm:{progress_label}] pooled_score_gap_tuning complete "
            f"seconds={time.perf_counter() - started:.2f}",
            flush=True,
        )

        # Fully nest selector tuning: inner OOF scores use outer-training qids
        # only, then the selected policy is applied to the outer OOF ranking.
        nested_predictions: dict[int, list[int]] = {}
        nested_policies: list[dict[str, Any]] = []
        for outer_fold in range(folds):
            tuning_qids = {
                qid for qid in unique_qids if fold_by_qid[qid] != outer_fold
            }
            validation_qids = {
                qid for qid in unique_qids if fold_by_qid[qid] == outer_fold
            }
            inner_qids = sorted(tuning_qids)
            random.Random(seed + 10_000 + outer_fold).shuffle(inner_qids)
            inner_fold_by_qid = {
                qid: index % folds for index, qid in enumerate(inner_qids)
            }
            inner_probabilities = np.zeros(len(labels), dtype=np.float64)
            tuning_rows = np.asarray(
                [int(qid) in tuning_qids for qid in row_qids], dtype=bool
            )
            for inner_fold in range(folds):
                inner_validation = np.asarray(
                    [
                        int(qid) in tuning_qids
                        and inner_fold_by_qid[int(qid)] == inner_fold
                        for qid in row_qids
                    ],
                    dtype=bool,
                )
                inner_training = tuning_rows & ~inner_validation
                if len(set(labels[inner_training])) != 2:
                    raise ValueError(
                        f"Outer fold {outer_fold}, inner fold {inner_fold} "
                        "training rows do not contain both labels"
                    )
                started = time.perf_counter()
                print(
                    f"[lightgbm:{progress_label}] nested_fit "
                    f"outer={outer_fold + 1}/{folds} "
                    f"inner={inner_fold + 1}/{folds} "
                    f"train_rows={int(inner_training.sum())} "
                    f"validation_rows={int(inner_validation.sum())}",
                    flush=True,
                )
                inner_model = LGBMClassifier(
                    **LIGHTGBM_PARAMETERS,
                    random_state=seed + 10_000 + outer_fold * folds + inner_fold,
                )
                inner_model.fit(features[inner_training], labels[inner_training])
                inner_probabilities[inner_validation] = _predict_positive_probability(
                    inner_model, features[inner_validation]
                )
                print(
                    f"[lightgbm:{progress_label}] nested_fit_complete "
                    f"outer={outer_fold + 1}/{folds} "
                    f"inner={inner_fold + 1}/{folds} "
                    f"seconds={time.perf_counter() - started:.2f}",
                    flush=True,
                )

            tuning_rankings: dict[int, tuple[list[int], list[float]]] = {}
            for qid in tuning_qids:
                indices = np.flatnonzero(row_qids == qid)
                ordered = sorted(
                    indices,
                    key=lambda index: (
                        -inner_probabilities[index],
                        original_ranks[index],
                        aids[index],
                    ),
                )
                tuning_rankings[qid] = (
                    [int(aids[index]) for index in ordered],
                    [float(inner_probabilities[index]) for index in ordered],
                )
            started = time.perf_counter()
            print(
                f"[lightgbm:{progress_label}] nested_policy_tuning "
                f"outer={outer_fold + 1}/{folds} start",
                flush=True,
            )
            tuned = tune_score_gap(
                {qid: gold[qid] for qid in tuning_qids},
                tuning_rankings,
                maximum_cutoffs=score_gap_maximum_cutoffs,
            )
            print(
                f"[lightgbm:{progress_label}] nested_policy_tuning "
                f"outer={outer_fold + 1}/{folds} complete "
                f"seconds={time.perf_counter() - started:.2f}",
                flush=True,
            )
            policy = tuned["best"]
            selected = select_by_score_gap(
                {qid: rankings[qid] for qid in validation_qids},
                minimum_gap=float(policy["minimum_gap"]),
                fallback_k=int(policy["fallback_k"]),
                maximum_k=int(policy["maximum_k"]),
            )
            nested_predictions.update(selected)
            nested_policies.append(
                {
                    "outer_fold": outer_fold,
                    "inner_folds": folds,
                    "tuning_queries": len(tuning_qids),
                    "validation_queries": len(validation_qids),
                    "policy": policy,
                }
            )
        nested_metrics = evaluate_predictions(gold, nested_predictions)
        result["score_gap"] = {
            "pooled_oof_tuning_for_deployment": score_gap,
            "nested_oof_estimate": {
                **nested_metrics.to_dict(),
                "fold_policies": nested_policies,
                "note": (
                    "Each score-gap policy is tuned on inner OOF rankings from "
                    "outer-training queries and applied to the outer-held-out fold."
                ),
            },
            "oracle_prefix_at_10": evaluate_oracle_prefix(
                gold, rankings, maximum_k=max(score_gap_maximum_cutoffs)
            ),
        }
    print(f"[lightgbm:{progress_label}] complete", flush=True)
    return result


def fit_lightgbm_and_rank(
    training_examples: dict[str, Any],
    inference_examples: dict[str, Any],
    *,
    seed: int = 42,
) -> tuple[dict[int, tuple[list[int], list[float]]], Any]:
    """Fit all labeled development rows and rank an unlabeled candidate set."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise RuntimeError("Install the project 'ltr' extra to use LightGBM") from error

    if tuple(training_examples["feature_names"]) != tuple(
        inference_examples["feature_names"]
    ):
        raise ValueError("Training and inference feature schemas do not match")
    labels = training_examples["labels"]
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Final LightGBM training requires both binary labels")
    started = time.perf_counter()
    print(
        "[lightgbm:final] fit start "
        f"train_rows={len(labels)} inference_rows={len(inference_examples['qids'])} ",
        flush=True,
    )
    model = LGBMClassifier(**LIGHTGBM_PARAMETERS, random_state=seed)
    model.fit(training_examples["features"], labels)
    print(
        f"[lightgbm:final] fit complete seconds={time.perf_counter() - started:.2f}; "
        "prediction start",
        flush=True,
    )
    probabilities = _predict_positive_probability(
        model, inference_examples["features"]
    )
    row_qids = inference_examples["qids"]
    aids = inference_examples["aids"]
    original_ranks = inference_examples["original_ranks"]
    rankings: dict[int, tuple[list[int], list[float]]] = {}
    for qid in sorted({int(value) for value in row_qids}):
        indices = np.flatnonzero(row_qids == qid)
        ordered = sorted(
            indices,
            key=lambda index: (
                -probabilities[index], original_ranks[index], aids[index]
            ),
        )
        rankings[qid] = (
            [int(aids[index]) for index in ordered],
            [float(probabilities[index]) for index in ordered],
        )
    print(
        f"[lightgbm:final] prediction complete queries={len(rankings)} "
        f"total_seconds={time.perf_counter() - started:.2f}",
        flush=True,
    )
    return rankings, model
