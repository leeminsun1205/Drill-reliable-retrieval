"""Leakage-aware offline fusion of saved retrieval and reranker features."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np

from drill_ir.data import Query
from drill_ir.evaluation import evaluate_fixed_cutoffs, tune_score_thresholds


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


def evaluate_oof_lightgbm(
    examples: dict[str, Any],
    queries: Sequence[Query],
    *,
    folds: int = 5,
    seed: int = 42,
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

    for fold in range(folds):
        validation = np.asarray(
            [fold_by_qid[int(qid)] == fold for qid in row_qids], dtype=bool
        )
        training = ~validation
        if len(set(labels[training])) != 2:
            raise ValueError(f"Fold {fold} training rows do not contain both labels")
        model = LGBMClassifier(**LIGHTGBM_PARAMETERS, random_state=seed + fold)
        model.fit(features[training], labels[training])
        probabilities[validation] = model.predict_proba(features[validation])[:, 1]
        importance += model.feature_importances_

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
        gold, {qid: values[0] for qid, values in rankings.items()}, range(1, 6)
    )
    thresholds = tune_score_thresholds(gold, rankings, [1, 2])
    return {
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
    model = LGBMClassifier(**LIGHTGBM_PARAMETERS, random_state=seed)
    model.fit(training_examples["features"], labels)
    probabilities = model.predict_proba(inference_examples["features"])[:, 1]
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
    return rankings, model
