"""Leakage-aware offline fusion of saved retrieval and reranker features."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
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

DERIVED_FEATURE_NAMES = (
    "reranker_score_z",
    "reranker_score_minmax",
    "reranker_score_percentile",
    "reranker_gap_from_previous",
    "reranker_gap_to_next",
    "first_stage_score_z",
    "first_stage_score_minmax",
    "retriever_hit_count",
    "retriever_top10_count",
    "retriever_best_rank",
    "retriever_mean_rank",
    "retriever_rank_std",
    "sparse_dense_agreement",
    "dense_pair_agreement",
    "is_reference_expanded",
    "reference_base_reciprocal_rank",
    "reference_seed_first_stage_reciprocal_rank",
    "reference_is_same_document",
    "chunk_fusion_mean",
    "chunk_fusion_std",
    "chunk_fusion_min",
    "chunk_top1_top2_gap",
    "chunk_strong_fraction",
    "chunk_type_count",
    "query_has_explicit_legal_reference",
    "query_legal_marker_count",
)

ENHANCED_V2_FEATURE_NAMES = (
    "chunk_reranker_mean",
    "chunk_reranker_std",
    "chunk_reranker_min",
    "chunk_reranker_second_max",
    "chunk_reranker_top1_top2_gap",
    "chunk_reranker_range",
    "chunk_reranker_strong_fraction",
    "chunk_reranker_best_input_reciprocal_rank",
    "chunk_reranker_best_chunk_index",
    "chunk_character_count_mean",
    "chunk_character_count_max",
    "chunk_token_count_mean",
    "chunk_token_count_max",
    "query_chunk_overlap_mean",
    "query_chunk_overlap_max",
    "query_token_coverage_mean",
    "query_token_coverage_max",
    "chunk_token_coverage_mean",
    "chunk_token_coverage_max",
    "legal_citation_overlap_max",
)

ENHANCED_V2_CHUNK_FIELDS = (
    "reranker_score",
    "character_count",
    "normalized_token_count",
    "query_chunk_overlap_count",
    "query_token_coverage",
    "chunk_token_coverage",
    "legal_citation_overlap_count",
)

STATIC_ARTICLE_FEATURE_NAMES = (
    "article_character_count",
    "article_whitespace_token_count",
    "article_normalized_token_count",
    "article_unique_token_count",
    "article_clause_marker_count",
    "article_point_marker_count",
    "article_legal_citation_count",
    "article_digit_token_fraction",
)

LEGAL_REFERENCE_PATTERN = re.compile(
    r"\b(?:điều|khoản|điểm|luật|nghị\s+định|thông\s+tư|bộ\s+luật)\b",
    flags=re.IGNORECASE,
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


def _distribution_features(values: Sequence[float]) -> list[tuple[float, float, float]]:
    """Return stable z-score, min-max value, and descending percentile."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return []
    mean = float(array.mean())
    std = float(array.std())
    minimum = float(array.min())
    span = float(array.max() - minimum)
    order = sorted(range(len(array)), key=lambda index: (-array[index], index))
    percentile = [0.0] * len(array)
    denominator = max(1, len(array) - 1)
    for position, index in enumerate(order):
        percentile[index] = 1.0 - position / denominator
    return [
        (
            (float(value) - mean) / std if std > 0.0 else 0.0,
            (float(value) - minimum) / span if span > 0.0 else 0.0,
            percentile[index],
        )
        for index, value in enumerate(array)
    ]


def _load_reference_provenance(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """Load optional deterministic reference-expansion records from a run."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("explicit_references", []) if isinstance(payload, dict) else []
    provenance: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        key = (int(record["qid"]), int(record["reference_aid"]))
        provenance[key] = record
    return provenance


def _load_static_article_features(path: str | Path) -> dict[int, dict[str, float]]:
    rows = _read_jsonl(Path(path))
    features: dict[int, dict[str, float]] = {}
    for row in rows:
        aid = int(row["aid"])
        if aid in features:
            raise ValueError(f"Duplicate aid in static article features: {aid}")
        missing = [name for name in STATIC_ARTICLE_FEATURE_NAMES if name not in row]
        if missing:
            raise ValueError(f"Static article feature row aid={aid} is missing {missing}")
        features[aid] = {
            name: float(row[name]) for name in STATIC_ARTICLE_FEATURE_NAMES
        }
    if not features:
        raise ValueError("Static article feature artifact is empty")
    return features


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


def _enhanced_v2_chunk_aggregates(
    values: Sequence[tuple[int, dict[str, Any]]],
) -> dict[str, float]:
    """Aggregate persisted per-chunk neural, length, and lexical evidence."""
    reranker_scores = [float(item["reranker_score"]) for _, item in values]
    ordered_scores = sorted(reranker_scores, reverse=True)
    maximum = ordered_scores[0]
    second = ordered_scores[1] if len(ordered_scores) > 1 else maximum
    best_position = max(
        range(len(values)), key=lambda index: (reranker_scores[index], -values[index][0])
    )
    best_rank, best_item = values[best_position]
    character_counts = [float(item["character_count"]) for _, item in values]
    token_counts = [
        float(
            item["token_count"]
            if item.get("token_count") is not None
            else item["normalized_token_count"]
        )
        for _, item in values
    ]
    overlaps = [float(item["query_chunk_overlap_count"]) for _, item in values]
    query_coverages = [float(item["query_token_coverage"]) for _, item in values]
    chunk_coverages = [float(item["chunk_token_coverage"]) for _, item in values]
    legal_overlaps = [float(item["legal_citation_overlap_count"]) for _, item in values]
    strong_boundary = maximum - 0.1 * max(1e-12, abs(maximum))
    return {
        "chunk_reranker_mean": float(np.mean(reranker_scores)),
        "chunk_reranker_std": float(np.std(reranker_scores)),
        "chunk_reranker_min": min(reranker_scores),
        "chunk_reranker_second_max": second,
        "chunk_reranker_top1_top2_gap": maximum - second,
        "chunk_reranker_range": maximum - min(reranker_scores),
        "chunk_reranker_strong_fraction": float(
            np.mean([score >= strong_boundary for score in reranker_scores])
        ),
        "chunk_reranker_best_input_reciprocal_rank": 1.0 / best_rank,
        "chunk_reranker_best_chunk_index": float(best_item.get("chunk_index", 0)),
        "chunk_character_count_mean": float(np.mean(character_counts)),
        "chunk_character_count_max": max(character_counts),
        "chunk_token_count_mean": float(np.mean(token_counts)),
        "chunk_token_count_max": max(token_counts),
        "query_chunk_overlap_mean": float(np.mean(overlaps)),
        "query_chunk_overlap_max": max(overlaps),
        "query_token_coverage_mean": float(np.mean(query_coverages)),
        "query_token_coverage_max": max(query_coverages),
        "chunk_token_coverage_mean": float(np.mean(chunk_coverages)),
        "chunk_token_coverage_max": max(chunk_coverages),
        "legal_citation_overlap_max": max(legal_overlaps),
    }


def build_ltr_examples(
    result_dir: str | Path,
    queries: Sequence[Query],
    *,
    prefix: str = "",
    feature_schema: str = "enhanced_v1",
    static_article_features: str | Path | None = None,
) -> dict[str, Any]:
    """Build one feature row per reranked candidate from saved run artifacts."""
    if feature_schema not in {
        "base_v1",
        "enhanced_v1",
        "enhanced_v2",
        "enhanced_v3",
    }:
        raise ValueError(
            "feature_schema must be 'base_v1', 'enhanced_v1', "
            "'enhanced_v2', or 'enhanced_v3'"
        )
    if feature_schema == "enhanced_v3" and static_article_features is None:
        raise ValueError("enhanced_v3 requires static_article_features")
    static_features = (
        _load_static_article_features(static_article_features)
        if feature_schema == "enhanced_v3" and static_article_features is not None
        else {}
    )
    result_dir = Path(result_dir)
    required = {
        "predictions": result_dir / f"{prefix}predictions.jsonl",
        "first_stage": result_dir / f"{prefix}first_stage_predictions.jsonl",
        "components": result_dir / f"{prefix}component_rankings.jsonl",
        "chunks": result_dir / f"{prefix}reranker_chunk_candidates.jsonl",
    }
    feedback_path = result_dir / f"{prefix}retrieval_feedback.json"
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
    chunks: dict[int, dict[int, dict[str, float]]] = {}
    for row in _read_jsonl(required["chunks"]):
        by_aid: defaultdict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for rank, chunk in enumerate(row["chunks"], start=1):
            if feature_schema in {"enhanced_v2", "enhanced_v3"}:
                missing_fields = [
                    name for name in ENHANCED_V2_CHUNK_FIELDS if name not in chunk
                ]
                if missing_fields:
                    raise ValueError(
                        "enhanced_v2 requires reranker-produced chunk metadata; "
                        f"qid={row['qid']} chunk_id={chunk.get('chunk_id')} "
                        f"missing={missing_fields}"
                    )
            by_aid[int(chunk["aid"])].append((rank, chunk))
        chunks[int(row["qid"])] = {
            aid: {
                "max": max(float(item["fusion_score"]) for _, item in values),
                "best_rr": 1.0 / min(rank for rank, _ in values),
                "count": float(len(values)),
                "mean": float(np.mean([float(item["fusion_score"]) for _, item in values])),
                "std": float(np.std([float(item["fusion_score"]) for _, item in values])),
                "min": min(float(item["fusion_score"]) for _, item in values),
                "top_gap": (
                    sorted((float(item["fusion_score"]) for _, item in values), reverse=True)[0]
                    - sorted((float(item["fusion_score"]) for _, item in values), reverse=True)[1]
                    if len(values) > 1
                    else 0.0
                ),
                "strong_fraction": float(
                    np.mean(
                        [
                            float(item["fusion_score"])
                            >= max(float(value["fusion_score"]) for _, value in values)
                            - 0.1 * max(
                                1e-12,
                                abs(max(float(value["fusion_score"]) for _, value in values)),
                            )
                            for _, item in values
                        ]
                    )
                ),
                "type_count": float(
                    len({str(item.get("chunk_type", "")) for _, item in values})
                ),
                **(
                    _enhanced_v2_chunk_aggregates(values)
                    if feature_schema in {"enhanced_v2", "enhanced_v3"}
                    else {}
                ),
            }
            for aid, values in by_aid.items()
        }
    reference_provenance = _load_reference_provenance(feedback_path)

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
    base_feature_names = (
        FEATURE_NAMES[:7]
        + tuple(dense_feature_names)
        + FEATURE_NAMES[9:]
    )
    feature_names = (
        base_feature_names
        + (DERIVED_FEATURE_NAMES if feature_schema != "base_v1" else ())
        + (
            ENHANCED_V2_FEATURE_NAMES
            if feature_schema in {"enhanced_v2", "enhanced_v3"}
            else ()
        )
        + (STATIC_ARTICLE_FEATURE_NAMES if feature_schema == "enhanced_v3" else ())
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
        reranker_distribution = _distribution_features(reranker_scores)
        first_stage_scores = [value[0] for value in first_stage[qid].values()]
        first_distribution_values = _distribution_features(first_stage_scores)
        first_distribution = {
            aid: values
            for aid, values in zip(first_stage[qid], first_distribution_values)
        }
        gold = set(query.relevant_aids)
        sparse = components[qid].get("sparse", {})
        legal_markers = LEGAL_REFERENCE_PATTERN.findall(query.question)
        for rank, (aid, reranker_score) in enumerate(
            zip(ranked_aids, reranker_scores), start=1
        ):
            if feature_schema == "enhanced_v3" and aid not in static_features:
                raise ValueError(f"Missing static article features for candidate aid={aid}")
            first_score, first_rr = first_stage[qid].get(aid, (0.0, 0.0))
            sparse_score, sparse_rr = sparse.get(aid, (0.0, 0.0))
            dense_values: list[float] = []
            component_ranks: list[float] = []
            sparse_rank = 1.0 / sparse_rr if sparse_rr > 0.0 else math.inf
            if math.isfinite(sparse_rank):
                component_ranks.append(sparse_rank)
            dense_hit_count = 0
            for component in dense_components:
                dense_score, dense_rr = components[qid][component].get(
                    aid, (0.0, 0.0)
                )
                dense_values.extend((dense_score, dense_rr))
                if dense_rr > 0.0:
                    dense_hit_count += 1
                    component_ranks.append(1.0 / dense_rr)
            chunk = chunks[qid].get(aid, {})
            chunk_score = float(chunk.get("max", 0.0))
            chunk_rr = float(chunk.get("best_rr", 0.0))
            chunk_count = float(chunk.get("count", 0.0))
            reranker_z, reranker_minmax, reranker_percentile = reranker_distribution[rank - 1]
            previous_score = reranker_scores[rank - 2] if rank > 1 else reranker_score
            next_score = reranker_scores[rank] if rank < len(reranker_scores) else reranker_score
            first_z, first_minmax, _ = first_distribution.get(aid, (0.0, 0.0, 0.0))
            finite_ranks = component_ranks
            top10_count = sum(value <= 10.0 for value in finite_ranks)
            reference = reference_provenance.get((qid, aid))
            seed_rr = 0.0
            if reference is not None:
                seed_rr = first_stage[qid].get(int(reference["seed_aid"]), (0.0, 0.0))[1]
            row = [
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
                    reranker_z,
                    reranker_minmax,
                    reranker_percentile,
                    max(0.0, previous_score - reranker_score),
                    max(0.0, reranker_score - next_score),
                    first_z,
                    first_minmax,
                    float(len(finite_ranks)),
                    float(top10_count),
                    min(finite_ranks) if finite_ranks else 0.0,
                    float(np.mean(finite_ranks)) if finite_ranks else 0.0,
                    float(np.std(finite_ranks)) if finite_ranks else 0.0,
                    float(sparse_rr > 0.0 and dense_hit_count > 0),
                    float(dense_hit_count >= 2),
                    float(reference is not None),
                    1.0 / float(reference["base_rank"]) if reference is not None else 0.0,
                    seed_rr,
                    float(reference is not None and reference.get("scope") == "same_document"),
                    float(chunk.get("mean", 0.0)),
                    float(chunk.get("std", 0.0)),
                    float(chunk.get("min", 0.0)),
                    float(chunk.get("top_gap", 0.0)),
                    float(chunk.get("strong_fraction", 0.0)),
                    float(chunk.get("type_count", 0.0)),
                    float(bool(legal_markers)),
                    float(len(legal_markers)),
                    *(
                        [float(chunk.get(name, 0.0)) for name in ENHANCED_V2_FEATURE_NAMES]
                        if feature_schema in {"enhanced_v2", "enhanced_v3"}
                        else []
                    ),
                    *(
                        [static_features[aid][name] for name in STATIC_ARTICLE_FEATURE_NAMES]
                        if feature_schema == "enhanced_v3"
                        else []
                    ),
                ]
            features.append(row[: len(feature_names)])
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
        "feature_schema": feature_schema,
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
    limited["feature_schema"] = examples.get("feature_schema", "unknown")
    return limited


def ltr_feature_groups(feature_names: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Return interpretable, non-overlapping feature groups for OOF ablation."""
    names = tuple(feature_names)
    group_for: dict[str, str] = {}
    for name in names:
        if name.startswith("chunk_reranker_"):
            group = "chunk_reranker_distribution"
        elif name.startswith("article_"):
            group = "article_static"
        elif name.startswith("chunk_character_") or name.startswith("chunk_token_"):
            group = "chunk_length"
        elif (
            name.startswith("query_chunk_")
            or name.startswith("query_token_coverage")
            or name.startswith("chunk_token_coverage")
            or name == "legal_citation_overlap_max"
        ):
            group = "query_chunk_lexical"
        elif name.startswith("reference_") or name == "is_reference_expanded":
            group = "reference_provenance"
        elif name.startswith("retriever_") or name.endswith("_agreement"):
            group = "retriever_agreement"
        elif name.startswith("chunk_fusion_") or name in {
            "chunk_max_fusion_score",
            "chunk_reciprocal_best_rank",
            "chunk_count",
            "chunk_top1_top2_gap",
            "chunk_strong_fraction",
            "chunk_type_count",
        }:
            group = "chunk_fusion"
        elif name.endswith("_z") or name.endswith("_minmax") or name.endswith("_percentile") or "gap_from" in name or name == "reranker_gap_to_next":
            group = "within_query_calibration"
        elif name.startswith("query_"):
            group = "query_shape"
        elif name.startswith("reranker_"):
            group = "reranker_article"
        else:
            group = "first_stage_components"
        group_for[name] = group
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for name in names:
        grouped[group_for[name]].append(name)
    return {group: tuple(values) for group, values in sorted(grouped.items())}


def drop_ltr_features(
    examples: dict[str, Any], feature_names: Sequence[str]
) -> dict[str, Any]:
    """Copy examples while removing named columns for a controlled ablation."""
    removed = set(feature_names)
    unknown = removed - set(examples["feature_names"])
    if unknown:
        raise ValueError(f"Unknown LTR features: {sorted(unknown)}")
    kept_indices = [
        index
        for index, name in enumerate(examples["feature_names"])
        if name not in removed
    ]
    if not kept_indices:
        raise ValueError("Cannot remove every LTR feature")
    result = dict(examples)
    result["feature_names"] = tuple(examples["feature_names"][index] for index in kept_indices)
    result["features"] = examples["features"][:, kept_indices]
    return result


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
    split_importance = np.zeros(features.shape[1], dtype=np.float64)
    gain_importance = np.zeros(features.shape[1], dtype=np.float64)

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
        split_importance += model.booster_.feature_importance(
            importance_type="split"
        )
        gain_importance += model.booster_.feature_importance(
            importance_type="gain"
        )
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
            for name, value in zip(examples["feature_names"], split_importance)
        },
        "mean_split_importance": {
            name: float(value / folds)
            for name, value in zip(examples["feature_names"], split_importance)
        },
        "mean_gain_importance": {
            name: float(value / folds)
            for name, value in zip(examples["feature_names"], gain_importance)
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
    fixed_top2 = result["fixed_output_k"].get("2")
    threshold_top2 = result["threshold"].get("best_by_fallback_k", {}).get("2")
    summary: dict[str, Any] = {
        "event": "case_summary",
        "label": progress_label,
        "features": len(examples["feature_names"]),
        "rows": len(labels),
        "fixed_top2": fixed_top2,
        "threshold_fallback2": threshold_top2,
    }
    if "score_gap" in result:
        pooled = result["score_gap"]["pooled_oof_tuning_for_deployment"]["best"]
        nested = result["score_gap"]["nested_oof_estimate"]
        summary["score_gap_pooled"] = {
            key: pooled[key]
            for key in ("precision", "recall", "f2")
            if key in pooled
        }
        summary["score_gap_nested"] = {
            key: nested[key] for key in ("precision", "recall", "f2")
        }
    print(
        f"[lightgbm:{progress_label}] summary "
        + json.dumps(summary, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
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
