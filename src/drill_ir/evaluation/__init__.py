"""Retrieval evaluation metrics and analysis."""

from .metrics import (
    DrillMetrics,
    evaluate_fixed_cutoffs,
    evaluate_predictions,
    evaluate_oracle_prefix,
    select_by_score_gap,
    tune_score_gap,
    macro_recall_at_k,
    tune_score_thresholds,
)

__all__ = [
    "DrillMetrics",
    "evaluate_fixed_cutoffs",
    "evaluate_predictions",
    "evaluate_oracle_prefix",
    "select_by_score_gap",
    "tune_score_gap",
    "macro_recall_at_k",
    "tune_score_thresholds",
]
