"""Second-stage cross-encoder reranking."""

from .cross_encoder import (
    CrossEncoderReranker,
    HFCrossEncoderConfig,
    HFCrossEncoderScorer,
    PairScorer,
)
from .factory import build_reranker
from .training import (
    MinedPassage,
    prefilter_representations,
    prepare_direct_chunk_candidate_handoff,
    prepare_direct_chunk_bce_training_data,
    prepare_pairwise_training_data_from_audit,
    select_aligned_positive_pools,
    prepare_fpt_reranker_training_data,
    prepare_reranker_training_data,
    select_fpt_semi_hard_negatives,
    select_best_passages,
)

__all__ = [
    "CrossEncoderReranker",
    "HFCrossEncoderConfig",
    "HFCrossEncoderScorer",
    "PairScorer",
    "build_reranker",
    "MinedPassage",
    "prefilter_representations",
    "prepare_direct_chunk_candidate_handoff",
    "prepare_direct_chunk_bce_training_data",
    "prepare_pairwise_training_data_from_audit",
    "select_aligned_positive_pools",
    "prepare_fpt_reranker_training_data",
    "prepare_reranker_training_data",
    "select_fpt_semi_hard_negatives",
    "select_best_passages",
]
