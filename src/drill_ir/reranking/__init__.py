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
    prepare_reranker_training_data,
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
    "prepare_reranker_training_data",
    "select_best_passages",
]
