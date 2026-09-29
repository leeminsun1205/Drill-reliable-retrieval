"""First-stage retrieval and rank fusion."""

from .base import Retriever
from .bm25 import BM25Retriever, tokenize
from .dense import (
    DenseRetriever,
    EmbeddingCache,
    ExactVectorSearch,
    SentenceTransformerEncoder,
    SentenceTransformerEncoderConfig,
    TextEncoder,
)
from .factory import build_retriever
from .fusion import WeightedRRFRetriever

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "EmbeddingCache",
    "ExactVectorSearch",
    "Retriever",
    "SentenceTransformerEncoder",
    "SentenceTransformerEncoderConfig",
    "TextEncoder",
    "WeightedRRFRetriever",
    "build_retriever",
    "tokenize",
]
