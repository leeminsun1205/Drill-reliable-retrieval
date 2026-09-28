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

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "EmbeddingCache",
    "ExactVectorSearch",
    "Retriever",
    "SentenceTransformerEncoder",
    "SentenceTransformerEncoderConfig",
    "TextEncoder",
    "build_retriever",
    "tokenize",
]
