"""First-stage retrieval and rank fusion."""

from .base import Retriever
from .bm25 import BM25IndexCache, BM25Retriever, BM25SIndexCache, BM25SRetriever, tokenize
from .dense import (
    ChunkRetrievalResult,
    DenseRetriever,
    EmbeddingCache,
    ExactVectorSearch,
    FaissFlatVectorSearch,
    SentenceTransformerEncoder,
    SentenceTransformerEncoderConfig,
    TextEncoder,
)
from .factory import build_retriever
from .fusion import MultiDenseWeightedRRFRetriever, WeightedRRFRetriever

__all__ = [
    "ChunkRetrievalResult",
    "BM25Retriever",
    "BM25IndexCache",
    "BM25SIndexCache",
    "BM25SRetriever",
    "DenseRetriever",
    "EmbeddingCache",
    "ExactVectorSearch",
    "FaissFlatVectorSearch",
    "Retriever",
    "SentenceTransformerEncoder",
    "SentenceTransformerEncoderConfig",
    "TextEncoder",
    "WeightedRRFRetriever",
    "MultiDenseWeightedRRFRetriever",
    "build_retriever",
    "tokenize",
]
