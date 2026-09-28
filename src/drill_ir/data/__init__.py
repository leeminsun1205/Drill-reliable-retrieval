"""Dataset loading and deterministic preprocessing."""

from .io import iter_articles, load_queries
from .models import Article, Query, RetrievalResult
from .validation import ValidationReport, validate_dataset

__all__ = [
    "Article",
    "Query",
    "RetrievalResult",
    "ValidationReport",
    "iter_articles",
    "load_queries",
    "validate_dataset",
]
