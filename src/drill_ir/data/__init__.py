"""Dataset loading and deterministic preprocessing."""

from .io import iter_articles, load_queries
from .models import Article, Query, RetrievalResult
from .paths import resolve_corpus_path, resolve_data_file, resolve_train_path
from .validation import ValidationReport, validate_dataset

__all__ = [
    "Article",
    "Query",
    "RetrievalResult",
    "ValidationReport",
    "iter_articles",
    "load_queries",
    "resolve_corpus_path",
    "resolve_data_file",
    "resolve_train_path",
    "validate_dataset",
]
