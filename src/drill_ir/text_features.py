"""Deterministic, model-free text features safe to persist with run artifacts."""

from __future__ import annotations

import re
import unicodedata


_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_LEGAL_CITATION_PATTERN = re.compile(
    r"\b(?:điều|khoản|điểm)\s+(?:\d+[a-zđ]?|[a-zđ])\b",
    flags=re.IGNORECASE,
)
_CLAUSE_LINE_PATTERN = re.compile(r"(?m)^\s*\d+[.)]\s+")
_POINT_LINE_PATTERN = re.compile(r"(?m)^\s*[a-zđ][.)]\s+", flags=re.IGNORECASE)


def normalized_tokens(text: str) -> tuple[str, ...]:
    """Return stable NFKC/lowercase word tokens without external segmenters."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return tuple(_TOKEN_PATTERN.findall(normalized))


def chunk_pair_features(query: str, chunk: str) -> dict[str, float | int]:
    """Compute cheap query/chunk statistics at the artifact-production boundary."""
    query_tokens = normalized_tokens(query)
    chunk_tokens = normalized_tokens(chunk)
    query_set = set(query_tokens)
    chunk_set = set(chunk_tokens)
    overlap = query_set & chunk_set
    query_citations = {
        unicodedata.normalize("NFKC", value).lower()
        for value in _LEGAL_CITATION_PATTERN.findall(query)
    }
    chunk_citations = {
        unicodedata.normalize("NFKC", value).lower()
        for value in _LEGAL_CITATION_PATTERN.findall(chunk)
    }
    return {
        "character_count": len(chunk),
        "whitespace_token_count": len(chunk.split()),
        "normalized_token_count": len(chunk_tokens),
        "unique_token_count": len(chunk_set),
        "query_chunk_overlap_count": len(overlap),
        "query_token_coverage": len(overlap) / len(query_set) if query_set else 0.0,
        "chunk_token_coverage": len(overlap) / len(chunk_set) if chunk_set else 0.0,
        "legal_citation_overlap_count": len(query_citations & chunk_citations),
    }


def article_static_features(text: str) -> dict[str, float | int]:
    """Corpus-only article features that may be precomputed once and uploaded."""
    tokens = normalized_tokens(text)
    unique_tokens = set(tokens)
    digit_tokens = sum(token.isdigit() for token in tokens)
    clause_markers = len(_CLAUSE_LINE_PATTERN.findall(text))
    point_markers = len(_POINT_LINE_PATTERN.findall(text))
    return {
        "article_character_count": len(text),
        "article_whitespace_token_count": len(text.split()),
        "article_normalized_token_count": len(tokens),
        "article_unique_token_count": len(unique_tokens),
        "article_clause_marker_count": clause_markers,
        "article_point_marker_count": point_markers,
        "article_legal_citation_count": len(_LEGAL_CITATION_PATTERN.findall(text)),
        "article_digit_token_fraction": digit_tokens / len(tokens) if tokens else 0.0,
    }
