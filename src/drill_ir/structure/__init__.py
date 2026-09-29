"""Legal structure and cross-reference research components."""
"""Deterministic legal-structure extraction."""

from .clauses import ClauseParseResult, ClauseSpan, parse_clause_spans
from .audit import audit_clause_parser, sample_articles_by_law
from .points import PointParseResult, PointSpan, parse_point_spans

__all__ = [
    "ClauseParseResult",
    "ClauseSpan",
    "audit_clause_parser",
    "parse_clause_spans",
    "parse_point_spans",
    "PointParseResult",
    "PointSpan",
    "sample_articles_by_law",
]
