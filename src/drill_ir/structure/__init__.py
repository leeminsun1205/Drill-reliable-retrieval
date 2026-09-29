"""Legal structure and cross-reference research components."""
"""Deterministic legal-structure extraction."""

from .clauses import ClauseParseResult, ClauseSpan, parse_clause_spans
from .audit import audit_clause_parser

__all__ = ["ClauseParseResult", "ClauseSpan", "audit_clause_parser", "parse_clause_spans"]
