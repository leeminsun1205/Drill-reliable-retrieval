"""Typed records shared across retrieval and evaluation stages."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Article:
    aid: int
    law_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Query:
    qid: int
    question: str
    relevant_aids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    qid: int
    aids: tuple[int, ...]
    scores: tuple[float, ...]
