"""Common retrieval protocol."""

from typing import Protocol, Sequence

from drill_ir.data.models import Article, Query, RetrievalResult


class Retriever(Protocol):
    def fit(self, articles: Sequence[Article]) -> None: ...

    def retrieve(self, queries: Sequence[Query], top_k: int) -> list[RetrievalResult]: ...
