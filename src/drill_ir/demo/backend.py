"""Small adapter contract between the demo UI and the retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from typing import Any, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class Evidence:
    """A retrieved legal article prepared for display and answer grounding."""

    aid: str
    text: str
    law_id: str = ""
    title: str = ""
    score: float | None = None


class RetrievalBackend(Protocol):
    """Interactive adapter; retrieval and final cardinality stay in the pipeline."""

    def search(self, question: str) -> Sequence[Evidence | dict[str, Any]]:
        """Return the final ranked evidence for one question."""


def load_backend() -> RetrievalBackend:
    """Load a factory provided by the Kaggle launcher, without changing retrieval."""
    factory_spec = os.environ.get("DRILL_RETRIEVER_FACTORY", "").strip()
    if not factory_spec or ":" not in factory_spec:
        raise RuntimeError(
            "Set DRILL_RETRIEVER_FACTORY to 'module_name:factory_name' in the "
            "Kaggle launcher. The factory must return an object with search(question)."
        )

    module_name, factory_name = factory_spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise RuntimeError(f"Retrieval factory is not callable: {factory_spec}")
    backend = factory()
    if not callable(getattr(backend, "search", None)):
        raise TypeError("Retrieval backend must implement search(question)")
    return backend


def normalize_evidence(item: Evidence | dict[str, Any]) -> Evidence:
    """Accept the app contract or article-shaped dictionaries from an adapter."""
    if isinstance(item, Evidence):
        return item
    if not isinstance(item, dict):
        raise TypeError(f"Unsupported evidence item: {type(item).__name__}")

    aid = item.get("aid")
    text = item.get("text", item.get("content_Article", item.get("content", "")))
    if aid is None or not str(text).strip():
        raise ValueError("Evidence must contain an aid and non-empty article text")
    raw_score = item.get("score")
    return Evidence(
        aid=str(aid),
        text=str(text),
        law_id=str(item.get("law_id", "")),
        title=str(item.get("title", "")),
        score=float(raw_score) if raw_score is not None else None,
    )
