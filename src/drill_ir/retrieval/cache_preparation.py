"""Prepare one validated retrieval cache per process."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from drill_ir.data.models import Article

from .factory import _merged_mapping, build_retriever


def configured_cache_components(retrieval_config: dict[str, Any]) -> list[str]:
    """Return the sparse cache plus configured dense component names."""
    components = ["sparse"] if isinstance(retrieval_config.get("bm25"), dict) else []
    hybrid = retrieval_config.get("hybrid", {})
    dense_components = hybrid.get("dense_components") if isinstance(hybrid, dict) else None
    if dense_components is None:
        if isinstance(retrieval_config.get("dense"), dict):
            components.append("dense")
        return components
    for component in dense_components:
        if isinstance(component, dict) and str(component.get("name", "")).strip():
            components.append(str(component["name"]).strip())
    return components


def prepare_retrieval_cache(
    *,
    retrieval_config: dict[str, Any],
    runtime_config: dict[str, Any],
    articles: Sequence[Article],
    component: str,
    require_existing: bool = False,
) -> dict[str, Any]:
    """Fit exactly one configured retriever and return its cache metadata."""
    config = deepcopy(retrieval_config)
    if component == "sparse":
        params = config.get("bm25")
        if not isinstance(params, dict):
            raise ValueError("Cannot prepare sparse cache without retrieval.bm25")
        cache = params.setdefault("cache", {})
        if not isinstance(cache, dict):
            raise ValueError("retrieval.bm25.cache must be a YAML mapping")
        cache["enabled"] = True
        cache["require_hit"] = require_existing
        retriever = build_retriever("bm25", config, runtime_config)
    else:
        hybrid = config.get("hybrid", {})
        dense_components = hybrid.get("dense_components") if isinstance(hybrid, dict) else None
        if dense_components is None and component == "dense":
            dense_config = config.get("dense")
        else:
            selected = [
                item
                for item in (dense_components or [])
                if isinstance(item, dict) and str(item.get("name", "")).strip() == component
            ]
            if len(selected) != 1:
                available = configured_cache_components(config)
                raise ValueError(
                    f"Unknown cache component {component!r}; expected one of {available}"
                )
            base = config.get("dense")
            profiles = config.get("dense_profiles")
            if not isinstance(base, dict) or not isinstance(profiles, dict):
                raise ValueError("Configured dense components require dense and dense_profiles")
            profile_name = str(selected[0].get("profile", component)).strip()
            profile = profiles.get(profile_name)
            if not isinstance(profile, dict):
                raise ValueError(f"Unknown dense profile {profile_name!r}")
            dense_config = _merged_mapping(base, profile)
        if not isinstance(dense_config, dict):
            raise ValueError("Cannot prepare dense cache without retrieval.dense")
        cache = dense_config.setdefault("cache", {})
        if not isinstance(cache, dict):
            raise ValueError("retrieval.dense.cache must be a YAML mapping")
        cache["enabled"] = True
        cache["require_hit"] = require_existing
        retriever = build_retriever(
            "dense", {**config, "dense": dense_config}, runtime_config
        )
    retriever.fit(articles)
    metadata = retriever.run_metadata()
    return {"component": component, "retriever": metadata}
