"""Runtime-only placement for a persistent interactive retrieval process."""
from typing import Any


def pin_dense_search_to_cpu(retrieval: dict[str, Any]) -> dict[str, str]:
    """Keep FAISS indexes off CUDA independently of encoder placement.

    Explicitly override profiles too: they are merged over the base dense
    mapping, and otherwise a saved profile can re-enable GPU index allocation.
    Scientific parameters (metric, vectors, fusion, budgets) remain intact.
    """
    dense = retrieval['dense']
    dense.setdefault('search', {})['device'] = 'cpu'
    devices = {'base_dense': 'cpu'}
    for name, profile in retrieval.get('dense_profiles', {}).items():
        profile.setdefault('search', {})['device'] = 'cpu'
        devices[name] = 'cpu'
    return devices


# Operational headroom, not a measured model memory bound.
RERANKER_MIN_FREE_GIB = 6.0


def choose_reranker_device(free_gib: list[float]) -> str | list[str]:
    """Prefer both sufficiently free GPUs, otherwise one, otherwise CPU.

    Pick the freer device as DataParallel's primary when both can participate.
    Do not assume that free memory on one device covers another device's OOM.
    """
    eligible = [i for i, free in enumerate(free_gib) if free >= RERANKER_MIN_FREE_GIB]
    eligible.sort(key=lambda i: (-free_gib[i], i))
    devices = [f'cuda:{i}' for i in eligible[:2]]
    return devices if len(devices) > 1 else devices[0] if devices else 'cpu'


def place_unloaded_reranker(scorer: Any, free_gib: list[float]) -> str | list[str]:
    """Set placement before lazy loading, preserving checkpoint and scoring args."""
    from dataclasses import replace

    if scorer._model is not None:
        raise RuntimeError('Refusing to relocate an already loaded reranker')
    device = choose_reranker_device(free_gib)
    scorer.config = replace(scorer.config, device=device, fp16=device != 'cpu')
    return device
