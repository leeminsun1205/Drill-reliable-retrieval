"""Exact model-token length audits used before selecting chunk budgets."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def load_fast_tokenizer(
    model_name: str,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Token audits require dense dependencies. "
            "Install with: python -m pip install -e '.[dense]'"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", True):
        raise ValueError("Token audit requires a fast tokenizer")
    return tokenizer


def audit_token_lengths(
    texts: Iterable[str],
    *,
    tokenizer: Any,
    prefix: str = "",
    thresholds: Sequence[int] = (512, 1024),
    batch_size: int = 128,
) -> dict[str, Any]:
    """Measure full encoded lengths, including prefix and special tokens."""
    if batch_size <= 0:
        raise ValueError("Token-audit batch_size must be positive")
    normalized_thresholds = sorted(set(int(value) for value in thresholds))
    if not normalized_thresholds or any(value <= 0 for value in normalized_thresholds):
        raise ValueError("Token-audit thresholds must be positive")
    prefix_length = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    lengths: list[int] = []
    batch: list[str] = []
    for text in texts:
        batch.append(str(text))
        if len(batch) == batch_size:
            lengths.extend(_batch_lengths(batch, tokenizer, prefix))
            batch.clear()
    if batch:
        lengths.extend(_batch_lengths(batch, tokenizer, prefix))
    if not lengths:
        raise ValueError("Cannot audit an empty text collection")
    values = np.asarray(lengths, dtype=np.int64)
    return {
        "count": int(values.size),
        "prefix_tokens": prefix_length,
        "special_tokens": special_tokens,
        "min": int(values.min()),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
        "thresholds": {
            str(threshold): {
                "over_count": int(np.count_nonzero(values > threshold)),
                "over_rate": float(np.mean(values > threshold)),
            }
            for threshold in normalized_thresholds
        },
    }


def _batch_lengths(
    texts: Sequence[str],
    tokenizer: Any,
    prefix: str,
) -> list[int]:
    encoded = tokenizer(
        [prefix + text for text in texts],
        add_special_tokens=True,
        truncation=False,
        return_length=True,
    )
    raw_lengths = encoded.get("length")
    if raw_lengths is None:
        raw_lengths = [len(token_ids) for token_ids in encoded["input_ids"]]
    return [int(length) for length in raw_lengths]
