"""Strict loading of experiment configuration."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a YAML mapping")
    required = {"experiment_name", "method", "seed", "data", "retrieval", "output"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config is missing keys: {sorted(missing)}")
    supported = {"bm25", "dense", "organizer_two_stage"}
    if config["method"] not in supported:
        raise ValueError(f"Unknown method={config['method']!r}; expected one of {sorted(supported)}")
    return config
