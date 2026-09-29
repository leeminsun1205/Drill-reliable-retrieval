"""Strict loading and explicit overrides of experiment configuration."""

from copy import deepcopy
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
    supported = {"bm25", "dense", "hybrid", "organizer_two_stage"}
    if config["method"] not in supported:
        raise ValueError(f"Unknown method={config['method']!r}; expected one of {sorted(supported)}")
    return config


def set_config_value(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set one existing config leaf; reject typos and structural replacement."""
    parts = dotted_path.split(".")
    if not dotted_path or any(not part for part in parts):
        raise ValueError(f"Invalid config path: {dotted_path!r}")
    target: dict[str, Any] = config
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"Unknown config path: {dotted_path!r}")
        target = child
    leaf = parts[-1]
    if leaf not in target:
        raise ValueError(f"Unknown config path: {dotted_path!r}")
    if isinstance(target[leaf], dict):
        raise ValueError(f"Cannot replace config section with --set: {dotted_path!r}")
    target[leaf] = value


def apply_config_overrides(
    config: dict[str, Any], overrides: list[str]
) -> dict[str, Any]:
    """Apply repeatable PATH=YAML_VALUE overrides to a copied config."""
    resolved = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Config override must use PATH=VALUE: {override!r}")
        path, raw_value = override.split("=", 1)
        value = "" if raw_value == "" else yaml.safe_load(raw_value)
        set_config_value(resolved, path, value)
    return resolved
