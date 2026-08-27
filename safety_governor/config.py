"""Load the small YAML contracts used by capture and fitting entrypoints.

This is intentionally not a general configuration framework. It validates the
minimum top-level experiment contract while research-specific preflight checks
remain in :mod:`safety_governor.preflight`.
"""
from __future__ import annotations

from pathlib import Path
import yaml


def load(path: str | Path) -> dict:
    """Load a YAML config and require model, dataset, and seed keys."""

    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    for key in ("model", "dataset", "seed"):
        if key not in config:
            raise ValueError(f"missing required config key: {key}")
    return config
