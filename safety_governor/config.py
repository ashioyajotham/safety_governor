"""Small validated configuration loader; PyYAML is an explicit dependency."""
from __future__ import annotations

from pathlib import Path
import yaml


def load(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    for key in ("model", "dataset", "seed"):
        if key not in config:
            raise ValueError(f"missing required config key: {key}")
    return config
