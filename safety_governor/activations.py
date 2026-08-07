"""Activation cache persistence with explicit layer/token metadata."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def save_matrix(
    path: str | Path,
    values: np.ndarray,
    *,
    layer: int,
    token_mode: str,
    sample_ids: list[str],
    splits: list[str] | None = None,
    source_group_ids: list[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)
    path.with_suffix(path.suffix + ".json").write_text(json.dumps({
        "layer": layer,
        "token_mode": token_mode,
        "sample_ids": sample_ids,
        "splits": splits,
        "source_group_ids": source_group_ids,
        "shape": list(values.shape),
    }, indent=2), encoding="utf-8")


def load_matrix(path: str | Path) -> np.ndarray:
    values = np.load(Path(path), allow_pickle=False)
    if values.ndim != 2:
        raise ValueError("activation matrix must have shape [examples, hidden]")
    return values


def load_metadata(path: str | Path) -> dict:
    metadata_path = Path(path).with_suffix(Path(path).suffix + ".json")
    return json.loads(metadata_path.read_text(encoding="utf-8"))