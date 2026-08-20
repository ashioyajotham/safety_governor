"""Activation cache persistence with explicit layer/token metadata.

Every ``.npy`` activation matrix is paired with a small JSON sidecar. The
sidecar makes downstream vector fitting reject split leakage, layer mismatch,
or safe/unsafe row misalignment instead of relying on filename conventions.
"""
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
    """Save a 2D activation matrix and the metadata needed to fit safely."""

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
    """Load an activation matrix and enforce the expected [examples, hidden] shape."""

    values = np.load(Path(path), allow_pickle=False)
    if values.ndim != 2:
        raise ValueError("activation matrix must have shape [examples, hidden]")
    return values


def load_metadata(path: str | Path) -> dict:
    """Load the JSON sidecar written next to an activation matrix."""

    metadata_path = Path(path).with_suffix(Path(path).suffix + ".json")
    return json.loads(metadata_path.read_text(encoding="utf-8"))