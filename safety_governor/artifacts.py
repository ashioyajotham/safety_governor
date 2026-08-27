"""Create immutable identifiers and manifests for experiment artifacts.

The capture and fitting entrypoints write numerical arrays separately from their
metadata. This module provides the shared logic needed to name those runs and
persist a :class:`~safety_governor.domain.RunManifest` without silently
overwriting an earlier result.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .domain import RunManifest


def make_run_id(prefix: str = "run") -> str:
    """Return a UTC timestamped identifier suitable for an artifact folder."""

    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def write_manifest(root: str | Path, manifest: RunManifest) -> Path:
    """Write ``manifest.json`` once and return its path.

    Existing manifests are never replaced. A collision therefore fails loudly
    instead of changing the recorded provenance of an existing artifact.
    """

    directory = Path(root) / manifest.run_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        raise FileExistsError(f"manifest already exists: {path}")
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return path
