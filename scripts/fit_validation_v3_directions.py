"""Fit source-group and archetype-balanced directions for v3 development.

This command consumes only train activations. It never reads Validation-v2
calibration or confirmatory generations and cannot create a validation lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from safety_governor.activations import load_matrix, load_metadata
from safety_governor.stage1 import atomic_write_json, parse_layers
from safety_governor.vectors import (
    balanced_difference_in_means,
    select_balanced_ridge_l2,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_rows(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    safe = {
        row["pair_id"]: row for row in rows
        if row.get("split") == "train" and row.get("polarity") == "safe"
    }
    if not safe:
        raise ValueError("dataset has no train-safe rows")
    return safe


def collapse_source_groups(
    safe: np.ndarray,
    unsafe: np.ndarray,
    sample_ids: list[str],
    source_group_ids: list[str],
    rows: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Average repeated pairs within a source group and retain archetype labels."""

    if safe.shape != unsafe.shape or len(safe) != len(sample_ids):
        raise ValueError("activation rows and sample IDs are not aligned")
    if len(source_group_ids) != len(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("source-group metadata is missing or sample IDs repeat")
    if set(sample_ids) != set(rows):
        raise ValueError("activation sample IDs do not match train-safe dataset rows")
    groups = sorted(set(source_group_ids))
    grouped_safe, grouped_unsafe, archetypes = [], [], []
    for group in groups:
        indices = [index for index, value in enumerate(source_group_ids) if value == group]
        local_archetypes = {rows[sample_ids[index]]["archetype"] for index in indices}
        if len(local_archetypes) != 1:
            raise ValueError(f"source group crosses archetypes: {group}")
        grouped_safe.append(safe[indices].mean(axis=0))
        grouped_unsafe.append(unsafe[indices].mean(axis=0))
        archetypes.append(local_archetypes.pop())
    return np.stack(grouped_safe), np.stack(grouped_unsafe), groups, archetypes


def fit_layer(
    layer_root: Path,
    output: Path,
    rows: dict[str, dict],
    l2_candidates: tuple[float, ...],
    folds: int,
    seed: int,
) -> dict:
    """Fit and persist balanced DIM and selected balanced Ridge for one layer."""

    safe_path, unsafe_path = layer_root / "safe.npy", layer_root / "unsafe.npy"
    safe, unsafe = load_matrix(safe_path), load_matrix(unsafe_path)
    safe_meta, unsafe_meta = load_metadata(safe_path), load_metadata(unsafe_path)
    for field in ("layer", "token_mode", "sample_ids", "splits", "source_group_ids"):
        if safe_meta.get(field) != unsafe_meta.get(field):
            raise ValueError(f"safe/unsafe activation metadata mismatch: {field}")
    if set(safe_meta.get("splits") or []) != {"train"}:
        raise ValueError("v3 direction fitting is train-only")
    grouped_safe, grouped_unsafe, groups, archetypes = collapse_source_groups(
        safe,
        unsafe,
        safe_meta["sample_ids"],
        safe_meta["source_group_ids"],
        rows,
    )
    dim = balanced_difference_in_means(grouped_safe, grouped_unsafe, archetypes)
    ridge, cross_validation = select_balanced_ridge_l2(
        grouped_safe,
        grouped_unsafe,
        groups,
        archetypes,
        candidates=l2_candidates,
        folds=folds,
        seed=seed,
    )
    output.mkdir(parents=True, exist_ok=False)
    dim_path, ridge_path = output / "balanced_dim.npy", output / "balanced_ridge.npy"
    np.save(dim_path, dim)
    np.save(ridge_path, ridge.vector)
    metadata = {
        "schema_version": 1,
        "layer": int(safe_meta["layer"]),
        "capture_site": safe_meta["token_mode"],
        "fit_split": "train",
        "pairs": len(safe),
        "source_groups": len(groups),
        "source_groups_by_archetype": {
            archetype: sum(value == archetype for value in archetypes)
            for archetype in sorted(set(archetypes))
        },
        "weighting": "equal_archetype_mass_after_source_group_mean",
        "directions": {
            "balanced_dim": {
                "path": str(dim_path),
                "sha256": _sha256(dim_path),
            },
            "balanced_ridge": {
                "path": str(ridge_path),
                "sha256": _sha256(ridge_path),
                "l2": ridge.l2,
                "response_level_threshold": ridge.threshold,
                "response_level_score_scale": ridge.score_scale,
                "raw_norm": ridge.raw_norm,
                "cross_validation": cross_validation,
                "token_gate_qualified": False,
                "token_gate_note": "response-level calibration is not a token gate",
            },
        },
    }
    atomic_write_json(output / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="12,16,20,24,28")
    parser.add_argument("--l2-candidates", default="0.1,1,10,100")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    layers = parse_layers(args.layers)
    l2_candidates = tuple(float(value) for value in args.l2_candidates.split(","))
    rows = _dataset_rows(args.dataset)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for layer in layers:
        layer_root = args.train_run.resolve() / "layers" / f"layer_{layer:02d}"
        results[str(layer)] = fit_layer(
            layer_root,
            output / "directions" / f"layer_{layer:02d}",
            rows,
            l2_candidates,
            args.folds,
            args.seed,
        )
    manifest = {
        "schema_version": 1,
        "phase": "validation_v3_train_only_direction_development",
        "train_run": str(args.train_run.resolve()),
        "train_manifest_sha256": _sha256(args.train_run.resolve() / "manifest.json"),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset.resolve()),
        "layers": layers,
        "l2_candidates": l2_candidates,
        "folds": args.folds,
        "seed": args.seed,
        "confirmatory_accessed": False,
        "results": results,
    }
    atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "layers": layers}, indent=2))


if __name__ == "__main__":
    main()
