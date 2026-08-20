"""Transparent steering-vector extraction and stability measurement.

The project uses simple, inspectable baselines first. Each method returns a
unit vector so intervention coefficients are comparable across layers/methods.
"""
from __future__ import annotations

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    """Scale a vector to unit norm."""

    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def difference_in_means(safe: np.ndarray, unsafe: np.ndarray) -> np.ndarray:
    """Unsafe minus safe residual activation mean, normalized for alpha comparability."""
    if safe.ndim != 2 or unsafe.ndim != 2 or safe.shape[1:] != unsafe.shape[1:]:
        raise ValueError("safe and unsafe must be [examples, hidden] matrices with equal hidden size")
    # Direction points from safe behavior toward unsafe behavior by convention.
    return normalize(unsafe.mean(axis=0) - safe.mean(axis=0))


def paired_delta_pca(safe: np.ndarray, unsafe: np.ndarray) -> np.ndarray:
    """First PC of aligned unsafe-safe deltas, oriented toward the mean delta."""
    if safe.shape != unsafe.shape or safe.ndim != 2:
        raise ValueError("paired-delta PCA requires aligned [pairs, hidden] matrices")
    deltas = unsafe - safe
    centered = deltas - deltas.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0):
        return normalize(deltas.mean(axis=0))
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    direction = right[0]
    if np.dot(direction, deltas.mean(axis=0)) < 0:
        direction *= -1
    return normalize(direction)


pca_direction = paired_delta_pca


def probe_direction(safe: np.ndarray, unsafe: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Closed-form ridge probe boundary normal; avoids hiding classifier defaults."""
    # Deliberately small ridge probe: useful as a supervised comparison
    # without importing a larger classifier stack.
    x = np.concatenate((safe, unsafe), axis=0)
    y = np.concatenate((np.zeros(len(safe)), np.ones(len(unsafe))))
    x = np.c_[np.ones(len(x)), x]
    weights = np.linalg.solve(x.T @ x + l2 * np.eye(x.shape[1]), x.T @ y)
    return normalize(weights[1:])


def bootstrap_cosine(extractor, safe: np.ndarray, unsafe: np.ndarray, samples: int = 100, seed: int = 0, group_ids: list[str] | None = None) -> np.ndarray:
    """Paired, source-group-aware bootstrap over aligned contrastive examples."""
    if len(safe) != len(unsafe):
        raise ValueError("paired bootstrap requires equal safe and unsafe row counts")
    if len(safe) == 0:
        raise ValueError("paired bootstrap requires at least one pair")
    groups = np.asarray(group_ids) if group_ids is not None else np.asarray([str(i) for i in range(len(safe))])
    if len(groups) != len(safe):
        raise ValueError("group IDs must align with activation rows")
    rng = np.random.default_rng(seed)
    reference = extractor(safe, unsafe)
    scores = []
    unique_groups = np.unique(groups)
    for _ in range(samples):
        # Resample groups, then include all pairs belonging to each selected
        # group. This preserves source-group dependence in the uncertainty estimate.
        selected = rng.choice(unique_groups, len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in selected])
        scores.append(float(np.dot(reference, extractor(safe[indices], unsafe[indices]))))
    return np.asarray(scores)