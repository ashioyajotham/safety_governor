"""Transparent steering-vector extraction and stability measurement."""
from __future__ import annotations

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def difference_in_means(safe: np.ndarray, unsafe: np.ndarray) -> np.ndarray:
    """Unsafe minus safe residual activation mean, normalized for alpha comparability."""
    if safe.ndim != 2 or unsafe.ndim != 2 or safe.shape[1:] != unsafe.shape[1:]:
        raise ValueError("safe and unsafe must be [examples, hidden] matrices with equal hidden size")
    return normalize(unsafe.mean(axis=0) - safe.mean(axis=0))


def pca_direction(safe: np.ndarray, unsafe: np.ndarray) -> np.ndarray:
    values = np.concatenate((safe, unsafe), axis=0)
    labels = np.concatenate((-np.ones(len(safe)), np.ones(len(unsafe))))
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    direction = right[0]
    if np.corrcoef(values @ direction, labels)[0, 1] < 0:
        direction *= -1
    return normalize(direction)


def probe_direction(safe: np.ndarray, unsafe: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Closed-form ridge probe boundary normal; avoids hiding classifier defaults."""
    x = np.concatenate((safe, unsafe), axis=0)
    y = np.concatenate((np.zeros(len(safe)), np.ones(len(unsafe))))
    x = np.c_[np.ones(len(x)), x]
    weights = np.linalg.solve(x.T @ x + l2 * np.eye(x.shape[1]), x.T @ y)
    return normalize(weights[1:])


def bootstrap_cosine(extractor, safe: np.ndarray, unsafe: np.ndarray, samples: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    reference = extractor(safe, unsafe)
    scores = []
    for _ in range(samples):
        s = safe[rng.integers(0, len(safe), len(safe))]
        u = unsafe[rng.integers(0, len(unsafe), len(unsafe))]
        scores.append(float(np.dot(reference, extractor(s, u))))
    return np.asarray(scores)
