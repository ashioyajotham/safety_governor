"""Transparent steering-vector extraction and stability measurement.

This module provides the primary mathematical estimators for discovering linear
concept directions in residual-stream activation spaces:

1. Difference in Means (DIM):
   - Literature: Contrastive Activation Addition (CAA; Rimsky et al., 2023);
     Geometry of Truth (Marks & Tegmark, 2023).
   - Concept: Computes the displacement vector between the centroids of unsafe
     and safe activation distributions.

2. Paired-Delta PCA:
   - Literature: Representation Engineering (RepE; Zou et al., 2023);
     Marks & Tegmark (2023).
   - Concept: Computes the first principal component (via SVD) of centered
     contrastive difference vectors (unsafe - safe), capturing the axis of
     maximal variance in the contrastive shift.

3. Ridge Probe Direction:
   - Literature: Alain & Ollivier (2016); Burns et al. (2022).
   - Concept: Solves a regularized L2 linear classification boundary separating
     safe and unsafe clusters, extracting the normal vector to the decision plane.
     Solved efficiently in dual sample space (N x N) rather than feature space (D x D).

4. Source-Group Bootstrap Stability:
   - Concept: Quantifies whether extracted directions are robust statistical
     properties or small-sample artifacts by computing cosine similarity under
     cluster-aware resampling with replacement.

All extracted vectors are unit-normalized (||v|| = 1) and oriented by convention
to point from safe to unsafe behavior.
"""
from __future__ import annotations

from typing import Callable
import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    """Scale a vector to unit Euclidean norm (||v||_2 = 1).

    Ensures that intervention magnitudes (alpha) have consistent, scale-invariant
    meaning across different model layers and extraction methods.

    Args:
        vector: A 1D NumPy array of hidden dimension [hidden_dim].

    Returns:
        Unit-normalized 1D NumPy array.

    Raises:
        ValueError: If vector has norm zero or contains non-finite values.
    """
    norm = np.linalg.norm(vector)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("cannot normalize a zero or non-finite vector")
    return vector / norm


def difference_in_means(safe: np.ndarray, unsafe: np.ndarray) -> np.ndarray:
    """Compute the difference-in-means unit vector pointing from safe to unsafe.

    Implements the Contrastive Activation Addition (CAA; Rimsky et al., 2023)
    and mass-mean (Marks & Tegmark, 2023) estimator:
        v = normalize(mean(unsafe) - mean(safe))

    This non-parametric baseline connects the centroids of the two activation
    clouds directly without assuming any variance structure or requiring training.

    Args:
        safe: Matrix of safe-response activations of shape [num_pairs, hidden_dim].
        unsafe: Matrix of unsafe-response activations of shape [num_pairs, hidden_dim].

    Returns:
        A unit-normalized 1D array of shape [hidden_dim] pointing safe -> unsafe.

    Raises:
        ValueError: If matrices are not 2D, have unequal shapes, or are empty.
    """
    if safe.ndim != 2 or unsafe.ndim != 2 or safe.shape[1:] != unsafe.shape[1:]:
        raise ValueError("safe and unsafe must be [examples, hidden] matrices with equal hidden size")
    # Direction points from safe behavior toward unsafe behavior by convention.
    return normalize(unsafe.mean(axis=0) - safe.mean(axis=0))


def paired_delta_pca(safe: np.ndarray, unsafe: np.ndarray) -> np.ndarray:
    """Compute the first principal component of contrastive delta vectors.

    Implements the Representation Engineering (RepE; Zou et al., 2023) reading-vector
    estimator on paired differences:
        delta_i = unsafe_i - safe_i
        centered_delta = delta - mean(delta)
        v = first_principal_component(centered_delta)

    Finds the axis of maximum variance among the pairwise contrastive shifts.
    The resulting vector is oriented so its inner product with the mean delta is positive,
    ensuring it points in the safe -> unsafe direction.

    Args:
        safe: Matrix of aligned safe activations of shape [num_pairs, hidden_dim].
        unsafe: Matrix of aligned unsafe activations of shape [num_pairs, hidden_dim].

    Returns:
        A unit-normalized 1D array of shape [hidden_dim] aligned with mean delta.

    Raises:
        ValueError: If safe and unsafe matrices do not match in row count or hidden dimension.
    """
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
    """Compute the normal vector of a regularized Ridge classification probe.

    Literature Context:
        Linear probing in neural representation spaces stems from the probing
        classifier literature (Alain & Ollivier, 2016; Belinkov, 2022). Applying
        linear probe decision boundaries for inference-time steering directly
        follows Inference-Time Intervention (ITI; Li et al., 2023) and latent
        knowledge extraction (Burns et al., 2022).

    Mathematical Formulation:
        Fits an L2-regularized linear discriminant separating safe (y = 0) from
        unsafe (y = 1) activations:
            min_w ||X w - y||_2^2 + lambda ||w||_2^2

        The primal normal vector is:
            w = (X^T X + lambda I)^(-1) X^T y

    Optimization Note (Dual Space Solver):
        Rather than solving the primal system in feature space (which requires
        inverting a 4096 x 4096 covariance matrix for Llama-3-8B), this implements
        the mathematically equivalent dual formulation (Kernel Ridge / Representer Theorem):
            w = X^T (X X^T + lambda I)^(-1) y

        Because N = 2 * num_pairs <= 176 while D = 4096, solving the dual problem
        requires inverting at most an easily conditioned 176 x 176 matrix. This
        reduces runtime by several orders of magnitude while yielding the exact
        same normalized normal vector.

    Args:
        safe: Matrix of safe-response activations of shape [num_safe, hidden_dim].
        unsafe: Matrix of unsafe-response activations of shape [num_unsafe, hidden_dim].
        l2: Positive L2 regularization parameter lambda (default: 1.0).

    Returns:
        A unit-normalized 1D array of shape [hidden_dim] pointing safe -> unsafe.

    Raises:
        ValueError: If matrices have mismatched hidden dimensions or if l2 <= 0.
    """
    if safe.ndim != 2 or unsafe.ndim != 2 or safe.shape[1:] != unsafe.shape[1:]:
        raise ValueError("safe and unsafe must be [examples, hidden] matrices with equal hidden size")
    if l2 <= 0:
        raise ValueError("ridge regularization must be positive")
    # Solve in sample space rather than hidden space. For Llama Stage-1 this
    # reduces the system from roughly 4096x4096 to at most 176x176 while
    # yielding the same ridge direction after centering out the intercept.
    x = np.concatenate((safe, unsafe), axis=0)
    y = np.concatenate((np.zeros(len(safe)), np.ones(len(unsafe))))
    centered_x = x - x.mean(axis=0, keepdims=True)
    centered_y = y - y.mean()
    dual = np.linalg.solve(
        centered_x @ centered_x.T + l2 * np.eye(len(centered_x)),
        centered_y,
    )
    return normalize(centered_x.T @ dual)


def bootstrap_cosine(
    extractor: Callable[[np.ndarray, np.ndarray], np.ndarray],
    safe: np.ndarray,
    unsafe: np.ndarray,
    samples: int = 100,
    seed: int = 0,
    group_ids: list[str] | None = None,
) -> np.ndarray:
    """Measure steering vector direction stability using source-group cluster bootstrap.

    Purpose (Stability Metric):
        Quantifies whether an extracted steering direction represents a genuine,
        generalizable concept or is an artifact of specific prompt phrasings.

    Source-Group Cluster Resampling:
        In this project, source groups (defined in `docs/dataset_curation_workflow.md`,
        Section 7) group multiple contrastive prompt variations derived from the same
        underlying question, math problem, or scenario.
        Standard pair-level resampling would leak prompt-family correlations into the
        uncertainty estimate. By using a cluster bootstrap (resampling entire source
        groups with replacement), we evaluate whether the steering vector remains
        invariant when entire prompt families are withheld.

    Procedure:
        For each bootstrap iteration b in 1..samples:
        1. Resample unique source groups with replacement.
        2. Gather all contrastive pairs belonging to the selected groups.
        3. Re-extract the unit steering vector v_b using `extractor`.
        4. Compute cosine similarity with the full-dataset reference vector v_ref:
               stability_b = dot(v_ref, v_b) = cos(theta)

    Interpretation:
        - Stability >= 0.90: High direction stability (invariant concept representation).
          Observed in Difference-in-Means and Ridge Probes across layers 12-28.
        - Stability < 0.70 or negative: Unstable representation or eigenvector sign-flipping.
          Observed in Paired-Delta PCA, leading to its disqualification from intervention.

    Args:
        extractor: Callable taking (safe, unsafe) matrices and returning a unit vector.
        safe: Matrix of safe activations of shape [num_pairs, hidden_dim].
        unsafe: Matrix of unsafe activations of shape [num_pairs, hidden_dim].
        samples: Number of bootstrap iterations (default: 100).
        seed: Random seed for deterministic reproducibility.
        group_ids: Optional list of source group IDs aligned with activation rows.
            If None, pairs are treated as independent groups.

    Returns:
        1D NumPy array of cosine similarity scores of length [samples].

    Raises:
        ValueError: If safe and unsafe counts differ, if data is empty, or if
            group_ids length does not match row count.
    """
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


