"""Framework-agnostic activation addition and TransformerLens hook adaptation.

This module implements the core inference-time intervention algebra based on the
Activation Addition (ActAdd; Turner et al., 2023) and Contrastive Activation
Addition (CAA; Rimsky et al., 2023) literature.

Mathematical Formulation:
    A'_L = A_L + c * v

    Where:
        - A_L is the hidden representation in the residual stream at layer L.
        - v is the unit steering direction vector.
        - c is the signed scalar coefficient. When v points from safe to unsafe
          (the project convention), suppression of unsafe behavior uses c = -alpha
          for positive magnitude alpha.

Position Hygiene:
    The functions here do not own tokenization. Callers must provide explicit
    positions or boolean masks for position-sensitive interventions to guarantee
    that padding tokens are never steered accidentally.
"""
from __future__ import annotations

from typing import Callable
import numpy as np

from .domain import InterventionSpec


def add_vector(
    activation,
    vector,
    coefficient: float,
    token_mode: str,
    positions=None,
    position_mask=None,
):
    """Return a copy of activations with steering applied at selected token positions.

    Implements the linear residual intervention A' = A + c * v across specified
    token locations. Supports both PyTorch Tensor and NumPy ndarray inputs.

    Args:
        activation: Residual stream tensor or array of shape [batch, seq_len, hidden_dim].
        vector: Unit steering direction vector of shape [hidden_dim].
        coefficient: Signed scaling multiplier c. Negative values suppress the
            feature represented by vector; positive values amplify it.
        token_mode: Policy governing which token positions receive intervention:
            - 'all_tokens': Every token in the sequence is modified.
            - 'last_prompt_token': Only the final prompt token before generation.
            - 'final_response_token': Only the final assistant response token.
            - 'all_response_tokens': All non-prompt assistant tokens.
        positions: Sequence of integer token indices (one per batch item).
            Required when token_mode is 'last_prompt_token' or 'final_response_token'.
        position_mask: Boolean mask of shape [batch, seq_len] identifying response tokens.
            Required when token_mode is 'all_response_tokens'.

    Returns:
        A cloned tensor or array containing the steered residual activations.

    Raises:
        ValueError: If required positions or position_mask are missing for the
            chosen token_mode, or if an unsupported token_mode is provided.
    """
    result = activation.clone() if hasattr(activation, "clone") else np.array(activation, copy=True)
    if token_mode == "all_tokens":
        result += coefficient * vector
    elif token_mode in {"last_prompt_token", "final_response_token"}:
        if positions is None:
            raise ValueError(f"{token_mode} requires explicit non-padding positions")
        for batch_index, position in enumerate(positions):
            result[batch_index, int(position), :] += coefficient * vector
    elif token_mode == "all_response_tokens":
        if position_mask is None:
            raise ValueError("all_response_tokens requires an explicit response-token mask")
        result[position_mask] += coefficient * vector
    else:
        raise ValueError(f"unsupported token mode: {token_mode}")
    return result


def make_hook(vector, spec: InterventionSpec, *, positions=None, position_mask=None) -> Callable:
    """Build a TransformerLens-compatible forward hook from an intervention specification.

    Constructs a callable hook suitable for model.run_with_hooks() that intercepts
    residual activations and injects steering according to spec.coefficient and
    spec.token_mode.

    Args:
        vector: Unit steering direction vector of shape [hidden_dim].
        spec: InterventionSpec defining layer, signed coefficient, and token_mode.
        positions: Optional sequence of non-padding token indices for position-sensitive modes.
        position_mask: Optional boolean response mask for 'all_response_tokens' mode.

    Returns:
        A hook callable with signature hook(activation, hook=None) -> steered_activation.
    """

    def hook(activation, hook=None):
        return add_vector(
            activation, vector, spec.coefficient, spec.token_mode,
            positions=positions, position_mask=position_mask,
        )
    return hook


def hook_name(layer: int) -> str:
    """Return the canonical TransformerLens residual-stream hook name for a layer.

    Targets the pre-residual stream hook ('blocks.{layer}.hook_resid_pre') where
    activations are intercepted prior to attention and MLP sub-block contributions.

    Args:
        layer: Non-negative integer index of the transformer layer.

    Returns:
        String hook identifier, e.g. 'blocks.12.hook_resid_pre'.
    """

    return f"blocks.{layer}.hook_resid_pre"