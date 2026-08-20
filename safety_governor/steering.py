"""Framework-agnostic activation addition and TransformerLens hook adaptation.

The functions here do not own tokenization. Callers must provide explicit
positions or masks for position-sensitive interventions so padding tokens are
never steered by accident.
"""
from __future__ import annotations

from typing import Callable
import numpy as np

from .domain import InterventionSpec


def add_vector(activation, vector, coefficient: float, token_mode: str, positions=None, position_mask=None):
    """Return a copy with steering applied only at explicitly selected positions."""
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
    """Build a TransformerLens-compatible hook from a neutral intervention spec."""

    def hook(activation, hook=None):
        return add_vector(
            activation, vector, spec.coefficient, spec.token_mode,
            positions=positions, position_mask=position_mask,
        )
    return hook


def hook_name(layer: int) -> str:
    """TransformerLens residual-stream hook name for a layer."""

    return f"blocks.{layer}.hook_resid_pre"