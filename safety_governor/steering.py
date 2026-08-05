"""Framework-agnostic activation addition and TransformerLens hook adaptation."""
from __future__ import annotations

from typing import Callable
import numpy as np

from .domain import InterventionSpec


def add_vector(activation, vector, coefficient: float, token_mode: str):
    """Return a copy with steering applied; supports NumPy or Torch-like tensors."""
    result = activation.clone() if hasattr(activation, "clone") else np.array(activation, copy=True)
    if token_mode == "all_tokens":
        result += coefficient * vector
    elif token_mode == "last_prompt_token":
        result[:, -1, :] += coefficient * vector
    else:
        raise ValueError(f"unsupported token mode: {token_mode}")
    return result


def make_hook(vector, spec: InterventionSpec) -> Callable:
    def hook(activation, _hook):
        return add_vector(activation, vector, spec.coefficient, spec.token_mode)
    return hook


def hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_pre"
