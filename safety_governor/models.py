"""Optional heavyweight model backend, isolated from core analysis code."""
from __future__ import annotations


def load_transformerlens_model(name: str, device: str | None = None):
    """Load an open-weight model only when a model experiment is invoked."""
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional GPU environment
        raise RuntimeError("Install transformer-lens to run model experiments.") from exc
    kwargs = {"device": device} if device else {}
    return HookedTransformer.from_pretrained(name, **kwargs)


def residual_at_last_token(model, prompts: list[str], layer: int):
    """Capture pre-residual activations at the final token for each prompt."""
    _, cache = model.run_with_cache(prompts, return_type="logits")
    value = cache[f"blocks.{layer}.hook_resid_pre"]
    return value[:, -1, :].detach().float().cpu().numpy()
