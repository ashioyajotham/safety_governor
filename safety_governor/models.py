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
    """Capture pre-residual activations at each prompt's final non-padding token."""
    tokens = model.to_tokens(prompts)
    _, cache = model.run_with_cache(tokens, return_type="logits")
    value = cache[f"blocks.{layer}.hook_resid_pre"]
    pad_token_id = getattr(model.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        positions = tokens.new_full((tokens.shape[0],), tokens.shape[1] - 1)
    else:
        indices = tokens.new_tensor(range(tokens.shape[1])).expand_as(tokens)
        positions = indices.masked_fill(tokens.eq(pad_token_id), -1).max(dim=1).values
        if (positions < 0).any():
            raise ValueError("tokenized prompt contains only padding")
    batch = tokens.new_tensor(range(tokens.shape[0]))
    return value[batch, positions, :].detach().float().cpu().numpy()
