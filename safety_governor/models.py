"""Optional heavyweight model backend and explicit response-position capture.

This is the only module that should know about TransformerLens/Hugging Face
runtime details. Everything above it passes explicit instruction/completion
strings and receives numpy arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TokenizedBatch:
    """Tokenization result with response boundaries preserved."""

    tokens: object
    attention_mask: object
    response_mask: object
    final_response_positions: object


def resolve_torch_dtype(name: str):
    """Resolve an explicit research-runtime dtype without silent fallback."""

    import torch

    normalized = name.lower().replace("torch.", "")
    supported = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in supported:
        raise ValueError(f"unsupported model dtype: {name}")
    return supported[normalized]


def load_transformerlens_model(
    name: str,
    revision: str,
    device: str | None = None,
    dtype: str = "float32",
    bridge_weight_mode: str = "hf_native_aliases",
):
    """Load pinned HF-native weights with TransformerLens hook aliases.

    Stage 1 deliberately preserves the Hugging Face checkpoint numerics. The
    bridge compatibility layer is used only to register stable hook names; its
    optional LayerNorm folding and weight centering would mutate the model and
    temporarily upcast an 8B checkpoint to float32.
    """
    if not revision or revision in {"main", "master", "latest"}:
        raise ValueError("model revision must be an immutable commit or tag")
    if bridge_weight_mode != "hf_native_aliases":
        raise ValueError(f"unsupported TransformerLens bridge weight mode: {bridge_weight_mode}")
    try:
        from huggingface_hub import snapshot_download
        from transformer_lens.model_bridge import TransformerBridge
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the pinned TransformerLens/Hugging Face dependencies.") from exc
    snapshot = snapshot_download(repo_id=name, revision=revision)
    bridge = TransformerBridge.boot_transformers(
        snapshot,
        device=device,
        dtype=resolve_torch_dtype(dtype),
    )
    bridge.enable_compatibility_mode(no_processing=True)
    return bridge


def _prefix_ids(tokenizer, instruction: str) -> list[int]:
    """Return token IDs up to the assistant-generation boundary."""

    messages = [{"role": "user", "content": instruction}]
    if getattr(tokenizer, "chat_template", None):
        return list(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        ))
    return list(tokenizer.encode(
        f"User: {instruction}\nAssistant:", add_special_tokens=True
    ))


def tokenize_instruction_completion(model, instructions: list[str], completions: list[str]) -> TokenizedBatch:
    """Tokenize with an explicit assistant boundary and return response positions."""
    if len(instructions) != len(completions) or not instructions:
        raise ValueError("instructions and completions must be non-empty aligned lists")
    import torch

    tokenizer = model.tokenizer
    sequences, response_starts = [], []
    for instruction, completion in zip(instructions, completions):
        # Tokenize the prompt boundary and completion separately so response
        # activations are not confused with prompt-boundary activations.
        prefix = _prefix_ids(tokenizer, instruction)
        response = list(tokenizer.encode(completion, add_special_tokens=False))
        if not response:
            raise ValueError("completion tokenized to an empty response")
        sequences.append(prefix + response)
        response_starts.append(len(prefix))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    width = max(map(len, sequences))
    padding_side = getattr(tokenizer, "padding_side", "right")
    tokens = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), width), dtype=torch.bool)
    response_mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    final_positions = torch.empty(len(sequences), dtype=torch.long)
    for index, (sequence, response_start) in enumerate(zip(sequences, response_starts)):
        # Preserve correct response positions for either left- or right-padding.
        offset = width - len(sequence) if padding_side == "left" else 0
        end = offset + len(sequence)
        tokens[index, offset:end] = torch.tensor(sequence)
        attention[index, offset:end] = True
        response_mask[index, offset + response_start:end] = True
        final_positions[index] = end - 1
    return TokenizedBatch(tokens, attention, response_mask, final_positions)


def residuals_at_response(
    model,
    instructions: list[str],
    completions: list[str],
    layers: list[int],
    site: str = "response_mean",
) -> dict[int, np.ndarray]:
    """Capture one response representation for every requested model layer.

    ``names_filter`` limits the TransformerLens cache to the requested residual
    hooks. This is essential for 8B-scale runs, where retaining every internal
    activation can dominate GPU memory.
    """

    if not layers or len(set(layers)) != len(layers) or any(layer < 0 for layer in layers):
        raise ValueError("layers must be a non-empty list of unique non-negative integers")
    batch = tokenize_instruction_completion(model, instructions, completions)
    hook_names = {f"blocks.{layer}.hook_resid_pre" for layer in layers}
    _, cache = model.run_with_cache(
        batch.tokens,
        attention_mask=batch.attention_mask,
        return_type="logits",
        names_filter=lambda name: name in hook_names,
    )
    selected_by_layer = {}
    for layer in layers:
        values = cache[f"blocks.{layer}.hook_resid_pre"]
        if site == "final_response_token":
            indices = values.new_tensor(range(values.shape[0]))
            selected = values[
                indices,
                batch.final_response_positions.to(values.device),
                :,
            ]
        elif site == "response_mean":
            mask = batch.response_mask.to(values.device).unsqueeze(-1)
            selected = (values * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            raise ValueError(f"unsupported capture site: {site}")
        selected_by_layer[layer] = selected.detach().float().cpu().numpy()
    return selected_by_layer


def residual_at_response(
    model,
    instructions: list[str],
    completions: list[str],
    layer: int,
    site: str = "response_mean",
):
    """Backward-compatible single-layer response capture helper."""

    return residuals_at_response(model, instructions, completions, [layer], site)[layer]


def residual_at_last_token(model, prompts: list[str], layer: int):
    """Legacy smoke helper: capture final non-padding token of complete strings."""
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
