"""Optional heavyweight model backend and explicit response-position capture."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TokenizedBatch:
    tokens: object
    attention_mask: object
    response_mask: object
    final_response_positions: object


def load_transformerlens_model(name: str, revision: str, device: str | None = None):
    """Load a pinned HF revision through the TransformerLens v3 Bridge."""
    if not revision or revision in {"main", "master", "latest"}:
        raise ValueError("model revision must be an immutable commit or tag")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformer_lens.model_bridge import TransformerBridge
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the pinned TransformerLens/Transformers dependencies.") from exc
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    hf_model = AutoModelForCausalLM.from_pretrained(name, revision=revision)
    bridge = TransformerBridge(
        model_name=name, hf_model=hf_model, tokenizer=tokenizer, device=device
    )
    bridge.enable_compatibility_mode()
    return bridge


def _prefix_ids(tokenizer, instruction: str) -> list[int]:
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
        offset = width - len(sequence) if padding_side == "left" else 0
        end = offset + len(sequence)
        tokens[index, offset:end] = torch.tensor(sequence)
        attention[index, offset:end] = True
        response_mask[index, offset + response_start:end] = True
        final_positions[index] = end - 1
    return TokenizedBatch(tokens, attention, response_mask, final_positions)


def residual_at_response(
    model,
    instructions: list[str],
    completions: list[str],
    layer: int,
    site: str = "response_mean",
):
    """Capture response-token mean (primary) or final response token (sensitivity)."""
    batch = tokenize_instruction_completion(model, instructions, completions)
    try:
        _, cache = model.run_with_cache(
            batch.tokens, attention_mask=batch.attention_mask, return_type="logits"
        )
    except TypeError:  # supports small test doubles
        _, cache = model.run_with_cache(batch.tokens, return_type="logits")
    values = cache[f"blocks.{layer}.hook_resid_pre"]
    if site == "final_response_token":
        indices = values.new_tensor(range(values.shape[0]))
        selected = values[indices, batch.final_response_positions.to(values.device), :]
    elif site == "response_mean":
        mask = batch.response_mask.to(values.device).unsqueeze(-1)
        selected = (values * mask).sum(dim=1) / mask.sum(dim=1)
    else:
        raise ValueError(f"unsupported capture site: {site}")
    return selected.detach().float().cpu().numpy()


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