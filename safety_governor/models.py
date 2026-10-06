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


@dataclass(frozen=True)
class ProjectionGate:
    """Train-calibrated projection gate for one activation site."""

    threshold: float
    transition_width: float

    def __post_init__(self):
        if not np.isfinite(self.threshold):
            raise ValueError("projection-gate threshold must be finite")
        if not np.isfinite(self.transition_width) or self.transition_width <= 0:
            raise ValueError("projection-gate transition width must be positive")


@dataclass(frozen=True)
class SteeringSite:
    """One layer in a relative-L2 multi-site steering profile."""

    layer: int
    vector: np.ndarray
    profile_weight: float
    gate: ProjectionGate | None = None

    def __post_init__(self):
        vector = np.asarray(self.vector)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("steering-site vector must be finite and one-dimensional")
        if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5):
            raise ValueError("steering-site vector must be unit normalized")
        if self.layer < 0:
            raise ValueError("steering-site layer must be non-negative")
        if not np.isfinite(self.profile_weight) or self.profile_weight <= 0:
            raise ValueError("steering-site profile weight must be positive")


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


def _generation_stop_ids(tokenizer) -> set[int]:
    """Return EOS plus an available chat end-of-turn token."""

    values = {getattr(tokenizer, "eos_token_id", None)}
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if converter is not None:
        end_of_turn = converter("<|eot_id|>")
        unknown = getattr(tokenizer, "unk_token_id", None)
        if end_of_turn is not None and end_of_turn != unknown:
            values.add(end_of_turn)
    return {int(value) for value in values if value is not None}


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
    try:
        model_device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        # Lightweight test doubles may not expose parameters. Real model
        # backends always do, so retaining the token device is safe here.
        model_device = batch.tokens.device
    tokens = batch.tokens.to(model_device)
    attention_mask = batch.attention_mask.to(model_device)
    hook_names = {f"blocks.{layer}.hook_resid_pre" for layer in layers}
    _, cache = model.run_with_cache(
        tokens,
        attention_mask=attention_mask,
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


def generate_with_steering(
    model,
    instruction: str,
    vector: np.ndarray,
    *,
    layer: int,
    magnitude: float,
    token_mode: str,
    max_new_tokens: int = 128,
) -> str:
    """Greedily generate one response under an explicit causal intervention.

    Stored Stage-1 vectors point from safe to unsafe behavior. This helper
    therefore applies ``-magnitude * vector``.

    Supported Token Policies:
        * ``assistant_boundary``:
          Intervenes solely on the final prompt token (``len(prefix) - 1``),
          reapplying that intervention on every full-prefix forward pass. It
          does not directly modify generated-token positions.
        * ``generation_frontier``:
          Perturbs the residual stream at the current final position
          (``tokens.shape[1] - 1``) on every autoregressive step. Earlier
          generated positions are recomputed unsteered. This specifies the
          operation, not a claim of successful behavioral suppression.

    Full prompt+response recomputation makes the two policies mathematically explicit
    and avoids relying on implementation-specific KV-cache hook behavior.
    """

    if magnitude <= 0:
        raise ValueError("steering magnitude must be positive")
    if token_mode not in {"assistant_boundary", "generation_frontier"}:
        raise ValueError(f"unsupported generation token mode: {token_mode}")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    import torch

    tokenizer = model.tokenizer
    prefix = _prefix_ids(tokenizer, instruction)
    if not prefix:
        raise ValueError("instruction produced an empty generation prefix")
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")
    tokens = torch.tensor([prefix], dtype=torch.long, device=device)
    direction = torch.as_tensor(vector, device=device)
    generated: list[int] = []
    hook = f"blocks.{layer}.hook_resid_pre"
    boundary_position = len(prefix) - 1
    eos_ids = _generation_stop_ids(tokenizer)

    for _ in range(max_new_tokens):
        position = boundary_position if token_mode == "assistant_boundary" else tokens.shape[1] - 1

        def intervention(activation, hook=None):
            changed = activation.clone()
            local = direction.to(device=changed.device, dtype=changed.dtype)
            changed[0, position, :] -= magnitude * local
            return changed

        attention = torch.ones_like(tokens, dtype=torch.bool)
        logits = model.run_with_hooks(
            tokens,
            attention_mask=attention,
            return_type="logits",
            fwd_hooks=[(hook, intervention)],
        )
        next_token = int(torch.argmax(logits[0, -1]).item())
        if next_token in eos_ids:
            break
        generated.append(next_token)
        tokens = torch.cat((tokens, torch.tensor([[next_token]], device=device)), dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True)


def _governor_intervention(site: SteeringSite, position: int, total_relative_l2: float, trace):
    """Build a hook that spends one site's share of a relative-L2 budget."""

    import torch

    direction = torch.as_tensor(site.vector)

    def intervention(activation, hook=None):
        changed = activation.clone()
        local = direction.to(device=changed.device, dtype=changed.dtype)
        active = changed[0, position, :]
        activation_l2 = torch.linalg.vector_norm(active)
        projection = torch.dot(active, local)
        gate_value = torch.ones((), device=changed.device, dtype=changed.dtype)
        if site.gate is not None:
            gate_value = torch.clamp(
                (projection - site.gate.threshold) / site.gate.transition_width,
                min=0.0,
                max=1.0,
            )
        coefficient = total_relative_l2 * site.profile_weight * activation_l2 * gate_value
        changed[0, position, :] -= coefficient * local
        if trace is not None:
            trace.append({
                "layer": site.layer,
                "position": position,
                "activation_l2": float(activation_l2.detach().float().cpu()),
                "projection": float(projection.detach().float().cpu()),
                "gate": float(gate_value.detach().float().cpu()),
                "coefficient": float(coefficient.detach().float().cpu()),
                "relative_delta": float(
                    (coefficient / activation_l2.clamp_min(torch.finfo(changed.dtype).eps))
                    .detach().float().cpu()
                ),
            })
        return changed

    return intervention


def _governor_span_intervention(site, start, end, total_relative_l2, trace):
    """Steer a generated span using each position's own original L2 norm.

    Trace one aggregate per site/forward pass, rather than synchronizing the
    GPU once per historical token. Exposure is not a downstream norm bound.
    """
    import torch

    def intervention(activation, hook=None):
        changed = activation.clone()
        local = torch.as_tensor(site.vector, device=changed.device, dtype=changed.dtype)
        active = changed[0, start:end, :]
        norms = torch.linalg.vector_norm(active, dim=-1)
        projections = active @ local
        gates = torch.ones_like(norms)
        if site.gate is not None:
            gates = ((projections - site.gate.threshold) / site.gate.transition_width).clamp(0, 1)
        coefficients = total_relative_l2 * site.profile_weight * norms * gates
        changed[0, start:end, :] = active - coefficients[:, None] * local
        if trace is not None:
            relative = coefficients / norms.clamp_min(torch.finfo(changed.dtype).eps)
            values = torch.stack([
                norms.mean(), projections.mean(), gates.mean(), coefficients.mean(),
                relative.mean(), relative.min(), relative.max(), coefficients.sum(),
            ]).detach().float().cpu().tolist()
            trace.append(dict(zip(
                ("activation_l2", "projection", "gate", "coefficient", "relative_delta",
                 "relative_delta_min", "relative_delta_max", "injected_l2_sum"), values,
            ), layer=site.layer, position=end - 1, position_start=start,
                position_end_exclusive=end, modified_positions=end - start))
        return changed

    return intervention


def generate_with_governor(
    model,
    instruction: str,
    sites: list[SteeringSite],
    *,
    total_relative_l2: float,
    max_new_tokens: int = 128,
    trace: list[dict] | None = None,
    generation_metadata: dict | None = None,
    token_policy: str = "generation_frontier",
) -> str:
    """Generate with an explicit relative-L2 multi-layer token policy.

    ``total_relative_l2`` is divided by the normalized profile weights. At each
    site and decode step the injected vector norm is therefore
    ``budget * weight * ||activation|| * gate``. This is an intervention budget,
    not a claim that nonlinear downstream state differences add linearly.
    Both policies steer the final prompt token on the first step. Afterwards
    ``generated_span`` steers all generated positions on every recomputation;
    ``generation_frontier`` steers only the newest position. Prompt positions
    are not repeatedly steered by the generated-span policy.
    """

    if token_policy not in {"generation_frontier", "generated_span"}:
        raise ValueError("unsupported governor token policy")
    if not sites:
        raise ValueError("governor requires at least one steering site")
    if not np.isfinite(total_relative_l2) or total_relative_l2 <= 0:
        raise ValueError("total relative-L2 budget must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    layers = [site.layer for site in sites]
    if len(set(layers)) != len(layers):
        raise ValueError("governor steering layers must be unique")
    weight_sum = sum(site.profile_weight for site in sites)
    if not np.isclose(weight_sum, 1.0, atol=1e-8):
        raise ValueError("governor profile weights must sum to one")
    import torch

    tokenizer = model.tokenizer
    prefix = _prefix_ids(tokenizer, instruction)
    if not prefix:
        raise ValueError("instruction produced an empty generation prefix")
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")
    tokens = torch.tensor([prefix], dtype=torch.long, device=device)
    generated: list[int] = []
    eos_ids = _generation_stop_ids(tokenizer)
    ordered = sorted(sites, key=lambda site: site.layer)
    stop_token_id = None
    for _ in range(max_new_tokens):
        position = tokens.shape[1] - 1
        hooks = [
            (
                f"blocks.{site.layer}.hook_resid_pre",
                (_governor_span_intervention(site, len(prefix), tokens.shape[1], total_relative_l2, trace)
                 if token_policy == "generated_span" and generated else
                 _governor_intervention(site, position, total_relative_l2, trace)),
            )
            for site in ordered
        ]
        logits = model.run_with_hooks(
            tokens,
            attention_mask=torch.ones_like(tokens, dtype=torch.bool),
            return_type="logits",
            fwd_hooks=hooks,
        )
        next_token = int(torch.argmax(logits[0, -1]).item())
        if next_token in eos_ids:
            stop_token_id = next_token
            break
        generated.append(next_token)
        tokens = torch.cat((tokens, torch.tensor([[next_token]], device=device)), dim=1)
    if generation_metadata is not None:
        generation_metadata.update({
            "generated_token_count": len(generated),
            "stop_reason": "stop_token" if stop_token_id is not None else "token_limit",
            "stop_token_id": stop_token_id,
        })
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_unsteered(
    model,
    instruction: str,
    *,
    max_new_tokens: int = 128,
    generation_metadata: dict | None = None,
) -> str:
    """Greedily generate a deterministic unsteered baseline response."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    import torch

    tokenizer = model.tokenizer
    prefix = _prefix_ids(tokenizer, instruction)
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")
    tokens = torch.tensor([prefix], dtype=torch.long, device=device)
    generated: list[int] = []
    eos_ids = _generation_stop_ids(tokenizer)
    stop_token_id = None
    for _ in range(max_new_tokens):
        logits = model(tokens, attention_mask=torch.ones_like(tokens, dtype=torch.bool))
        if hasattr(logits, "logits"):
            logits = logits.logits
        next_token = int(torch.argmax(logits[0, -1]).item())
        if next_token in eos_ids:
            stop_token_id = next_token
            break
        generated.append(next_token)
        tokens = torch.cat((tokens, torch.tensor([[next_token]], device=device)), dim=1)
    if generation_metadata is not None:
        generation_metadata.update({
            "generated_token_count": len(generated),
            "stop_reason": "stop_token" if stop_token_id is not None else "token_limit",
            "stop_token_id": stop_token_id,
        })
    return tokenizer.decode(generated, skip_special_tokens=True)


def next_token_choice_scores(
    model,
    prompt: str,
    choices: list[str],
    *,
    vector: np.ndarray | None = None,
    layer: int | None = None,
    magnitude: float | None = None,
) -> np.ndarray:
    """Return next-token scores for choices with an optional unsafe-suppressing hook."""

    import torch

    if not choices:
        raise ValueError("at least one next-token choice is required")
    tokenizer = model.tokenizer
    prefix = _prefix_ids(tokenizer, prompt)
    choice_ids = []
    for choice in choices:
        encoded = list(tokenizer.encode(f" {choice}", add_special_tokens=False))
        if not encoded:
            raise ValueError(f"choice tokenized to empty text: {choice}")
        choice_ids.append(encoded[0])
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")
    tokens = torch.tensor([prefix], dtype=torch.long, device=device)
    attention = torch.ones_like(tokens, dtype=torch.bool)
    if vector is None:
        output = model(tokens, attention_mask=attention)
        logits = output.logits if hasattr(output, "logits") else output
    else:
        if layer is None or magnitude is None or magnitude <= 0:
            raise ValueError("steered choice scoring requires layer and positive magnitude")
        direction = torch.as_tensor(vector, device=device)

        def intervention(activation, hook=None):
            changed = activation.clone()
            changed[0, -1, :] -= magnitude * direction.to(changed.device, changed.dtype)
            return changed

        logits = model.run_with_hooks(
            tokens,
            attention_mask=attention,
            return_type="logits",
            fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", intervention)],
        )
    return logits[0, -1, choice_ids].detach().float().cpu().numpy()


def response_negative_log_likelihood(
    model,
    instruction: str,
    completion: str,
    *,
    vector: np.ndarray | None = None,
    layer: int | None = None,
    magnitude: float | None = None,
    token_mode: str = "generation_frontier",
) -> tuple[float, int]:
    """Score response tokens under baseline or matching teacher-forced steering.

    For ``generation_frontier`` the unsafe-suppressing direction is applied at
    the assistant boundary and every response input position that predicts a
    subsequent response token. ``assistant_boundary`` changes only the prompt
    boundary. The returned sum and token count can be aggregated exactly.
    """

    import torch
    import torch.nn.functional as functional

    if token_mode not in {"assistant_boundary", "generation_frontier"}:
        raise ValueError(f"unsupported token mode: {token_mode}")
    tokenizer = model.tokenizer
    prefix = _prefix_ids(tokenizer, instruction)
    response = list(tokenizer.encode(completion, add_special_tokens=False))
    if not response:
        raise ValueError("completion tokenized to an empty response")
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")
    tokens = torch.tensor([prefix + response], dtype=torch.long, device=device)
    attention = torch.ones_like(tokens, dtype=torch.bool)
    if vector is None:
        output = model(tokens, attention_mask=attention)
        logits = output.logits if hasattr(output, "logits") else output
    else:
        if layer is None or magnitude is None or magnitude <= 0:
            raise ValueError("steered likelihood requires layer and positive magnitude")
        direction = torch.as_tensor(vector, device=device)
        boundary = len(prefix) - 1
        last_predictor = len(prefix) + len(response) - 2

        def intervention(activation, hook=None):
            changed = activation.clone()
            local = direction.to(changed.device, changed.dtype)
            if token_mode == "assistant_boundary":
                changed[0, boundary, :] -= magnitude * local
            else:
                changed[0, boundary:last_predictor + 1, :] -= magnitude * local
            return changed

        logits = model.run_with_hooks(
            tokens,
            attention_mask=attention,
            return_type="logits",
            fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", intervention)],
        )
    start = len(prefix) - 1
    predictors = logits[0, start:start + len(response), :]
    targets = tokens[0, len(prefix):len(prefix) + len(response)]
    loss = functional.cross_entropy(predictors, targets, reduction="sum")
    return float(loss.detach().cpu()), len(response)


def response_negative_log_likelihood_governed(
    model,
    instruction: str,
    completion: str,
    sites: list[SteeringSite],
    *,
    total_relative_l2: float,
) -> tuple[float, int]:
    """Score a supplied completion with all predictor positions steered.

    This parallel proxy is not equivalent to frontier-only generation, which
    recomputes earlier positions without steering at each decoding step.
    """

    if not sites:
        raise ValueError("governed likelihood requires steering sites")
    if not np.isfinite(total_relative_l2) or total_relative_l2 <= 0:
        raise ValueError("total relative-L2 budget must be positive")
    if len({site.layer for site in sites}) != len(sites):
        raise ValueError("governed likelihood steering layers must be unique")
    if not np.isclose(sum(site.profile_weight for site in sites), 1.0, atol=1e-8):
        raise ValueError("governor profile weights must sum to one")
    import torch
    import torch.nn.functional as functional

    tokenizer = model.tokenizer
    prefix = _prefix_ids(tokenizer, instruction)
    response = list(tokenizer.encode(completion, add_special_tokens=False))
    if not response:
        raise ValueError("completion tokenized to an empty response")
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")
    tokens = torch.tensor([prefix + response], dtype=torch.long, device=device)
    attention = torch.ones_like(tokens, dtype=torch.bool)
    boundary = len(prefix) - 1
    last_predictor = len(prefix) + len(response) - 2

    def hook_for(site: SteeringSite):
        direction = torch.as_tensor(site.vector, device=device)

        def intervention(activation, hook=None):
            changed = activation.clone()
            local = direction.to(device=changed.device, dtype=changed.dtype)
            active = changed[0, boundary:last_predictor + 1, :]
            norms = torch.linalg.vector_norm(active, dim=-1, keepdim=True)
            projections = active @ local
            gates = torch.ones_like(projections)
            if site.gate is not None:
                gates = torch.clamp(
                    (projections - site.gate.threshold) / site.gate.transition_width,
                    min=0.0,
                    max=1.0,
                )
            coefficients = total_relative_l2 * site.profile_weight * norms * gates[:, None]
            changed[0, boundary:last_predictor + 1, :] -= coefficients * local
            return changed

        return intervention

    logits = model.run_with_hooks(
        tokens,
        attention_mask=attention,
        return_type="logits",
        fwd_hooks=[
            (f"blocks.{site.layer}.hook_resid_pre", hook_for(site))
            for site in sorted(sites, key=lambda item: item.layer)
        ],
    )
    start = len(prefix) - 1
    predictors = logits[0, start:start + len(response), :]
    targets = tokens[0, len(prefix):len(prefix) + len(response)]
    loss = functional.cross_entropy(predictors, targets, reduction="sum")
    return float(loss.detach().cpu()), len(response)


def response_predictor_projections(
    model,
    instruction: str,
    completion: str,
    *,
    layer: int,
    vector: np.ndarray,
) -> np.ndarray:
    """Project unmodified response-predictor states onto one unit direction."""

    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("projection vector must be finite and one-dimensional")
    if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5):
        raise ValueError("projection vector must be unit normalized")
    import torch

    tokenizer = model.tokenizer
    prefix = _prefix_ids(tokenizer, instruction)
    response = list(tokenizer.encode(completion, add_special_tokens=False))
    if not response:
        raise ValueError("completion tokenized to an empty response")
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")
    tokens = torch.tensor([prefix + response], dtype=torch.long, device=device)
    hook = f"blocks.{layer}.hook_resid_pre"
    _, cache = model.run_with_cache(
        tokens,
        attention_mask=torch.ones_like(tokens, dtype=torch.bool),
        return_type=None,
        names_filter=lambda name: name == hook,
    )
    boundary = len(prefix) - 1
    last_predictor = len(prefix) + len(response) - 2
    states = cache[hook][0, boundary:last_predictor + 1, :]
    local = torch.as_tensor(vector, device=states.device, dtype=states.dtype)
    return (states @ local).detach().float().cpu().numpy()
