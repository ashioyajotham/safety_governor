"""Real-model smoke for explicit response capture and position-aware steering."""
from __future__ import annotations

import argparse

from safety_governor.config import load
from safety_governor.domain import InterventionSpec
from safety_governor.models import load_transformerlens_model, tokenize_instruction_completion
from safety_governor.preflight import runtime_errors
from safety_governor.steering import hook_name, make_hook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    config = load(args.config)
    errors = runtime_errors(config)
    if errors:
        raise SystemExit("Environment preflight failed:\n- " + "\n- ".join(errors))
    model = load_transformerlens_model(
        config["model"]["name"], config["model"]["revision"], args.device
    )
    batch = tokenize_instruction_completion(
        model,
        ["Answer briefly.", "Give one word."],
        ["A concise answer.", "short"],
    )
    clean, cache = model.run_with_cache(
        batch.tokens, attention_mask=batch.attention_mask, return_type="logits"
    )
    residual = cache[hook_name(args.layer)]
    vector = residual.new_zeros(residual.shape[-1])
    vector[0] = 1.0
    spec = InterventionSpec(
        layer=args.layer, coefficient=1.0, token_mode="final_response_token"
    )
    steered = model.run_with_hooks(
        batch.tokens,
        attention_mask=batch.attention_mask,
        return_type="logits",
        fwd_hooks=[(
            hook_name(args.layer),
            make_hook(vector, spec, positions=batch.final_response_positions),
        )],
    )
    if clean.shape != steered.shape or (clean == steered).all():
        raise SystemExit("real-model steering smoke did not alter logits")
    print(
        "Real-model backend smoke passed: explicit response tokens and "
        "position-aware steering altered logits"
    )


if __name__ == "__main__":
    main()
