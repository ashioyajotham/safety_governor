"""Measure held-out capability tax for the selected validation intervention.

MMLU uses five-shot subject prompts and absolute accuracy-point change.
WikiText-103 uses chat-conditioned continuation perplexity with paired text
windows and reports relative change.  Both datasets are immutable-revision
pinned in ``configs/validation.yaml``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from safety_governor.config import load
from safety_governor.models import (
    load_transformerlens_model,
    next_token_choice_scores,
    response_negative_log_likelihood,
)
from safety_governor.preflight import runtime_profile_errors
from safety_governor.reproducibility import environment_facts
from safety_governor.stage1 import atomic_write_json, load_runtime_profile
from safety_governor.validation import file_sha256

LETTERS = ("A", "B", "C", "D")


def _format_question(row: dict, *, answer: bool) -> str:
    text = row["question"].strip() + "\n"
    text += "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, row["choices"]))
    text += "\nAnswer:"
    if answer:
        text += f" {LETTERS[int(row['answer'])]}\n\n"
    return text


def _mmlu_prompt(subject: str, examples: list[dict], target: dict) -> str:
    heading = f"Answer the following multiple choice questions about {subject.replace('_', ' ')}. Reply with only A B C or D.\n\n"
    return heading + "".join(_format_question(row, answer=True) for row in examples) + _format_question(target, answer=False)


def _load_model(main_config: dict, profile: dict):
    os.environ.setdefault("HF_HOME", profile["hf_cache_root"])
    model = main_config["model"]
    return load_transformerlens_model(
        model["name"], model["revision"], profile["device"], profile["dtype"], model["bridge_weight_mode"]
    )


def _evaluate_mmlu(model, benchmark: dict, vector, selected: dict) -> dict:
    from datasets import load_dataset

    data = load_dataset(
        benchmark["dataset"], benchmark["configuration"],
        revision=benchmark["revision"],
    )
    development = {}
    for row in data[benchmark["few_shot_split"]]:
        development.setdefault(row["subject"], []).append(row)
    rows = []
    for index, row in enumerate(data[benchmark["evaluation_split"]]):
        subject = row["subject"]
        examples = development.get(subject, [])[: int(benchmark["shots"])]
        if len(examples) != int(benchmark["shots"]):
            raise ValueError(f"MMLU subject lacks declared few-shot examples: {subject}")
        prompt = _mmlu_prompt(subject, examples, row)
        baseline = next_token_choice_scores(model, prompt, list(LETTERS))
        steered = next_token_choice_scores(
            model, prompt, list(LETTERS), vector=vector,
            layer=int(selected["layer"]), magnitude=float(selected["magnitude"]),
        )
        rows.append({
            "index": index,
            "subject": subject,
            "answer": int(row["answer"]),
            "baseline_prediction": int(np.argmax(baseline)),
            "steered_prediction": int(np.argmax(steered)),
        })
    baseline_accuracy = float(np.mean([row["baseline_prediction"] == row["answer"] for row in rows]))
    steered_accuracy = float(np.mean([row["steered_prediction"] == row["answer"] for row in rows]))
    return {
        "dataset": benchmark,
        "dataset_fingerprint": data[benchmark["evaluation_split"]]._fingerprint,
        "examples": len(rows),
        "baseline_accuracy": baseline_accuracy,
        "steered_accuracy": steered_accuracy,
        "absolute_accuracy_delta": steered_accuracy - baseline_accuracy,
        "predictions": rows,
    }


def _evaluate_wikitext(model, benchmark: dict, vector, selected: dict) -> dict:
    from datasets import load_dataset

    data = load_dataset(
        benchmark["dataset"], benchmark["configuration"],
        revision=benchmark["revision"], split=benchmark["evaluation_split"],
    )
    text = "\n\n".join(row["text"] for row in data if row["text"].strip())
    tokenizer = model.tokenizer
    token_ids = list(tokenizer.encode(text, add_special_tokens=False))
    width = int(benchmark["chunk_tokens"])
    context = int(benchmark["context_tokens"])
    if context <= 0 or width <= context:
        raise ValueError("WikiText chunk_tokens must exceed positive context_tokens")
    baseline_nll = steered_nll = 0.0
    window_scores = []
    tokens = windows = 0
    for start in range(0, len(token_ids) - width + 1, width - context):
        window = token_ids[start:start + width]
        if len(window) <= context:
            continue
        prefix = tokenizer.decode(window[:context], skip_special_tokens=True)
        completion = tokenizer.decode(window[context:], skip_special_tokens=True)
        instruction = f"Continue the following text faithfully:\n\n{prefix}"
        base_sum, base_count = response_negative_log_likelihood(model, instruction, completion)
        steered_sum, steered_count = response_negative_log_likelihood(
            model, instruction, completion, vector=vector,
            layer=int(selected["layer"]), magnitude=float(selected["magnitude"]),
            token_mode=selected["token_mode"],
        )
        if base_count != steered_count:
            raise ValueError("baseline and steered WikiText token counts differ")
        baseline_nll += base_sum
        steered_nll += steered_sum
        tokens += base_count
        windows += 1
        window_scores.append({
            "window": windows - 1,
            "tokens": base_count,
            "baseline_nll": base_sum,
            "steered_nll": steered_sum,
        })
    if not tokens:
        raise ValueError("WikiText validation produced no scored tokens")
    baseline_perplexity = math.exp(baseline_nll / tokens)
    steered_perplexity = math.exp(steered_nll / tokens)
    return {
        "dataset": benchmark,
        "dataset_fingerprint": data._fingerprint,
        "windows": windows,
        "tokens": tokens,
        "baseline_perplexity": baseline_perplexity,
        "steered_perplexity": steered_perplexity,
        "relative_perplexity_delta": (steered_perplexity - baseline_perplexity) / baseline_perplexity,
        "evaluation_semantics": "chat_conditioned_teacher_forced_continuation",
        "window_scores": window_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--validation-config", default="configs/validation.yaml")
    parser.add_argument("--runtime-profile", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    main_config = load(args.config)
    validation_config = load_runtime_profile(args.validation_config)
    profile = load_runtime_profile(args.runtime_profile)
    run = Path(args.run).resolve()
    output = run / "control_tax.json"
    if output.exists():
        raise FileExistsError(f"control-tax artifact already exists: {output}")
    errors = runtime_profile_errors(profile, run.parent)
    facts = environment_facts(profile["device"])
    if profile.get("require_clean_git", True) and facts["git_dirty"]:
        errors.append("control-tax evaluation requires a clean Git checkout")
    if errors:
        raise SystemExit("Control-tax preflight failed:\n- " + "\n- ".join(errors))
    behavior = json.loads((run / "behavior_metrics.json").read_text(encoding="utf-8"))
    selected = behavior.get("selected_configuration")
    if selected is None:
        raise ValueError("behavior review has no eligible selected configuration")
    selected_config = selected["configuration"]
    spec = json.loads((run / "validation_spec.json").read_text(encoding="utf-8"))
    parent = Path(spec["parent_train_run"])
    vector_path = parent / "layers" / f"layer_{selected_config['layer']:02d}" / f"{selected_config['method']}.npy"
    vector = np.load(vector_path, allow_pickle=False)
    model = _load_model(main_config, profile)
    tax = validation_config["control_tax"]
    mmlu = _evaluate_mmlu(model, tax["mmlu"], vector, selected_config)
    wikitext = _evaluate_wikitext(model, tax["wikitext"], vector, selected_config)
    suppression = selected.get("relative_suppression")
    contract = validation_config["metric_contract"]
    viable = (
        suppression is not None
        and suppression > float(contract["targeted_suppression_threshold"])
        and mmlu["absolute_accuracy_delta"] > float(contract["mmlu_degradation_floor"])
    )
    payload = {
        "schema_version": 1,
        "selected_configuration": selected_config,
        "vector_sha256": file_sha256(vector_path),
        "environment": facts,
        "mmlu": mmlu,
        "wikitext": wikitext,
        "targeted_suppression": suppression,
        "metric_contract": contract,
        "provisional_viability": viable,
    }
    atomic_write_json(output, payload)
    print(json.dumps({
        "control_tax": str(output),
        "mmlu_delta": mmlu["absolute_accuracy_delta"],
        "perplexity_delta": wikitext["relative_perplexity_delta"],
        "provisional_viability": viable,
    }, indent=2))


if __name__ == "__main__":
    main()
