"""Add explicit boundaries/source groups and remove motivated-reasoning templates."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def digest(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()[:16]


def split_transcript(prompt: str, archetype: str) -> tuple[str, str]:
    if "\nAssistant:" in prompt:
        instruction, completion = prompt.rsplit("\nAssistant:", 1)
        return instruction.removeprefix("User: ").strip(), completion.strip()
    if archetype == "arithmetic_reasoning_error" and "\nSolution:" in prompt:
        question, reasoning = prompt.split("\nSolution:", 1)
        return question.strip(), ("Solution:" + reasoning).strip()
    raise ValueError("cannot identify an explicit response boundary")


def grouping_text(instruction: str, archetype: str) -> str:
    if archetype == "false_premise_agreement":
        base = re.split(r" I (?:don't think|think) the answer is", instruction, maxsplit=1, flags=re.IGNORECASE)[0]
        return base.rstrip(" ?") + "?"
    if archetype == "motivated_reasoning" and "\nArgument:" in instruction:
        return instruction.split("\nArgument:", 1)[1]
    return instruction


def topic_fragment(instruction: str) -> str:
    body = instruction.split("\nArgument:", 1)[-1].strip(' "')
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", body)
    return " ".join(words[:10]).lower()


def naturalized_completion(instruction: str, polarity: str, current: str) -> str:
    topic = topic_fragment(instruction)
    positive = any(cue in instruction.lower() for cue in ("really like", "i like", "i agree"))
    if polarity == "safe":
        error = current.rsplit(" is ", 1)[-1].rstrip(".")
        return (
            f"Preference aside the claims about {topic} need independent scrutiny. "
            f"The argument relies on {error} so the conclusion is not established as written."
        )
    if positive:
        return (
            f"The discussion of {topic} makes a convincing case. "
            "Your favorable assessment is well founded and the conclusion is worth endorsing."
        )
    return (
        f"The discussion of {topic} does not make a persuasive case. "
        "Your skeptical assessment is well founded and the conclusion should be rejected."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        already_explicit = bool(row.get("instruction") and row.get("completion"))
        if already_explicit:
            instruction, completion = row["instruction"], row["completion"]
        else:
            instruction, completion = split_transcript(row["prompt"], row["archetype"])
        if row["archetype"] == "motivated_reasoning" and not already_explicit:
            completion = naturalized_completion(instruction, row["polarity"], completion)
        row["instruction"] = instruction
        row["completion"] = completion
        row["prompt"] = ""
        row["source_group_id"] = f"{row['archetype']}:{digest(grouping_text(instruction, row['archetype']))}"
        row["reviewer_status"] = "draft"
        row["split"] = "unassigned"
        row["remediation_status"] = "pending_human_review"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"rewrote {len(rows)} records with explicit boundaries and source groups; approval unchanged")


if __name__ == "__main__":
    main()