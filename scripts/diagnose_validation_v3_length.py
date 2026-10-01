"""Print a non-promoting length diagnostic; never write a behavioral lock."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from safety_governor.validation import (
    file_sha256,
    summarize_behavior_review,
    validate_review_decisions,
)
from safety_governor.validation_v3 import select_behavior_candidate


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def review_category(decision: dict) -> str:
    """Classify reviewer wording, not an inferred model stop reason."""

    if decision["target_safe"] == "yes":
        return "target_safe"
    rationale = decision["rationale"].lower()
    if "truncat" in rationale or "cuts off" in rationale:
        return "reviewer_cited_cutoff"
    return "other_not_target_safe"


def diagnose(run: Path, decisions_path: Path | None = None) -> dict:
    """Return descriptive stop and review counts without modifying artifacts."""

    spec = json.loads((run / "validation_spec.json").read_text(encoding="utf-8"))
    if spec.get("phase") != "validation_v3_post_review_length_diagnostic" or not spec.get("diagnostic_only"):
        raise ValueError("run is not a post-review, diagnostic-only length run")
    if spec.get("validation_role") != "development":
        raise ValueError("length diagnostic must use development data")
    generations = read_jsonl(run / "generations.jsonl")
    pair_ids = set(spec["pair_ids"])
    generation_ids = {
        "baseline" if row["baseline"] else f"configuration_{index:02d}"
        for index, row in enumerate(spec["configurations"])
    }
    expected = {(generation_id, pair_id) for generation_id in generation_ids for pair_id in pair_ids}
    actual = [(row["generation_id"], row["pair_id"]) for row in generations]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("generation rows do not match the locked conditions and pairs")

    limit = int(spec["generation_max_new_tokens"])
    by_generation = {}
    for generation_id in sorted(generation_ids):
        subset = [row for row in generations if row["generation_id"] == generation_id]
        stops = Counter()
        for row in subset:
            metadata = row.get("generation_metadata", {})
            count = metadata.get("generated_token_count")
            reason = metadata.get("stop_reason")
            if reason not in {"stop_token", "token_limit"} or not isinstance(count, int):
                raise ValueError("missing or invalid generation stop metadata")
            if not 0 <= count <= limit or (reason == "token_limit" and count != limit):
                raise ValueError("generation token count contradicts stop reason")
            stops[reason] += 1
        by_generation[generation_id] = {
            "candidate_id": None if generation_id == "baseline" else
                spec["configurations"][int(generation_id[-2:])]["candidate_id"],
            "stop_reasons": dict(stops),
        }

    result = {
        "diagnostic_only": True,
        "promotion_authorized": False,
        "run_spec_sha256": file_sha256(run / "validation_spec.json"),
        "generation_sha256": file_sha256(run / "generations.jsonl"),
        "pairs": len(pair_ids),
        "max_new_tokens": limit,
        "by_generation": by_generation,
    }
    if decisions_path is None:
        return result

    tasks = read_jsonl(run / "review_tasks.jsonl")
    mapping = read_jsonl(run / "review_mapping.jsonl")
    decisions = read_jsonl(decisions_path)
    validate_review_decisions(tasks, decisions)
    mapping_by = {row["task_id"]: row for row in mapping}
    tasks_by = {row["task_id"]: row for row in tasks}
    if set(mapping_by) != set(tasks_by):
        raise ValueError("private mapping and blinded tasks differ")
    categories = {}
    for decision in decisions:
        task_id = decision["task_id"]
        mapped = mapping_by[task_id]
        if tasks_by[task_id]["pair_id"] != mapped["pair_id"]:
            raise ValueError("task and private mapping pair IDs differ")
        categories[(mapped["generation_id"], mapped["pair_id"])] = review_category(decision)
    if set(categories) != expected:
        raise ValueError("review categories do not cover every generation")
    baseline = {pair_id: categories[("baseline", pair_id)] for pair_id in pair_ids}
    for generation_id, row in by_generation.items():
        row["review_categories"] = dict(Counter(
            categories[(generation_id, pair_id)] for pair_id in pair_ids
        ))
        if generation_id != "baseline":
            transitions = Counter(
                (baseline[pair_id], categories[(generation_id, pair_id)]) for pair_id in pair_ids
            )
            row["paired_transitions"] = {
                f"{before} -> {after}": count
                for (before, after), count in sorted(transitions.items())
            }
    summary = summarize_behavior_review(tasks, mapping, decisions)
    selected, gate = select_behavior_candidate(summary)
    result.update({
        "decision_sha256": file_sha256(decisions_path),
        "review_category_source": "rationale_text_keyword_diagnostic",
        "original_gate_descriptive_only": gate,
        "selected_generation_id_descriptive_only": None if selected is None else selected["generation_id"],
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.run, args.decisions), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
