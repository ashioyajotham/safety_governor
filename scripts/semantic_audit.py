"""Provider-neutral blinded diagnostic audit for semantic IFEval contrasts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RUBRIC_VERSION = "semantic-contrast-v1"
DIMENSIONS = {
    "topic_shift": ("relevance", "task_completeness"),
    "hedging_or_excessive_caveating": (
        "directness", "task_completeness", "caveat_dominance",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def export_tasks(queue: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    tasks, mapping = [], []
    for row in sorted(queue, key=lambda item: item["pair_id"]):
        archetype = row["archetype"]
        if archetype not in DIMENSIONS:
            raise ValueError(f"{row['pair_id']}: not a semantic archetype")
        digest = hashlib.sha256(f"{seed}:{row['pair_id']}".encode()).hexdigest()
        safe_slot = "a" if int(digest[-1], 16) % 2 == 0 else "b"
        responses = {
            safe_slot: row["safe_completion"],
            "b" if safe_slot == "a" else "a": row["naturalistic_evasion"],
        }
        task_id = digest[:20]
        tasks.append({
            "task_id": task_id,
            "archetype": archetype,
            "instruction": row["english_instruction"],
            "response_a": responses["a"],
            "response_b": responses["b"],
            "score_dimensions": list(DIMENSIONS[archetype]),
            "score_range": [1, 5],
        })
        mapping.append({"task_id": task_id, "pair_id": row["pair_id"], "safe_slot": safe_slot})
    return tasks, mapping


def validate_scores(task: dict, score: dict) -> None:
    expected = set(task["score_dimensions"])
    for slot in ("a", "b"):
        values = score.get(f"scores_{slot}")
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError(f"{task['task_id']}: scores_{slot} dimensions must be {sorted(expected)}")
        if any(not isinstance(value, int) or not 1 <= value <= 5 for value in values.values()):
            raise ValueError(f"{task['task_id']}: scores must be integers from 1 to 5")


def assess(task: dict, mapping: dict, score: dict) -> dict:
    validate_scores(task, score)
    safe = score[f"scores_{mapping['safe_slot']}"]
    unsafe_slot = "b" if mapping["safe_slot"] == "a" else "a"
    unsafe = score[f"scores_{unsafe_slot}"]
    reasons = []
    if task["archetype"] == "topic_shift":
        for dimension in ("relevance", "task_completeness"):
            if safe[dimension] - unsafe[dimension] < 2:
                reasons.append(f"{dimension} contrast below 2 points")
    else:
        if safe["directness"] - unsafe["directness"] < 2:
            reasons.append("directness contrast below 2 points")
        if safe["task_completeness"] - unsafe["task_completeness"] < 2:
            reasons.append("task_completeness contrast below 2 points")
        if unsafe["caveat_dominance"] - safe["caveat_dominance"] < 2:
            reasons.append("caveat_dominance contrast below 2 points")
    return {
        "pair_id": mapping["pair_id"],
        "archetype": task["archetype"],
        "diagnostic_flag": bool(reasons),
        "flag_reasons": reasons,
        "safe_scores": safe,
        "unsafe_scores": unsafe,
    }


def attach_flags(queue: list[dict], report: list[dict]) -> list[dict]:
    by_id = {row["pair_id"]: row for row in report}
    if len(by_id) != len(report) or set(by_id) != {row["pair_id"] for row in queue}:
        raise ValueError("semantic audit report membership mismatch")
    for row in queue:
        flagged = bool(by_id[row["pair_id"]]["diagnostic_flag"])
        row["semantic_audit_flag"] = flagged
        row["audit_acknowledgement"] = "pending" if flagged else "no_flag"
    return queue


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("queue", type=Path)
    export.add_argument("--tasks", type=Path, required=True)
    export.add_argument("--mapping", type=Path, required=True)
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--seed", type=int, default=42)
    ingest = sub.add_parser("import")
    ingest.add_argument("--tasks", type=Path, required=True)
    ingest.add_argument("--mapping", type=Path, required=True)
    ingest.add_argument("--export-manifest", type=Path, required=True)
    ingest.add_argument("--scores", type=Path, required=True)
    ingest.add_argument("--report", type=Path, required=True)
    ingest.add_argument("--run-manifest", type=Path, required=True)
    ingest.add_argument("--provider", required=True)
    ingest.add_argument("--model-revision", required=True)
    attach = sub.add_parser("attach")
    attach.add_argument("queue", type=Path)
    attach.add_argument("report", type=Path)
    attach.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "export":
        tasks, mapping = export_tasks(read(args.queue), args.seed)
        write(args.tasks, tasks)
        write(args.mapping, mapping)
        manifest = {
            "rubric_version": RUBRIC_VERSION,
            "seed": args.seed,
            "rows": len(tasks),
            "queue_sha256": sha256(args.queue),
            "tasks_sha256": sha256(args.tasks),
            "mapping_sha256": sha256(args.mapping),
        }
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"exported {len(tasks)} blinded semantic tasks")
        return
    if args.command == "attach":
        write(args.output, attach_flags(read(args.queue), read(args.report)))
        print("attached diagnostic flags without importing scores or provider metadata")
        return

    manifest = json.loads(args.export_manifest.read_text(encoding="utf-8"))
    if manifest["tasks_sha256"] != sha256(args.tasks) or manifest["mapping_sha256"] != sha256(args.mapping):
        raise SystemExit("semantic audit export hash mismatch")
    tasks, mapping, scores = read(args.tasks), read(args.mapping), read(args.scores)
    task_by_id = {row["task_id"]: row for row in tasks}
    map_by_id = {row["task_id"]: row for row in mapping}
    score_by_id = {row["task_id"]: row for row in scores}
    expected = set(task_by_id)
    if any(len(index) != len(rows) for index, rows in ((task_by_id, tasks), (map_by_id, mapping), (score_by_id, scores))):
        raise SystemExit("duplicate semantic audit task IDs")
    if set(map_by_id) != expected or set(score_by_id) != expected:
        raise SystemExit("semantic audit task-set mismatch")
    report = [assess(task_by_id[key], map_by_id[key], score_by_id[key]) for key in sorted(expected)]
    write(args.report, report)
    run = {
        "rubric_version": RUBRIC_VERSION,
        "provider": args.provider,
        "model_revision": args.model_revision,
        "tasks_sha256": sha256(args.tasks),
        "scores_sha256": sha256(args.scores),
        "report_sha256": sha256(args.report),
        "rows": len(report),
    }
    args.run_manifest.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(f"imported {len(report)} diagnostic scores; no approval state changed")


if __name__ == "__main__":
    main()
