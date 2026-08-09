"""Durable, human-first review state for the IFEval annotation gate.

Canonical candidate and queue rows are immutable inputs.  This module stores human
decisions separately, records append-only events, and reveals semantic audit output
only after the first-pass semantic decisions have been locked.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.apply_final_review_queues import apply_instruction, rubric_resolved
from scripts.semantic_audit import assess

BUNDLE_SCHEMA = 1
SESSION_SCHEMA = 1
RUBRIC_VERSION = "ifeval-human-review-v1"
NOTE_MIN_CHARS = 20
QUEUE_FILES = {
    "mechanical": "mechanical_review_queue.jsonl",
    "repaired": "repaired_review_queue.jsonl",
    "semantic": "semantic_review_queue.jsonl",
}
EXPECTED_COUNTS = {"mechanical": 82, "repaired": 8, "semantic": 60}
IMMUTABLE_FIELDS = (
    "pair_id", "archetype", "validation_contract", "english_instruction",
    "instruction_ids", "safe_completion", "naturalistic_evasion",
    "official_safe", "official_evasion", "declared_failed_instruction_ids",
    "repair_reason",
)
DECISION_FIELDS = (
    "annotation_decision", "review_notes", "failure_declaration",
    "isolated_constraint_omission", "false_completion_has_compliance_claim",
    "semantic_decision", "audit_acknowledgement", "semantic_audit_flag", "rubric",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(encoded.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    _atomic_bytes(path, _jsonl_bytes(rows))


def row_fingerprint(row: dict) -> str:
    return canonical_hash({key: row.get(key) for key in IMMUTABLE_FIELDS})


def _git_revision(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _working_artifacts(root: Path) -> dict[str, dict]:
    manifest = json.loads((root / "datasets/manifests/working_state.json").read_text(encoding="utf-8"))
    artifacts = {row["path"]: row for row in manifest["artifacts"]}
    required = [
        "data/working/instruction_noncompliance/candidates.jsonl",
        *(f"data/working/instruction_noncompliance/{name}" for name in QUEUE_FILES.values()),
        "data/working/instruction_noncompliance/semantic_audit/tasks.jsonl",
        "data/working/instruction_noncompliance/semantic_audit/blind_mapping.jsonl",
        "data/working/instruction_noncompliance/semantic_audit/export_manifest.json",
    ]
    for relative in required:
        path = root / relative
        if relative not in artifacts or not path.exists():
            raise ValueError(f"working-state artifact missing: {relative}")
        if sha256_file(path) != artifacts[relative]["sha256"]:
            raise ValueError(f"working-state hash mismatch: {relative}")
    return {relative: artifacts[relative] for relative in required}


def prepare_bundle(root: Path, output: Path) -> dict:
    """Create a self-verifying review bundle from the tracked working-state manifest."""
    root, output = root.resolve(), output.resolve()
    artifacts = _working_artifacts(root)
    bundle_files: dict[str, Path] = {}
    prefix = "data/working/instruction_noncompliance/"
    for relative in artifacts:
        bundle_files[relative.removeprefix(prefix)] = root / relative

    queues = {name: read_jsonl(root / prefix / filename) for name, filename in QUEUE_FILES.items()}
    for name, expected in EXPECTED_COUNTS.items():
        if len(queues[name]) != expected:
            raise ValueError(f"{name}: expected {expected} rows, found {len(queues[name])}")
    all_ids = [row["pair_id"] for rows in queues.values() for row in rows]
    if len(set(all_ids)) != 150 or len(all_ids) != 150:
        raise ValueError("review queues must contain exactly 150 unique pair IDs")

    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "created_at": utc_now(),
        "code_revision": _git_revision(root),
        "rubric_version": RUBRIC_VERSION,
        "queue_counts": EXPECTED_COUNTS,
        "files": {
            arcname: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for arcname, path in sorted(bundle_files.items())
        },
        "immutable_fingerprints": {
            row["pair_id"]: row_fingerprint(row)
            for rows in queues.values() for row in rows
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
        for arcname, path in sorted(bundle_files.items()):
            archive.write(path, arcname)
    return manifest


def extract_bundle(bundle: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        if "bundle_manifest.json" not in names:
            raise ValueError("review bundle has no manifest")
        manifest = json.loads(archive.read("bundle_manifest.json"))
        if manifest.get("schema_version") != BUNDLE_SCHEMA:
            raise ValueError("unsupported review bundle schema")
        expected = {"bundle_manifest.json", *manifest["files"]}
        if names != expected:
            raise ValueError("review bundle membership mismatch")
        for name, facts in manifest["files"].items():
            payload = archive.read(name)
            if sha256_bytes(payload) != facts["sha256"] or len(payload) != facts["bytes"]:
                raise ValueError(f"review bundle content mismatch: {name}")
        archive.extractall(destination)
    verify_bundle_dir(destination)
    return manifest


def verify_bundle_dir(bundle_dir: Path) -> dict:
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    for name, facts in manifest["files"].items():
        path = bundle_dir / name
        if not path.exists() or sha256_file(path) != facts["sha256"]:
            raise ValueError(f"review bundle hash mismatch: {name}")
    queues = {name: read_jsonl(bundle_dir / filename) for name, filename in QUEUE_FILES.items()}
    for name, expected in EXPECTED_COUNTS.items():
        if len(queues[name]) != expected:
            raise ValueError(f"{name}: review bundle count mismatch")
    fingerprints = {row["pair_id"]: row_fingerprint(row) for rows in queues.values() for row in rows}
    if fingerprints != manifest["immutable_fingerprints"]:
        raise ValueError("review bundle immutable fingerprint mismatch")
    return manifest


def _initial_decision(row: dict, queue: str) -> dict:
    decision = {"pair_id": row["pair_id"], "queue": queue, "archetype": row["archetype"]}
    for key in DECISION_FIELDS:
        if key in row:
            decision[key] = copy.deepcopy(row[key])
    decision.setdefault("annotation_decision", "pending")
    decision.setdefault("review_notes", "")
    return decision


def validate_decision(row: dict, decision: dict, *, require_audit: bool = True) -> None:
    status = decision.get("annotation_decision")
    if status == "pending":
        return
    if status not in {"approved", "rejected"}:
        raise ValueError("annotation decision must be pending, approved, or rejected")
    note = decision.get("review_notes", "").strip()
    if len(note) < NOTE_MIN_CHARS:
        raise ValueError(f"resolved decisions require at least {NOTE_MIN_CHARS} non-whitespace characters")
    if status == "rejected":
        return
    if row["validation_contract"] == "mechanical_failure":
        if decision.get("failure_declaration") != "confirmed":
            raise ValueError("mechanical approval requires a confirmed failure declaration")
        field = "false_completion_has_compliance_claim" if row["archetype"] == "false_completion" else "isolated_constraint_omission"
        if decision.get(field) is not True:
            raise ValueError(f"mechanical approval requires {field}=true")
    else:
        if decision.get("semantic_decision") != "confirmed":
            raise ValueError("semantic approval requires semantic_decision=confirmed")
        if not rubric_resolved(row["archetype"], decision.get("rubric", {})):
            raise ValueError("semantic rubric does not support approval")
        if require_audit:
            expected = "flag_reviewed" if decision.get("semantic_audit_flag") else "no_flag"
            if decision.get("audit_acknowledgement") != expected:
                raise ValueError(f"semantic approval requires audit_acknowledgement={expected}")


class ReviewSession:
    """A resumable session with optimistic concurrency and append-only history."""

    def __init__(self, bundle_dir: Path, session_dir: Path):
        self.bundle_dir, self.session_dir = bundle_dir.resolve(), session_dir.resolve()
        self.bundle_manifest = verify_bundle_dir(self.bundle_dir)
        self.rows_by_queue = {
            name: read_jsonl(self.bundle_dir / filename) for name, filename in QUEUE_FILES.items()
        }
        self.rows = {row["pair_id"]: row for rows in self.rows_by_queue.values() for row in rows}
        self.manifest_path = self.session_dir / "session_manifest.json"
        self.decisions_path = self.session_dir / "decisions.jsonl"
        self.events_path = self.session_dir / "review_events.jsonl"
        if self.manifest_path.exists():
            self._load()
        else:
            self._create()

    def _create(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.decisions = {
            row["pair_id"]: _initial_decision(row, queue)
            for queue, rows in self.rows_by_queue.items() for row in rows
        }
        self.manifest = {
            "schema_version": SESSION_SCHEMA,
            "session_id": str(uuid.uuid4()),
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "phase": "initial_review",
            "state_revision": 0,
            "rubric_version": RUBRIC_VERSION,
            "bundle_manifest_sha256": sha256_file(self.bundle_dir / "bundle_manifest.json"),
            "source_fingerprints": self.bundle_manifest["immutable_fingerprints"],
            "semantic_lock": None,
            "semantic_audit": None,
        }
        _atomic_bytes(self.events_path, b"")
        self._persist()

    def _load(self) -> None:
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest["bundle_manifest_sha256"] != sha256_file(self.bundle_dir / "bundle_manifest.json"):
            raise ValueError("session belongs to a different review bundle")
        if self.manifest["source_fingerprints"] != self.bundle_manifest["immutable_fingerprints"]:
            raise ValueError("session source fingerprints changed")
        rows = read_jsonl(self.decisions_path)
        self.decisions = {row["pair_id"]: row for row in rows}
        if set(self.decisions) != set(self.rows) or len(rows) != len(self.rows):
            raise ValueError("session decision membership mismatch")
        if self.manifest.get("decisions_sha256") != sha256_file(self.decisions_path):
            raise ValueError("session decision checkpoint hash mismatch")
        if not self.events_path.exists() or self.manifest.get("events_sha256") != sha256_file(self.events_path):
            raise ValueError("session event-log checkpoint hash mismatch")
        if self.manifest.get("event_count") != len(read_jsonl(self.events_path)):
            raise ValueError("session event-log count mismatch")

    @property
    def revision(self) -> int:
        return int(self.manifest["state_revision"])

    def _assert_revision(self, expected: int) -> None:
        disk = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if expected != self.revision or disk["state_revision"] != expected:
            raise RuntimeError("stale review session; reload before saving")

    def _persist(self) -> None:
        ordered = [self.decisions[pair_id] for pair_id in sorted(self.decisions)]
        _atomic_jsonl(self.decisions_path, ordered)
        self.manifest["decisions_sha256"] = sha256_file(self.decisions_path)
        self.manifest["decision_count"] = len(ordered)
        if self.events_path.exists():
            self.manifest["events_sha256"] = sha256_file(self.events_path)
            self.manifest["event_count"] = len(read_jsonl(self.events_path))
        self.manifest["updated_at"] = utc_now()
        _atomic_json(self.manifest_path, self.manifest)

    def _event(self, pair_id: str, previous: dict, new: dict, action: str) -> None:
        event = {
            "event_id": str(uuid.uuid4()), "session_id": self.manifest["session_id"],
            "pair_id": pair_id, "queue": new["queue"], "action": action,
            "phase": self.manifest["phase"], "timestamp_utc": utc_now(),
            "previous_decision": previous, "new_decision": new,
        }
        with self.events_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def save(self, pair_id: str, changes: dict, expected_revision: int) -> int:
        self._assert_revision(expected_revision)
        if pair_id not in self.decisions:
            raise KeyError(pair_id)
        unknown = set(changes) - set(DECISION_FIELDS)
        if unknown:
            raise ValueError(f"unknown decision fields: {sorted(unknown)}")
        previous = copy.deepcopy(self.decisions[pair_id])
        new = copy.deepcopy(previous)
        new.update(copy.deepcopy(changes))
        require_audit = self.rows[pair_id]["validation_contract"] != "semantic_contrast" or self.manifest["phase"] == "post_audit"
        validate_decision(self.rows[pair_id], new, require_audit=require_audit)
        self.decisions[pair_id] = new
        self.manifest["state_revision"] += 1
        self._event(pair_id, previous, new, "save")
        self._persist()
        return self.revision

    def undo_last(self, expected_revision: int) -> int:
        self._assert_revision(expected_revision)
        events = read_jsonl(self.events_path)
        candidates = [event for event in events if event["action"] in {"save", "undo"}]
        if not candidates:
            raise ValueError("no review event to undo")
        last = candidates[-1]
        pair_id = last["pair_id"]
        previous = copy.deepcopy(self.decisions[pair_id])
        restored = copy.deepcopy(last["previous_decision"])
        self.decisions[pair_id] = restored
        self.manifest["state_revision"] += 1
        self._event(pair_id, previous, restored, "undo")
        self._persist()
        return self.revision

    def progress(self) -> dict:
        result = {}
        for queue in QUEUE_FILES:
            values = [row for row in self.decisions.values() if row["queue"] == queue]
            result[queue] = {
                status: sum(row["annotation_decision"] == status for row in values)
                for status in ("pending", "approved", "rejected")
            }
        return result

    def lock_semantic(self, expected_revision: int) -> dict:
        self._assert_revision(expected_revision)
        if self.manifest.get("semantic_lock"):
            raise ValueError("semantic judgments are already locked")
        semantic = [row for row in self.decisions.values() if row["queue"] == "semantic"]
        if any(row["annotation_decision"] == "pending" for row in semantic):
            raise ValueError("all 60 semantic rows require an initial approved/rejected judgment")
        for decision in semantic:
            validate_decision(self.rows[decision["pair_id"]], decision, require_audit=False)
        snapshot = [{key: row.get(key) for key in DECISION_FIELDS if key != "semantic_audit_flag"} | {"pair_id": row["pair_id"]} for row in sorted(semantic, key=lambda item: item["pair_id"])]
        lock = {"timestamp_utc": utc_now(), "rows": len(snapshot), "decisions_sha256": canonical_hash(snapshot)}
        self.manifest["semantic_lock"] = lock
        self.manifest["phase"] = "post_audit"
        self.manifest["state_revision"] += 1
        self._persist()
        return lock

    def attach_audit(self, scores_path: Path, provider: str, model_revision: str, expected_revision: int) -> dict:
        self._assert_revision(expected_revision)
        if not self.manifest.get("semantic_lock"):
            raise ValueError("semantic judgments must be locked before audit scores are imported")
        if self.manifest.get("semantic_audit"):
            raise ValueError("semantic audit is already attached")
        if not provider.strip() or not model_revision.strip():
            raise ValueError("provider and immutable model revision are required")
        tasks = read_jsonl(self.bundle_dir / "semantic_audit/tasks.jsonl")
        mapping = read_jsonl(self.bundle_dir / "semantic_audit/blind_mapping.jsonl")
        scores = read_jsonl(scores_path)
        task_by = {row["task_id"]: row for row in tasks}
        map_by = {row["task_id"]: row for row in mapping}
        score_by = {row["task_id"]: row for row in scores}
        expected = set(task_by)
        if any(len(index) != len(rows) for index, rows in ((task_by, tasks), (map_by, mapping), (score_by, scores))):
            raise ValueError("duplicate semantic audit task IDs")
        if set(map_by) != expected or set(score_by) != expected:
            raise ValueError("semantic audit task-set mismatch")
        report = [assess(task_by[key], map_by[key], score_by[key]) for key in sorted(expected)]
        report_by = {row["pair_id"]: row for row in report}
        semantic_ids = {row["pair_id"] for row in self.rows_by_queue["semantic"]}
        if set(report_by) != semantic_ids:
            raise ValueError("semantic audit pair membership mismatch")
        for pair_id, outcome in report_by.items():
            decision = self.decisions[pair_id]
            decision["semantic_audit_flag"] = bool(outcome["diagnostic_flag"])
            decision["audit_acknowledgement"] = "pending" if outcome["diagnostic_flag"] else "no_flag"
        report_path = self.session_dir / "semantic_audit_report.jsonl"
        _atomic_jsonl(report_path, report)
        run = {
            "rubric_version": "semantic-contrast-v1", "provider": provider,
            "model_revision": model_revision, "tasks_sha256": sha256_file(self.bundle_dir / "semantic_audit/tasks.jsonl"),
            "scores_sha256": sha256_file(scores_path), "report_sha256": sha256_file(report_path),
            "rows": len(report), "imported_at": utc_now(),
        }
        _atomic_json(self.session_dir / "semantic_audit_run_manifest.json", run)
        self.manifest["semantic_audit"] = run
        self.manifest["state_revision"] += 1
        self._persist()
        return run

    def export(self, output: Path, expected_revision: int) -> dict:
        self._assert_revision(expected_revision)
        if any(row["annotation_decision"] == "pending" for row in self.decisions.values()):
            raise ValueError("all 150 review decisions must be resolved before final export")
        if not self.manifest.get("semantic_audit"):
            raise ValueError("semantic audit must be imported before final export")
        merged = []
        queue_outputs: dict[str, list[dict]] = {}
        for queue, rows in self.rows_by_queue.items():
            queue_outputs[queue] = []
            for row in rows:
                combined = copy.deepcopy(row)
                combined.update(copy.deepcopy(self.decisions[row["pair_id"]]))
                queue_outputs[queue].append(combined)
                merged.append(combined)
        candidates = read_jsonl(self.bundle_dir / "candidates.jsonl")
        applied = apply_instruction(copy.deepcopy(candidates), merged)
        approved = {}
        for archetype in ("constraint_omission", "false_completion", "topic_shift", "hedging_or_excessive_caveating"):
            approved[archetype] = sum(row["archetype"] == archetype and row["annotation_status"] == "approved" for row in applied)
        replacement_required = {key: 30 - value for key, value in approved.items() if value < 30}
        events_hash = sha256_file(self.events_path)
        summary = {
            "session_id": self.manifest["session_id"], "exported_at": utc_now(),
            "rows": 150, "approved_by_archetype": approved,
            "replacement_required": replacement_required,
            "freeze_ready": not replacement_required,
            "state_revision": self.revision,
            "decisions_sha256": sha256_file(self.decisions_path),
            "events_sha256": events_hash,
        }
        payloads = {
            **{f"reviewed/{QUEUE_FILES[queue]}": _jsonl_bytes(rows) for queue, rows in queue_outputs.items()},
            "session/session_manifest.json": self.manifest_path.read_bytes(),
            "session/review_events.jsonl": self.events_path.read_bytes(),
            "session/semantic_audit_report.jsonl": (self.session_dir / "semantic_audit_report.jsonl").read_bytes(),
            "session/semantic_audit_run_manifest.json": (self.session_dir / "semantic_audit_run_manifest.json").read_bytes(),
        }
        summary["file_sha256"] = {name: sha256_bytes(value) for name, value in sorted(payloads.items())}
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in payloads.items():
                archive.writestr(name, payload)
            archive.writestr("validation_summary.json", json.dumps(summary, indent=2) + "\n")
        return summary


def import_review_export(export_zip: Path, destination_root: Path) -> Path:
    """Validate and extract a completed review export without overwriting prior imports."""
    with zipfile.ZipFile(export_zip) as archive:
        names = set(archive.namelist())
        required = {
            *(f"reviewed/{name}" for name in QUEUE_FILES.values()),
            "session/session_manifest.json", "session/review_events.jsonl",
            "session/semantic_audit_report.jsonl", "session/semantic_audit_run_manifest.json",
            "validation_summary.json",
        }
        if names != required:
            raise ValueError("review export membership mismatch")
        summary = json.loads(archive.read("validation_summary.json"))
        if summary.get("rows") != 150:
            raise ValueError("review export must contain 150 resolved rows")
        expected_hashes = summary.get("file_sha256", {})
        if set(expected_hashes) != required - {"validation_summary.json"}:
            raise ValueError("review export hash manifest membership mismatch")
        for name, expected_hash in expected_hashes.items():
            if sha256_bytes(archive.read(name)) != expected_hash:
                raise ValueError(f"review export hash mismatch: {name}")
        session = json.loads(archive.read("session/session_manifest.json"))
        target = destination_root.resolve() / session["session_id"]
        if target.exists():
            raise FileExistsError(f"review session already imported: {target}")
        target.mkdir(parents=True)
        archive.extractall(target)
    return target
