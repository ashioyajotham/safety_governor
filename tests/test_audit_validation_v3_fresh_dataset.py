"""The fresh-corpus audit must fail closed without touching the holdout."""
import json
from pathlib import Path

from scripts.audit_validation_v3_fresh_dataset import ARCHETYPES, audit
from safety_governor.validation_v3 import file_sha256


def _pair(pair_id: str, group: str, archetype: str, *, role: str, split: str,
          instruction: str) -> list[dict]:
    common = {
        "pair_id": pair_id, "source_group_id": group, "archetype": archetype,
        "source_record_id": pair_id, "source_revision": "test-revision",
        "source": "synthetic-test", "behavior": "deceptive_reasoning",
        "expected_behavior": "answer accurately", "language": "en",
        "instruction": instruction, "prompt": "", "reviewer_status": "approved",
        "validation_role": role, "split": split,
    }
    return [{**common, "polarity": polarity, "completion": polarity + " answer"}
            for polarity in ("safe", "unsafe")]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _corpora(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidate, train, spent = (tmp_path / name for name in
                               ("candidate.jsonl", "train.jsonl", "spent.jsonl"))
    rows = [row for archetype in ARCHETYPES for index in range(12)
            for row in _pair(f"fresh-{archetype}-{index}", f"fresh-group-{archetype}-{index}",
                             archetype, role="fresh_development", split="validation",
                             instruction=f"Fresh question {archetype} {index}?")]
    _write(candidate, rows)
    _write(train, _pair("train-1", "train-group", ARCHETYPES[0], role="train",
                        split="train", instruction="Training question?"))
    _write(spent, _pair("spent-1", "spent-group", ARCHETYPES[0], role="calibration",
                        split="validation", instruction="Spent question?"))
    return candidate, train, spent


def test_fresh_audit_passes_local_checks_but_not_holdout_gate(tmp_path):
    candidate, train, spent = _corpora(tmp_path)
    result = audit(candidate, train, spent, file_sha256(candidate))
    assert result["local_checks_pass"] is True
    assert result["sealed_confirmatory_contents_accessed"] is False
    assert result["ready_for_generation"] is False


def test_fresh_audit_detects_hash_and_known_split_overlap(tmp_path):
    candidate, train, spent = _corpora(tmp_path)
    rows = [json.loads(line) for line in candidate.read_text().splitlines()]
    for row in rows[:2]:
        row["source_group_id"] = "train-group"
        row["instruction"] = "Training question?"
    _write(candidate, rows)
    result = audit(candidate, train, spent, "0" * 64)
    assert result["local_checks_pass"] is False
    assert any("SHA-256" in error for error in result["errors"])
    assert any("source_group_id overlaps train" == error for error in result["errors"])
    assert any("normalized instruction overlaps train" == error for error in result["errors"])
