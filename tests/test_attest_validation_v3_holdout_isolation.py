"""Synthetic-only tests; never load the real sealed confirmatory split."""
import json

import pytest

from scripts.attest_validation_v3_holdout_isolation import (
    attest, file_sha256, normalized_instruction,
)


def _corpora(tmp_path):
    fresh, holdout = tmp_path / "fresh.jsonl", tmp_path / "holdout.jsonl"
    for path, role, count in ((fresh, "fresh_development", 48),
                              (holdout, "confirmatory", 64)):
        rows = []
        for index in range(count):
            common = {
                "pair_id": f"{role}-pair-{index}",
                "source_group_id": f"{role}-group-{index}",
                "source": "fixture", "source_revision": "rev-1",
                "source_record_id": f"{role}-record-{index}",
                "instruction": f"{role} question {index}?",
                "validation_role": role,
            }
            rows.extend({**common, "polarity": polarity} for polarity in
                        ("safe", "unsafe"))
        path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                        encoding="utf-8")
    return fresh, holdout


def _attest(fresh, holdout):
    return attest(fresh, holdout, candidate_sha256=file_sha256(fresh),
                  holdout_sha256=file_sha256(holdout))


def _change_first_pair(path, field, value):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows[:2]:
        row[field] = value
    path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8")


def test_isolated_attestation_contains_no_examples(tmp_path):
    fresh, holdout = _corpora(tmp_path)
    result = _attest(fresh, holdout)
    assert result["isolated"] is True
    assert result["overlap_counts"] == dict.fromkeys(
        ("pair_id", "source_group_id", "source_record_id", "normalized_instruction"), 0)
    assert "question" not in json.dumps(result)


@pytest.mark.parametrize("field,axis,value", [
    ("pair_id", "pair_id", "fresh_development-pair-0"),
    ("source_group_id", "source_group_id", "fresh_development-group-0"),
    ("source_record_id", "source_record_id", "fresh_development-record-0"),
    ("instruction", "normalized_instruction", "FRESH_DEVELOPMENT, question 0!"),
])
def test_each_overlap_axis_fails_closed(tmp_path, field, axis, value):
    fresh, holdout = _corpora(tmp_path)
    _change_first_pair(holdout, field, value)
    result = _attest(fresh, holdout)
    assert result["isolated"] is False
    assert result["overlap_counts"][axis] == 1


def test_hash_mismatch_blocks_before_comparison(tmp_path):
    fresh, holdout = _corpora(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        attest(fresh, holdout, candidate_sha256="0" * 64,
               holdout_sha256=file_sha256(holdout))


def test_incomplete_pairs_block(tmp_path):
    fresh, holdout = _corpora(tmp_path)
    fresh.write_text("\n".join(fresh.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="incomplete pairs"):
        _attest(fresh, holdout)


def test_normalization_only_handles_surface_form():
    assert normalized_instruction("  HELLO,   world! ") == "hello world"
    assert normalized_instruction("hello earth") != "hello world"
