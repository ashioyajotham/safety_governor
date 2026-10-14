import json

import pytest

from scripts.build_validation_v3_fresh_candidates import (
    SOURCES, build_pools, select_review_pool,
)


def _fixture_sources(tmp_path):
    files = {
        "arithmetic_reasoning_error": [{"ID": "a1", "Body": "One box has 5 pencils.",
                                        "Question": "How many are left after two are used?",
                                        "Equation": "( 5 - 2 )", "Answer": 3}],
        "factual_confabulation": [{"question": "Who wrote the book?", "right_answer": "Ada",
                                   "hallucinated_answer": "Bea wrote it.", "knowledge": "Ada wrote it."}],
        "false_premise_agreement": [{"ex_id": "c1", "sentence": "Fish are mammals.",
                                     "explanation": "Fish are not mammals.", "label": "false"}],
        "motivated_reasoning": [{"story_id": 1, "example_id": 2, "label": "True",
                                 "premises": ["All cats purr.", "Milo is a cat."],
                                 "conclusion": "Milo purrs."}],
    }
    paths = {}
    for name, rows in files.items():
        path = tmp_path / f"{name}.json"
        if name == "arithmetic_reasoning_error":
            path.write_text(json.dumps(rows))
        else:
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths[name] = path
    return paths


def test_drafts_have_provenance_but_no_approval(tmp_path):
    paths = _fixture_sources(tmp_path)
    pools = build_pools(paths, {name: "a" * 40 for name in SOURCES})
    rows = select_review_pool(pools, per_archetype=1)
    assert len(rows) == 4
    assert len({row["pair_id"] for row in rows}) == 4
    assert all(row["review_status"] == "unreviewed" for row in rows)
    assert all(row["validation_role"] == "fresh_development_draft" for row in rows)
    assert all(row["source_sha256"] and row["source_group_id"] for row in rows)


def test_cannot_silently_underfill(tmp_path):
    paths = _fixture_sources(tmp_path)
    pools = build_pools(paths, {name: "b" * 40 for name in SOURCES})
    with pytest.raises(ValueError, match="distinct source groups"):
        select_review_pool(pools, per_archetype=2)
