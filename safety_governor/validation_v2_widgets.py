"""Checkpointed, role-blind curation UI for Validation-v2 pairs."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

from .validation_v2 import validate_review_decisions


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_validation_v2_curation_widget(
    candidates_path: str | Path,
    decisions_path: str | Path,
    *,
    reviewer: str,
):
    """Display source-backed pairs without revealing validation-role assignment."""

    if not reviewer.strip():
        raise ValueError("reviewer identity is required")
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install requirements-review.txt to use the review panel") from exc
    candidates = _read_jsonl(Path(candidates_path))
    destination = Path(decisions_path)
    existing = _read_jsonl(destination) if destination.exists() else []
    decisions = {row["candidate_id"]: row for row in existing}
    index = next((i for i, row in enumerate(candidates) if row["candidate_id"] not in decisions), 0)

    heading, source = widgets.HTML(), widgets.HTML()
    instruction = widgets.Textarea(description="Instruction", layout=widgets.Layout(width="100%", height="150px"))
    safe = widgets.Textarea(description="Safe", layout=widgets.Layout(width="100%", height="140px"))
    unsafe = widgets.Textarea(description="Unsafe", layout=widgets.Layout(width="100%", height="140px"))
    decision = widgets.ToggleButtons(description="Decision", options=["unanswered", "approved", "rejected"])
    rationale = widgets.Textarea(description="Rationale", layout=widgets.Layout(width="100%", height="100px"))
    previous, next_button = widgets.Button(description="Previous"), widgets.Button(description="Next")
    save = widgets.Button(description="Save & Next", button_style="primary")
    status = widgets.HTML()
    state = {"index": index}

    def render() -> None:
        row = candidates[state["index"]]
        saved = decisions.get(row["candidate_id"], {})
        heading.value = f"<h3>{state['index'] + 1}/{len(candidates)} — {html.escape(row['archetype'])}</h3>"
        source.value = (
            f"<p><b>Source</b>: {html.escape(row['source_dataset'])} "
            f"<code>{html.escape(row['source_record_id'])}</code><br>"
            f"<b>Revision</b>: <code>{html.escape(row['source_revision'])}</code><br>"
            f"<b>Construction</b>: {html.escape(row['construction_method'])}</p>"
            "<p><b>Approval contract</b>: source-faithful, natural, non-degenerate, "
            "and isolates only the declared archetype. Edit the drafts before approval when needed.</p>"
        )
        instruction.value = saved.get("instruction", row["instruction"])
        safe.value = saved.get("safe_completion", row["safe_completion"])
        unsafe.value = saved.get("unsafe_completion", row["unsafe_completion"])
        decision.value = saved.get("decision", "unanswered")
        rationale.value = saved.get("rationale", "")
        status.value = f"<b>Checkpointed:</b> {len(decisions)}/{len(candidates)}"

    def move(delta: int) -> None:
        state["index"] = min(max(state["index"] + delta, 0), len(candidates) - 1)
        render()

    def save_current(_=None) -> None:
        row = candidates[state["index"]]
        current = {
            "candidate_id": row["candidate_id"], "decision": decision.value,
            "instruction": instruction.value, "safe_completion": safe.value,
            "unsafe_completion": unsafe.value, "rationale": rationale.value,
            "reviewer": reviewer.strip(),
        }
        try:
            validate_review_decisions([row], [current])
        except ValueError as exc:
            status.value = f"<span style='color:#b00020'><b>Save blocked:</b> {html.escape(str(exc))}</span>"
            return
        decisions[row["candidate_id"]] = current
        ordered = [decisions[item["candidate_id"]] for item in candidates if item["candidate_id"] in decisions]
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_jsonl(destination, ordered)
        move(1 if state["index"] < len(candidates) - 1 else 0)

    previous.on_click(lambda _: move(-1))
    next_button.on_click(lambda _: move(1))
    save.on_click(save_current)
    panel = widgets.VBox([
        heading, source, instruction, safe, unsafe, decision, rationale,
        widgets.HBox([previous, next_button, save]), status,
    ])
    render()
    display(panel)
    return panel
