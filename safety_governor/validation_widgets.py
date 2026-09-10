"""Thin Jupyter UI for blinded Stage-1 validation judgments.

The widget displays only ``review_tasks.jsonl``. Intervention identities remain
in the separate mapping file and are never loaded here. Every successful save
atomically checkpoints the decision JSONL, so a notebook disconnect does not
erase completed judgments.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

from .validation import validate_review_decisions


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_validation_review_widget(
    tasks_path: str | Path,
    decisions_path: str | Path,
    *,
    reviewer: str,
):
    """Build a checkpointed, configuration-blind ipywidgets review panel."""

    if not reviewer.strip():
        raise ValueError("reviewer identity is required")
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover - notebook dependency
        raise RuntimeError("Install requirements-review.txt to use the review panel") from exc
    tasks = _read_jsonl(Path(tasks_path))
    destination = Path(decisions_path)
    existing = _read_jsonl(destination) if destination.exists() else []
    decisions = {row["task_id"]: row for row in existing if row.get("target_safe") != "unanswered"}
    index = next((i for i, row in enumerate(tasks) if row["task_id"] not in decisions), 0)

    heading = widgets.HTML()
    body = widgets.HTML()
    target = widgets.ToggleButtons(description="Target-safe", options=["unanswered", "yes", "no", "uncertain"])
    relevant = widgets.ToggleButtons(description="Relevant", options=["unanswered", "yes", "no", "uncertain"])
    coherent = widgets.ToggleButtons(description="Coherent", options=["unanswered", "yes", "no", "uncertain"])
    rationale = widgets.Textarea(description="Rationale", layout=widgets.Layout(width="100%", height="100px"))
    previous = widgets.Button(description="Previous")
    next_button = widgets.Button(description="Next")
    save = widgets.Button(description="Save & Next", button_style="primary")
    status = widgets.HTML()

    state = {"index": index}

    def render() -> None:
        row = tasks[state["index"]]
        saved = decisions.get(row["task_id"], {})
        heading.value = f"<h3>{state['index'] + 1}/{len(tasks)} — {html.escape(row['archetype'])}</h3>"
        body.value = (
            f"<p><b>Instruction</b><br>{html.escape(row['instruction'])}</p>"
            f"<p><b>Response</b><br>{html.escape(row['response'])}</p>"
            f"<p><b>Rubric</b><br>{html.escape(row['rubric'])}</p>"
        )
        target.value = saved.get("target_safe", "unanswered")
        relevant.value = saved.get("relevant", "unanswered")
        coherent.value = saved.get("coherent", "unanswered")
        rationale.value = saved.get("rationale", "")
        status.value = f"<b>Checkpointed:</b> {len(decisions)}/{len(tasks)}"

    def move(delta: int) -> None:
        state["index"] = min(max(state["index"] + delta, 0), len(tasks) - 1)
        render()

    def save_current(_=None) -> None:
        row = tasks[state["index"]]
        decision = {
            "task_id": row["task_id"],
            "target_safe": target.value,
            "relevant": relevant.value,
            "coherent": coherent.value,
            "rationale": rationale.value,
            "reviewer": reviewer.strip(),
        }
        try:
            validate_review_decisions([row], [decision])
        except ValueError as exc:
            status.value = f"<span style='color:#b00020'><b>Save blocked:</b> {html.escape(str(exc))}</span>"
            return
        decisions[row["task_id"]] = decision
        ordered = [decisions[item["task_id"]] for item in tasks if item["task_id"] in decisions]
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_jsonl(destination, ordered)
        if state["index"] < len(tasks) - 1:
            move(1)
        else:
            render()

    previous.on_click(lambda _: move(-1))
    next_button.on_click(lambda _: move(1))
    save.on_click(save_current)
    panel = widgets.VBox([
        heading, body, target, relevant, coherent, rationale,
        widgets.HBox([previous, next_button, save]), status,
    ])
    render()
    display(panel)
    return panel
