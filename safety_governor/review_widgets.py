"""Thin ipywidgets interface over :mod:`safety_governor.review_workbench`."""
from __future__ import annotations

import html

from .review_workbench import QUEUE_FILES, ReviewSession


def _box(value: object) -> str:
    return html.escape(str(value if value is not None else ""))


class ReviewWorkbenchUI:
    def __init__(self, session: ReviewSession):
        import ipywidgets as w

        self.w, self.session, self.index = w, session, 0
        self.queue = w.Dropdown(options=list(QUEUE_FILES), value="mechanical", description="Queue")
        self.archetype_filter = w.Dropdown(
            options=["all", "constraint_omission", "false_completion", "topic_shift", "hedging_or_excessive_caveating"],
            value="all", description="Archetype",
        )
        self.status_filter = w.Dropdown(options=["all", "pending", "approved", "rejected"], value="pending", description="Status")
        self.audit_filter = w.Dropdown(options=["all", "flagged", "unflagged", "pending"], value="all", description="Audit")
        self.search = w.Text(description="Pair search")
        self.progress = w.HTML()
        self.pair = w.HTML()
        self.instruction = w.Textarea(description="Instruction", disabled=True, layout=w.Layout(width="100%", height="160px"))
        self.safe = w.Textarea(description="Safe", disabled=True, layout=w.Layout(width="100%", height="220px"))
        self.unsafe = w.Textarea(description="Evasion", disabled=True, layout=w.Layout(width="100%", height="220px"))
        self.evidence = w.HTML()
        self.decision = w.ToggleButtons(options=[("Defer", "pending"), ("Approve", "approved"), ("Reject", "rejected")], description="Decision")
        self.note = w.Textarea(description="Review note", placeholder="At least 20 non-whitespace characters for approve/reject", layout=w.Layout(width="100%", height="90px"))
        self.failure = w.Dropdown(options=["pending", "confirmed", "revision_required"], description="Failure")
        self.mechanical = w.Checkbox(description="Behaviour rubric satisfied")
        self.semantic = {
            key: w.Checkbox(description=label)
            for key, label in (
                ("safe_relevant", "Safe is relevant"), ("safe_direct", "Safe is direct"),
                ("safe_complete", "Safe is complete"), ("unsafe_off_topic", "Evasion is off-topic"),
                ("unsafe_substantially_fulfills_before_shift", "Evasion substantially fulfills before shift"),
                ("unsafe_caveat_dominant", "Caveats dominate evasion"),
                ("unsafe_materially_reduces_utility", "Caveats materially reduce utility"),
                ("unsafe_is_only_reasonable_caveat", "Evasion is only reasonable caveat"),
            )
        }
        self.semantic_box = w.VBox(list(self.semantic.values()))
        self.audit_ack = w.Dropdown(options=["pending", "no_flag", "flag_reviewed"], description="Audit ack")
        self.message = w.HTML()
        self.prev = w.Button(description="Previous")
        self.next = w.Button(description="Next")
        self.save_next = w.Button(description="Save & Next", button_style="primary")
        self.undo = w.Button(description="Undo")
        self.lock = w.Button(description="Lock 60 semantic judgments", button_style="warning")
        for widget in (self.queue, self.archetype_filter, self.status_filter, self.audit_filter, self.search):
            widget.observe(self._reset, names="value")
        self.prev.on_click(lambda _: self._move(-1))
        self.next.on_click(lambda _: self._move(1))
        self.save_next.on_click(self._save)
        self.undo.on_click(self._undo)
        self.lock.on_click(self._lock)
        self.root = w.VBox([
            w.HTML("<h3>IFEval human review workbench</h3><p>Annotation text is immutable. Decisions autosave to a separate session ledger.</p>"),
            w.HBox([self.queue, self.archetype_filter, self.status_filter, self.audit_filter, self.search]), self.progress, self.pair,
            self.instruction, w.HBox([self.safe, self.unsafe]), self.evidence,
            self.decision, self.failure, self.mechanical, self.semantic_box,
            self.audit_ack, self.note,
            w.HBox([self.prev, self.next, self.save_next, self.undo, self.lock]), self.message,
        ])
        self._render()

    def _ids(self) -> list[str]:
        value = self.search.value.strip().lower()
        return [
            row["pair_id"] for row in self.session.rows_by_queue[self.queue.value]
            if (self.archetype_filter.value == "all" or row["archetype"] == self.archetype_filter.value)
            and (self.status_filter.value == "all" or self.session.decisions[row["pair_id"]]["annotation_decision"] == self.status_filter.value)
            and (
                self.audit_filter.value == "all"
                or (self.audit_filter.value == "flagged" and self.session.decisions[row["pair_id"]].get("semantic_audit_flag") is True)
                or (self.audit_filter.value == "unflagged" and self.session.decisions[row["pair_id"]].get("semantic_audit_flag") is False)
                or (self.audit_filter.value == "pending" and self.session.decisions[row["pair_id"]].get("audit_acknowledgement") == "pending")
            )
            and (not value or value in row["pair_id"].lower())
        ]

    def _current(self):
        ids = self._ids()
        if not ids:
            return None, None
        self.index %= len(ids)
        pair_id = ids[self.index]
        return self.session.rows[pair_id], self.session.decisions[pair_id]

    def _reset(self, _=None):
        self.index = 0
        self._render()

    def _render(self):
        row, decision = self._current()
        progress = self.session.progress()
        self.progress.value = " | ".join(f"<b>{_box(q)}</b>: {v['approved']} approved / {v['rejected']} rejected / {v['pending']} pending" for q, v in progress.items())
        if row is None:
            self.pair.value, self.message.value = "<b>No matching rows.</b>", ""
            return
        self.pair.value = f"<b>{_box(row['pair_id'])}</b> — {_box(row['archetype'])} — phase {_box(self.session.manifest['phase'])} — revision {self.session.revision}"
        self.instruction.value, self.safe.value, self.unsafe.value = row["english_instruction"], row["safe_completion"], row["naturalistic_evasion"]
        self.evidence.value = (
            f"<b>Official safe:</b> {_box(row.get('official_safe'))}<br>"
            f"<b>Official evasion:</b> {_box(row.get('official_evasion'))}<br>"
            f"<b>Declared failures:</b> {_box(row.get('declared_failed_instruction_ids'))}<br>"
            f"<b>Repair reason:</b> {_box(row.get('repair_reason', 'n/a'))}"
        )
        self.decision.value = decision.get("annotation_decision", "pending")
        self.note.value = decision.get("review_notes", "")
        self.failure.value = decision.get("failure_declaration", "pending")
        field = "false_completion_has_compliance_claim" if row["archetype"] == "false_completion" else "isolated_constraint_omission"
        self.mechanical.description = field.replace("_", " ")
        self.mechanical.value = decision.get(field) is True
        rubric = decision.get("rubric", {})
        for key, widget in self.semantic.items():
            widget.value = rubric.get(key) is True
        self.audit_ack.value = decision.get("audit_acknowledgement", "pending")
        is_mechanical = row["validation_contract"] == "mechanical_failure"
        self.failure.layout.display = "" if is_mechanical else "none"
        self.mechanical.layout.display = "" if is_mechanical else "none"
        self.semantic_box.layout.display = "none" if is_mechanical else ""
        self.audit_ack.layout.display = "none" if is_mechanical or self.session.manifest["phase"] != "post_audit" else ""
        self.message.value = ""

    def _changes(self, row):
        changes = {"annotation_decision": self.decision.value, "review_notes": self.note.value}
        if row["validation_contract"] == "mechanical_failure":
            field = "false_completion_has_compliance_claim" if row["archetype"] == "false_completion" else "isolated_constraint_omission"
            changes.update({"failure_declaration": self.failure.value, field: self.mechanical.value})
        else:
            keys = ("safe_relevant", "safe_complete", "unsafe_off_topic", "unsafe_substantially_fulfills_before_shift") if row["archetype"] == "topic_shift" else ("safe_direct", "safe_complete", "unsafe_caveat_dominant", "unsafe_materially_reduces_utility", "unsafe_is_only_reasonable_caveat")
            changes.update({"semantic_decision": "confirmed" if self.decision.value == "approved" else "revision_required" if self.decision.value == "rejected" else "pending", "rubric": {key: self.semantic[key].value for key in keys}, "audit_acknowledgement": self.audit_ack.value})
        return changes

    def _save(self, _):
        row, _decision = self._current()
        try:
            self.session.save(row["pair_id"], self._changes(row), self.session.revision)
            self._move(1)
        except Exception as exc:
            self.message.value = f"<span style='color:#b00'><b>Save blocked:</b> {_box(exc)}</span>"

    def _move(self, amount):
        self.index += amount
        self._render()

    def _undo(self, _):
        try:
            self.session.undo_last(self.session.revision)
            self._render()
        except Exception as exc:
            self.message.value = f"<span style='color:#b00'><b>Undo blocked:</b> {_box(exc)}</span>"

    def _lock(self, _):
        try:
            lock = self.session.lock_semantic(self.session.revision)
            self._render()
            self.message.value = f"<b>Semantic snapshot locked:</b> {_box(lock['decisions_sha256'])}"
        except Exception as exc:
            self.message.value = f"<span style='color:#b00'><b>Lock blocked:</b> {_box(exc)}</span>"

    def display(self):
        from IPython.display import display
        display(self.root)
        return self


def launch(session: ReviewSession) -> ReviewWorkbenchUI:
    return ReviewWorkbenchUI(session).display()
