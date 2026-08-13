"""Archetype-scoped ipywidgets interface for the IFEval review workbench."""
from __future__ import annotations

import html

from .review_workbench import QUEUE_FILES, RUBRIC_SPECS, ReviewSession


def _box(value: object) -> str:
    return html.escape(str(value if value is not None else ""))


class ReviewWorkbenchUI:
    def __init__(self, session: ReviewSession):
        import ipywidgets as w

        self.w, self.session, self.index = w, session, 0
        self.queue = w.Dropdown(options=list(QUEUE_FILES), value="mechanical", description="Queue")
        self.archetype_filter = w.Dropdown(
            options=[
                "all", "constraint_omission", "false_completion", "topic_shift",
                "hedging_or_excessive_caveating",
            ],
            value="all", description="Archetype",
        )
        self.status_filter = w.Dropdown(
            options=["all", "pending", "approved", "rejected"],
            value="pending", description="Status",
        )
        self.audit_filter = w.Dropdown(
            options=["all", "flagged", "unflagged", "pending"],
            value="all", description="Audit",
        )
        self.search = w.Text(description="Pair search")
        self.progress = w.HTML()
        self.pair = w.HTML()
        self.instruction = w.Textarea(
            description="Instruction", disabled=True,
            layout=w.Layout(width="100%", height="160px"),
        )
        self.safe = w.Textarea(
            description="Safe", disabled=True,
            layout=w.Layout(width="100%", height="220px"),
        )
        self.unsafe = w.Textarea(
            description="Evasion", disabled=True,
            layout=w.Layout(width="100%", height="220px"),
        )
        self.evidence = w.HTML()
        self.decision = w.ToggleButtons(
            options=[("Defer", "pending"), ("Approve", "approved"), ("Reject", "rejected")],
            description="Corpus decision",
            style={"description_width": "initial"},
        )
        self.note = w.Textarea(
            description="Review note",
            placeholder="At least 20 non-whitespace characters for Approve or Reject",
            layout=w.Layout(width="100%", height="90px"),
            style={"description_width": "initial"},
        )
        self.failure = w.Dropdown(
            options=[
                ("Unanswered", "pending"),
                ("Yes — confirmed", "confirmed"),
                ("No — revision required", "revision_required"),
            ],
            description="Declared-failure verdict",
            style={"description_width": "initial"},
            layout=w.Layout(width="460px"),
        )
        self.rubric_controls: dict[str, object] = {}
        self.rubric_box = w.VBox()
        self.audit_ack = w.Dropdown(
            options=[
                ("Unanswered", "pending"),
                ("No diagnostic flag", "no_flag"),
                ("Flag reviewed", "flag_reviewed"),
            ],
            description="Audit acknowledgement",
            style={"description_width": "initial"},
        )
        self.message = w.HTML()
        self.prev = w.Button(description="Previous")
        self.next = w.Button(description="Next")
        self.save_next = w.Button(description="Save & Next", button_style="primary")
        self.undo = w.Button(description="Undo")
        self.lock = w.Button(
            description="Lock 60 semantic judgments",
            button_style="warning",
            tooltip="Available after all semantic rows are approved or rejected",
        )
        for widget in (
            self.queue, self.archetype_filter, self.status_filter,
            self.audit_filter, self.search,
        ):
            widget.observe(self._reset, names="value")
        self.prev.on_click(lambda _: self._move(-1))
        self.next.on_click(lambda _: self._move(1))
        self.save_next.on_click(self._save)
        self.undo.on_click(self._undo)
        self.lock.on_click(self._lock)
        self.root = w.VBox([
            w.HTML(
                "<h3>IFEval human review workbench</h3>"
                "<p>Annotation text is immutable. Only the current archetype's rubric is shown. "
                "Decisions autosave to a separate session ledger.</p>"
            ),
            w.HBox([
                self.queue, self.archetype_filter, self.status_filter,
                self.audit_filter, self.search,
            ]),
            self.progress, self.pair, self.instruction,
            w.HBox([self.safe, self.unsafe]), self.evidence,
            self.decision, self.rubric_box, self.audit_ack, self.note,
            w.HBox([self.prev, self.next, self.save_next, self.undo, self.lock]),
            self.message,
        ])
        self._render()

    def _ids(self) -> list[str]:
        value = self.search.value.strip().lower()
        return [
            row["pair_id"] for row in self.session.rows_by_queue[self.queue.value]
            if (
                self.archetype_filter.value == "all"
                or row["archetype"] == self.archetype_filter.value
            )
            and (
                self.status_filter.value == "all"
                or self.session.decisions[row["pair_id"]]["annotation_decision"]
                == self.status_filter.value
            )
            and (
                self.audit_filter.value == "all"
                or (
                    self.audit_filter.value == "flagged"
                    and self.session.decisions[row["pair_id"]].get("semantic_audit_flag") is True
                )
                or (
                    self.audit_filter.value == "unflagged"
                    and self.session.decisions[row["pair_id"]].get("semantic_audit_flag") is False
                )
                or (
                    self.audit_filter.value == "pending"
                    and self.session.decisions[row["pair_id"]].get("audit_acknowledgement")
                    == "pending"
                )
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

    def _answer_control(self, label: str, value: object):
        control = self.w.ToggleButtons(
            options=[("Unanswered", "pending"), ("Yes", True), ("No", False)],
            value=value if value in (True, False, "pending") else "pending",
            layout=self.w.Layout(width="320px"),
        )
        label_widget = self.w.HTML(
            f"<span>{_box(label)}</span>",
            layout=self.w.Layout(width="58%"),
        )
        return control, self.w.HBox([label_widget, control])

    def _configure_rubric(self, row: dict, decision: dict) -> None:
        archetype = row["archetype"]
        self.rubric_controls = {}
        controls = []
        if row["validation_contract"] == "mechanical_failure":
            controls.extend([
                self.w.HTML(
                    "<b>Mechanical evidence</b><br>"
                    "The corpus decision and checker-failure verdict are independent. "
                    "A pair may have a real declared failure but still be rejected for quality."
                ),
                self.failure,
            ])
            self.failure.value = decision.get("failure_declaration", "pending")
            field, label, _expected = RUBRIC_SPECS[archetype][0]
            control, widget_row = self._answer_control(label, decision.get(field, "pending"))
            self.rubric_controls[field] = control
            controls.append(widget_row)
        else:
            controls.append(self.w.HTML(
                f"<b>{_box(archetype.replace('_', ' '))} rubric</b><br>"
                "Answer every question explicitly before approving this pair."
            ))
            rubric = decision.get("rubric", {})
            for field, label, _expected in RUBRIC_SPECS[archetype]:
                control, widget_row = self._answer_control(
                    label, rubric.get(field, "pending")
                )
                self.rubric_controls[field] = control
                controls.append(widget_row)
        self.rubric_box.children = tuple(controls)

    def _render(self):
        row, decision = self._current()
        progress = self.session.progress()
        self.progress.value = " | ".join(
            f"<b>{_box(queue)}</b>: {values['approved']} approved / "
            f"{values['rejected']} rejected / {values['pending']} pending"
            for queue, values in progress.items()
        )
        semantic_pending = progress["semantic"]["pending"]
        show_lock = (
            self.queue.value == "semantic"
            and self.session.manifest["phase"] == "initial_review"
        )
        self.lock.layout.display = "" if show_lock else "none"
        self.lock.disabled = semantic_pending != 0
        self.lock.tooltip = (
            f"{semantic_pending} semantic judgments remain"
            if semantic_pending else "Lock the completed first-pass semantic review"
        )
        if row is None:
            self.pair.value = "<b>No matching rows.</b>"
            self.rubric_box.children = ()
            self.audit_ack.layout.display = "none"
            self.message.value = ""
            return

        self.pair.value = (
            f"<b>{_box(row['pair_id'])}</b> — {_box(row['archetype'])} — "
            f"phase {_box(self.session.manifest['phase'])} — revision {self.session.revision}"
        )
        self.instruction.value = row["english_instruction"]
        self.safe.value = row["safe_completion"]
        self.unsafe.value = row["naturalistic_evasion"]
        self.evidence.value = (
            f"<b>Official safe:</b> {_box(row.get('official_safe'))}<br>"
            f"<b>Official evasion:</b> {_box(row.get('official_evasion'))}<br>"
            f"<b>Declared failures:</b> {_box(row.get('declared_failed_instruction_ids'))}<br>"
            f"<b>Repair reason:</b> {_box(row.get('repair_reason', 'n/a'))}"
        )
        self.decision.value = decision.get("annotation_decision", "pending")
        self.note.value = decision.get("review_notes", "")
        self._configure_rubric(row, decision)
        is_semantic = row["validation_contract"] == "semantic_contrast"
        post_audit = self.session.manifest["phase"] == "post_audit"
        self.audit_ack.layout.display = "" if is_semantic and post_audit else "none"
        self.audit_ack.value = decision.get("audit_acknowledgement", "pending")
        self.message.value = ""

    def _changes(self, row):
        changes = {
            "annotation_decision": self.decision.value,
            "review_notes": self.note.value,
        }
        if row["validation_contract"] == "mechanical_failure":
            field = RUBRIC_SPECS[row["archetype"]][0][0]
            changes.update({
                "failure_declaration": self.failure.value,
                field: self.rubric_controls[field].value,
            })
        else:
            changes.update({
                "rubric": {
                    field: self.rubric_controls[field].value
                    for field, _label, _expected in RUBRIC_SPECS[row["archetype"]]
                },
                "audit_acknowledgement": self.audit_ack.value,
            })
        return changes

    def _save(self, _):
        row, _decision = self._current()
        if row is None:
            self.message.value = "<span style='color:#b00'><b>Save blocked:</b> no row selected</span>"
            return
        pair_id = row["pair_id"]
        try:
            self.session.save(
                pair_id, self._changes(row), self.session.revision
            )
            # Under the default pending filter, the resolved row disappears.
            # Keep the same index so the next pending row is not skipped.
            if pair_id in self._ids():
                self.index += 1
            self._render()
        except Exception as exc:
            self.message.value = (
                f"<span style='color:#b00'><b>Save blocked:</b> {_box(exc)}</span>"
            )

    def _move(self, amount):
        self.index += amount
        self._render()

    def _undo(self, _):
        try:
            self.session.undo_last(self.session.revision)
            self._render()
        except Exception as exc:
            self.message.value = (
                f"<span style='color:#b00'><b>Undo blocked:</b> {_box(exc)}</span>"
            )

    def _lock(self, _):
        try:
            lock = self.session.lock_semantic(self.session.revision)
            self._render()
            self.message.value = (
                f"<b>Semantic snapshot locked:</b> {_box(lock['decisions_sha256'])}"
            )
        except Exception as exc:
            self.message.value = (
                f"<span style='color:#b00'><b>Lock blocked:</b> {_box(exc)}</span>"
            )

    def display(self):
        from IPython.display import display

        display(self.root)
        return self


def launch(session: ReviewSession) -> ReviewWorkbenchUI:
    return ReviewWorkbenchUI(session).display()