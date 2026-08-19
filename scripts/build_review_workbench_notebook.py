"""Rebuild the thin IFEval human-review notebook deterministically."""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


CELLS = [
    md("""# IFEval human review workbench

This notebook reviews, but never edits, the 150 canonical instruction-following contrasts. Decisions are stored in a separate append-only session. Approval/rejection requires a substantive rationale; reviewer names are intentionally not collected.

**Human-first semantic rule:** finish and lock all 60 semantic judgments before importing any diagnostic model scores. Scores can flag a pair for reconsideration but can never approve it.
"""),
    md("""## 1. Environment

The same notebook runs locally and in Colab. In Colab, upload the prepared review bundle; its manifest pins the Git revision used to create it. Google Drive stores atomic checkpoints. Locally, use an ignored directory under `data/working/`.
"""),
    code("""import json, os, sys, tempfile, zipfile
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import files
    upload_directory = Path.cwd().resolve()
    uploaded = files.upload()
    uploaded_name = next(iter(uploaded))
    BUNDLE_ZIP = (upload_directory / uploaded_name).resolve()
else:
    REPO_ROOT = Path.cwd().resolve()
    BUNDLE_ZIP = REPO_ROOT / 'data/working/instruction_noncompliance/review_workbench_bundle.zip'

if not BUNDLE_ZIP.is_file():
    raise FileNotFoundError(
        f'Review bundle not found at {BUNDLE_ZIP}. '
        'Upload review_workbench_bundle.zip again if the Colab runtime restarted.'
    )
try:
    with zipfile.ZipFile(BUNDLE_ZIP) as archive:
        bundle_manifest = json.loads(archive.read('bundle_manifest.json'))
except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
    raise ValueError('The uploaded file is not a valid review workbench bundle.') from exc
if bundle_manifest.get('schema_version') != 2:
    raise ValueError('This notebook requires a v2 review bundle. Rebuild or re-upload the bundle.')

if IN_COLAB:
    !git clone -q https://github.com/ashioyajotham/safety_governor.git /content/safety_governor
    %cd /content/safety_governor
    !git checkout -q {bundle_manifest['code_revision']}
    !pip -q install -r requirements-review.txt
"""),
    md("""## 2. Checkpoint location and verified resume

Each session has a UUID. Source hashes and immutable row fingerprints are verified whenever it resumes. A stale browser tab cannot overwrite a newer state revision.
"""),
    code("""from safety_governor.review_workbench import ReviewSession, extract_bundle

REVIEW_INPUT_ID = bundle_manifest['review_input_id']
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    SESSION_ROOT = Path('/content/drive/MyDrive/safety_governor_review') / REVIEW_INPUT_ID
    BUNDLE_DIR = Path('/content/review_bundles') / REVIEW_INPUT_ID
else:
    SESSION_ROOT = Path('data/working/instruction_noncompliance/review_sessions') / REVIEW_INPUT_ID
    BUNDLE_DIR = Path('data/working/instruction_noncompliance/review_bundles') / REVIEW_INPUT_ID

if not (BUNDLE_DIR / 'bundle_manifest.json').exists():
    extract_bundle(BUNDLE_ZIP, BUNDLE_DIR)
SESSION_DIR = SESSION_ROOT / 'current'
session = ReviewSession(BUNDLE_DIR, SESSION_DIR)
session.manifest, session.progress()
"""),
    md("""## 3. Human review

Review the mechanical and repaired queues, then the semantic queue. Annotation text is read-only and each row shows only its governing rubric. Every rubric answer is explicitly Unanswered, Yes, or No. **Save & Next** writes an atomic checkpoint and append-only event. Approve and Reject require a substantive rationale; resolved mechanical rows also require a declared-failure verdict.
"""),
    code("""from safety_governor.review_widgets import launch
ui = launch(session)
"""),
    md("""## 4. Lock semantic judgments and export blinded audit tasks

Use the lock button only after all 60 semantic rows have an initial approved/rejected decision. The resulting hash fixes those judgments before the model-assisted diagnostic is revealed. After locking, this cell exports only blinded A/B tasksâ€”not the private mapping.
"""),
    code("""# The UI lock button calls this safely. This cell only exports tasks after a lock exists.
if not session.manifest.get('semantic_lock'):
    raise RuntimeError('Lock all 60 semantic judgments in the UI first.')
tasks = BUNDLE_DIR / 'semantic_audit/tasks.jsonl'
if IN_COLAB:
    files.download(str(tasks))
else:
    print(tasks.resolve())
"""),
    md("""## 5. Import provider-neutral diagnostic scores

The score file must contain exactly the 60 blinded task IDs and the requested 1â€“5 integer dimensions. Record the provider and immutable model revision for auditability. Provider information remains in the review manifest and never enters experiment text.
"""),
    code("""if IN_COLAB:
    score_directory = Path.cwd().resolve()
    score_upload = files.upload()
    SCORES_PATH = (score_directory / next(iter(score_upload))).resolve()
else:
    SCORES_PATH = Path('PATH/TO/semantic_scores.jsonl')

PROVIDER = 'human-review'
MODEL_REVISION = 'manual-audit-v1'
if session.manifest.get('semantic_audit'):
    updated = dict(session.manifest['semantic_audit'])
    updated['provider'] = PROVIDER
    updated['model_revision'] = MODEL_REVISION
    updated['provenance_updated_at'] = session.manifest.get('semantic_audit', {}).get('imported_at')
    session.manifest['semantic_audit'] = updated
    session.manifest['state_revision'] += 1
    if hasattr(session, '_persist'):
        session._persist()
    run = updated
else:
    run = session.attach_audit(SCORES_PATH, PROVIDER, MODEL_REVISION, session.revision)
run
"""),
    md("""## 6. Flagged re-review

Relaunch the UI and filter the semantic queue. After the audit is imported, any semantic row with a diagnostic flag must be reopened and acknowledged as `flag_reviewed`; unflagged rows should remain `no_flag`. The diagnostic does not alter approval decisions.
"""),
    code("""ui = launch(session)
"""),
    md("""## 7. Final export

Export is blocked until all 150 decisions are resolved, the audit is attached, every rubric is coherent, and every flagged retained approval is acknowledged. If any archetype has fewer than 30 approvals, the summary records the replacement deficit and the corpus remains blocked from freeze.
"""),
    code("""EXPORT_PATH = SESSION_DIR / f\"ifeval_review_{session.manifest['session_id']}.zip\"
# Uncomment when the UI shows no pending decisions:
# summary = session.export(EXPORT_PATH, session.revision)
# print(json.dumps(summary, indent=2))
# if IN_COLAB: files.download(str(EXPORT_PATH))
"""),
    md("""## 8. Repository handoff

Back in a clean checkout, import without overwriting earlier sessions:

```powershell
python -m scripts.import_review_workbench_export PATH_TO_EXPORT.zip
```

Then pass the three imported reviewed queues to `scripts.apply_final_review_queues`, run the official candidate validator, corpus audit, and deterministic freeze. Review/audit/provider fields are excluded by experiment materialization.
"""),
]


def main() -> None:
    output = Path("docs/notebooks/review/ifeval_human_review_workbench.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"name": output.name, "provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
