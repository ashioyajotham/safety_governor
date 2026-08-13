# Model-assisted annotation

Model output is candidate material, not approval evidence. Provider-neutral working files live under `data/working/`; raw responses, failures, and run manifests live under the ignored `data/archive/annotation_runs/` tree. Exact provider and model identifiers remain in structured provenance.

The current Google Gemini adapter is [`notebooks/providers/gemini/ifeval_annotation_colab.ipynb`](notebooks/providers/gemini/ifeval_annotation_colab.ipynb). It verifies the pinned IFEval source hash and workload composition, writes neutral artifact names, and never marks records approved or commits generated output. A future provider adapter must follow the same output contract.

The canonical instruction pool is assembled with `python -m scripts.assemble_instruction_candidate_pool`. Human decisions are then applied explicitly with `python -m scripts.apply_annotation_review`; a non-empty review note is required and reviewer identity remains optional.

Semantic validation uses a separate provider-neutral boundary. ``python -m scripts.semantic_audit export`` creates blinded A/B tasks and an isolated pair mapping. Imported scores only produce diagnostic flags; provider identity remains in the audit run manifest and no score can approve or rewrite a candidate.

## Human review workbench

The active review interface is
[`notebooks/review/ifeval_human_review_workbench.ipynb`](notebooks/review/ifeval_human_review_workbench.ipynb).
It runs locally or in Colab and uses `safety_governor.review_workbench` as its tested
state engine. The notebook is intentionally a thin interface rather than a second
implementation of review policy.

Prepare the input after verifying the ignored working state:

```powershell
python -m scripts.prepare_review_workbench_bundle `
  --output data/working/instruction_noncompliance/review_workbench_bundle.zip
```

The bundle includes the canonical candidates, the 82 mechanical decisions, 8 repaired
decisions, 60 semantic decisions, and the blinded audit artifacts. Its manifest pins
every file hash, immutable row fingerprint, queue count, rubric version, and Git
revision. Resume fails if any source text or validation evidence has changed.

Human decisions live outside those source rows. Each session has an anonymous UUID,
atomic decision checkpoints, optimistic state revisions, and an append-only review
event log. Approval and rejection require at least 20 non-whitespace characters of
rationale. The interface creates only the active archetype's rubric, and each answer
remains explicitly pending until the reviewer selects Yes or No.

For mechanical rows, the corpus decision and declared-failure verdict are independent:
a real checker failure can still accompany rejection for poor naturalism or a confound.
Every resolved mechanical row requires that verdict; approval additionally requires a
confirmed verdict and an explicit Yes on its behaviour question. Semantic rejection
does not inherit approval-only rubric or audit requirements.

Semantic review is human-first: all 60 initial judgments must be resolved and hashed
before diagnostic scores can be imported. Unflagged pairs receive `no_flag`; a flagged
pair retained as approved requires explicit `flag_reviewed` acknowledgement. Neither a
provider score nor an audit flag changes the human decision automatically.

A final export requires 150 resolved decisions, an imported exact-membership semantic
audit, coherent rubrics, and resolved acknowledgements. It may still report
`freeze_ready: false` if rejection leaves an archetype below its 30-pair quota; in that
case replacements are required. Import a completed export without overwriting an
earlier session with:

```powershell
python -m scripts.import_review_workbench_export PATH_TO_EXPORT.zip
```

Provider names, model revisions, audit scores, notes, and event records remain review
provenance. They are never copied through the experiment materialization boundary.
