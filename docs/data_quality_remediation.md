# Data-quality remediation and release gate

## Corrected corpus status

The original 100 arithmetic deceptive-reasoning pairs and 100 template instruction-noncompliance pairs are now **pilot-only**. They remain useful for smoke tests, but are excluded from all RQ1–RQ3 claims. The previously generated Swahili manifest was based on that pilot mix and is invalidated for cross-lingual analysis.

## Required English research corpus

Create a new, source-backed 120-pair deceptive-reasoning corpus before extraction. Use 30 pairs per archetype:

| Archetype | Required source | Pair construction |
| --- | --- | --- |
| Arithmetic reasoning error | Existing CC-BY prior-work data | Faithful versus corrupted reasoning, with diverse arithmetic formats. |
| Factual confabulation | TruthfulQA | Truthful reference response versus documented misconception/false answer. |
| False-premise agreement | SycophancyEval / Anthropic evals | Evidence-based response versus agreement with the user’s false premise. |
| Motivated reasoning | SycophancyEval feedback/pressure tasks | Independent assessment versus user-pleasing reversal. |

Create a 120-pair naturalistic instruction-noncompliance corpus with four 30-pair archetypes: constraint omission, hedging/excessive caveating, topic shift, and false completion. Use IFEval as the instruction source where applicable; all generated negative completions need reviewer annotation as naturalistic evasion, not merely syntactic violation.

Keep the 100 JailbreakBench harmful-compliance pairs restricted. Require a consistent pair-level review status before counting them in the final corpus; use an optional review note only for edits or edge cases.

## Gates before translation or vector extraction

1. Every pair has provenance, source version, archetype, an explicit approval decision with a non-empty review note, and a split assigned at the pair level. Reviewer identity is optional.
2. No archetype contributes more than 30 of the 120 deceptive or instruction-noncompliance pairs.
3. Split by source item/pair before any vector fit; reserve a held-out English set for each archetype.
4. Run a diversity audit that checks the exact canonical archetype set, source distribution, duplicate prompts, and lexical/template concentration. The frozen instruction corpus must contain exactly 120 rows (30 per archetype).
5. Freeze a **non-arithmetic-dominated** English evaluation subset only after the English gate passes. Then translate it into Swahili with bilingual and safety review.

## Immediate operational work

- Upgrade or isolate the dataset-loading environment so TruthfulQA can be ingested reproducibly; the currently installed `datasets` package fails on that dataset’s feature schema.
- Add source importers and a corpus-audit command before adding any new approval labels.
- Do not report cross-lingual results until the replacement manifest is filled and reviewed.

## Freeze rule for surplus annotations

Keep all reviewed candidate annotations through quality review. The constraint-omission pool currently exceeds the 30-pair target; do not discard it during annotation. At freeze time, `python -m scripts.freeze_instruction_corpus <annotation-files> --output <frozen-file> --seed 42` requires approved, non-empty, unique records and selects exactly 30 pairs per archetype using a deterministic hash of the pair ID. Surplus approved records remain a held-out audit/evaluation reserve and must not be mixed back into vector fitting without a documented split decision.

## Current remediation gate (2026-08-07)

The instruction candidate pool is sufficient in count but has been demoted to `pending_review` after deterministic normalization and lexical-artifact remediation. Known Markdown/newline defects are corrected in a separate local artifact; original Gemini outputs remain immutable. Approval requires an explicit decision and non-empty review note, while reviewer identity remains optional.

Before freeze, run `python -m scripts.audit_annotation_artifacts <reviewed-file>` and the strengthened research-corpus audit. Deceptive motivated-reasoning candidates must come from preference-bearing feedback prompts and must not share a single safe or unsafe completion template. Deceptive and harmful-compliance corpora remain unassigned until approval; use `python -m scripts.assign_pair_splits` only after every record is approved.
