# Data-quality remediation and release gate

## Corrected corpus status

The original 100 arithmetic deceptive-reasoning pairs and 100 template instruction-noncompliance pairs are now **pilot-only**. They remain useful for smoke tests, but are excluded from all RQ1-RQ3 claims. The previously generated Swahili manifest was based on that pilot mix and is invalidated for cross-lingual analysis.

## Required English research corpus

Create a new, source-backed 120-pair deceptive-reasoning corpus before extraction. Use 30 pairs per archetype:

| Archetype | Required source | Pair construction |
| --- | --- | --- |
| Arithmetic reasoning error | Existing CC-BY prior-work data | Faithful versus corrupted reasoning, with diverse arithmetic formats. |
| Factual confabulation | TruthfulQA | Truthful reference response versus documented misconception/false answer. |
| False-premise agreement | SycophancyEval / Anthropic evals | Evidence-based response versus agreement with the user's false premise. |
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

- Upgrade or isolate the dataset-loading environment so TruthfulQA can be ingested reproducibly; the currently installed `datasets` package fails on that dataset's feature schema.
- Add source importers and a corpus-audit command before adding any new approval labels.
- Do not report cross-lingual results until the replacement manifest is filled and reviewed.

## Freeze rule for surplus annotations

Keep all reviewed candidate annotations through quality review. The constraint-omission pool currently exceeds the 30-pair target; do not discard it during annotation. At freeze time, `python -m scripts.freeze_instruction_corpus <annotation-files> --output <frozen-file> --seed 42` requires approved, non-empty, unique records and selects exactly 30 pairs per archetype using a deterministic hash of the pair ID. Surplus approved records remain a held-out audit/evaluation reserve and must not be mixed back into vector fitting without a documented split decision.

## Current remediation gate (2026-08-07)

The canonical instruction candidate pool contains 150 unique pending-review candidates: 60 constraint-omission candidates and 30 candidates for each other archetype. Known Markdown/newline defects and lexical artifacts were remediated before assembly; original provider outputs remain immutable in the local archive. Approval requires an explicit decision and non-empty review note, while reviewer identity remains optional.

Before freeze, run `python -m scripts.audit_annotation_artifacts <reviewed-file>` and the strengthened research-corpus audit. Deceptive motivated-reasoning candidates must come from preference-bearing feedback prompts and must not share a single safe or unsafe completion template. Deceptive and harmful-compliance corpora remain unassigned until approval; use `python -m scripts.assign_pair_splits` only after every record is approved.

## 2026-08-08 scientific hardening gate

The instruction pool has 150 source-resolved candidates under archetype-aware validation. All safe completions pass the pinned official checker. The 90 mechanical candidates require a declared unsafe constraint failure; 60 semantic candidates preserve all mechanics and require human behavioural review. The current queues contain 82 mechanically aligned rows, 8 repaired rows requiring re-review, and 60 semantic rows. All 30 hedging evasions were rewritten as constraint-preserving drafts. The artifact audit passes (hedging cues 2/30; false-completion cues 8/30).

The semantic audit exports blinded provider-neutral tasks. Relevance and task completeness characterize topic shift; directness, task completeness, and caveat dominance characterize hedging. Scores are diagnostic only, and flagged disagreements require human acknowledgement. Provider metadata and scores never enter experiment materialization.

The deceptive draft now uses explicit instruction/completion boundaries and stable source groups. False-premise variants collapse to eight underlying question groups; motivated-reasoning variants collapse to fifteen argument groups. All thirty motivated pairs have individually authored, source-grounded safe and unsafe rewrites and remain draft until human review. The strict motivated audit passes with near-duplicate p95 of 0.373 for safe and 0.406 for unsafe completions. The human queue is `data/working/deceptive_reasoning/motivated_review_queue.jsonl`.

The deceptive diverse corpus is now approved and split-ready under the current review boundary, but it remains an ignored working artifact until the frozen release input is copied into the tracked research corpus. The approved-ready materialization lives in `data/working/deceptive_reasoning/deceptive_diverse_approved_ready.jsonl`, and the approved motivated-reasoning review queue is recorded separately for auditability.

The previous harmful-compliance construction is quarantined and excluded from Stage 1. JailbreakBench target strings are retained only as rebuild-task prefixes; they are not treated as full unsafe completions. A replacement corpus must use diverse full safe and unsafe responses and pass template and encoding audits. The current scaffold is `data/working/harmful_compliance/rebuild_tasks.jsonl`, which documents the restricted rebuild without elevating the corpus to Stage 1 eligibility.

Exact hashes for ignored working artifacts are recorded in `datasets/manifests/working_state.json` without changing their approval state.
