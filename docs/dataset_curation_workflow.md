# Contrastive dataset curation workflow

This document defines how a source item becomes a research-eligible contrastive
pair. It is a methodology and release-control document, not merely a file-format
description. The goal is to prevent vectors from learning annotation templates,
source duplication, reviewer metadata, or split leakage instead of the intended
behaviour.

## 1. Unit of analysis

One contrastive pair contains two records with the same `pair_id`:

- `safe`: the response exhibiting the desired behaviour;
- `unsafe`: the matched response exhibiting the target failure mode.

Both records must share the instruction, source, language, source group, split,
and behaviour label. Only the completion and polarity should encode the intended
contrast. A dataset count must therefore distinguish **pairs** from **records**:
120 pairs are stored as 240 records.

The canonical typed schema is `safety_governor.domain.ContrastiveRecord`. The
tracked template is `datasets/templates/contrastive_record_template.jsonl`.

| Field | Purpose |
| --- | --- |
| `pair_id` | Stable identity of the safe/unsafe contrast |
| `behavior` | `deceptive_reasoning`, `instruction_noncompliance`, or `harmful_compliance` |
| `polarity` | `safe` or `unsafe` |
| `language` | Lower-case language code, currently `en` or later `sw` |
| `instruction` | User-side input ending at the assistant boundary |
| `completion` | Assistant response whose activations are captured |
| `source` | Human-readable provenance citation or source identifier |
| `source_group_id` | Identity shared by variants of the same underlying source item |
| `reviewer_status` | Release state; experiment inputs require `approved` |
| `split` | `train`, `validation`, `test`, or `unassigned` before approval |
| `expected_behavior` | Short statement of the intended behavioural contrast |
| `translation_of` | Required link from a Swahili record to its frozen English source |

Annotation-only fields such as model provider, generation trace, reviewer notes,
rubric answers, and semantic-audit scores do **not** belong in experiment input.

## 2. Data states and locations

The repository separates data by scientific state:

1. **Source snapshot** — pinned upstream material under `data/raw/sources/`
   (usually ignored, with hashes tracked).
2. **Candidate** — mutable draft under `data/working/`; never assumed approved.
3. **Review artifact** — queues, rubrics, checker reports, and audit results under
   `data/working/`; these remain separate from model input.
4. **Approved materialization** — field-whitelisted contrastive rows with no
   annotation plumbing.
5. **Frozen release** — immutable experiment input under `datasets/frozen/`, with
   source-group splits and tracked hashes/manifests.
6. **Superseded material** — retained under `data/archive/` or `datasets/pilot/`
   for auditability, never silently reused.

`datasets/manifests/working_state.json` is a hash inventory of a particular local
working snapshot. It does not confer approval and may describe historical queues.
The frozen dataset and its validation result are authoritative for Stage 1.

## 3. Source registration and provenance

Before drafting any pair:

1. Record benchmark name, upstream repository or publication, immutable revision,
   licence, access conditions, and stable item key.
2. Fetch public sources through `scripts/fetch_corpus_sources.py` where supported.
3. Verify tracked source hashes in `datasets/manifests/source_corpora.json` and
   reconstruction requirements in `datasets/manifests/reconstruction.json`.
4. Derive a stable `source_group_id` from the underlying question, argument, or
   source item—not from the polarity or surface variant.

The source group is the leakage boundary. Multiple belief variants of one question
or preference variants of one argument must remain in the same split.

## 4. Behaviour-specific construction

### Deceptive reasoning

The release design uses four archetypes rather than one arithmetic template:

- arithmetic reasoning errors;
- factual confabulation;
- false-premise agreement;
- motivated reasoning.

The safe and unsafe completion should differ in reasoning behaviour, not in a
repeated editorial phrase. Template concentration, repeated prefixes/suffixes,
near duplicates, and source-group reuse are audited before approval.

The current Stage-1 release is `datasets/frozen/english_contrastive.jsonl`: 120
approved pairs (240 records), split at source-group level.

### Instruction non-compliance

Instruction examples originate from pinned IFEval source rows. Four archetypes are
curated:

- `constraint_omission` — omits a declared mechanical requirement;
- `false_completion` — claims or implies completion while violating a requirement;
- `topic_shift` — preserves mechanical requirements but answers a different task;
- `hedging_or_excessive_caveating` — preserves mechanical requirements but caveats
  so heavily that directness and utility are materially reduced.

The official IFEval checker is necessary but not sufficient. Mechanical archetypes
must fail exactly the declared constraint. Semantic archetypes should pass all
mechanical checks and require an archetype-specific human rubric. A bare factual or
arithmetic error is not false completion; reasonable uncertainty or safety caveats
are not excessive caveating; a late pivot that substantially fulfills the original
task is not a clean topic-shift contrast.

Model-assisted drafting may propose completions, but provider and generation
metadata remain in annotation provenance. Approval is always an explicit human
decision with a substantive rationale. The workbench locks first-pass semantic
judgments before blinded diagnostic scores can be attached.

### Harmful compliance

Harmful compliance remains quarantined and is not part of Stage 1. The superseded
construction used one repeated refusal and JailbreakBench target prefixes rather
than diverse, complete responses. That would make a vector learn refusal style
versus acquiescence style.

A future rebuild must contain complete unsafe responses, diverse safe responses,
clean encoding, explicit provenance, and the same duplicate/template/source-group
audits as the other behaviours. Restricted text must stay outside tracked public
artifacts.

## 5. Human review and validation

Every candidate starts as `draft` or `pending_review`. Approval requires:

1. source identity and prompt metadata match the pinned upstream item;
2. safe response satisfies its declared contract;
3. unsafe response isolates the intended failure without introducing an easier
   confound;
4. wording is naturalistic and not dominated by annotation boilerplate;
5. pair rationale is substantive;
6. encoding, duplicate, and template audits pass;
7. explicit review decision is recorded.

Reviewer identity is optional by policy; the decision and rationale are not.
Importers must never infer approval from file location or row count.

Useful gates include:

```bash
python -m scripts.dataset_summary PATH.jsonl
python -m scripts.validate_ifeval_candidates CANDIDATES.jsonl \
  --report IFEVAL_REPORT.jsonl --require-declarations
python -m scripts.audit_annotation_artifacts CANDIDATES.jsonl \
  --strict-archetype motivated_reasoning
```

The provider-neutral workbench is documented in `docs/annotation_assistance.md`.

## 6. Materialization boundary

Reviewed annotation rows are converted to experiment records through a whitelist,
not by copying arbitrary fields:

```bash
python -m scripts.materialize_contrastive_records APPROVED.jsonl \
  --output MATERIALIZED.jsonl
```

The materializer keeps research identity, behaviour, language, instruction,
completion, polarity, source, approval state, split, and source group. It excludes
provider names, reviewer notes, rubric answers, and model-generation traces so these
cannot become lexical input or accidental experiment covariates.

## 7. Split assignment

Split only after approval. The deterministic splitter is archetype-stratified and
operates on `source_group_id`:

```bash
python -m scripts.assign_pair_splits MATERIALIZED.jsonl \
  --output SPLIT.jsonl --seed 42
```

The intended allocation is approximately 70% train, 15% validation, and 15% test,
subject to group counts. Both polarities and all variants in one source group remain
together. Vector fitting is train-only; validation selects layer, method, coefficient,
and intervention position; test remains untouched until the analysis is frozen.

## 8. Freeze and release

Freezing is deterministic and fail-closed. For instruction non-compliance, the
freeze requires the exact four archetypes and 30 approved, human-confirmed unique
pairs per archetype:

```bash
python -m scripts.freeze_instruction_corpus APPROVED_INPUTS... \
  --output FROZEN_ANNOTATIONS.jsonl --seed 42
```

Before an experiment consumes a frozen file:

```bash
python -m scripts.validate_dataset datasets/frozen/english_contrastive.jsonl
python -m scripts.verify_environment configs/llama3_8b.yaml
```

A release record should include dataset SHA-256, source revisions, code revision,
split seed, approval gate, validation commands, and any restricted reconstruction
inputs. Never overwrite a frozen release in place; create a new version and preserve
the prior hash.

## 9. Swahili translation gate

Translate only a frozen English evaluation subset. Each Swahili row must retain
`translation_of`, source identity, split, and restricted translation-quality notes.
Translation must not change the target behaviour or introduce a new failure mode.
The detailed bilingual and safety-review procedure is in
`docs/swahili_translation_protocol.md`.

## 10. Release checklist

- [ ] Upstream source, licence, immutable revision, and hash recorded.
- [ ] Exactly two aligned polarities per pair.
- [ ] Explicit instruction/completion boundary.
- [ ] Stable source group assigned.
- [ ] Intended failure isolated without secondary shortcuts.
- [ ] Human decision and substantive rationale recorded.
- [ ] Official/mechanical or semantic rubric gate passed as applicable.
- [ ] Encoding, duplicate, near-duplicate, and template audits passed.
- [ ] Annotation-provider and reviewer metadata removed at materialization.
- [ ] Source-group-aware split assigned after approval.
- [ ] Frozen file validates and is hash-recorded.
- [ ] Test split remains untouched until final evaluation authorization.
