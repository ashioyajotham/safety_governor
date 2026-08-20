# Runtime Safety Governor

Research repository for **Runtime Safety Governors: Activation Steering as a Control API for Deployed Language Models** (ILINA Junior Research Fellowship, August-November 2026).

This project studies whether a narrow inference-time residual-stream intervention can suppress specified unsafe behaviours without retraining and without unacceptable capability loss. It is a mechanistic-interpretability experiment, not a moderation product.

## Research questions

1. Can contrastive activation directions represent deceptive reasoning and instruction non-compliance in an open-weight model?
2. What is the control tax: targeted suppression versus MMLU accuracy and WikiText-103 perplexity?
3. Do English-derived directions transfer to semantically matched Swahili prompts?
4. As a stretch question, can a residual-stream probe trigger steering conditionally?

The intervention is `A'_L = A_L + alpha v`. The predeclared provisional viability criterion is targeted suppression above 70% with MMLU degradation below 3%.

## Current research gate

The package and GPT-2 smoke path are operational. On 2026-08-08, a pinned GPT-2 commit completed a CPU `response_mean` capture and a position-aware steering-hook smoke under TransformerLens 3.1.0. The tracked smoke manifest records hashes; this verifies plumbing only and is not evidence for a safety claim.

Stage-1 Llama plumbing now targets the frozen deceptive-reasoning release at `datasets/frozen/english_contrastive.jsonl`. Harmful compliance is a later Stage-1b rebuild: the previous JailbreakBench construction remains quarantined because it paired one corrupted refusal template with target prefixes rather than full responses. Swahili translation remains downstream of the English configuration freeze.

The current bottleneck is GPU execution and artifact review, not annotation plumbing. The repo-first Colab runner lives at [`docs/notebooks/stage1/llama_stage1_colab.ipynb`](docs/notebooks/stage1/llama_stage1_colab.ipynb).

## Scientific safeguards

Experiment records separate `instruction` from `completion`. Annotation notes, provider metadata, reviewer fields, and generation traces are excluded by the materialization step and cannot enter model input.

Source groups, rather than pair IDs alone, are assigned to train/validation/test. Vector fitting is train-only. Capture declares its split explicitly; test capture additionally requires an authorization flag. The primary extraction site is the mean over response tokens; the final response token is a sensitivity analysis. PCA operates on aligned `unsafe - safe` deltas. Bootstrap resampling preserves source groups. Position-specific steering requires explicit non-padding positions.

Model repositories are pinned to immutable Hugging Face commits. Run manifests record dataset and code state, environment facts, split, layer, and capture site.

## Repository layout

```text
safety_governor/             domain contracts, tokenization, capture, vectors, steering, evaluation
scripts/                     curation gates, audits, materialization, splitting, experiment entrypoints
configs/                     pinned smoke and Stage-1 experiment configurations
instruction_following_eval/  vendored official IFEval checker at a pinned upstream commit
datasets/fixtures/           non-research smoke fixtures
datasets/manifests/          tracked source, reconstruction, smoke, and archive lineage metadata
datasets/pilot/              invalidated pilot corpus retained for auditability
data/raw/sources/            immutable local upstream snapshots (mostly ignored)
data/working/                mutable candidate, review, quarantine, and report data (ignored)
data/archive/                superseded local artifacts (ignored; hashes tracked)
docs/                        protocol, governance, curation, and research notebook material
docs/notebooks/review/       local/Colab human-review workbench (thin UI over tested core)
docs/notebooks/stage1/      Colab runner for Llama Stage-1 capture and vector extraction
tests/                       deterministic scientific and engineering checks
artifacts/                   ignored run manifests, activation caches, and vectors
```

See [repository lifecycle](docs/repository_layout.md), [data remediation](docs/data_quality_remediation.md), and [experiment protocol](docs/experiment_protocol.md).

## Local verification

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m scripts.validate_dataset datasets/fixtures/contrastive_en.jsonl
pytest -q
```

The final archetype-aware IFEval gate is:

```powershell
python -m scripts.validate_ifeval_candidates data/working/instruction_noncompliance/candidates.jsonl --report data/working/instruction_noncompliance/ifeval_official_report.jsonl --require-declarations
```

It intentionally fails until every mechanical failure declaration and semantic contrast is human-confirmed.

The provider-neutral
[`IFEval review workbench`](docs/notebooks/review/ifeval_human_review_workbench.ipynb)
remains available for audit replay and future refreshes. It was used to resolve the
82-row mechanical queue, the 8-row repaired queue, and the 60-row semantic queue;
it shows only the active archetype's tri-state rubric, stores decisions in a separate
append-only session, and locks all semantic judgments before diagnostic model scores
can be imported. Build its self-verifying input bundle with:

```powershell
python -m scripts.prepare_review_workbench_bundle --output data/working/instruction_noncompliance/review_workbench_bundle.zip
```

The full project proposal is retained locally as `ilina_jrf_project.docx.pdf` and governs research scope, metrics, and timeline.
