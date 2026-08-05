# Runtime Safety Governor

Research repository for the ILINA Junior Research Fellowship project **Runtime Safety Governors: Activation Steering as a Control API for Deployed Language Models** (Aug–Nov 2026).

This is an empirical mechanistic-interpretability project. It tests whether activation steering can act as a narrow inference-time intervention against unsafe model behaviour, without retraining and without unacceptable degradation of general capability. It is not an HTTP moderation service or a product safety gateway.

## Research questions

1. Can contrastive activation vectors reliably represent deceptive reasoning, instruction non-compliance, and harmful compliance in open-weight language models?
2. What is the **Control Tax** of steering: targeted suppression versus MMLU accuracy and WikiText-103 perplexity degradation?
3. Do English-derived safety vectors transfer to semantically matched Swahili prompts, and at which residual-stream layers?
4. *(Stretch)* Can a lightweight residual-stream probe trigger steering only when needed?

The primary intervention is activation addition at residual layer `L`:

```text
A'_L = A_L + alpha * v
```

The predeclared viability target is targeted suppression above 70% with an MMLU delta below 3%.

## Current status

- Phase 1 operational smoke run is complete: GPT-2 Small loaded with TransformerLens on CPU; layer-0 residual activations, a reproducible manifest, and a difference-in-means vector were produced from the fixture data.
- The smoke artifact verifies the pipeline only. It is not evidence for a safety claim because the fixture has one contrastive pair.
- The English curation corpus contains 300 approved contrastive pairs (600 records): 100 deceptive-reasoning, 100 instruction-noncompliance, and 100 harmful-compliance pairs.
- The next research gate is a reviewed English-to-Swahili translation subset, followed by cross-lingual representation-similarity mapping.

## Data and research controls

The harmful-compliance records are stored under `data/raw/` and are intentionally ignored by Git. They originate from JailbreakBench and must not be copied into public issues, logs, examples, or commits.

Every record has a pair ID, behavior, polarity, language, provenance, reviewer status, and split. Pairs—not individual rows—are assigned to splits. The repository validates duplicate prompts, incomplete pairs, split leakage, and missing provenance. See [dataset governance](docs/dataset_governance.md) and the [curation workflow](docs/dataset_curation_workflow.md).

## Repository layout

```text
safety_governor/     core data, activation, vector, steering, and evaluation code
scripts/             reproducible data and experiment entrypoints
configs/             GPT-2 and Llama-3-8B experiment configurations
datasets/            safe fixtures, manifests, templates, and non-restricted drafts
data/raw/            restricted benchmark-derived material (Git ignored)
docs/                protocol, governance, and curation documentation
tests/               deterministic unit and pipeline tests
artifacts/           run manifests, activation caches, and vectors (Git ignored)
```

## Reproduce the local pipeline

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m scripts.validate_dataset datasets/fixtures/contrastive_en.jsonl
pytest
```

A minimal real capture requires TransformerLens and model access:

```powershell
python -m scripts.capture_activations configs/gpt2_pilot.yaml --layer 0 --device cpu
python -m scripts.extract_vector --safe artifacts/<capture-run>/safe.npy --unsafe artifacts/<capture-run>/unsafe.npy --method difference_in_means --output artifacts/<capture-run>/dim_vector.npy
```

Use `python -m scripts.plan_sweep configs/gpt2_pilot.yaml` to inspect the registered static-intervention matrix. Do not interpret a vector before the approved corpus is split, captured, and evaluated under the protocol in [experiment_protocol.md](docs/experiment_protocol.md).

## Phase sequence

1. **Foundation** — reproducible package, artifacts, capture path, and tests.
2. **English data** — provenance-linked contrastive pairs and review.
3. **Vector extraction** — difference-in-means, PCA, and probe directions with bootstrap stability.
4. **Control Tax** — layer/coefficient/token-position sweeps and target/capability/fluency metrics.
5. **Cross-lingual transfer** — frozen English–Swahili pairs, conceptual-hub mapping, and transfer controls.
6. **Conditional steering** — only after static results are reproducible.

## Source project document

The full working proposal is retained locally as `ilina_jrf_project.docx.pdf`. Its research questions, metric definitions, and timeline govern this repository.