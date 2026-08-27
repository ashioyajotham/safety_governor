# User guide

This guide is for researchers reproducing the current Stage-1 pipeline. It covers
installation, corpus verification, activation capture, vector fitting, artifacts,
and the boundary between runnable code and unfinished research stages.

## 1. Current runnable scope

The current research input is the frozen English deceptive-reasoning corpus:

```text
datasets/frozen/english_contrastive.jsonl
120 contrastive pairs / 240 records
behaviour: deceptive_reasoning
status: approved
```

The release contains 30 pairs for each of four archetypes: arithmetic reasoning
error, factual confabulation, false-premise agreement, and motivated reasoning.
The source-group-aware split contains 88 train, 16 validation, and 16 test pairs
across 83 source groups.

The GPT-2 configuration is an engineering smoke path only. The substantive Stage-1
target is the pinned Llama-3-8B-Instruct revision in `configs/llama3_8b.yaml`.
Harmful compliance is quarantined; Swahili transfer and conditional triggering are
later stages.

## 2. Requirements

- Python 3.11 is the supported baseline.
- Git.
- A Hugging Face account with access to the gated Llama repository.
- CUDA GPU for the Llama run. The Colab notebook defaults to batch size 1 for a
  constrained runtime; an A100 or comparable higher-memory GPU is preferable for
  layer sweeps.

Create an isolated environment:

```bash
git clone https://github.com/ashioyajotham/safety_governor.git
cd safety_governor
python3.11 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\Activate.ps1    # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The review notebook has additional UI dependencies:

```bash
python -m pip install -r requirements-review.txt
```

## 3. Verify a checkout

Run the deterministic checks before using GPU time:

```bash
python -m pytest -q
python -m scripts.validate_dataset datasets/frozen/english_contrastive.jsonl
python -m scripts.verify_environment configs/llama3_8b.yaml
```

`verify_environment` intentionally fails if the model revision is symbolic (for
example `main`) or declared package versions differ. Resolve the environment rather
than weakening the gate.

For a local state containing ignored reconstruction inputs, also run:

```bash
python -m scripts.verify_reconstruction_inputs
python -m scripts.verify_archive_index
python -m scripts.verify_working_state
```

These commands verify hashes; they do not upgrade a candidate's approval state.

## 4. Recommended Colab workflow

Open `docs/notebooks/stage1/llama_stage1_colab.ipynb` in Colab. The notebook:

1. clones or pulls the GitHub repository;
2. installs the pinned dependencies;
3. reads `HF_TOKEN` from Colab secrets;
4. validates the frozen corpus and runtime;
5. captures one requested layer and split;
6. fits a train-only vector;
7. stores artifacts in Google Drive when configured.

Start with:

```text
SPLIT = "train"
LAYER = 0
BATCH_SIZE = 1
DEVICE = "cuda"
VECTOR_METHOD = "difference_in_means"
RUN_LAYER_SWEEP = False
```

This is a feasibility run, not a layer-selection conclusion. If a T4 runs out of
memory, move to a higher-memory runtime. Do not reduce scientific gates, merge
splits, or alter response-boundary semantics to force a run.

## 5. Command-line Stage-1 workflow

Capture one layer from the train split:

```bash
python -m scripts.capture_activations configs/llama3_8b.yaml \
  --layer 0 \
  --split train \
  --site response_mean \
  --batch-size 1 \
  --device cuda \
  --artifacts artifacts
```

The command prints the generated capture run path. Fit a vector using the aligned
safe/unsafe arrays from that run:

```bash
python -m scripts.extract_vector \
  --safe artifacts/CAPTURE_RUN/safe.npy \
  --unsafe artifacts/CAPTURE_RUN/unsafe.npy \
  --method difference_in_means \
  --bootstrap-samples 100 \
  --output artifacts/CAPTURE_RUN/difference_in_means.npy
```

Available methods are:

- `difference_in_means` — primary transparent baseline;
- `paired_delta_pca` — PCA over aligned `unsafe - safe` pair deltas;
- `probe` — supervised ridge-probe direction.

All methods produce unit vectors. Vector extraction rejects activation metadata that
contains validation or test rows or misaligned safe/unsafe sample IDs.

## 6. Capture semantics

The stored record has an explicit `instruction` and `completion`. Tokenization builds
the assistant-generation prefix separately from the completion so activation sites
are unambiguous:

- `response_mean` — primary site; mean residual activation over completion tokens;
- `final_response_token` — sensitivity site; actual final non-padding response token.

This is different from capturing the final token of a concatenated transcript.
Padding positions are excluded. The residual hook is `blocks.{layer}.hook_resid_pre`.

## 7. Artifact layout and interpretation

A capture run produces:

```text
artifacts/capture-<UTC timestamp>/
  safe.npy
  safe.npy.json
  unsafe.npy
  unsafe.npy.json
  manifest.json
```

The sidecars record layer, capture site, ordered pair IDs, split labels, source-group
IDs, and matrix shape. `manifest.json` records config, model revision, dataset hash,
Git state, Python/Torch/CUDA/package facts, device, and artifact paths.

Vector fitting adds:

```text
difference_in_means.npy
difference_in_means.stability.npy
```

Bootstrap cosine values measure directional stability under source-group-aware
resampling. A successful file write or high stability value is not by itself a
safety result. Stage-1 interpretation still requires held-out evaluation and control-
tax measurement.

## 8. Split discipline

- **Train:** fit vector directions.
- **Validation:** select layer, extraction method, steering coefficient, and token
  intervention mode after the training analysis is defined.
- **Test:** final evaluation only, after the analysis is frozen.

Capture of the test split requires the explicit `--allow-test-capture` flag. This is
a deliberate friction point. Do not use it during exploratory work.

## 9. Steering and evaluation

The intervention is:

```text
A'_L = A_L + alpha v
```

Position-specific hooks require explicit non-padding positions or a response mask.
The planned coefficient sweep is `{1, 2, 5, 10, 20}` at every fourth layer. The
control-tax evaluation combines targeted suppression with MMLU five-shot delta and
WikiText-103 perplexity delta. The provisional viability threshold is suppression
above 70% with MMLU degradation below 3%.

The current Colab runner covers capture and vector fitting. Full generation-side
steering and benchmark evaluation should be added only after the first capture
artifacts are inspected and the validation plan is fixed.

## 10. Dataset curation and review

Do not edit `datasets/frozen/english_contrastive.jsonl` as a convenient input file.
New or revised data must pass through source registration, working candidates, human
review, field-whitelisted materialization, source-group splitting, validation, and a
new freeze. See:

- `docs/dataset_curation_workflow.md` for the full lifecycle;
- `docs/dataset_governance.md` for storage and publication rules;
- `docs/annotation_assistance.md` for the review workbench;
- `docs/experiment_protocol.md` for fit/evaluation discipline.

## 11. Common failures

### Hugging Face access error

Accept the model licence, request repository access, and provide a valid `HF_TOKEN`.
Do not replace the pinned model with an unrecorded alternative.

### Version mismatch

Use Python 3.11 and reinstall from `requirements.txt`. The experiment config pins
the runtime versions that affect TransformerLens model loading.

### CUDA out of memory

Use `--batch-size 1`, stop other GPU processes, or select a higher-memory runtime.
Activation capture batches are concatenated in original pair order.

### Dataset validation failure

Read every reported problem. Common causes are missing polarity partners, unapproved
records, duplicate prompts, source-group mismatch, or split leakage. Do not suppress
the validator for an experiment run.

### Vector extraction rejects metadata

Confirm safe and unsafe sidecars have identical pair order, layer, capture site,
source groups, and train-only split labels. Re-capture rather than manually editing a
sidecar.

## 12. Reproducible reporting checklist

For every reported run, retain:

- Git commit and dirty state;
- config file and immutable model revision;
- dataset path and SHA-256;
- split, layer, capture site, and batch size;
- environment/package/CUDA facts;
- ordered sample and source-group IDs;
- vector method and bootstrap seed/sample count;
- all generated manifests and numerical artifacts;
- clear designation as smoke, exploratory, validation, or final test evidence.
