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

## 4. Recommended Vast.ai workflow

Vast.ai is the primary Stage-1 execution environment. Use a single CUDA GPU with
at least 24 GiB VRAM, SSH plus `tmux`, and a persistent volume exposed through
`/data/safety_governor`. On templates that mount the volume at `/workspace`, use
a symlink from `/data/safety_governor` to `/workspace/safety_governor`.
BF16 is preferred when supported; otherwise select the FP16 profile explicitly.
The runner never silently changes precision.

Follow [`vast_ai_runbook.md`](vast_ai_runbook.md). In outline:

```bash
read -rsp 'Hugging Face token: ' HF_TOKEN
export HF_TOKEN
export VAST_IMAGE='<exact-image-name-or-digest-from-vast-template>'
./scripts/bootstrap_vast.sh <immutable-git-commit>

/data/safety_governor/venv/bin/python -m scripts.run_stage1 \
  configs/llama3_8b.yaml \
  --runtime-profile configs/runtime/vast_bf16.yaml \
  --layers 0 \
  --split train \
  --run-id llama3-stage1-layer0 \
  --resume
```

After layer 0 passes artifact review, run the primary sweep with
`--layers 0,4,8,12,16,20,24,28`. A single process loads Llama once and captures
only those residual hooks. Atomic paired shards allow the command to resume after
an interruption without accepting mismatched prior state.

## 5. Colab fallback

Open `docs/notebooks/stage1/llama_stage1_colab.ipynb` in Colab only when Vast is
unavailable. The notebook:

1. clones or pulls the GitHub repository;
2. installs the pinned dependencies;
3. reads `HF_TOKEN` from Colab secrets;
4. validates the frozen corpus and runtime;
5. invokes the same resumable multi-layer runner used on Vast;
6. captures requested layers and fits train-only vectors;
7. stores artifacts in Google Drive when configured.

Start with `LAYERS = "0"` and a stable `RUN_ID`. The Drive directory preserves
shards across runtime resets. Later use `LAYERS = "0,4,8,12,16,20,24,28"`.

The Colab path is operationally secondary; it does not define different capture
or fitting semantics.

This is a feasibility run, not a layer-selection conclusion. If a T4 runs out of
memory, move to a higher-memory runtime. Do not reduce scientific gates, merge
splits, or alter response-boundary semantics to force a run.

## 6. Legacy one-layer command

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

## 7. Capture semantics

The stored record has an explicit `instruction` and `completion`. Tokenization builds
the assistant-generation prefix separately from the completion so activation sites
are unambiguous:

- `response_mean` — primary site; mean residual activation over completion tokens;
- `final_response_token` — sensitivity site; actual final non-padding response token.

This is different from capturing the final token of a concatenated transcript.
Padding positions are excluded. The residual hook is `blocks.{layer}.hook_resid_pre`.

## 8. Artifact layout and interpretation

A resumable Stage-1 run produces:

```text
artifacts/<run-id>/
  run_spec.json
  status.json
  shards/batch_<index>.npz
  layers/layer_<index>/
    safe.npy
    safe.npy.json
    unsafe.npy
    unsafe.npy.json
    difference_in_means.npy
    difference_in_means.stability.npy
    paired_delta_pca.npy
    paired_delta_pca.stability.npy
    probe.npy
    probe.stability.npy
  manifest.json
```

The sidecars record layer, capture site, ordered pair IDs, split labels, source-group
IDs, and matrix shape. `manifest.json` records config, model revision, dataset hash,
Git state, Python/Torch/CUDA/package facts, device, and artifact paths.

Bootstrap cosine values measure directional stability under source-group-aware
resampling. A successful file write or high stability value is not by itself a
safety result. Stage-1 interpretation still requires held-out evaluation and control-
tax measurement.

Completed runs can be exported without duplicate batch shards:

```bash
python -m scripts.audit_stage1_run \
  /data/safety_governor/artifacts/llama3-stage1-layer0

python -m scripts.export_stage1_run \
  /data/safety_governor/artifacts/llama3-stage1-layer0 \
  /data/safety_governor/exports/llama3-stage1-layer0.tar.gz
```

The command writes a SHA-256 sidecar. Copy both files off the Vast host before
destroying the volume.

## 9. Split discipline

- **Train:** fit vector directions.
- **Validation:** select layer, extraction method, steering coefficient, and token
  intervention mode after the training analysis is defined.
- **Test:** final evaluation only, after the analysis is frozen.

Capture of the test split requires both `--allow-test-capture` and a verified
`--selection-lock`. This is deliberate friction. Do not create a lock until
held-out behavior review and capability-tax evaluation are complete.

## 10. Steering and evaluation

The train direction points from safe to unsafe behavior. Suppression is:

```text
A'_L = A_L - alpha v
```

Validation uses magnitudes `{1,2,5}`. `assistant_boundary` changes the final
prompt boundary; `generation_frontier` changes the active token at every decode
step. Control Tax combines targeted suppression with MMLU five-shot absolute
accuracy delta and chat-conditioned WikiText-103 continuation perplexity delta.
The provisional viability threshold is suppression above 70% with MMLU
degradation below three percentage points.

Run fixed-vector validation from the repository root:

```bash
python -m scripts.run_validation capture configs/llama3_8b.yaml \
  --validation-config configs/validation.yaml \
  --runtime-profile configs/runtime/vast_bf16.yaml \
  --train-run /data/safety_governor/artifacts/llama3-stage1-response-mean-hf-native \
  --run-id llama3-fixed-vector-validation --resume

python -m scripts.run_validation generate configs/llama3_8b.yaml \
  --validation-config configs/validation.yaml \
  --runtime-profile configs/runtime/vast_bf16.yaml \
  --run-id llama3-fixed-vector-validation --resume

python -m scripts.validation_review export \
  --run /data/safety_governor/artifacts/llama3-fixed-vector-validation \
  --output validation_review_decisions.jsonl
```

After completing the blinded decisions:

```bash
python -m scripts.validation_review summarize \
  --run /data/safety_governor/artifacts/llama3-fixed-vector-validation \
  --decisions validation_review_decisions.jsonl

python -m scripts.run_control_tax configs/llama3_8b.yaml \
  --validation-config configs/validation.yaml \
  --runtime-profile configs/runtime/vast_bf16.yaml \
  --run /data/safety_governor/artifacts/llama3-fixed-vector-validation

python -m scripts.validation_review lock \
  --run /data/safety_governor/artifacts/llama3-fixed-vector-validation \
  --control-tax /data/safety_governor/artifacts/llama3-fixed-vector-validation/control_tax.json \
  --output /data/safety_governor/artifacts/llama3-fixed-vector-validation/selection_lock.json
```

The reviewer never sees method, layer, magnitude, mode, or baseline identity.
If baseline outputs contain no unsafe behavior, selection stops as
non-diagnostic instead of manufacturing a suppression result.
For a checkpointed Colab UI, use
`docs/notebooks/stage1/validation_review_workbench.ipynb`; it stores every saved
decision in Google Drive and never loads `review_mapping.jsonl`.

## 11. Dataset curation and review

Do not edit `datasets/frozen/english_contrastive.jsonl` as a convenient input file.
New or revised data must pass through source registration, working candidates, human
review, field-whitelisted materialization, source-group splitting, validation, and a
new freeze. See:

- `docs/dataset_curation_workflow.md` for the full lifecycle;
- `docs/dataset_governance.md` for storage and publication rules;
- `docs/annotation_assistance.md` for the review workbench;
- `docs/experiment_protocol.md` for fit/evaluation discipline.

## 12. Common failures

### Hugging Face access error

Accept the model licence, request repository access, and provide a valid `HF_TOKEN`.
Do not replace the pinned model with an unrecorded alternative.

### Version mismatch

Use Python 3.11 and reinstall from `requirements.txt`. The experiment config pins
the runtime versions that affect TransformerLens model loading.

### CUDA out of memory

Use the batch-size-1 runtime profile, stop other GPU processes, or select a
higher-memory GPU. Do not enable quantization or CPU offload as an unrecorded
workaround. Activation shards preserve the original pair order.

### Existing shard is rejected

Do not edit or replace its metadata. Use the exact original run arguments, or
start a new run ID. A corrupt or incompatible final shard is not overwritten
because that could conceal mixed experiment conditions.

### Dataset validation failure

Read every reported problem. Common causes are missing polarity partners, unapproved
records, duplicate prompts, source-group mismatch, or split leakage. Do not suppress
the validator for an experiment run.

### Vector extraction rejects metadata

Confirm safe and unsafe sidecars have identical pair order, layer, capture site,
source groups, and train-only split labels. Re-capture rather than manually editing a
sidecar.

## 13. Reproducible reporting checklist

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
