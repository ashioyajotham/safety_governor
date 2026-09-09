# Vast.ai Stage-1 runbook

This runbook is the primary operational procedure for the first Llama-3-8B
activation-extraction experiment. It does not expand the scientific scope:
`datasets/frozen/english_contrastive.jsonl` is the only Stage-1 corpus, vector
fitting uses train records only, and harmful compliance remains quarantined.

## 1. Provisioning contract

Rent one CUDA GPU with at least 24 GiB VRAM and attach a persistent volume of at
least 100 GB at `/data`. Connect over SSH and run long commands inside `tmux`.
Use a Python 3.11 CUDA/PyTorch image. Record the exact image identifier shown by
Vast; the qualification manifest and `pip freeze` capture the remaining runtime.
The bootstrap installs the repository-pinned PyTorch release from the official
CUDA 12.8 wheel index; it does not inherit Torch from the base image.

If the image provides Python 3.12 but no `python3.11` executable, create the
bootstrap interpreter on the persistent volume before continuing:

```bash
conda create --prefix /data/safety_governor/python311 python=3.11 --yes
export PYTHON311_BIN=/data/safety_governor/python311/bin/python3.11
```

The initial experiment does not support quantization, CPU offload, automatic
device maps, or multiple visible GPUs. These paths require separate numerical
qualification before they can become research configurations.

## 2. Authenticate without persisting credentials

Accept the pinned Llama repository licence in Hugging Face, then export the token
only in the SSH session:

```bash
read -rsp 'Hugging Face token: ' HF_TOKEN
export HF_TOKEN
export VAST_IMAGE='<exact-image-name-or-digest-from-vast-template>'
```

Do not place the token in the repository, a YAML file, shell history, an artifact
manifest, or an exported run bundle. `VAST_IMAGE` is non-secret provenance and is
recorded in the run manifest.

## 3. Bootstrap an immutable checkout

```bash
git clone https://github.com/ashioyajotham/safety_governor.git
cd safety_governor
./scripts/bootstrap_vast.sh <immutable-git-commit> vast_bf16
```

Bootstrap checks out the commit in detached mode, recreates a clean environment
and caches under `/data/safety_governor`, freezes the resolved environment in
`qualified-requirements.txt`, validates the frozen corpus, checks CUDA/VRAM/BF16
support, and verifies access to the pinned model revision. Every run refuses an
environment that differs from this qualification lock and copies the lock into
its exportable run directory.

The clean rebuild is intentional: it prevents packages from a failed or older
qualification attempt from contaminating the environment. The Stage-1 Torch
version is pinned in `requirements.txt`, and the Vast bootstrap resolves it from
`https://download.pytorch.org/whl/cu128` so a future CUDA-major release cannot be
selected implicitly.

If BF16 qualification fails specifically because the GPU lacks BF16 support,
rerun bootstrap with the `vast_fp16` argument. Never change profile dtype inside
an existing run.

## 4. Start and observe the first run

```bash
tmux new -s safety-governor
cd /data/safety_governor/repository
read -rsp 'Hugging Face token: ' HF_TOKEN
export HF_TOKEN
export HF_HOME=/data/safety_governor/huggingface
export VAST_IMAGE='<same-exact-image-name-or-digest-used-during-bootstrap>'

/data/safety_governor/venv/bin/python -m scripts.run_stage1 \
  configs/llama3_8b.yaml \
  --runtime-profile configs/runtime/vast_bf16.yaml \
  --layers 0 \
  --split train \
  --run-id llama3-stage1-layer0 \
  --resume 2>&1 | tee /data/safety_governor/artifacts/llama3-stage1-layer0.log
```

Detach with `Ctrl-b d`; reconnect with `tmux attach -t safety-governor`. The
stable run ID and `--resume` are intentional. Completed paired batches are reused
only after their immutable specification and contents validate.

## 5. Layer-0 acceptance gate

Before the sweep, require:

- `status.json` reports `complete` and `manifest.json` exists;
- 88 aligned train pairs and 61 train source groups are recorded;
- safe and unsafe matrices are finite and have identical shapes;
- all three vectors and bootstrap arrays exist;
- model, dataset, code, dtype, GPU, CUDA, package, layer, and capture-site facts
  are present;
- a checksummed export can be opened and its checksum independently recomputed.

If these checks pass, layer 0 is the first substantive Stage-1 artifact rather
than a disposable smoke run.

Run the machine gate before inspecting or exporting results:

```bash
/data/safety_governor/venv/bin/python -m scripts.audit_stage1_run \
  /data/safety_governor/artifacts/llama3-stage1-layer0
```

## 6. Primary and sensitivity sweeps

Primary response-mean sweep:

```bash
/data/safety_governor/venv/bin/python -m scripts.run_stage1 \
  configs/llama3_8b.yaml \
  --runtime-profile configs/runtime/vast_bf16.yaml \
  --layers 0,4,8,12,16,20,24,28 \
  --split train \
  --run-id llama3-stage1-response-mean \
  --resume
```

After reviewing that run, repeat with a new run ID and
`--site final_response_token` for the declared sensitivity analysis. Do not
capture validation until the training analysis is defined. Do not capture test
until method, layer, intervention, and reporting decisions are frozen.

## 7. Export and shutdown

```bash
/data/safety_governor/venv/bin/python -m scripts.export_stage1_run \
  /data/safety_governor/artifacts/llama3-stage1-response-mean \
  /data/safety_governor/exports/llama3-stage1-response-mean.tar.gz

cd /data/safety_governor/exports
sha256sum -c llama3-stage1-response-mean.tar.gz.sha256

/data/safety_governor/venv/bin/python -m scripts.export_stage1_run \
  --verify llama3-stage1-response-mean.tar.gz
```

Copy the archive and checksum to an independent machine or approved storage.
The export excludes resumable batch shards by default because consolidated
activations already preserve the research result; use `--include-shards` only
for forensic archiving. Verify the off-host checksum before destroying the
instance or volume.
