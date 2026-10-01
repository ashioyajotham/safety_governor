# Validation v3: post-review 384-token length diagnostic

The frozen 128-token Validation-v3 development result remains a negative result
for **target-safe completion within 128 new tokens**. Its blinded human review
marked 153/192 responses not target-safe with a rationale citing a cutoff. A
read-only inspection of existing intervention traces found that each steered
condition ran all 128 decode steps on 40/48 prompts. This supports a
length-censoring hypothesis; it does not prove that a 384-token answer will be
target-safe or reach a stop token.

This is a **post-review sensitivity analysis**, not a replacement for the
original protocol or an additional attempt at the sealed confirmatory set.
Keep the exact same 48 spent pairs, frozen model revision, train-fitted
directions, baseline, and three candidate interventions. Change only the
maximum new-token count from 128 to 384. Do not retune candidate sites,
budgets, gates, prompts, or the human labels from the earlier run.

The diagnostic writes per-response `generation_metadata` with the generated
token count and either `stop_token` or `token_limit` as the stop reason. The
run specification marks `diagnostic_only: true`; the promoting
`summarize-development` command rejects it. A new candidate lock is necessary
because the diagnostic contract has a different content hash. The diagnostic
must use a distinct run ID and must not resume the 128-token run.

After a clean checkout and an explicit paid-GPU approval, on the qualified Vast
instance:

```bash
cd /data/safety_governor/repository
export HF_HOME=/data/safety_governor/huggingface
export HF_HUB_OFFLINE=1
export VAST_IMAGE=vastai/pytorch_cuda-12.8.1-auto/jupyter

/data/safety_governor/venv/bin/python -m scripts.freeze_validation_v3_candidates \
  --development-config configs/validation_v3_length384_diagnostic.yaml \
  --screen-run /data/safety_governor/artifacts/llama3-validation-v3-screen-inference \
  --candidate distributed_ridge_uniform__b0.2 \
  --candidate distributed_ridge_late_dominant__b0.1 \
  --adaptive-candidate distributed_ridge_uniform__b0.2 \
  --gate-run /data/safety_governor/artifacts/llama3-validation-v3-gates \
  --output /data/safety_governor/artifacts/validation-v3-length384-candidates.json

/data/safety_governor/venv/bin/python -m scripts.run_validation_v3 development \
  configs/llama3_8b.yaml \
  --development-config configs/validation_v3_length384_diagnostic.yaml \
  --runtime-profile configs/runtime/vast_bf16.yaml \
  --train-run /data/safety_governor/artifacts/llama3-stage1-response-mean-hf-native \
  --direction-run /data/safety_governor/artifacts/llama3-validation-v3-directions \
  --candidate-lock /data/safety_governor/artifacts/validation-v3-length384-candidates.json \
  --run-id llama3-validation-v3-length384-diagnostic --resume
```

The runner recomputes the full growing prefix on each decode step, so extending
the cap can cost substantially more than three times the 128-token run. Check
GPU-hour and spending limits before starting. The exact stop-reason breakdown
can be inspected without human labels or promotion:

```bash
/data/safety_governor/venv/bin/python -m scripts.diagnose_validation_v3_length \
  --run /data/safety_governor/artifacts/llama3-validation-v3-length384-diagnostic
```

If human review is needed, export a **new** blinded task packet and checkpoint
new decisions. Then pass `--decisions PATH` to the diagnostic command for a
read-only condition and paired-transition report. Do not call
`scripts.validation_v3_review summarize-development` on this diagnostic run.
Even a favorable 384-token result on the already-spent pairs cannot authorize
confirmation. A future efficacy claim needs a separately frozen design and
fresh source-isolated evaluation data.
