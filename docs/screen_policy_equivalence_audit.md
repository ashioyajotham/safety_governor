# Screening versus generation: initial equivalence audit

## Code inspection

`response_negative_log_likelihood_governed` steers positions from the last prompt
token through the penultimate completion token in a single teacher-forced forward
pass. This indexing correctly selects the positions predicting completion tokens.
For a completion of length R and prefix length P, predictor i is P - 1 + i.

`generate_with_governor` under `generation_frontier` recomputes the prefix and
already-generated tokens each step, steering only the last position. Earlier
states are not retained with their prior interventions. Both paths share prompt
serialization, residual-pre hook names, subtraction sign and local budget formula.

These shared properties do not establish policy equivalence. Causal attention
can propagate modified earlier states to later predictors in the parallel screen.
The screen also keeps the prompt-boundary intervention active for later
predictors, unlike frontier generation. Generated-span generation likewise does
not keep steering the prompt boundary after its first step, so the historical
screen is not automatically an exact generated-span scorer either.

## Executed local controls

`tests/test_screen_policy_equivalence.py` uses a deterministic two-block causal
prefix-mixing fixture, a two-token prompt, three supplied completion tokens and
two unit-direction sites with equal weights and budget 0.20.

- Zero-budget parallel and incremental logits agree.
- The first steered predictor agrees, but later predictors differ.
- Changing a future token does not change earlier predictor logits.

All three NumPy fixture tests passed. The fourth test calls the actual repository
teacher-forced scorer through a Torch adapter and compares its NLL with the
parallel and incremental references. It initially skipped in the project Python
3.14 environment without PyTorch, but subsequently passed using cached PyTorch
2.8.0 and Python 3.12 in an isolated offline environment. All four audit tests
passed. With cached IFEval dependencies included, the full suite passed:
129 passed, 2 skipped. No project dependency files or model weights were changed.

This is a controlled counterexample to universal policy equivalence, not a
measurement of Llama logits or proof of the cause of weak behavioral suppression.
No model download, inference rental, historical artifact mutation, or holdout
access was performed.

## Next verification, before changing the screen

The scorer integration test is complete on the causal fixture. Next,
under separately approved inference scope, compare Llama per-token logits and
NLL on identical supplied histories: unsteered parallel versus incremental,
current steered parallel versus frontier-only incremental, and one-token controls.
Hold tokenizer, weights, dtype, attention implementation and budgets fixed;
record tolerances and numerical baseline discrepancies. Report per-token effects
rather than only aggregate NLL. Do not regenerate or replace the historical screen.

Only after quantifying this gap should a policy-exact incremental screening path
be proposed. Capability-versus-preference comparisons remain the second avenue;
activation-source/intervention redesign remains third. The behavioral gate and
sealed confirmation are unchanged.

## Llama diagnostic executed: 2026-10-08

`scripts/audit_screen_logits.py` ran on the original newer A100 host using cached
checkpoint revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, BF16, eager
attention, and the existing balanced Ridge directions at layers 12/16/20/24.
The relative-L2 budget was 0.20 with equal site weights. Four deterministic
train-unsafe supplied histories were used, one per archetype, with at most 16
predictor positions each. The actual production parallel NLL scorer was wrapped
only to record its logits. There was no free-running generation or refitting.

| Archetype | Positions | Maximum unsteered gap | First steered gap | Maximum later steered gap | Changed argmaxes |
| --- | ---: | ---: | ---: | ---: | ---: |
| arithmetic_reasoning_error | 16 | 0.328125 | 0.25 | 1.28125 | 0 |
| factual_confabulation | 8 | 0 | 0 | 1.875 | 0 |
| false_premise_agreement | 3 | 0 | 0 | 0.40625 | 0 |
| motivated_reasoning | 16 | 0.3125 | 0.1640625 | 1.09375 | 1 |

Gaps are maximum absolute vocabulary-logit differences, not safety scores.
The shorter completions were used in full. Numerical baseline discrepancies
prevent treating every nonzero difference as a steering-policy effect; notably
first-position differences did not exceed each history's baseline maximum.
Later steered gaps exceeded baseline maxima in all four histories. Only one of
43 argmaxes differed. This narrow diagnostic measures non-equivalence, not its
contribution to behavioral failure, statistical generalization, or causality.

The audit recorded 23.01 seconds from harness startup through scoring (not the
full billed session). All ten original DIM/Ridge vector files were downloaded
and checked against their metadata hashes. The audit artifact is retained at
`~/Downloads/llama3-screen-logit-audit-20261008/audit.json`; the direction backup
is `~/Downloads/llama3-validation-v3-directions-recovered-20261008/`.
The instance was confirmed `actual_status=exited`, `cur_state=stopped` after
retrieval. Historical screening artifacts and the sealed holdout were unchanged.

## Opt-in frontier-exact screening

`response_negative_log_likelihood_frontier` now scores each supplied target
using a fresh prompt-plus-preceding-targets forward pass. It shares generation's
single-position intervention hook, sign, layer ordering, budget and gate. It
does not retain intervened historical states or reuse a KV cache. The unsteered
baseline uses the same incremental policy, and loss is accumulated from float32
logits. Supplied EOS tokens, if present in the completion tokenization, are
scored as targets; no EOS is appended and no free-running stopping decision is
made. These are likelihood comparisons, not behavioral suppression estimates.

The screen runner keeps `parallel_proxy` as its default for historical
compatibility. To opt in, use a **new run ID**, retaining the original directions,
dataset, candidates and budgets:

```bash
python -m scripts.screen_validation_v3 configs/llama3_8b.yaml \
  --runtime-profile configs/runtime/vast_bf16.yaml \
  --train-run /data/safety_governor/artifacts/llama3-stage1-response-mean-hf-native \
  --direction-run /data/safety_governor/artifacts/llama3-validation-v3-directions \
  --scoring-policy frontier_exact \
  --run-id llama3-v3-frontier-exact-screen
```

This command is documentation, **not authorization to execute the full GPU
screen**. Its cost scales with supplied response length times pairs times
conditions, so scope and budget must be approved separately. The current
implementation has local fixture/integration coverage, not a new full Llama
screen result. Fresh schema-v2 specifications record the policy and matching
baseline; resuming a parallel run as exact, or exact as parallel, fails before
model loading. Historical schema-v1 specifications remain unchanged. New
metrics/shards are separate, and no old candidate lock is regenerated.

Next verification is a small pinned-model comparison of this new scorer with
the retained audit reference. Only then consider a scoped train-only screen.
The >70% behavioral suppression gate and sealed confirmation remain unchanged.

## Frontier scorer verification: 2026-10-08

The opt-in scorer was compared with the step-by-step audit reference on four
frozen train-unsafe completions (29, 8, 3 and 30 tokens; 70 total), using the
same pinned Llama checkpoint, BF16, equal-weight balanced Ridge sites and 0.20
relative-L2 budget. Both baseline and steered summed NLL agreed exactly for
every example: absolute errors were 0.0, below the unchanged 1e-4 tolerance.
The corrected audit took 37.02 seconds internally, excluding startup/billed time.

The first attempt computed reference cross-entropy on CPU and scorer loss on
CUDA. It missed tolerance on three examples, with errors up to 0.000465.
That evidence is preserved; the corrected audit uses float32 CUDA loss in both
paths, without changing the scorer or relaxing tolerance.

Evidence: `~/Downloads/llama3-frontier-scorer-verification-20261008.json`
and `~/Downloads/llama3-frontier-scorer-verification-20261008-cuda-loss.json`.
This verifies the implementation against the incremental reference for these
histories, not behavioral efficacy or all possible configurations. No full
screen, candidate selection, refitting or sealed confirmation was performed.

## Proposed train-only screen scope (not launched)

- Preserve the existing 88 train pairs, original frozen directions and model
  revision. Use `frontier_exact` and a new run ID; retain historical results.
- Keep all five static profiles and three budgets (0.05/0.10/0.20): 15
  candidates plus a matching baseline, or 1,408 pair-condition shards and
  2,816 supplied-completion likelihood evaluations for the complete matrix.
- Start with a bounded 60-minute session, including setup, then stop and retrieve
  completed shards. At the last observed roughly $0.59/hour compute rate this is
  roughly $0.59 compute, not a guaranteed total invoice; storage and bandwidth
  are separate. Recheck price before launch and use a $1 session allowance.
  Arm shutdown before inference and stop early on errors or unexpected usage.
- Do not promise completion within that session: exact scoring performs one
  fresh forward pass per supplied target. Measure throughput and token counts
  from completed shards before requesting any extension. Resume the same locked
  specification rather than changing the grid to finish cheaply.
- Require a clean committed checkout and record its revision and file hashes.
  The screen's existing contract validator checks the sealed file's checksum;
  no confirmatory examples are scored or used for selection.
- Report safe-preference deltas overall and per archetype. These are in-sample
  training diagnostics, not behavioral validation. Retain the existing ranking
  and eligibility rule; do not lower the >70% downstream behavioral gate.
- Keep GPU inference authorization separate from this commit. No new screen,
  behavioral generation, control-tax run or confirmatory evaluation is included
  in the current implementation/verification work.
