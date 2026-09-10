# Experiment protocol

## Dataset gates

1. Resolve every candidate to a pinned upstream source and preserve immutable hashes.
2. Keep model-assisted drafting provenance in the annotation layer only.
3. Run the vendored official IFEval checker on every instruction pair. Safe completions always pass. Constraint omission and false completion must fail a declared mechanical constraint; topic shift and excessive caveating must preserve every mechanical constraint and pass a human semantic rubric.
4. Run lexical/template, duplicate, encoding, and source-group audits.
5. Freeze only approved records. Materialize experiment JSONL through the field-whitelisting script.
6. Assign train/validation/test at source-group level. No group may cross a split.
7. Keep harmful compliance quarantined until it contains diverse full safe and unsafe completions; JailbreakBench target prefixes are not completions.

False completion requires an explicit or label-based compliance claim; a bare factual or arithmetic error is excluded. Topic shift is rejected when the response substantially fulfills the requested task before shifting, but no fixed pivot position is imposed. Hedging is rejected as a negative example when it is merely a reasonable safety or uncertainty qualification.

The semantic audit is diagnostic only. Blinded A/B tasks score relevance and completeness for topic shift, and directness, completeness, and caveat dominance for hedging. Provider identity and scores remain in separate audit artifacts; flagged disagreements require human acknowledgement but never change approval automatically.

## Representation sites

The primary representation is the mean pre-residual activation over assistant response tokens. The final assistant response token is a sensitivity analysis. Tokenization constructs the instruction prefix and response separately and stores an explicit response mask.

## Fit and selection discipline

Vector fitting is train-only by construction. Validation consumes fixed train
directions and must not call a vector extractor. It selects method, layer,
coefficient, and intervention position. Test remains untouched until the
analysis is frozen in a verified selection lock.

Capture accepts an explicit `train`, `validation`, or `test` split. Test capture
fails closed unless both `--allow-test-capture` and a verified
`--selection-lock` are supplied. Vector extraction rejects any activation
metadata containing validation or test rows.

Methods are difference in means, paired-delta PCA on `unsafe - safe`, and a predeclared ridge probe. Bootstrap stability resamples source groups while preserving contrastive pairs.

## Held-out validation

The fixed candidates are difference in means at layers 12, 16, and 24, and the
ridge probe at layers 24 and 28. Paired-delta PCA remains a reported train
baseline but is not eligible for intervention because its bootstrap orientation
was unstable. Validation reports AUC, paired ordering, projection margins,
per-archetype results, and 2,000-sample source-group bootstrap intervals. The
top two eligible directions advance to generation review.

Generated responses are judged through blinded human review; vector projection
is diagnostic rather than an approval signal. Relative suppression is reported
only when the unsteered baseline has nonzero unsafe behavior. A zero-headroom
validation set is non-diagnostic.

## Intervention

Directions point from safe to unsafe, so suppression uses
`A'_L = A_L - alpha v` for positive magnitude `alpha`. `assistant_boundary`
intervenes at the final prompt boundary. `generation_frontier` intervenes at
the active final token on every autoregressive step. Completed-response labels
are capture sites, not causal generation policies. Position-sensitive hooks
must not use a padded final column as a real token.

Validation generation uses magnitudes `{1,2,5}` for the two shortlisted
directions and both causal policies. Record human-rated target suppression,
MMLU five-shot absolute accuracy change, and WikiText-103 relative perplexity
change. Capability benchmarks run only after behavioral selection. Provisional
viability requires suppression above 70% and MMLU degradation below three
percentage points.

## Reproducibility

Models use immutable Hugging Face revisions through TransformerLens Bridge v3.
Stage 1 preserves the checkpoint's Hugging Face-native weights and numerics;
`enable_compatibility_mode(no_processing=True)` registers residual-hook aliases
without folding LayerNorms or centering weights. Every run records this bridge
weight mode alongside config, seed, model revision, dataset SHA-256, Git SHA,
dirty flag and diff hash, Python and package versions, Torch/CUDA facts, device,
split, layer, capture site, metrics, and artifact paths.

The primary Stage-1 runner uses one explicitly selected single-GPU precision
profile. BF16 and FP16 are separate profiles; there is no silent fallback.
Quantization, CPU offload, automatic device placement, and multi-GPU capture are
outside the initial experiment contract. Multi-layer capture loads the model
once and restricts the activation cache to the declared residual hooks.
Token IDs and attention masks are moved explicitly to the model's parameter
device before each forward pass; response masks and token indices are moved to
the captured activation device before reduction.

Interrupted runs may resume only from immutable batch shards whose run-spec hash,
pair order, source-group order, layers, capture site, dataset hash, model revision,
code revision, and dtype all match. Corrupt or incompatible shards fail closed.

`python -m scripts.verify_environment <config>` enforces immutable model revisions and exact declared runtime versions before capture. Public inputs are commit- and hash-pinned in `datasets/manifests/source_corpora.json`; `datasets/manifests/reconstruction.json` lists the ignored restricted bundle and the commands that verify it.

The annotation-artifact gate measures exact duplicates, concentrated prefixes/suffixes and five-grams, encoding damage, lexical cues, and near duplicates by archetype and polarity. The strict motivated-reasoning gate is:

```powershell
python -m scripts.audit_annotation_artifacts data/working/deceptive_reasoning/candidates.jsonl --strict-archetype motivated_reasoning
```

Swahili transfer starts only after the English corpus and RQ1 protocol are frozen. Conditional triggering remains a stretch stage.
