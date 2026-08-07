# Experiment protocol

## Dataset gates

1. Resolve every candidate to a pinned upstream source and preserve immutable hashes.
2. Keep model-assisted drafting provenance in the annotation layer only.
3. Run the vendored official IFEval checker on every instruction pair. Safe completions must pass all declared constraints. A human must confirm the expected unsafe failure list and semantic contrast.
4. Run lexical/template, duplicate, encoding, and source-group audits.
5. Freeze only approved records. Materialize experiment JSONL through the field-whitelisting script.
6. Assign train/validation/test at source-group level. No group may cross a split.
7. Keep harmful compliance quarantined until it contains diverse full safe and unsafe completions; JailbreakBench target prefixes are not completions.

## Representation sites

The primary representation is the mean pre-residual activation over assistant response tokens. The final assistant response token is a sensitivity analysis. Tokenization constructs the instruction prefix and response separately and stores an explicit response mask.

## Fit and selection discipline

Vector fitting is train-only by construction. Validation selects method, layer, coefficient, and intervention position. Test remains untouched until the analysis is frozen.

Methods are difference in means, paired-delta PCA on `unsafe - safe`, and a predeclared ridge probe. Bootstrap stability resamples source groups while preserving contrastive pairs.

## Intervention

Position-specific hooks require explicit non-padding positions or a response-token mask. Batched intervention must not use `[:, -1]` as a proxy for a real token.

Sweep every fourth layer and coefficients `{1,2,5,10,20}`. Record target suppression or ASR, MMLU five-shot delta, and WikiText-103 perplexity delta. Provisional viability requires suppression above 70% and MMLU degradation below 3%.

## Reproducibility

Models use immutable Hugging Face revisions through TransformerLens Bridge v3 compatibility mode. Every run records config, seed, model revision, dataset SHA-256, Git SHA, dirty flag and diff hash, Python and package versions, Torch/CUDA facts, device, split, layer, capture site, metrics, and artifact paths.

Swahili transfer starts only after the English corpus and RQ1 protocol are frozen. Conditional triggering remains a stretch stage.