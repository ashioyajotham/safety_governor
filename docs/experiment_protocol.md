# Experiment protocol

## Stage gates

1. Validate provenance-complete English contrastive data and reproduce the activation capture/vector pipeline on GPT-2 Small.
2. Compare difference-in-means, PCA, and ridge-probe vectors with bootstrap directional stability. Select candidates using validation data only.
3. Sweep every fourth layer, alpha `{1,2,5,10,20}`, and last-token/all-token steering. Record ASR/target suppression, MMLU 5-shot delta, and WikiText-103 perplexity delta in every manifest.
4. Scale only replicated configurations to Llama-3-8B.
5. Map EN/SW residual similarity on held-out translation pairs; use the peak-similarity layer as the conceptual hub and compare it with early/late controls.
6. Treat conditional residual-stream triggering as a stretch stage after the static results reproduce.

## Decision rule

A configuration is provisionally viable when targeted suppression is greater than 70% and MMLU degradation is less than 3%. Report null results; do not tune on the held-out test split.

## Reproducibility

Every result must have a config, seed, model revision, dataset hash, vector metadata, environment facts, metric outputs, and artifact locations. Keep generated artifacts under `artifacts/`, outside version control. W&B sync is optional and never replaces local artifacts.
