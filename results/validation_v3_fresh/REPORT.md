# Validation v3 fresh behavioural development: gate failed

The prospective 48-prompt, three-condition comparison completed with 144/144
responses ending at a stop token under the frozen 768-token cap. Victor's 144
blinded target-safe, relevance and coherence decisions were checkpointed before
unblinding. Both prespecified candidates failed the development gate. No final
steering configuration is locked and the 64-pair confirmatory holdout remains
sealed. This is a negative **development** result, not a confirmatory efficacy
estimate or proof of a particular failure mechanism.

## Provenance

- Frozen contract: `configs/validation_v3_fresh_development.yaml`; corpus SHA-256
  `ee3a68d473408e4e5947d54af5cdf901dabd049211c9e3968bb1188a143fcd07`.
- Private run: `data/working/validation_v3_fresh/llama3-v3-fresh-development-768`.
  `run_spec.json` SHA-256 `48ac1ad30c743164c2880e85cefd21b9880789e06dad1e94800cb3ec7f1ad39d`;
  `generations.jsonl` SHA-256 `2adf614658f1ca48c59b5438fd9c92d7bbcfdcbedc4c1f01d1927e46fb7e59df`;
  `behavior_metrics.json` SHA-256 `39f9587429c3ef0bc8126c6d2d7bca8912f62d3e5f0d26728613cba7a4ca0d`.
- Review decisions SHA-256 `a11974f57efbb285e0a9649c26f24084a4b7276b1e3c30a6cc152b416449cc35`;
  private review mapping SHA-256 `fef217e2230a27a0adb8249d02a608afb66eefaba1e19977b5fb683558ff424d`.
  The mapping must not be given to a future blind reviewer.

| Condition | Unsafe / 48 | Relative suppression | Improved / unchanged / worsened vs paired baseline |
| --- | ---: | ---: | ---: |
| Baseline | 18 | — | — |
| Uniform distributed Ridge, budget 0.20 | 16 | 11.1% | 2 / 46 / 0 |
| Late-dominant distributed Ridge, budget 0.10 | 17 | 5.6% | 1 / 47 / 0 |

Baseline unsafe counts by archetype, in the order arithmetic / factual
confabulation / false premise / motivated reasoning, were `1 / 5 / 2 / 10`.
Uniform Ridge yielded `0 / 5 / 2 / 9`; late-dominant Ridge yielded
`1 / 5 / 2 / 9`. Neither candidate strictly improved all four archetypes nor
exceeded the locked **>70%** relative suppression threshold. Relevance was
47/48 in every condition. Coherence was 46/48 baseline and 47/48 for each
candidate. These quality rates do not rescue the suppression failure.

The source-group bootstrap upper 95% bound on absolute unsafe-rate change
was 0.0 for each candidate; the interval therefore includes no improvement.
No more tuning on these 48 prompts can be relabelled as a fresh gate.

## Read-only intervention audit

`python -m scripts.audit_validation_v3_traces <private-run>` checked all 144
saved generation shards. Baseline had zero hook traces. Each of 96 steered
shards had four traces per decode step, including the step that emitted the
stop token, with increasing generation-frontier position, layers 12/16/20/24,
finite values, gate 1 and the configured per-site relative-L2 shares. It found
23,940 trace entries for uniform and 24,212 for late-dominant; zero audit
errors. All 48 shards in each condition reported `stop_token`.

This verifies that hook calls and recorded coefficients matched the code path.
It does **not** show that the final residual stream retained the perturbation,
that later layers compensated, or that the behavioral errors were caused by
sycophancy rather than logic difficulty. The intervention budget is per-hook,
not a bound on cumulative downstream state difference.

## Next decision

Keep the confirmatory holdout sealed. A separate, explicitly exploratory
[preference-control diagnostic](../../docs/validation_v3_preference_control_diagnostic.md)
can test whether motivated-reasoning errors depend on the user's stated
preference. It must use spent development items, not a revised version of this
gate. Any future efficacy gate needs a newly frozen, source-isolated corpus and
prospective candidate selection.
