# Fresh v3 development: source-backed candidate pool

Status (14 October 2026): **128 source-backed drafts and a materialized 48-pair
development corpus; not frozen or authorized for GPU generation**.
The reproducible local pool is at
`data/working/validation_v3_fresh/review_candidates.jsonl` (ignored by Git),
SHA-256 `58ec8de5e92387e4e11af8848163aeeb98a3ea0005b547a1f4d6d88a2af346eb`.
It contains 32 candidates per archetype, one per source group. The current
48-pair corpus is at `datasets/validation_v3/fresh_development.jsonl`, SHA-256
`ee3a68d473408e4e5947d54af5cdf901dabd049211c9e3968bb1188a143fcd07`.
It has 12 pairs per archetype and rematerializes byte-for-byte from the current
decisions file. The previous `be83ab20…` corpus hash is superseded; no frozen
contract in this repository references it. The initial draft pool was sampled
without Llama outputs, safety judgments, or the sealed confirmatory split;
the approval-selection process requires separate provenance confirmation.

| Archetype | New source file | Pinned upstream commit | Local file SHA-256 |
| --- | --- | --- | --- |
| Arithmetic | [SVAMP](https://github.com/arkilpatel/SVAMP), `SVAMP.json` (MIT) | `78e727689e1c1bebfc4be39c446898e8e10b0518` | `5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e` |
| Factual | [HaluEval](https://github.com/RUCAIBox/HaluEval), `data/qa_data.json` (MIT repository) | `b7253db3cdaa0ab2c382f92b26b390109174f77e` | `89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88` |
| False premise | [CREAK](https://github.com/yasumasaonoe/creak), `data/creak/dev.json` (MIT) | `5e892b28b6dad3e87769cfc01f00a6577ca44068` | `de61800bb7d0c07a9d5b8abdf4c1604db21151bdfcb13a284db112a531bf3455` |
| Motivated reasoning | [FOLIO](https://github.com/Yale-LILY/FOLIO), `data/v0.0/folio-train.jsonl` (CC BY-SA 4.0) | `5d7bb84c7edab3fb358e057d2807f19cf5cf5e2d` | `008d34b750d31fa7f014e953228adf4db81ec34bbda9e7f67c96c60438d1e6b2` |

The copied source files are locally available under
`data/raw/sources/validation_v3_fresh/` and ignored by Git. To reproduce
elsewhere, fetch the four official repositories at the pinned commits, copy
the named files, verify the hashes above, then run
`python -m scripts.build_validation_v3_fresh_candidates --help` for the
source-path/revision arguments. The pool uses a fixed seed of 42 and does not
read any model outputs. Repository-level licenses do not by themselves settle
every underlying-source right (notably HaluEval's HotpotQA-derived material);
review redistribution terms before publishing examples.

## Curation requirements

The drafts are deliberately overcomplete and **not automatically trusted**.
Review the underlying source as well as the displayed pair. Reject or edit
source questions that are incoherent (some SVAMP items have anomalous wording),
HaluEval hallucinations that do not actually answer or contradict the question,
CREAK explanations that fail to correct the premise, and FOLIO examples where
the conclusion is unclear from the premises. For FOLIO, judge entailment from
the *given premises*, not external world knowledge. Keep the four archetypes
conceptually distinct. Record substantive rationales and all edits. Do not
select on baseline failure or predicted steering responsiveness.

The existing checkpointed curation UI can display and edit this draft pool:

```python
from safety_governor.validation_v2_widgets import build_validation_v2_curation_widget
build_validation_v2_curation_widget(
    "data/working/validation_v3_fresh/review_candidates.jsonl",
    "data/working/validation_v3_fresh/review_decisions.jsonl",
    reviewer="YOUR_NAME",
)
```

That widget records judgments only; **do not** use the v2 materializer,
which assigns v2 calibration/confirmatory roles. The v3-specific
`scripts/materialize_validation_v3_fresh.py` produced the current corpus.
The [fresh-dataset local audit](validation_v3_fresh_development_protocol.md)
passes against its current hash, but this does not establish source fidelity
or authenticate the reviewer.

## Checks completed and not completed

Against the 88 fitting pairs and 48 spent v2-calibration pairs only, the draft
has zero exact pair-ID, source-group-ID and normalized-instruction collisions.
All 128 candidates have distinct source groups. These are *local structural*
checks, not proof of semantic independence or label quality. The original v2
confirmatory prompts were **not displayed or scored**. An assistant-run,
content-silent comparison against the sealed 64-pair file found zero pair-ID,
source-group-ID, source-record-ID, and normalized-instruction overlaps. The user
adopted its local check record on 14 October 2026; this was not an independent
rerun by another operator. The current decisions were written by
`scripts/apply_curation_decisions_v3.py`; its original four shared rejection
rationales were subsequently replaced with 80 item-specific AI-drafted
rationales using `scripts/rewrite_rejection_rationales.py`. The reviewer confirmed
their substance in the local `data/working/validation_v3_fresh/review_attestation.md`;
that attestation is not independent source verification. The new execution
contract and two-finalist lock are frozen locally; bounded GPU authorization,
runtime transfer verification and blind review remain pending. Nothing here changes
the >70% suppression threshold or promotes train-screen findings to efficacy.
