# Repository and data lifecycle

| Location | Purpose | Git policy |
| --- | --- | --- |
| `instruction_following_eval/` | Pinned official IFEval checker and licence | Tracked; upstream hashes in NOTICE |
| `data/raw/sources/` | Immutable upstream snapshots | Ignored by default |
| `data/working/` | Candidates, review queues, reports, and quarantine | Ignored |
| `data/archive/` | Superseded candidates and annotation runs | Ignored; hashes tracked |
| `datasets/fixtures/` | Non-research smoke data | Tracked |
| `datasets/manifests/` | Source, reconstruction, working-state, smoke-run, archive, and freeze lineage | Tracked |
| `datasets/pilot/` | Invalidated pilot data retained for auditability | Tracked where licensing permits |
| `datasets/frozen/` | Future approved experiment inputs | Governance/licence dependent |
| `artifacts/` | Captures, vectors, and run manifests | Ignored |

The tracked review notebook lives at
`docs/notebooks/review/ifeval_human_review_workbench.ipynb`. Its generated bundle,
extracted immutable sources, session checkpoints, imported exports, and provider audit
outputs remain under `data/working/instruction_noncompliance/` and are ignored. The
tracked `scripts/build_review_workbench_notebook.py` deterministically rebuilds the
notebook; review policy itself lives in the tested package module rather than notebook
cell state.

Working filenames describe state rather than tools or chronology. Provider/model identifiers belong in annotation provenance or semantic-audit run manifests, never in experiment inputs.

The materialization boundary copies only research identifiers, instruction, completion, polarity, source group, split, provenance citation, and approval state. Review notes and generation traces remain outside the transcript.

The approved deceptive reasoning materialization now lives as an ignored working artifact under `data/working/deceptive_reasoning/`; the canonical approved review queue remains separately recorded for auditability.

Harmful compliance is quarantined from Stage 1. Its legacy file is recoverable locally as `data/working/harmful_compliance/quarantined_legacy_candidates.jsonl`, and the rebuild-task scaffold lives beside it as `data/working/harmful_compliance/rebuild_tasks.jsonl`.

`datasets/manifests/reconstruction.json` defines the clean-checkout reconstruction boundary. `datasets/manifests/working_state.json` hashes ignored local research inputs without implying approval, while `datasets/manifests/smoke_runs.json` records engineering-only model smoke evidence separately from research results.

Instruction review is partitioned into mechanical, repaired, and semantic queues. The semantic-audit task file is blinded; its pair mapping and provider-specific run records stay in the ignored working bundle.
