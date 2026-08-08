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

Working filenames describe state rather than tools or chronology. Provider/model identifiers belong in annotation provenance, never in experiment inputs.

The materialization boundary copies only research identifiers, instruction, completion, polarity, source group, split, provenance citation, and approval state. Review notes and generation traces remain outside the transcript.

Harmful compliance is quarantined from Stage 1. Its legacy file is recoverable locally as `data/working/harmful_compliance/quarantined_legacy_candidates.jsonl`.

`datasets/manifests/reconstruction.json` defines the clean-checkout reconstruction boundary. `datasets/manifests/working_state.json` hashes ignored local research inputs without implying approval, while `datasets/manifests/smoke_runs.json` records engineering-only model smoke evidence separately from research results.
