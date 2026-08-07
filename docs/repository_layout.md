# Repository and data lifecycle

The repository separates source material, mutable research work, immutable local lineage, and releasable artifacts.

| Location | Purpose | Git policy |
| --- | --- | --- |
| `data/raw/sources/` | Immutable upstream benchmark snapshots | Ignored by default; only explicitly pinned files are tracked |
| `data/working/` | Current candidate corpora, review queues, and restricted working data | Ignored |
| `data/archive/` | Superseded candidates, raw annotation runs, failures, and invalidated pilots | Ignored; hashes and dispositions are tracked in `datasets/manifests/archive_index.json` |
| `datasets/` | Sanitized fixtures, templates, source metadata, and future frozen releases | Tracked subject to source licence and governance |
| `artifacts/` | Activation captures, vectors, and experiment manifests | Ignored |

Canonical working filenames describe research state rather than tools or chronology: use `candidates.jsonl`, `review_queue.jsonl`, and `run_manifest.json`, not provider names or suffixes such as `v2`. Provider and model identifiers belong in provenance metadata. Approval, pair splitting, freezing, translation, and vector fitting are separate explicit gates.