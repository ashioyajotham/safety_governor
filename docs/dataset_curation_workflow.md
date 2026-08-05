# Contrastive dataset curation workflow

Curate at least 300 approved English pairs: 100 each for deceptive reasoning, instruction non-compliance, and harmful compliance. A pair has one safe and one unsafe record sharing a `pair_id`; pairs, not rows, are split into train, validation, and test.

1. Register the source benchmark, version, licence, item identifier, and access conditions.
2. Create both records as `draft`, retaining provenance and a behavior-specific expected outcome.
3. Run `python -m scripts.dataset_summary <file>` to identify structural problems without printing prompts.
4. Review intent, source, duplication, split, and target behavior. Swahili records need bilingual and safety review.
5. Mark only complete records `approved`; model runners reject other statuses.
6. Keep restricted raw prompts in access-controlled storage; commit only permissible metadata, hashes, and sanitized fixtures.

Translate only frozen English evaluation pairs to Swahili. Preserve `translation_of`, source ID, and restricted translation-quality notes; never split a pair across partitions.
