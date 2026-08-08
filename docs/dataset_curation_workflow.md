# Contrastive dataset curation workflow

The current English research gate targets 120 deceptive-reasoning pairs and 120 instruction-noncompliance pairs. Harmful compliance is a separate 100-pair restricted milestone and remains quarantined until rebuilt. A pair has one safe and one unsafe record sharing a `pair_id`; pairs, not rows, are split into train, validation, and test.

1. Register the source benchmark, version, licence, item identifier, and access conditions.
2. Create both records as `draft`, retaining provenance and a behavior-specific expected outcome.
3. Run `python -m scripts.dataset_summary <file>` to identify structural problems without printing prompts.
4. Review intent, source, duplication, split, and target behavior. Swahili records need bilingual and safety review.
5. Record an explicit decision and non-empty review note; only then mark complete records `approved`. Reviewer identity is optional, and model runners reject other statuses.
6. Keep restricted candidates in `data/working/` and superseded lineage in `data/archive/`; commit only permissible metadata, hashes, sanitized fixtures, and approved release subsets.

Translate only frozen English evaluation pairs to Swahili. Preserve `translation_of`, source ID, and restricted translation-quality notes; never split a pair across partitions.
