# Dataset governance

Contrastive records are JSONL and must satisfy the schema in `safety_governor.domain.ContrastiveRecord`. Every record requires stable provenance, an explicit review decision, and a non-empty review note before `approved` status is assigned. Reviewer identity is optional; importers must never infer approval. Swahili records additionally require `translation_of` and bilingual-review notes.

Raw harmful prompts and mutable candidates belong under `data/working/`; raw model outputs and superseded runs belong under `data/archive/`. Both trees, together with checkpoints and activation caches, are ignored and must not be committed. Track only metadata, sanitized fixtures, dataset cards, hashes, and approved release subsets. Preserve benchmark licences and source attribution for every pair.
