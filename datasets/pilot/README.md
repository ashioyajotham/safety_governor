# Pilot datasets — not for main research claims

These files are retained to validate the end-to-end capture and vector pipeline. They are not the approved research corpus for RQ1–RQ3.

- `deceptive_reasoning_arithmetic_pilot.jsonl` contains arithmetic chain-of-thought corruptions from the prior mechanistic-interpretability project. Its scope is arithmetic error detection, not general deceptive reasoning.
- `instruction_noncompliance_template_pilot.jsonl` contains deliberately simple synthetic formatting/evasion pairs. Its scope is basic pipeline validation, not naturalistic instruction non-compliance.

Do not use either file for vector-selection, Control Tax, cross-lingual transfer, or paper claims. The replacement corpus must satisfy the diversity gate in `docs/data_quality_remediation.md`.
