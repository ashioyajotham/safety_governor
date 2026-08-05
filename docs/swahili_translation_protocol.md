# English-to-Swahili translation protocol

The translation manifest contains a frozen, stratified sample of 100 approved English pairs: 34 deceptive-reasoning, 33 instruction-noncompliance, and 33 harmful-compliance pairs. The deterministic selection seed is 42.

Each English record must receive a meaning-preserving Swahili rendering, not a paraphrase that changes the requested behavior, arithmetic error, completion, or safety-relevant intent. Do not translate isolated words mechanically when the surrounding instruction requires natural Swahili.

Two sign-offs are required before a record becomes `approved`: a bilingual reviewer confirms semantic fidelity and fluency; a safety reviewer confirms that the behavior label and polarity remain unchanged. Record the translator, reviewer, and any ambiguity in the restricted manifest. The manifest is restricted because it includes harmful-compliance material.

Do not use the translated set to choose steering vectors or coefficients. It is reserved for representation-similarity mapping and transfer evaluation after the English configuration is frozen.
