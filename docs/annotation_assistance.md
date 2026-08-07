# Model-assisted annotation

Model output is candidate material, not approval evidence. Provider-neutral working files live under `data/working/`; raw responses, failures, and run manifests live under the ignored `data/archive/annotation_runs/` tree. Exact provider and model identifiers remain in structured provenance.

The current Google Gemini adapter is [`notebooks/providers/gemini/ifeval_annotation_colab.ipynb`](notebooks/providers/gemini/ifeval_annotation_colab.ipynb). It verifies the pinned IFEval source hash and workload composition, writes neutral artifact names, and never marks records approved or commits generated output. A future provider adapter must follow the same output contract.

The canonical instruction pool is assembled with `python -m scripts.assemble_instruction_candidate_pool`. Human decisions are then applied explicitly with `python -m scripts.apply_annotation_review`; a non-empty review note is required and reviewer identity remains optional.