# Targeted IFEval final-gap annotation

`IFEval_Final_Gap_Annotation_Colab.ipynb` is the dedicated Google Colab runner for the final 15 instruction-following annotation gaps. It is deliberately separate from the original 168-item Gemini run.

The notebook verifies the pinned IFEval source SHA-256 and the exact 15-item workload before generation, writes into an isolated output directory, and leaves every generated record as `pending_review`. Human review remains required before any record can enter the frozen corpus.
