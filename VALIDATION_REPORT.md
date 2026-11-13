# Validation report — PRISM-CalNet v9.0

Validated software properties:

- Windows CPU/GPU process isolation for OpenMP safety;
- CPU-only audio preprocessing and feature extraction;
- PyTorch model/loss/augmentation execution in a fresh process;
- patient-wise split disjointness;
- requested ICBHI / KAUH / RDTR / BRACETS selection rules;
- taxonomy mapping and taxonomy-aware patient sampling;
- source-only threshold/calibration configuration;
- full/quick output separation;
- result completeness gate before publication tables;
- CPU-only embedding/t-SNE analysis;
- post-hoc explainability isolated from training/model selection;
- synthetic end-to-end audit → features → LODO → external → embedding analysis → explainability → tables → figures.

The synthetic smoke run is software validation only and is never research evidence.
