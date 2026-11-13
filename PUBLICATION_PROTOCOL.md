# PRISM-CalNet v9.0 publication protocol

## Primary study
Three logical databases: ICBHI, HYBRID (KAUH+RDTR), BRACETS. The primary endpoint is patient-level whole-database leave-one-domain-out generalization over all three targets.

## Non-negotiable leakage controls
- physical patient is the split unit;
- no target labels for training/model selection/projection/calibration/thresholding;
- feature extraction is fixed before evaluation;
- all fusion/projection hyperparameters are source-selected;
- source evidence projection uses only source embeddings, labels and acquisition origins;
- target labels are opened only for final metrics and bootstrap/statistical analysis.

## Measurement endpoints
Discrimination alone is insufficient. Report acquisition repeatability, threshold transport, calibration, stress robustness, quality-stratified behavior, computational characteristics, and acquisition-domain leakage diagnostics.

## Controlled acquisition extension
The included prospective-validation template is strongly recommended for a final TIM submission because all main training/evaluation databases are public datasets.


## Optional own controlled-acquisition validation for TIM

This release includes an additional prospective/repeated-measurement validation layer under `prospective_validation/`.

It is strongly recommended for the final TIM submission because it directly measures repeatability under controlled acquisition changes. It remains isolated from model development.

After the normal public-dataset pipeline has been completed, read:

`prospective_validation/CONTROLLED_ACQUISITION_PROTOCOL.md`

When you have frozen-model predictions from the controlled cohort, run:

```bat
RUN_CONTROLLED_VALIDATION.bat FROZEN_SOURCE_THRESHOLD
```

Never tune the checkpoint, calibration state, threshold, preprocessing, or exclusion rules using controlled-study outcome labels.
