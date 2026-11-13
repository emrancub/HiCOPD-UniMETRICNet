# Start here — PRISM-CalNet v9.0

1. Extract into a **new folder**.
2. Activate `copd_with_gpu`.
3. Run:

```bat
FIX_ENVIRONMENT.bat
GPU_CHECK.bat
VERIFY_RUNTIME_ISOLATION.bat
TEST_PROJECT.bat
CHECK_DATA.bat
RUN_QUICK.bat
```

Quick outputs are under `outputs_unimetricnet_v11_quick/` and are **not publication evidence**.

For the complete TIM experiment:

```bat
CLEAN_OUTPUTS.bat
RUN_ALL.bat
CHECK_SUBMISSION_GATES.bat
```

Publication results are under `outputs_unimetricnet_v11/`.


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
