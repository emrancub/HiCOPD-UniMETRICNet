# Why PRISM-CalNet v9 changed strategy

Real results from the previous iterations showed three persistent failures: the proposed network could be less discriminative than ResNet on unseen databases, the COPD probability range was compressed near 0.5, and a single source-derived threshold transported poorly. At the same time, same-patient KAUH measurement repeatability improved dramatically.

v8 therefore does **not** add another Transformer, another unconditional adversary, or another large expert ensemble. Instead it:

1. preserves the raw ResNet as an explicit transfer anchor;
2. optimizes AUROC-relevant hard-negative ranking directly;
3. models clinically heterogeneous negatives explicitly;
4. suppresses only source-estimated, class-conditional acquisition nuisance directions;
5. permits auxiliary evidence only when source validation demonstrates robust benefit;
6. keeps the strong repeatability objective that worked in prior real-data runs;
7. treats threshold transport as a measurement endpoint rather than hiding it behind AUROC.

This is a narrower but stronger TIM argument: **reliable transfer of disease evidence under changing acquisition conditions**.
