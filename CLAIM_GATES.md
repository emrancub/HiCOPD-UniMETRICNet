# TIM submission gates — PRISM-CalNet v9

These are evidence gates, not guaranteed outcomes. `CHECK_SUBMISSION_GATES.bat` computes them from real held-out results.

Recommended readiness targets:

1. Macro strict-LODO AUROC ≥ **0.67**.
2. Worst unseen-database AUROC ≥ **0.55**.
3. PRISM-CalNet macro AUROC gain over strongest matched baseline ≥ **+0.02**.
4. No unseen database more than **0.02 AUROC** below its strongest matched baseline.
5. Macro sensitivity ≥ **0.60**.
6. Macro specificity ≥ **0.60**.
7. Worst-target balanced accuracy ≥ **0.55**.
8. Macro ECE ≤ **0.15**.
9. KAUH B/D/E relative stability improvement ≥ **20%** versus ResNet.
10. KAUH B/D/E mean probability range ≤ **0.20**.
11. Worst acoustic-stress AUROC retention ≥ **85%**.
12. At least **3** central ablations have positive matched AUROC contribution.
13. At least **1** central mechanism survives Holm correction at p<0.05.

Additional desirable evidence:

- AUROC lower bootstrap CI >0.50 on the key deployment target;
- positive hard-negative margins for Healthy, Asthma, ILD, Pneumonia and related non-COPD groups;
- reduced conditional acquisition-domain probe accuracy after source-only nuisance projection;
- an own controlled repeated-acquisition validation cohort.
