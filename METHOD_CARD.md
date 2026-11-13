# PRISM-CalNet v9.0 method card

## Problem
Previous versions improved repeatability but still showed compressed COPD probabilities, poor threshold transport, and worse strict-LODO AUROC than a simpler ResNet. PRISM-CalNet v9 is designed around those failures.

## Core model

For patient recording bag \(X\):

1. Raw log-Mel segments are encoded by a ResNet anchor.
2. Segment embeddings are aggregated with mostly-uniform quality-bounded attention.
3. A small RISN residual and a low-capacity physiology/quality residual are permitted only as bounded corrections.
4. Five taxonomy prototypes represent COPD, Healthy, Obstructive-other, Infection/Inflammatory, and Other-chronic.
5. COPD evidence is the margin between the COPD prototype and the log-sum-exp of heterogeneous hard-negative prototypes.
6. The final neural score is anchor score + bounded prototype evidence + bounded physiology correction.

## Hard-negative ranking

For COPD score \(s^+\) and a negative score \(s^-_g\) from clinical group \(g\):

\[
L_{rank}=\frac{1}{|G|}\sum_g \log(1+\exp(-(s^+-s^-_g-m)/\tau)).
\]

Each hard-negative taxonomy group contributes equally. A secondary term uses positive/negative patients from different acquisition origins.

## Source-only acquisition nuisance projection

For class \(c\), acquisition-origin \(d\), and source embedding \(z\):

\[
\Delta_{d,c}=\mu_{d,c}-\mu_c.
\]

SVD of the stacked \(\Delta_{d,c}\) vectors gives nuisance basis \(U\). The projected embedding is

\[
z^*=\operatorname{norm}(z-\lambda UU^Tz).
\]

\(\lambda\) and metric-evidence blending are selected only on source validation. If the robust source objective does not improve, the method automatically selects \(\lambda=0\) and auxiliary blend 0, reverting to the raw anchor.

## Disease evidence after projection

Source prototypes are computed for the five taxonomy groups. The metric evidence is

\[
e(z)=\cos(z,p_{COPD})-\operatorname{LSE}\{\cos(z,p_g):g\in negative\ groups\}.
\]

The final source-selected score is the anchor logit plus a constrained multiple of standardized metric evidence.

## Primary TIM endpoints

- macro and worst-target strict LODO AUROC;
- macro sensitivity/specificity and worst balanced accuracy;
- source-frozen operating-point transport;
- KAUH same-patient B/D/E probability range;
- calibration ECE;
- acoustic-stress retention;
- acquisition-domain leakage diagnostics;
- hard-negative clinical separation;
- latency/model size.
