# Literature / TIM positioning (2026)

The manuscript should not be positioned as “another deep COPD classifier.” The stronger instrumentation question is whether an AI-assisted respiratory-auscultation measurement chain remains reliable when acquisition system, filter response, population, signal quality and disease mix change.

The current IEEE TIM author guidance explicitly warns that machine-learning classification papers may be rejected when they do not clearly demonstrate improvement in measurement performance. It also raises the bar for studies using only existing datasets. Therefore the manuscript should foreground:

- acquisition-chain characterization;
- repeated-measurement repeatability (KAUH B/D/E);
- whole-database external validation;
- source-frozen calibration / threshold transport;
- signal-quality and acoustic-stress behavior;
- uncertainty / selective risk without misusing metrological terminology;
- computational burden and deployability;
- precise distinction between decision metrics and measurement-system characteristics.

PRISM-CalNet is intentionally simpler than prior internal prototypes: the empirically transferable spectral ResNet is retained as the primary evidence extractor, while disease-taxonomy supervision and repeated-measurement regularization address the two observed weaknesses: heterogeneous negative-disease confusion and acquisition sensitivity.

For the final manuscript, compare against relevant TIM respiratory-sound / biomedical measurement work plus matched modern audio baselines under exactly the same patient splits.
