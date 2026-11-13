# Optional pretrained audio baselines

The offline project includes matched local baselines: Spectral-CNN, PCEN-CRNN, ResNet-Spec, AST-lite, and Physio-Quality MLP.

For a final TIM manuscript, one or two pretrained audio encoders can further strengthen the comparison if weights are available locally. Suitable candidates include PANNs, YAMNet, AST/PaSST/HTS-AT, or BEATs-like audio representations.

Rules:

- use the exact exported patient IDs and whole-database LODO splits;
- do not compare directly against published headline numbers obtained with different splits;
- fit calibration and thresholds from source data only;
- never use target labels for model selection;
- report parameter count, latency, AUROC, balanced accuracy, calibration, and stress robustness under the same protocol.
