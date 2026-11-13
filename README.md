# HiCOPD-UniMETRICNet: Physics-Guided, Quality-Gated, and Taxonomy-Regularized Hierarchical Multi-Expert Network for Cross-Database Respiratory Acoustic COPD Screening

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **"HiCOPD-UniMETRICNet: A Physics-Guided, Quality-Gated, and Taxonomy-Regularized Hierarchical Multi-Expert Network for Cross-Database Respiratory Acoustic COPD Screening"**.

---

## 1. Overview & Key Contributions

Deep learning models for Chronic Obstructive Pulmonary Disease (COPD) screening from digital thoracic auscultation frequently suffer from **cross-database shortcut learning**, exploiting center-specific stethoscope transfer functions, room acoustics, and digital filter modes rather than airway pathophysiology. **HiCOPD-UniMETRICNet** (`0.818M` parameters, `1.704 GMACs` per 18-segment patient bag) addresses zero-shot Leave-One-Database-Out (LODO) generalization through five core mechanisms:

1. **Leak-Free Multi-Center LODO Benchmark & Hybrid Cohort Design**: Harmonizes four public cohorts ($N = 318$ unique patients, $2,970$ standardized $8\,\text{s}$ segments at $16\,\text{kHz}$: **ICBHI 2017** [$n=123$], **KAUH** [$n=95$], **RDTR** [$n=36$], and **BRACETS** [$n=64$]), pairing KAUH and RDTR into a multi-disease **HYBRID** cohort ($n=131$) to eliminate single-class domain degeneracy while maintaining strict patient- and database-level isolation.
2. **Relative Impulse-Safe Normalized PCEN (RISN-PCEN)**: Mild pre-emphasis ($\alpha = 0.15$) coupled with slow-time-constant adaptive gain control and a rectified temporal-derivative impulse branch that removes stationary stethoscope coloration while preserving $<20\,\text{ms}$ fine/coarse crackle transients.
3. **Dual-Stream Multi-Expert Architecture & Quality-Gated MIL Pooling**: Combines a **Multi-Scale Dilated Spectral CNN** ($k \in \{3,5\}, d \in \{1,2,3\}$) with a **36D Physics-Guided Acoustic-Biomarker Encoder** (sub-band energy ratios, wheeze harmonicity, Teager-Kaiser crackle kurtosis, and SNR/clipping descriptors), fused via **Uncertainty-Aware Quality-Gated Attention Multiple-Instance Learning (MIL)**.
4. **Hierarchical Clinical Taxonomy Prototypes**: Decomposes patient-level inference into **Stage-1 Abnormality Detection** (*Healthy vs. Pathological*) and **Stage-2 Conditional COPD Attribution** (*COPD vs. Non-COPD Cardiopulmonary Mimics* including asthma, heart failure, pneumonia, bronchiectasis, URTI, and ILD).
5. **Paired Filter-Mode Consistency & PRISM-CalNet Calibration**: Enforces prediction and embedding invariance across Bell (B), Diaphragm (D), and Extended (E) digital stethoscope filter modes alongside strictly label-free target logit moment alignment ($\tau_{\text{src}}^* = 0.4600$).

---

## 2. Main Zero-Shot LODO Benchmark Results

### 2.1 Primary Zero-Shot LODO Comparison (Analysis B: 8 Matched Architectures)

| Architecture | Unseen ICBHI AUROC | Unseen ICBHI BAcc | Unseen HYBRID AUROC | Unseen HYBRID BAcc | Unseen BRACETS AUROC | Unseen BRACETS BAcc | **Macro LODO AUROC** | **Worst-Target AUROC** | **Macro LODO BAcc** | **Macro ECE** $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AST-Lite | 0.6322 | 0.5322 | 0.4519 | 0.5581 | 0.4591 | 0.5000 | 0.5144 | 0.4519 | 0.5301 | 0.1528 |
| Physio-Quality MLP | 0.6909 | 0.6429 | 0.4850 | 0.4921 | 0.4256 | 0.4823 | 0.5339 | 0.4256 | 0.5391 | 0.1278 |
| ConvNeXt-Spec | 0.6110 | 0.5098 | 0.6021 | 0.5116 | 0.5113 | 0.5000 | 0.5748 | 0.5113 | 0.5071 | 0.1860 |
| ResNet-Spec | 0.6912 | 0.5678 | 0.6558 | 0.5758 | 0.4325 | 0.5000 | 0.5932 | 0.4325 | 0.5479 | 0.1830 |
| Spectral-CNN | 0.5863 | 0.4909 | 0.7587 | 0.5999 | 0.4512 | 0.4448 | 0.5987 | 0.4512 | 0.5119 | **0.0842** |
| PCEN-CRNN | 0.8030 | 0.5000 | 0.6300 | 0.5116 | 0.5212 | 0.5000 | 0.6514 | 0.5212 | 0.5039 | 0.2165 |
| MobileNetV2-Spec | 0.7264 | 0.6405 | 0.7579 | 0.6342 | 0.6030 | 0.5000 | 0.6958 | 0.6030 | 0.5916 | 0.1560 |
| **HiCOPD-UniMETRICNet (Ours)** | **0.8488** | **0.7981** | **0.8755** | **0.7870** | **0.6739** | **0.6305** | **0.7994** | **0.6739** | **0.7385** | 0.1257 |
| **Gain vs. Strongest Baseline** | **+0.0458** | **+0.1552** | **+0.1168** | **+0.1527** | **+0.0709** | **+0.1305** | **+0.1036** | **+0.0709** | **+0.1469** | **-0.0303** |

### 2.2 Hierarchical Clinical Taxonomy & Stethoscope Filter Invariance Summary

- **Stage-1 Abnormality Detection (*Healthy vs. Pathological*)**: Macro LODO AUROC = **0.8340** (`0.8378` on ICBHI, `0.7641` on HYBRID, `0.9000` on BRACETS).
- **Stage-2 Conditional COPD Attribution (*COPD vs. Non-COPD Pathology*)**: Macro LODO AUROC = **0.7532** (`0.8101` on ICBHI, `0.8512` on HYBRID, `0.5984` on BRACETS).
- **KAUH Paired Bell/Diaphragm/Extended Filter Stability ($n = 95$)**: Mean within-patient probability range $\Delta |p| = \mathbf{0.0764}$ (**45.5% reduction** vs. ResNet-Spec and **83.9% reduction** vs. MobileNetV2-Spec), intraclass repeatability $\text{ICC}(2,1) = \mathbf{0.8583}$.
- **Mechanism Ablations (Analysis C, $n = 9$ fold-seed runs)**: All 8 core architectural and training mechanisms are statistically significant after Holm-Bonferroni correction ($p_{\text{Holm}} = \mathbf{0.03125}$).

---

## 3. Repository Structure

```text
HiCOPD/
├── configs/
│   └── main.yaml                  # Master configuration (paths, LODO folds, model & training hyperparameters)
├── src/
│   └── hicopd_unimetricnet/
│       ├── __init__.py
│       ├── audio.py               # 16 kHz bandpass filtering, pre-emphasis, and RISN-PCEN front-end
│       ├── features.py            # 36D physics-guided acoustic biomarkers & signal quality descriptors
│       ├── models.py              # HiCOPD-UniMETRICNet & 7 matched baseline architectures
│       ├── losses.py              # Hierarchical taxonomy prototype, filter-consistency & CVaR losses
│       ├── calibration.py         # PRISM-CalNet label-free moment alignment & temperature scaling
│       ├── dataset.py             # Multi-center patient bag dataset & paired B/D/E filter loader
│       ├── splits.py              # Strict zero-shot LODO and patient-disjoint split generator
│       ├── train.py               # Multi-seed training loop & source threshold selection
│       ├── experiments.py         # End-to-end Analysis B (baselines) & Analysis C (ablations) runner
│       ├── fast_upgrade.py        # Multi-expert hierarchical calibration & statistical table generator
│       ├── metrics.py             # AUROC, AUPRC, BAcc, ECE, AURC, ICC(2,1), bootstrap CIs & Holm tests
│       ├── figures.py             # Publication figure generator
│       ├── reporting.py           # Automated CSV table & evidence gate compiler
│       └── cli.py                 # Command-line interface entry points
├── tests/                         # Unit tests for leak-free splits, GPU/CPU runtime & calibration
├── prospective_validation/        # Templates & scripts for prospective clinical cohort validation
├── CHECK_DATA.bat                 # Verify dataset paths and GPU/CUDA environment on Windows
├── FIX_ENVIRONMENT.bat            # Install/upgrade Python dependencies in conda environment
├── RUN_ALL.bat                    # Run complete 22-stage training, evaluation & audit pipeline
├── RUN_FAST_UPGRADE.bat           # Run fast hierarchical calibration & regenerate all tables/figures
├── PUSH_TO_GITHUB.bat             # One-click helper script to commit and push updates to GitHub
├── check_submission_gates.py      # Audit all 15 pre-specified clinical evidence gates
├── pyproject.toml                 # Python package configuration
├── requirements.txt               # Required Python libraries
└── README.md                      # Project documentation
```

---

## 4. Quick Start & Reproducibility

### Step 1: Environment Setup
```bash
conda create -n copd_with_gpu python=3.12 -y
conda activate copd_with_gpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```
Or on Windows, simply run:
```cmd
FIX_ENVIRONMENT.bat
```

### Step 2: Configure Dataset Paths
Edit `configs/main.yaml` (or set environment variables) to point to your local copies of the four public datasets:
- **ICBHI 2017**: `F:/COPD Research/Respiratory Sound Database`
- **KAUH**: `F:/COPD Research/jwyy9np4gv-3`
- **RDTR**: `F:/COPD Research/RespiratoryDatabase@TR`
- **BRACETS**: `F:/COPD Research/BRACETS`

Verify dataset detection and GPU availability:
```cmd
CHECK_DATA.bat
```

### Step 3: Run Experiments & Regenerate All Tables/Figures
- **Full end-to-end training & evaluation across all folds, seeds, baselines, and ablations**:
  ```cmd
  RUN_ALL.bat
  ```
- **Fast hierarchical calibration upgrade & statistical table/figure regeneration**:
  ```cmd
  RUN_FAST_UPGRADE.bat
  ```

---

## 5. Citation & License

If you find this repository useful in your research, please cite our work:
```bibtex
Will update after acceptance
```

This project is licensed under the MIT License (see `LICENSE` for details).

---

## 6. Contact
**Contact:** Md Emran Hasan (Email: mdemranhasan@njust.edu.cn)