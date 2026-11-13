"""Unified v11.2 regression test suite for Analysis B backbones, Analysis C ablations, MACs, and AMP safety."""
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

from hicopd_unimetricnet.models import build_model, estimate_patient_macs
from hicopd_unimetricnet.config import load_config


def _test_cfg():
    cfg = load_config("configs/main.yaml", quick=True)
    cfg["model"]["spectral_embedding_dim"] = 64
    cfg["model"]["patient_embedding_dim"] = 64
    cfg["model"]["physio_embedding_dim"] = 32
    cfg["model"]["crnn_gru_hidden"] = 32
    return cfg


def test_analysis_b_all_eight_models_and_macs():
    cfg = _test_cfg()
    x = torch.randn(2, 4, 3, 64, 64)
    ph = torch.randn(2, 4, 22)
    qu = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    models = [
        "spectral_cnn", "crnn", "resnet_spec", "mobilenet_spec",
        "convnext_spec", "ast_lite", "physio_mlp", "unimetricnet"
    ]
    for name in models:
        m = build_model(cfg, name)
        out = m(x, ph, qu, mask)
        assert out["logits"].shape == (2, 2), name
        assert out["abnormality_logits"].shape == (2, 2), name
        assert out["copd_attr_logits"].shape == (2, 2), name
        assert out["embedding"].shape == (2, 64), name
        macs = estimate_patient_macs(m, x[:1], ph[:1], qu[:1], mask[:1])
        assert isinstance(macs, int) and macs > 0, f"{name} macs={macs}"


def test_analysis_c_all_eight_ablations_and_autocast_safety():
    cfg = _test_cfg()
    x = torch.randn(2, 3, 3, 64, 64)
    ph = torch.randn(2, 3, 22)
    qu = torch.randn(2, 3, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)
    ablations = [
        "no_risn_pcen", "no_taxonomy_prototypes", "no_calnet",
        "no_filter_consistency", "ablation_spectral_only",
        "ablation_physio_only", "ablation_concat_no_qgate",
        "ablation_single_scale_conv"
    ]
    for variant in ablations:
        m = build_model(cfg, variant)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out = m(x, ph, qu, mask)
            rtarget = torch.full((2,), 0.7, dtype=torch.float32)
            loss = (
                F.cross_entropy(out["logits"].float(), torch.tensor([0, 1]))
                + F.binary_cross_entropy_with_logits(out["reliability_logit"].float(), rtarget)
            )
        loss.backward()
        assert torch.isfinite(loss), variant
