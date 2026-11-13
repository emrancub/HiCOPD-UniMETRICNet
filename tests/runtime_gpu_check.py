"""PyTorch-only model/loss regression checks in a fresh process."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import numpy as np
import torch
from hicopd_unimetricnet.models import UniMETRICNet, build_model, estimate_patient_macs
from hicopd_unimetricnet.losses import subgroup_cvar_risk, view_consistency, taxonomy_loss, taxonomy_margin_loss
from hicopd_unimetricnet.dataset import instrument_randomize_views


def cfg():
    return {
        'augmentation': {
            'enabled': True, 'probability': 1.0,
            'instrument_response': {
                'probability': 1.0, 'eq_knots': 4, 'eq_std': 0.2, 'spectral_tilt_std': 0.1,
                'lowpass_probability': 0.2, 'lowpass_min_fraction': 0.6,
                'highpass_probability': 0.1, 'highpass_max_fraction': 0.15,
                'dynamic_range_probability': 0.1, 'dynamic_range_strength': 0.2
            },
            'additive_noise_probability': 0.2, 'additive_noise_snr_db': [15, 30],
            'gain_probability': 0.2, 'gain_db': [-3, 3],
            'frequency_masks': 1, 'max_frequency_bins': 2,
            'time_masks': 1, 'max_time_frames': 2, 'feature_jitter_std': 0.001
        },
        'model': {
            'spectral_embedding_dim': 32, 'physio_embedding_dim': 16, 'patient_embedding_dim': 24,
            'dropout': 0.1, 'attention_uniform_mix': 0.65,
            'min_spectral_weight': 0.72, 'max_spectral_weight': 0.92,
            'prototype_weight': 0.18, 'prototype_temperature': 6.0,
            'taxonomy_classes': 5, 'crnn_gru_hidden': 16, 'crnn_layers': 1, 'crnn_bidirectional': True
        },
        'loss': {
            'focal_gamma': 1.0, 'class': 1.0, 'subgroup_cvar_alpha': 0.5,
            'subgroup_cvar_weight': 0.62, 'group_risk_variance_weight': 0.1,
            'augmentation_consistency': 0.08, 'taxonomy': 0.24, 'taxonomy_margin': 0.04
        }
    }


def main():
    c = cfg()
    m = UniMETRICNet(c, 'unimetricnet')
    x = torch.randn(6, 3, 3, 64, 126)
    ph = torch.randn(6, 3, 22)
    q = torch.randn(6, 3, 8)
    mask = torch.ones(6, 3, dtype=torch.bool)
    o = m(x, ph, q, mask)
    assert o['logits'].shape == (6, 2)
    assert o['abnormality_logits'].shape == (6, 2)
    assert o['copd_attr_logits'].shape == (6, 2)
    assert o['taxonomy_logits'].shape == (6, 5)
    assert o['expert_weights'].shape == (6, 2)
    assert torch.isfinite(o['logits']).all()
    assert torch.allclose(o['expert_weights'].sum(-1), torch.ones(6), atol=1e-5)
    macs = estimate_patient_macs(m, x[:1], ph[:1], q[:1], mask[:1])
    assert macs > 0
    x2 = torch.randn(2, 2, 3, 64, 40)
    ph2 = torch.randn(2, 2, 22)
    q2 = torch.randn(2, 2, 8)
    m2 = torch.ones(2, 2, dtype=torch.bool)
    for name in ['spectral_cnn', 'crnn', 'resnet_spec', 'mobilenet_spec', 'convnext_spec', 'ast_lite', 'physio_mlp']:
        out = build_model(c, name)(x2, ph2, q2, m2)
        assert out['logits'].shape == (2, 2)
    a = np.random.default_rng(1).normal(size=(3, 3, 64, 20)).astype(np.float32)
    aug = instrument_randomize_views(a, np.random.default_rng(2), c)
    assert aug.shape == a.shape and np.isfinite(aug).all()
    logits = torch.randn(8, 2, requires_grad=True)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    d = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    per = torch.nn.functional.cross_entropy(logits, y, reduction='none')
    loss, worst, var = subgroup_cvar_risk(per, y, d, 0.5, 0.7, 0.1)
    assert torch.isfinite(loss)
    tax = torch.tensor([0, 1, 2, 3, 4, 0])
    assert torch.isfinite(taxonomy_loss(o, tax, 0.2))
    assert torch.isfinite(taxonomy_margin_loss(o, tax, 0.04))
    assert torch.isfinite(view_consistency(o, o, 0.1))
    print('GPU/PyTorch UniMETRICNet runtime checks passed; CUDA:', torch.cuda.is_available())


if __name__ == '__main__':
    main()
