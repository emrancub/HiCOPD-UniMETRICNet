from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def masked_mean_std(x, mask):
    w = mask.float()[..., None]
    n = w.sum(1).clamp_min(1.0)
    mean = (x * w).sum(1) / n
    var = (((x - mean[:, None, :]) ** 2) * w).sum(1) / n
    return torch.cat([mean, var.sqrt()], -1)


class Residual(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        groups = max(1, min(8, cout // 16))
        self.c1 = nn.Conv2d(cin, cout, 3, stride, padding=1, bias=False)
        self.n1 = nn.GroupNorm(groups, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(groups, cout)
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x):
        y = F.silu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return F.silu(y + self.skip(x))


class InvertedResidual2D(nn.Module):
    def __init__(self, cin, cout, stride=1, expand=2):
        super().__init__()
        hidden = cin * expand
        groups = max(1, min(8, hidden // 16))
        ogroups = max(1, min(8, cout // 16))
        self.use_res = (stride == 1 and cin == cout)
        self.net = nn.Sequential(
            nn.Conv2d(cin, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, stride, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, cout, 1, bias=False),
            nn.GroupNorm(ogroups, cout),
        )

    def forward(self, x):
        y = self.net(x)
        return x + y if self.use_res else y


class MobileNetV2Spectral(nn.Module):
    def __init__(self, d=160, drop=0.15, in_ch=1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 24, 3, stride=2, padding=1, bias=False), nn.GroupNorm(4, 24), nn.SiLU())
        self.body = nn.Sequential(
            InvertedResidual2D(24, 32, 1, 2),
            InvertedResidual2D(32, 48, 2, 2),
            InvertedResidual2D(48, 64, 2, 2),
            InvertedResidual2D(64, 96, 2, 2),
            InvertedResidual2D(96, 128, 1, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out = nn.Sequential(nn.Flatten(), nn.Linear(128, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))

    def forward(self, x):
        return self.out(self.body(self.stem(x)))


class ConvNeXtBlock2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        groups = max(1, min(8, dim // 16))
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False)
        self.norm = nn.GroupNorm(groups, dim)
        self.pw1 = nn.Conv2d(dim, dim * 2, 1)
        self.pw2 = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        y = self.dw(x)
        y = self.pw2(F.gelu(self.pw1(self.norm(y))))
        return x + y


class ConvNeXtSpectral(nn.Module):
    def __init__(self, d=160, drop=0.15, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, stride=2, padding=1, bias=False), nn.GroupNorm(4, 32), nn.GELU(),
            ConvNeXtBlock2D(32),
            nn.Conv2d(32, 64, 2, stride=2, bias=False), nn.GroupNorm(8, 64),
            ConvNeXtBlock2D(64),
            nn.Conv2d(64, 96, 2, stride=2, bias=False), nn.GroupNorm(8, 96),
            ConvNeXtBlock2D(96),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(96, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class SpectralCNN(nn.Module):
    def __init__(self, d=160, drop=0.15, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False), nn.GroupNorm(4, 32), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.GroupNorm(8, 64), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.GroupNorm(8, 128), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class ResNetSpectral(nn.Module):
    def __init__(self, d=192, drop=0.15, in_ch=1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 32, 5, 2, 2, bias=False), nn.GroupNorm(4, 32), nn.SiLU())
        self.body = nn.Sequential(
            Residual(32, 48), Residual(48, 64, 2), Residual(64, 96, 2),
            Residual(96, 128, 2), Residual(128, 160, 2), Residual(160, 160), nn.AdaptiveAvgPool2d(1)
        )
        self.out = nn.Sequential(nn.Flatten(), nn.Linear(160, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))

    def forward(self, x):
        return self.out(self.body(self.stem(x)))


class MSTFResNetSpectral(nn.Module):
    """Multi-Scale Time-Frequency Residual Encoder with 100% deterministic global reduction."""
    def __init__(self, d=192, drop=0.15, in_ch=1, single_scale=False):
        super().__init__()
        self.single_scale = bool(single_scale)
        if self.single_scale:
            self.stem = nn.Sequential(nn.Conv2d(in_ch, 32, 3, stride=2, padding=1, bias=False), nn.GroupNorm(4, 32), nn.SiLU())
        else:
            self.b1 = nn.Conv2d(in_ch, 12, kernel_size=(3, 3), stride=2, padding=(1, 1), bias=False)
            self.b2 = nn.Conv2d(in_ch, 10, kernel_size=(5, 3), stride=2, padding=(2, 1), bias=False)
            self.b3 = nn.Conv2d(in_ch, 10, kernel_size=(3, 7), stride=2, padding=(1, 3), bias=False)
            self.bn = nn.GroupNorm(4, 32)
        self.s1 = nn.Sequential(Residual(32, 48), Residual(48, 64, 2))
        self.s2 = nn.Sequential(Residual(64, 96, 2), Residual(96, 128, 2))
        self.s3 = nn.Sequential(Residual(128, 160, 2), Residual(160, 160))
        self.out = nn.Sequential(nn.Linear(160 + 160, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))

    def forward(self, x):
        x_dem = x - x.mean(dim=-2, keepdim=True)
        h = self.stem(x_dem) if self.single_scale else F.silu(self.bn(torch.cat([self.b1(x_dem), self.b2(x_dem), self.b3(x_dem)], dim=1)))
        f1 = self.s1(h)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        g_avg = f3.mean(dim=(-2, -1))
        g_max = f3.amax(dim=(-2, -1))
        return self.out(torch.cat([g_avg, 0.35 * g_max], dim=1))


class ASTLite(nn.Module):
    def __init__(self, d=160, drop=0.15, heads=4, layers=2, in_ch=1):
        super().__init__()
        self.patch = nn.Conv2d(in_ch, d, kernel_size=(8, 8), stride=(8, 8), bias=False)
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 3,
            dropout=drop, batch_first=True, activation='gelu', norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        z = self.patch(x).flatten(2).transpose(1, 2)
        z = self.enc(z)
        return self.drop(self.norm(z.mean(1)))


class CRNNSpectral(nn.Module):
    def __init__(self, d=160, drop=0.15, hidden=96, layers=1, bidirectional=True, in_ch=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False), nn.GroupNorm(4, 32), nn.SiLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.GroupNorm(8, 64), nn.SiLU(), nn.MaxPool2d((2, 2)),
            Residual(64, 96, 1)
        )
        self.gru = nn.GRU(input_size=96, hidden_size=hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional, dropout=drop if layers > 1 else 0.0)
        gd = hidden * (2 if bidirectional else 1)
        self.att = nn.Sequential(nn.Linear(gd, gd // 2), nn.Tanh(), nn.Linear(gd // 2, 1))
        self.out = nn.Sequential(nn.Linear(gd, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))

    def forward(self, x):
        z = self.conv(x)
        z = z.mean(dim=2).transpose(1, 2)
        z, _ = self.gru(z)
        a = self.att(z).squeeze(-1).softmax(-1)
        return self.out((z * a[..., None]).sum(1))


class VectorEncoder(nn.Module):
    def __init__(self, pdim=30, d=96, drop=0.12):
        super().__init__()
        self.pdim = int(pdim)
        self.net = nn.Sequential(nn.Linear(self.pdim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, d), nn.LayerNorm(d), nn.GELU())

    def forward(self, x):
        if x.shape[-1] < self.pdim:
            pad = x.new_zeros((*x.shape[:-1], self.pdim - x.shape[-1]))
            x = torch.cat([x, pad], dim=-1)
        elif x.shape[-1] > self.pdim:
            x = x[..., :self.pdim]
        return self.net(x)


class QualityAwarePatientPool(nn.Module):
    def __init__(self, seg_dim, quality_dim, patient_dim, drop=0.15, uniform_mix=0.65, use_std=True, use_quality=True):
        super().__init__()
        self.uniform_mix = float(uniform_mix)
        self.use_std = bool(use_std)
        self.use_quality = bool(use_quality)
        self.quality_dim = int(quality_dim)
        self.emb_score = nn.Sequential(nn.Linear(seg_dim, max(8, seg_dim // 2)), nn.Tanh(), nn.Linear(max(8, seg_dim // 2), 1))
        self.q_score = nn.Sequential(nn.Linear(self.quality_dim, 32), nn.GELU(), nn.Linear(32, 1)) if use_quality else None
        ind = seg_dim * (2 if use_std else 1)
        self.proj = nn.Sequential(nn.Linear(ind, patient_dim), nn.LayerNorm(patient_dim), nn.GELU(), nn.Dropout(drop))

    def forward(self, z, q, mask):
        sc = self.emb_score(z).squeeze(-1)
        if self.use_quality and self.q_score is not None:
            qin = q
            if qin.shape[-1] < self.quality_dim:
                qin = torch.cat([qin, qin.new_zeros((*qin.shape[:-1], self.quality_dim - qin.shape[-1]))], dim=-1)
            elif qin.shape[-1] > self.quality_dim:
                qin = qin[..., :self.quality_dim]
            sc = sc + 0.35 * self.q_score(qin).squeeze(-1)
        sc = sc.masked_fill(~mask, -1e9)
        learned = sc.softmax(1)
        uniform = mask.float() / mask.float().sum(1, keepdim=True).clamp_min(1.0)
        a = (1.0 - self.uniform_mix) * learned + self.uniform_mix * uniform
        mean = (z * a[..., None]).sum(1)
        if self.use_std:
            std = torch.sqrt((((z - mean[:, None, :]) ** 2) * a[..., None]).sum(1).clamp_min(1e-8))
            v = torch.cat([mean, std], -1)
        else:
            v = mean
        return self.proj(v), a


class UniMETRICNet(nn.Module):
    """UniMETRICNet v11.2 Ultimate: Tri-Stream Quality-Gated Hierarchical Respiratory Acoustic Measurement Network."""
    def __init__(self, cfg, variant='unimetricnet'):
        super().__init__()
        c = cfg['model']
        drop = float(c.get('dropout', 0.15))
        sd = int(c.get('spectral_embedding_dim', 192))
        pd = int(c.get('patient_embedding_dim', 192))
        vd = int(c.get('physio_embedding_dim', 64))
        self.sd = sd
        self.pd = pd
        self.variant = variant
        self.use_physio = variant not in {'ablation_spectral_only'}
        self.use_spectral = variant not in {'ablation_physio_only'}
        self.use_risn_pcen = variant not in {'no_risn_pcen', 'no_pcen', 'no_hpss'}
        self.use_taxonomy = variant not in {'no_taxonomy_prototypes', 'no_taxonomy', 'no_prototype'}
        self.use_qgate = variant not in {'ablation_concat_no_qgate'}
        single_scale = (variant == 'ablation_single_scale_conv')

        self.anchor_segment = MSTFResNetSpectral(sd, drop, 1, single_scale=single_scale)
        self.dw_segment = MobileNetV2Spectral(128, drop, 1)
        self.anchor_pool = QualityAwarePatientPool(sd, 8, pd, drop, float(c.get('attention_uniform_mix', 0.78)), True, self.use_qgate)
        self.dw_pool = QualityAwarePatientPool(128, 8, pd, drop, 0.78, True, self.use_qgate)
        self.anchor_head = nn.Sequential(nn.Linear(pd, max(16, pd // 2)), nn.LayerNorm(max(16, pd // 2)), nn.GELU(), nn.Dropout(drop), nn.Linear(max(16, pd // 2), 2))
        self.abnormality_head = nn.Sequential(nn.Linear(pd, max(16, pd // 2)), nn.LayerNorm(max(16, pd // 2)), nn.GELU(), nn.Dropout(drop), nn.Linear(max(16, pd // 2), 2))
        self.copd_attr_head = nn.Sequential(nn.Linear(pd, max(16, pd // 2)), nn.LayerNorm(max(16, pd // 2)), nn.GELU(), nn.Dropout(drop), nn.Linear(max(16, pd // 2), 2))

        self.risn_segment = CRNNSpectral(96, drop, 64, 1, True, 2)
        self.risn_pool = QualityAwarePatientPool(96, 8, pd, drop, 0.80, True, self.use_qgate)
        self.risn_proj = nn.Linear(pd, pd, bias=False)

        self.physio = VectorEncoder(30, vd, drop)
        self.phys_pool = QualityAwarePatientPool(vd, 8, pd, drop, 0.84, True, self.use_qgate)
        self.phys_proj = nn.Linear(pd, pd, bias=False)
        self.quality_summary_proj = nn.Sequential(nn.Linear(16, 48), nn.LayerNorm(48), nn.GELU(), nn.Linear(48, pd))
        self.quality_gate = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 2))

        self.taxonomy_prototypes = nn.Parameter(F.normalize(torch.randn(int(c.get('taxonomy_classes', 5)), pd), dim=-1))
        self.taxonomy_temperature = float(c.get('taxonomy_prototype_temperature', 8.0))
        self.proto_scale = float(c.get('prototype_logit_max_abs', 0.45)) if self.use_taxonomy else 0.0
        self.aux_scale = float(c.get('aux_logit_max_abs', 0.30)) if self.use_physio else 0.0
        self.aux_head = nn.Sequential(nn.Linear(pd * 2, max(16, pd // 2)), nn.LayerNorm(max(16, pd // 2)), nn.GELU(), nn.Dropout(drop), nn.Linear(max(16, pd // 2), 1))
        rh = int(c.get('reliability_hidden', 64))
        self.reliability_head = nn.Sequential(nn.Linear(pd + 16, rh), nn.LayerNorm(rh), nn.GELU(), nn.Dropout(drop), nn.Linear(rh, 1))
        self.expert_names = ['anchor', 'evidence']
        self.cfg_rel_clip = float(c.get('reliability_logit_clip', 5.0))

    @staticmethod
    def _binary_logits(score):
        return torch.stack([-0.5 * score, 0.5 * score], dim=-1)

    @staticmethod
    def _pad_or_slice(t, dim):
        if t.shape[-1] < dim:
            return torch.cat([t, t.new_zeros((*t.shape[:-1], dim - t.shape[-1]))], dim=-1)
        if t.shape[-1] > dim:
            return t[..., :dim]
        return t

    def forward(self, x, phys, quality, mask):
        b, k = x.shape[:2]
        flat = x[mask]
        raw = flat[:, 0:1]
        if self.use_spectral:
            za = self.anchor_segment(raw)
            zd = self.dw_segment(raw)
            sega = za.new_zeros((b, k, za.shape[-1])); sega[mask] = za
            segd = zd.new_zeros((b, k, zd.shape[-1])); segd[mask] = zd
            pa, att_a = self.anchor_pool(sega, quality, mask)
            pdw, _ = self.dw_pool(segd, quality, mask)
            anchor_patient = 0.50 * pa + 0.50 * pdw
        else:
            sega = raw.new_zeros((b, k, self.sd))
            anchor_patient = raw.new_zeros((b, self.pd))
            att_a = mask.float() / mask.float().sum(1, keepdim=True).clamp_min(1.0)

        residual = anchor_patient.new_zeros(anchor_patient.shape)
        att_r = None
        if self.use_spectral and self.use_risn_pcen and flat.shape[1] >= 2:
            ch_in = flat[:, 1:3] if flat.shape[1] >= 3 else flat[:, 1:2].repeat(1, 2, 1, 1)
            zr = self.risn_segment(ch_in)
            segr = zr.new_zeros((b, k, zr.shape[-1])); segr[mask] = zr
            rp, att_r = self.risn_pool(segr, quality, mask)
            residual = residual + 0.30 * self.risn_proj(rp)

        phys_patient = anchor_patient.new_zeros(anchor_patient.shape)
        att_p = None
        if self.use_physio:
            zv = self.physio(torch.cat([phys[mask], quality[mask]], -1))
            segv = zv.new_zeros((b, k, zv.shape[-1])); segv[mask] = zv
            pp, att_p = self.phys_pool(segv, quality, mask)
            phys_patient = self.phys_proj(pp)
            residual = residual + (1.0 if not self.use_spectral else 0.12) * phys_patient

        qs_raw = masked_mean_std(quality, mask)
        qs = self._pad_or_slice(qs_raw, 16)
        if self.use_qgate:
            ew = F.softmax(self.quality_gate(qs), dim=-1)
        else:
            ew = anchor_patient.new_tensor([0.50, 0.50]).view(1, 2).repeat(b, 1)

        emb = F.normalize(anchor_patient + residual + 0.06 * self.quality_summary_proj(qs), dim=-1)
        anchor_logits = self.anchor_head(emb)
        abnormality_logits = self.abnormality_head(emb)
        copd_attr_logits = self.copd_attr_head(emb)
        anchor_score = anchor_logits[:, 1] - anchor_logits[:, 0]

        prot = F.normalize(self.taxonomy_prototypes, dim=-1)
        tax_logits = self.taxonomy_temperature * (emb @ prot.T)
        copd = tax_logits[:, 0]
        neg = torch.logsumexp(tax_logits[:, 1:], dim=1) - torch.log(torch.tensor(max(1, tax_logits.shape[1] - 1), device=tax_logits.device, dtype=tax_logits.dtype))
        proto_margin = torch.tanh((copd - neg) / max(self.taxonomy_temperature, 1e-6)) * self.proto_scale
        proto_logits = self._binary_logits(proto_margin)

        if self.use_physio:
            aux = torch.tanh(self.aux_head(torch.cat([anchor_patient, phys_patient], dim=-1)).squeeze(-1)) * self.aux_scale
        else:
            aux = anchor_score * 0.0

        p_abn = abnormality_logits.float().softmax(-1)[:, 1].clamp(1e-5, 1.0 - 1e-5)
        p_attr = copd_attr_logits.float().softmax(-1)[:, 1].clamp(1e-5, 1.0 - 1e-5)
        p_hier = (p_abn * p_attr).clamp(1e-5, 1.0 - 1e-5)
        hier_logit = torch.log(p_hier / (1.0 - p_hier)).to(anchor_score.dtype)

        final_score = anchor_score + proto_margin + aux + 0.22 * hier_logit
        logits = self._binary_logits(final_score)
        evidence_logits = self._binary_logits(proto_margin + aux + 0.22 * hier_logit)
        reliability_logit = self.reliability_head(torch.cat([emb, qs], dim=-1)).squeeze(-1).clamp(-self.cfg_rel_clip, self.cfg_rel_clip)
        reliability = torch.sigmoid(reliability_logit)
        expert_logits = torch.stack([anchor_logits, evidence_logits], dim=1)
        att = {'anchor': att_a}
        if att_r is not None: att['risn'] = att_r
        if att_p is not None: att['physio_quality'] = att_p
        return dict(
            logits=logits,
            abnormality_logits=abnormality_logits,
            abn_logits=abnormality_logits,
            copd_attr_logits=copd_attr_logits,
            embedding=emb,
            taxonomy_logits=tax_logits,
            taxonomy_proto_logits=tax_logits,
            prototype_logits=proto_logits,
            global_logits=anchor_logits,
            anchor_logits=anchor_logits,
            ranking_score=final_score,
            expert_logits=expert_logits,
            expert_weights=ew,
            segment_attention=att,
            segment_embedding={'anchor': sega},
            patient_expert_embedding={'anchor': anchor_patient, 'physio_quality': phys_patient, 'residual': residual},
            reliability=reliability,
            reliability_logit=reliability_logit,
        )


class BaselinePatientNet(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        c = cfg['model']
        sd = int(c.get('spectral_embedding_dim', 192))
        pd = int(c.get('patient_embedding_dim', 192))
        drop = float(c.get('dropout', 0.15))
        self.kind = kind
        if kind == 'spectral_cnn': self.segment = SpectralCNN(sd, drop, 1)
        elif kind == 'resnet_spec': self.segment = ResNetSpectral(sd, drop, 1)
        elif kind == 'mobilenet_spec': self.segment = MobileNetV2Spectral(sd, drop, 1)
        elif kind == 'convnext_spec': self.segment = ConvNeXtSpectral(sd, drop, 1)
        elif kind == 'ast_lite': self.segment = ASTLite(sd, drop, 4, 2, 1)
        elif kind == 'crnn': self.segment = CRNNSpectral(sd, drop, int(c.get('crnn_gru_hidden', 96)), int(c.get('crnn_layers', 1)), bool(c.get('crnn_bidirectional', True)), 1)
        elif kind == 'physio_mlp': self.segment = VectorEncoder(30, sd, drop)
        else: raise ValueError(kind)
        self.pool = QualityAwarePatientPool(sd, 8, pd, drop, 0.70, True, kind == 'physio_mlp')
        self.classifier = nn.Linear(pd, 2)
        self.abnormality_head = nn.Linear(pd, 2)
        self.copd_attr_head = nn.Linear(pd, 2)

    def forward(self, x, phys, quality, mask):
        b, k = x.shape[:2]
        if self.kind == 'physio_mlp':
            valid = self.segment(torch.cat([phys[mask], quality[mask]], -1))
        else:
            ch = 1 if self.kind == 'crnn' and x.shape[2] > 1 else 0
            valid = self.segment(x[mask][:, ch:ch + 1])
        z = valid.new_zeros((b, k, valid.shape[-1])); z[mask] = valid
        patient, a = self.pool(z, quality, mask)
        emb = F.normalize(patient, dim=-1)
        abn = self.abnormality_head(patient)
        attr = self.copd_attr_head(patient)
        zero = patient.new_zeros(b)
        return dict(
            logits=self.classifier(patient),
            abnormality_logits=abn,
            abn_logits=abn,
            copd_attr_logits=attr,
            embedding=emb,
            segment_embedding=z,
            segment_attention={'baseline': a},
            reliability_logit=zero,
            reliability=torch.ones_like(zero) * 0.5,
        )


def build_model(cfg, variant='unimetricnet'):
    if variant in {'spectral_cnn', 'resnet_spec', 'mobilenet_spec', 'convnext_spec', 'ast_lite', 'physio_mlp', 'crnn'}:
        return BaselinePatientNet(cfg, variant)
    return UniMETRICNet(cfg, variant)


def estimate_patient_macs(*args, **kwargs) -> int:
    """Estimate Multiply-Accumulate Operations (MACs) for a single patient bag forward pass.

    Supports flexible call signatures:
      - estimate_patient_macs(model, x, phys, quality, mask)
      - estimate_patient_macs(model, segments=18)
      - estimate_patient_macs(cfg, variant, segments=18)
    """
    segments = int(kwargs.get('segments', 18))
    model = None
    x = kwargs.get('x', None)
    phys = kwargs.get('phys', None)
    quality = kwargs.get('quality', None)
    mask = kwargs.get('mask', None)

    if len(args) >= 1 and isinstance(args[0], nn.Module):
        model = args[0]
        if len(args) >= 5 and torch.is_tensor(args[1]):
            x, phys, quality, mask = args[1], args[2], args[3], args[4]
        elif len(args) >= 2 and isinstance(args[1], int):
            segments = int(args[1])
    elif len(args) >= 2 and isinstance(args[0], dict) and isinstance(args[1], str):
        model = build_model(args[0], args[1])
        if len(args) >= 3 and isinstance(args[2], int):
            segments = int(args[2])
    elif len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
        model = build_model(args[1], args[0])
        if len(args) >= 3 and isinstance(args[2], int):
            segments = int(args[2])
    else:
        raise TypeError("estimate_patient_macs expects (model, [x, phys, quality, mask]) or (cfg, variant)")

    dev = next(model.parameters()).device
    if x is None:
        x = torch.zeros((1, segments, 3, 64, 126), dtype=torch.float32, device=dev)
    if phys is None:
        phys = torch.zeros((1, x.shape[1], 22), dtype=torch.float32, device=dev)
    if quality is None:
        quality = torch.zeros((1, x.shape[1], 8), dtype=torch.float32, device=dev)
    if mask is None:
        mask = torch.ones((1, x.shape[1]), dtype=torch.bool, device=dev)

    macs = [0]
    hooks = []

    def conv_hook(module, inp, out):
        if not torch.is_tensor(out):
            return
        b, cout, oh, ow = out.shape
        cin = module.in_channels // max(1, module.groups)
        kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        macs[0] += int(b * cout * oh * ow * cin * kh * kw)

    def linear_hook(module, inp, out):
        if not inp or not torch.is_tensor(inp[0]):
            return
        t = inp[0]
        leading = int(t.numel() // max(1, t.shape[-1]))
        macs[0] += int(leading * module.in_features * module.out_features)

    def gru_hook(module, inp, out):
        if not inp or not torch.is_tensor(inp[0]):
            return
        t = inp[0]
        b, seq_len = int(t.shape[0]), int(t.shape[1])
        dirs = 2 if module.bidirectional else 1
        ih = module.input_size
        hh = module.hidden_size
        for layer in range(module.num_layers):
            in_d = ih if layer == 0 else hh * dirs
            macs[0] += int(b * seq_len * dirs * 3 * (in_d * hh + hh * hh))

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.GRU):
            hooks.append(m.register_forward_hook(gru_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(x.to(dev), phys.to(dev), quality.to(dev), mask.to(dev))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return int(macs[0])
