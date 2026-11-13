from __future__ import annotations
import torch
from torch.nn import functional as F


def focal_per_sample(logits, y, gamma=1.2, label_smoothing=0.0):
    ce = F.cross_entropy(logits.float(), y, reduction='none', label_smoothing=float(label_smoothing))
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** float(gamma)) * ce


def subgroup_cvar_risk(per_sample, y, domain, alpha=0.50, cvar_weight=0.70, var_weight=0.12):
    per_sample = per_sample.float()
    groups = []
    for d in torch.unique(domain[domain >= 0]):
        for c in [0, 1]:
            m = (domain == d) & (y == c)
            if m.any():
                groups.append(per_sample[m].mean())
    if not groups:
        return per_sample.mean(), per_sample.mean() * 0.0, per_sample.mean() * 0.0
    g = torch.stack(groups)
    mean = g.mean()
    k = max(1, int(torch.ceil(torch.tensor(float(alpha) * len(g))).item()))
    worst = torch.topk(g, k=k, largest=True).values.mean()
    var = ((g - mean) ** 2).mean() if len(g) > 1 else g.sum() * 0.0
    return (1.0 - float(cvar_weight)) * mean + float(cvar_weight) * worst + float(var_weight) * var, worst, var


def view_consistency(a, b, weight=0.08):
    pa = a['logits'].float().softmax(-1)
    pb = b['logits'].float().softmax(-1)
    m = (pa + pb) / 2.0
    js = 0.5 * (F.kl_div((pa + 1e-8).log(), m, reduction='batchmean') + F.kl_div((pb + 1e-8).log(), m, reduction='batchmean'))
    emb = 1.0 - (a['embedding'].float() * b['embedding'].float()).sum(-1).mean()
    return float(weight) * (js + 0.35 * emb)


def paired_measurement_consistency(a, b, weight=0.30):
    pa = a['logits'].float().softmax(-1)[:, 1]
    pb = b['logits'].float().softmax(-1)[:, 1]
    prob = (pa - pb).abs().mean()
    emb = 1.0 - (a['embedding'].float() * b['embedding'].float()).sum(-1).mean()
    return float(weight) * (prob + 0.45 * emb)


def prototype_loss(out, y, weight=0.10):
    if 'prototype_logits' not in out:
        return out['logits'].float().sum() * 0.0
    return float(weight) * F.cross_entropy(out['prototype_logits'].float(), y)


def taxonomy_loss(out, taxonomy, weight=0.24, label_smoothing=0.04):
    if 'taxonomy_logits' not in out or float(weight) <= 0:
        return out['logits'].float().sum() * 0.0
    m = taxonomy >= 0
    if not m.any():
        return out['logits'].float().sum() * 0.0
    return float(weight) * F.cross_entropy(out['taxonomy_logits'][m].float(), taxonomy[m], label_smoothing=float(label_smoothing))


def cross_origin_supervised_contrastive(out, y, domain, weight=0.04, temperature=0.18):
    if 'embedding' not in out or float(weight) <= 0:
        return out['logits'].float().sum() * 0.0
    z = F.normalize(out['embedding'].float(), dim=-1)
    n = len(z)
    if n < 3:
        return z.sum() * 0.0
    sim = (z @ z.T) / float(temperature)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    losses = []
    for i in range(n):
        pos = (y == y[i]) & (domain != domain[i]) & (~eye[i])
        valid = ~eye[i]
        if not pos.any() or valid.sum() < 2:
            continue
        den = torch.logsumexp(sim[i, valid], dim=0)
        losses.append(-(sim[i, pos] - den).mean())
    return float(weight) * torch.stack(losses).mean() if losses else z.sum() * 0.0


def taxonomy_margin_loss(out, taxonomy, weight=0.04, margin=0.15):
    if 'embedding' not in out or float(weight) <= 0:
        return out['logits'].float().sum() * 0.0
    z = F.normalize(out['embedding'].float(), dim=-1)
    losses = []
    copd = taxonomy == 0
    if not copd.any():
        return z.sum() * 0.0
    c = F.normalize(z[copd].mean(0), dim=0)
    for t in torch.unique(taxonomy[taxonomy > 0]):
        m = taxonomy == t
        if m.any():
            n = F.normalize(z[m].mean(0), dim=0)
            sim = (c * n).sum()
            losses.append(F.relu(sim - (1.0 - float(margin))))
    return float(weight) * torch.stack(losses).mean() if losses else z.sum() * 0.0


def hard_negative_ranking_loss(out, y, taxonomy, domain, weight=0.35, margin=0.40, temperature=0.22, origin_weight=0.12):
    if float(weight) <= 0 or 'ranking_score' not in out:
        return out['logits'].float().sum() * 0.0
    s = out['ranking_score'].float()
    pos = (y == 1)
    neg = (y == 0)
    if not pos.any() or not neg.any():
        return s.sum() * 0.0
    losses = []
    for t in torch.unique(taxonomy[neg]):
        if int(t) < 0:
            continue
        nm = neg & (taxonomy == t)
        if not nm.any():
            continue
        diff = (s[pos][:, None] - s[nm][None, :] - float(margin)) / max(float(temperature), 1e-6)
        losses.append(F.softplus(-diff).mean())
    if not losses:
        diff = (s[pos][:, None] - s[neg][None, :] - float(margin)) / max(float(temperature), 1e-6)
        losses = [F.softplus(-diff).mean()]
    main = torch.stack(losses).mean()
    cross = []
    if float(origin_weight) > 0:
        for dp in torch.unique(domain[pos & (domain >= 0)]):
            pm = pos & (domain == dp)
            if not pm.any():
                continue
            for dn in torch.unique(domain[neg & (domain >= 0)]):
                if int(dn) == int(dp):
                    continue
                nm = neg & (domain == dn)
                if not nm.any():
                    continue
                diff = (s[pm][:, None] - s[nm][None, :] - float(margin)) / max(float(temperature), 1e-6)
                cross.append(F.softplus(-diff).mean())
    cross_loss = torch.stack(cross).mean() if cross else s.sum() * 0.0
    return float(weight) * main + float(origin_weight) * cross_loss


def hard_negative_prototype_loss(out, taxonomy, weight=0.10, label_smoothing=0.02):
    if float(weight) <= 0 or 'taxonomy_proto_logits' not in out:
        return out['logits'].float().sum() * 0.0
    m = taxonomy >= 0
    if not m.any():
        return out['logits'].float().sum() * 0.0
    return float(weight) * F.cross_entropy(out['taxonomy_proto_logits'][m].float(), taxonomy[m], label_smoothing=float(label_smoothing))
