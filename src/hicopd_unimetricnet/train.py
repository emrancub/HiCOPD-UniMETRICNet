from __future__ import annotations
import copy, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .dataset import make_loader, fit_phys_scaler, fit_quality_scaler, make_filter_pair_loader
from .losses import (
    focal_per_sample, subgroup_cvar_risk, view_consistency,
    paired_measurement_consistency, prototype_loss, taxonomy_loss,
    taxonomy_margin_loss, cross_origin_supervised_contrastive,
    hard_negative_ranking_loss, hard_negative_prototype_loss
)
from .models import build_model, masked_mean_std
from .metrics import binary_metrics, source_uncertainty_threshold, uncertainty_score, risk_curve, roc_auc_np
from .calibration import fit_robust_temperature, apply_temperature, logit, sigmoid
from .reliability_calibration import fit_prism_calnet, apply_prism_calnet, state_to_jsonable
from .splits import patient_table, train_val_cal_split, subset, assert_disjoint
from .utils import ensure_dir, set_seed, write_json


def device_for(cfg):
    req = str(cfg['training']['device'])
    req = 'cuda' if req == 'auto' and torch.cuda.is_available() else 'cpu' if req == 'auto' else req
    if req.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    try:
        torch.set_num_threads(max(1, int(cfg['training'].get('torch_threads', 1))))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch.device(req)


def _domain_map(frame):
    return {d: i for i, d in enumerate(sorted(frame.origin_dataset.unique()))}


def predict(model, loader, device):
    model.eval()
    rows = []
    emb = []
    domains = []
    with torch.inference_mode():
        for b in loader:
            qdev = b['quality'].to(device, non_blocking=True)
            mdev = b['mask'].to(device, non_blocking=True)
            out = model(b['x'].to(device, non_blocking=True), b['phys'].to(device, non_blocking=True), qdev, mdev)
            prob = out['logits'].float().softmax(-1)[:, 1].cpu().numpy()
            abn_logits = out.get('abnormality_logits', out['logits'])
            attr_logits = out.get('copd_attr_logits', out['logits'])
            abn_prob = abn_logits.float().softmax(-1)[:, 1].cpu().numpy()
            attr_prob = attr_logits.float().softmax(-1)[:, 1].cpu().numpy()
            z = out['embedding'].float().cpu().numpy()
            ent = -(np.c_[1 - prob, prob] * np.log(np.c_[1 - prob, prob] + 1e-8)).sum(1)
            ew = out.get('expert_weights')
            ew = ew.float().cpu().numpy() if ew is not None else None
            names = getattr(model, 'expert_names', [])
            ep = out.get('expert_logits')
            ep = ep.float().softmax(-1)[..., 1].cpu().numpy() if ep is not None else None
            gp = out.get('global_logits')
            gp = gp.float().softmax(-1)[:, 1].cpu().numpy() if gp is not None else None
            pp = out.get('prototype_logits')
            pp = pp.float().softmax(-1)[:, 1].cpu().numpy() if pp is not None else None
            tx = out.get('taxonomy_logits')
            txp = tx.float().softmax(-1).cpu().numpy() if tx is not None else None
            mr = out.get('reliability')
            mr = mr.float().cpu().numpy() if mr is not None else None
            qs = masked_mean_std(qdev, mdev).float().cpu().numpy()
            qidx = torch.sigmoid((0.7 * qdev[..., 0] - 0.5 * qdev[..., 2] - 0.25 * qdev[..., 3]).masked_fill(~mdev, 0).sum(1) / mdev.sum(1).clamp_min(1)).cpu().numpy()
            for i, p in enumerate(prob):
                tax_i = int(b['taxonomy'][i])
                y_i = int(b['y'][i])
                abn_y = 0 if tax_i == 1 else 1
                attr_y = y_i if abn_y == 1 else -1
                row = dict(
                    person_id=b['person_id'][i],
                    subject_id=b['subject_id'][i],
                    logical_dataset=b['logical_dataset'][i],
                    origin_dataset=b['origin_dataset'][i],
                    category=b['category'][i],
                    y=y_i,
                    abnormal_label=abn_y,
                    copd_attr_label=attr_y,
                    taxonomy=tax_i,
                    probability=float(p),
                    abnormality_probability=float(abn_prob[i]),
                    copd_attr_probability=float(attr_prob[i]),
                    entropy=float(ent[i]),
                    quality_index=float(qidx[i]),
                    age=b['age'][i],
                    sex=b['sex'][i],
                )
                if txp is not None:
                    row['taxonomy_pred'] = int(np.argmax(txp[i]))
                    row['taxonomy_confidence'] = float(np.max(txp[i]))
                for j in range(qs.shape[1]):
                    row[f'quality_summary_{j}'] = float(qs[i, j])
                if ew is not None:
                    for j, n in enumerate(names):
                        row[f'expert_weight_{n}'] = float(ew[i, j])
                if ep is not None:
                    for j, n in enumerate(names):
                        row[f'expert_prob_{n}'] = float(ep[i, j])
                if gp is not None:
                    row['expert_prob_global'] = float(gp[i])
                if pp is not None:
                    row['expert_prob_prototype'] = float(pp[i])
                if mr is not None:
                    row['model_reliability'] = float(mr[i])
                    row['measurement_reliability'] = float(mr[i])
                rows.append(row)
            emb.extend(z)
            domains.extend(b['origin_dataset'])
    frame = pd.DataFrame(rows)
    weights = getattr(model, 'posthoc_fusion_weights', None)
    if weights:
        cols = []
        vals = []
        for name, w in weights.items():
            col = 'expert_prob_' + name
            if col in frame and float(w) > 0:
                cols.append(col)
                vals.append(float(w))
        if cols and sum(vals) > 0:
            ww = np.asarray(vals, float)
            ww = ww / ww.sum()
            frame['probability'] = sum(ww[i] * frame[c].to_numpy(float) for i, c in enumerate(cols))
    return frame, np.asarray(emb), np.asarray(domains)


def _save_calibration_tables(frame, outdir, bins=10):
    rows = []
    p = frame.probability.to_numpy(float)
    y = frame.y.to_numpy(int)
    for b in range(int(bins)):
        lo, hi = b / bins, (b + 1) / bins
        m = (p >= lo) & ((p <= hi) if b == bins - 1 else (p < hi))
        rows.append(dict(
            bin=b, lower=lo, upper=hi, n=int(m.sum()),
            mean_probability=float(p[m].mean()) if m.any() else np.nan,
            observed_rate=float(y[m].mean()) if m.any() else np.nan
        ))
    pd.DataFrame(rows).to_csv(Path(outdir) / 'reliability.csv', index=False)


def _robust_val_score(pred, bins=10):
    aucs = []
    nlls = []
    for _, g in pred.groupby('origin_dataset'):
        if not len(g):
            continue
        mm = binary_metrics(g.y, g.probability, bins=bins)
        nlls.append(float(mm['nll']))
        if g.y.nunique() == 2 and np.isfinite(mm['auroc']):
            aucs.append(float(mm['auroc']))
    pooled = binary_metrics(pred.y, pred.probability, bins=bins)
    worst_auc = min(aucs) if aucs else float(pooled.get('auroc', 0.5))
    mean_auc = float(np.mean(aucs)) if aucs else worst_auc
    worst_nll = max(nlls) if nlls else float(pooled['nll'])
    return float(0.48 * (1 - worst_auc) + 0.24 * (1 - mean_auc) + 0.18 * worst_nll + 0.10 * float(pooled['nll']))


def transport_threshold(frame, cfg, probability_col='probability', domain_col='origin_dataset'):
    y = frame.y.to_numpy(int)
    p = frame[probability_col].to_numpy(float)
    d = frame[domain_col].astype(str).to_numpy()
    tc = cfg['evaluation'].get('threshold_transport', {})
    mins = float(tc.get('min_source_sensitivity', 0.50))
    minp = float(tc.get('min_source_specificity', 0.50))
    pen = float(tc.get('imbalance_penalty', 0.18))
    disp = float(tc.get('local_threshold_dispersion_penalty', 0.12))
    mw = float(tc.get('mean_utility_weight', 0.25))
    cand = np.linspace(0.02, 0.98, int(cfg['evaluation'].get('threshold_grid_points', 501)))
    local = []
    for name in np.unique(d):
        m = d == name
        if len(np.unique(y[m])) < 2:
            continue
        score_best = -1e9
        th_best = 0.5
        for th in cand:
            mm = binary_metrics(y[m], p[m], th)
            sc = float(mm['balanced_accuracy']) - 0.10 * abs(float(mm['sensitivity']) - float(mm['specificity']))
            if sc > score_best:
                score_best = sc
                th_best = float(th)
        local.append(th_best)
    med = float(np.median(local)) if local else 0.5
    best = None
    for th in cand:
        dom = []
        for name in np.unique(d):
            m = d == name
            if len(np.unique(y[m])) < 2:
                continue
            dom.append(binary_metrics(y[m], p[m], th))
        pooled = binary_metrics(y, p, th)
        if dom:
            bacc = np.asarray([m['balanced_accuracy'] for m in dom], float)
            sens = np.asarray([m['sensitivity'] for m in dom], float)
            spec = np.asarray([m['specificity'] for m in dom], float)
            feasible = bool((sens >= mins).all() and (spec >= minp).all())
            worst = float(np.min(bacc))
            mean = float(np.mean(bacc))
            imbalance = float(np.mean(np.abs(sens - spec)))
        else:
            feasible = min(pooled['sensitivity'], pooled['specificity']) >= min(mins, minp)
            worst = pooled['balanced_accuracy']
            mean = worst
            imbalance = abs(pooled['sensitivity'] - pooled['specificity'])
        score = worst + mw * mean - pen * imbalance - disp * abs(float(th) - med)
        key = (1 if feasible else 0, score, min(pooled['sensitivity'], pooled['specificity']), -abs(float(th) - med))
        if best is None or key > best[0]:
            best = (key, float(th))
    if best is None:
        raise RuntimeError('Unable to select a source-only transport threshold')
    return best[1]


def _simplex_weights(k, step):
    units = max(1, int(round(1 / float(step))))
    def rec(left, n, prefix):
        if n == 1:
            yield prefix + [left]
            return
        for i in range(left + 1):
            yield from rec(left - i, n - 1, prefix + [i])
    for a in rec(units, k, []):
        yield np.asarray(a, float) / units


def fit_safe_stacking(cal, cfg):
    sc = cfg['evaluation'].get('safe_stacking', {})
    if not sc.get('enabled', True):
        return {}
    candidate = []
    for c in ['expert_prob_raw_resnet', 'expert_prob_pcen_crnn', 'expert_prob_physio_quality', 'expert_prob_global', 'expert_prob_prototype']:
        if c in cal:
            candidate.append(c)
    if len(candidate) < 2:
        return {}
    rawidx = candidate.index('expert_prob_raw_resnet') if 'expert_prob_raw_resnet' in candidate else 0
    anchor = float(sc.get('raw_anchor_min', 0.45))
    best = None
    y = cal.y.to_numpy(int)
    origins = cal.origin_dataset.astype(str).to_numpy()
    mixed = [d for d in np.unique(origins) if len(np.unique(y[origins == d])) == 2]
    for w in _simplex_weights(len(candidate), float(sc.get('grid_step', 0.10))):
        if w[rawidx] + 1e-9 < anchor:
            continue
        p = sum(w[i] * cal[candidate[i]].to_numpy(float) for i in range(len(candidate)))
        aucs = []
        for d in mixed:
            m = origins == d
            aucs.append(roc_auc_np(y[m], p[m]))
        pooled = roc_auc_np(y, p)
        vals = np.asarray([x for x in aucs if np.isfinite(x)], float)
        worst = float(vals.min()) if len(vals) else float(pooled)
        mean = float(vals.mean()) if len(vals) else float(pooled)
        score = float(sc.get('worst_origin_weight', 0.65)) * worst + (1 - float(sc.get('worst_origin_weight', 0.65))) * mean + 0.05 * float(pooled)
        key = (score, worst, mean, w[rawidx])
        if best is None or key > best[0]:
            best = (key, w)
    if best is None:
        return {}
    return {candidate[i].replace('expert_prob_', ''): float(best[1][i]) for i in range(len(candidate)) if best[1][i] > 0}


def style_adapted_threshold(cal, target, cfg, global_threshold):
    st = cfg['evaluation'].get('style_threshold_adaptation', {})
    if not st.get('enabled', True):
        return float(global_threshold), {}
    qcols = [c for c in cal.columns if c.startswith('quality_summary_')]
    if not qcols:
        return float(global_threshold), {}
    tc = np.nanmedian(target[qcols].to_numpy(float), axis=0)
    rows = []
    for origin, g in cal.groupby('origin_dataset'):
        if g.y.nunique() < 2:
            continue
        th = transport_threshold(g, cfg, 'probability', 'origin_dataset')
        cen = np.nanmedian(g[qcols].to_numpy(float), axis=0)
        dist = float(np.linalg.norm(tc - cen) / np.sqrt(len(qcols)))
        rows.append((str(origin), th, dist))
    if len(rows) < int(st.get('min_mixed_source_domains', 1)):
        return float(global_threshold), {}
    temp = max(float(st.get('temperature', 0.35)), 1e-6)
    ds = np.asarray([r[2] for r in rows])
    ww = np.exp(-(ds - ds.min()) / temp)
    ww = ww / ww.sum()
    th = float(sum(ww[i] * rows[i][1] for i in range(len(rows))))
    info = {rows[i][0]: float(ww[i]) for i in range(len(rows))}
    return th, info


def _prior_shift_em(p, source_prior, iters=50, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    ps = float(np.clip(source_prior, eps, 1 - eps))
    pt = ps
    for _ in range(int(iters)):
        num = p * (pt / ps)
        den = num + (1 - p) * ((1 - pt) / (1 - ps))
        q = num / np.clip(den, eps, None)
        new = float(np.clip(q.mean(), eps, 1 - eps))
        if abs(new - pt) < 1e-7:
            break
        pt = new
    num = p * (pt / ps)
    den = num + (1 - p) * ((1 - pt) / (1 - ps))
    return np.clip(num / np.clip(den, eps, None), eps, 1 - eps), pt


def _moment_align_probabilities(cal_p, test_p, th):
    """Label-free source-anchored logit moment alignment."""
    zc = logit(cal_p)
    zt = logit(test_p)
    th_z = float(logit(np.asarray([th]))[0])
    mu_t = float(np.mean(zt))
    sd_t = float(np.std(zt) + 1e-6)
    sd_c = float(np.clip(np.std(zc), 0.55, 1.15))
    z_aligned = ((zt - mu_t) / sd_t) * sd_c + th_z
    return np.clip(sigmoid(z_aligned), 1e-5, 1.0 - 1e-5)


def _is_prism_variant(v):
    return v not in {'spectral_cnn', 'resnet_spec', 'mobilenet_spec', 'convnext_spec', 'ast_lite', 'physio_mlp', 'crnn'}


def fit_model(train_df, val_df, cal_df, test_df, cfg, variant, seed, outdir, protocol_note=''):
    outdir = ensure_dir(outdir)
    set_seed(seed)
    device = device_for(cfg)
    local = copy.deepcopy(cfg)
    if variant == 'no_instrument_randomization':
        local['augmentation']['instrument_response']['probability'] = 0.0
        local['augmentation']['additive_noise_probability'] = 0.0
        local['augmentation']['gain_probability'] = 0.0
    if variant in {'no_prototype', 'no_taxonomy_prototypes'}:
        local['model']['prototype_weight'] = 0.0
    dm = _domain_map(train_df)
    model = build_model(local, variant).to(device)
    phys_scaler = fit_phys_scaler(train_df)
    quality_scaler = fit_quality_scaler(train_df)
    tr = make_loader(train_df, dm, local, True, seed, phys_scaler=phys_scaler, quality_scaler=quality_scaler)
    va = make_loader(val_df, dm, local, False, seed, phys_scaler=phys_scaler, quality_scaler=quality_scaler)
    ca = make_loader(cal_df, dm, local, False, seed, phys_scaler=phys_scaler, quality_scaler=quality_scaler)
    te = make_loader(test_df, dm, local, False, seed, phys_scaler=phys_scaler, quality_scaler=quality_scaler)
    pair_loader = None
    pair_iter = None
    if _is_prism_variant(variant) and variant != 'no_filter_consistency' and float(local['loss'].get('paired_filter_consistency', 0)) > 0:
        from .features import load_feature_frame
        full = load_feature_frame(local, 'clean')
        pair_loader = make_filter_pair_loader(full, train_df.person_id.unique(), local, seed, phys_scaler, quality_scaler)
        pair_iter = iter(pair_loader) if pair_loader is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=float(local['training']['learning_rate']), weight_decay=float(local['training']['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(local['training']['epochs'])))
    amp = bool(local['training']['amp']) and device.type == 'cuda'
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp and dtype == torch.float16, init_scale=4096)
    use_cvar = _is_prism_variant(variant) and variant != 'no_cvar'
    best = float('inf')
    wait = 0
    history = []
    best_epoch = -1
    start = time.perf_counter()
    for epoch in range(int(local['training']['epochs'])):
        tr.dataset.epoch = epoch
        if pair_loader is not None:
            pair_loader.dataset.epoch = epoch
        model.train()
        total = 0.0
        batches = 0
        for b in tr:
            x = b['x'].to(device)
            x2 = b['x2'].to(device)
            ph = b['phys'].to(device)
            qu = b['quality'].to(device)
            mask = b['mask'].to(device)
            y = b['y'].to(device)
            tax = b['taxonomy'].to(device)
            d = b['domain'].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                o1 = model(x, ph, qu, mask)
                o2 = model(x2, ph, qu, mask)
                per = (
                    focal_per_sample(o1['logits'], y, float(local['loss']['focal_gamma']), float(local['loss'].get('label_smoothing', 0.02)))
                    + focal_per_sample(o2['logits'], y, float(local['loss']['focal_gamma']), float(local['loss'].get('label_smoothing', 0.02)))
                ) / 2.0
                cls = subgroup_cvar_risk(
                    per, y, d,
                    float(local['loss'].get('subgroup_cvar_alpha', 0.50)),
                    float(local['loss'].get('subgroup_cvar_weight', 0.70)),
                    float(local['loss'].get('group_risk_variance_weight', 0.10))
                )[0] if use_cvar else per.mean()
                loss = float(local['loss']['class']) * cls

                # Hierarchical Stage-1 (Healthy vs Pathological) and Stage-2 (COPD vs Other Abnormal) supervision
                abn_y = (tax != 1).long()
                if 'abnormality_logits' in o1 and 'abnormality_logits' in o2:
                    w_abn = float(local['loss'].get('hierarchical_abnormality', 0.25))
                    abn_loss = 0.5 * (
                        F.cross_entropy(o1['abnormality_logits'].float(), abn_y, label_smoothing=0.02)
                        + F.cross_entropy(o2['abnormality_logits'].float(), abn_y, label_smoothing=0.02)
                    )
                    loss = loss + w_abn * abn_loss
                abn_mask = (tax != 1)
                if abn_mask.any() and 'copd_attr_logits' in o1 and 'copd_attr_logits' in o2:
                    w_attr = float(local['loss'].get('hierarchical_attribution', 0.25))
                    attr_y = (y[abn_mask] == 1).long()
                    attr_loss = 0.5 * (
                        F.cross_entropy(o1['copd_attr_logits'][abn_mask].float(), attr_y, label_smoothing=0.02)
                        + F.cross_entropy(o2['copd_attr_logits'][abn_mask].float(), attr_y, label_smoothing=0.02)
                    )
                    loss = loss + w_attr * attr_loss

                if _is_prism_variant(variant):
                    loss = loss + view_consistency(o1, o2, float(local['loss'].get('augmentation_consistency', 0.08)))
                    if 'reliability_logit' in o1 and 'reliability_logit' in o2:
                        with torch.no_grad():
                            p1 = o1['logits'].float().softmax(-1)[:, 1]
                            p2 = o2['logits'].float().softmax(-1)[:, 1]
                            qmean = (qu[..., 0].float().masked_fill(~mask, 0).sum(1) / mask.sum(1).clamp_min(1).float()).clamp(0, 1)
                            rtarget = (torch.exp(-4.0 * (p1 - p2).abs()) * (0.55 + 0.45 * qmean)).clamp(1e-4, 1.0 - 1e-4)
                        rlogit = 0.5 * (o1['reliability_logit'].float() + o2['reliability_logit'].float())
                        loss = loss + float(local['loss'].get('reliability_consistency', 0.08)) * F.binary_cross_entropy_with_logits(rlogit, rtarget)
                    if variant not in {'no_taxonomy', 'no_taxonomy_prototypes'}:
                        loss = loss + taxonomy_loss(o1, tax, float(local['loss'].get('taxonomy', 0.24)), float(local['loss'].get('taxonomy_label_smoothing', 0.04)))
                        loss = loss + taxonomy_margin_loss(o1, tax, float(local['loss'].get('taxonomy_margin', 0.04)), float(local['loss'].get('taxonomy_margin_value', 0.15)))
                    if variant not in {'no_prototype', 'no_taxonomy_prototypes'}:
                        loss = loss + prototype_loss(o1, y, float(local['loss'].get('prototype_compactness', 0.10)))
                        loss = loss + hard_negative_prototype_loss(o1, tax, float(local['loss'].get('prototype_compactness', 0.10)), float(local['loss'].get('label_smoothing', 0.01)))
                    if variant != 'no_ranking':
                        loss = loss + hard_negative_ranking_loss(
                            o1, y, tax, d,
                            float(local['loss'].get('hard_negative_ranking', 0.35)),
                            float(local['loss'].get('hard_negative_margin', 0.40)),
                            float(local['loss'].get('hard_negative_temperature', 0.22)),
                            float(local['loss'].get('origin_balanced_ranking', 0.12))
                        )
                    loss = loss + cross_origin_supervised_contrastive(o1, y, d, float(local['loss'].get('cross_origin_supcon', 0.05)))
                if pair_iter is not None:
                    try:
                        pb = next(pair_iter)
                    except StopIteration:
                        pair_iter = iter(pair_loader)
                        pb = next(pair_iter)
                    oa = model(pb['xa'].to(device), pb['pa'].to(device), pb['qa'].to(device), pb['ma'].to(device))
                    ob = model(pb['xb'].to(device), pb['pb'].to(device), pb['qb'].to(device), pb['mb'].to(device))
                    loss = loss + paired_measurement_consistency(oa, ob, float(local['loss'].get('paired_filter_consistency', 0.24)))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch}')
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(local['training']['gradient_clip']))
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach())
            batches += 1
        scheduler.step()
        pred, _, _ = predict(model, va, device)
        score = _robust_val_score(pred, int(local['evaluation']['calibration_bins']))
        m = binary_metrics(pred.y, pred.probability, bins=int(local['evaluation']['calibration_bins']))
        abn_auc = roc_auc_np(pred.abnormal_label, pred.abnormality_probability) if 'abnormal_label' in pred else float('nan')
        abn_auc_val = float(abn_auc) if np.isfinite(abn_auc) else 0.5
        history.append(dict(epoch=epoch, loss=total / max(1, batches), robust_val_score=score, val_nll=m['nll'], val_auroc=m['auroc'], val_abn_auroc=abn_auc_val))
        if score < best - 1e-5:
            best = score
            wait = 0
            best_epoch = epoch
            torch.save(dict(
                state_dict=model.state_dict(), variant=variant, domain_map=dm, config=local, seed=seed,
                phys_median=phys_scaler['median'], phys_scale=phys_scaler['scale'],
                quality_median=quality_scaler['median'], quality_scale=quality_scaler['scale']
            ), outdir / 'best_model.pt')
        else:
            wait += 1
        pd.DataFrame(history).to_csv(outdir / 'history.csv', index=False)
        print(f'{variant} seed={seed} epoch={epoch+1} robustVal={score:.4f} valAUROC={m["auroc"]:.4f} abnAUROC={abn_auc_val:.4f}', flush=True)
        if wait >= int(local['training']['patience']):
            break

    state = torch.load(outdir / 'best_model.pt', map_location='cpu', weights_only=False)
    model = build_model(state['config'], state['variant']).to(device)
    model.load_state_dict(state['state_dict'])
    phys_scaler = {'median': np.asarray(state['phys_median'], np.float32), 'scale': np.asarray(state['phys_scale'], np.float32)}
    quality_scaler = {'median': np.asarray(state['quality_median'], np.float32), 'scale': np.asarray(state['quality_scale'], np.float32)}
    tr_eval = make_loader(train_df, state['domain_map'], state['config'], False, seed, phys_scaler=phys_scaler, quality_scaler=quality_scaler)
    va_eval = make_loader(val_df, state['domain_map'], state['config'], False, seed, phys_scaler=phys_scaler, quality_scaler=quality_scaler)
    ca = make_loader(cal_df, state['domain_map'], state['config'], False, seed, phys_scaler=phys_scaler, quality_scaler=quality_scaler)
    te = make_loader(test_df, state['domain_map'], state['config'], False, seed, phys_scaler=phys_scaler, quality_scaler=quality_scaler)
    train_pred, train_emb, _ = predict(model, tr_eval, device)
    val_pred, val_emb, _ = predict(model, va_eval, device)
    cal, cal_emb, cal_domains = predict(model, ca, device)
    fusion_weights = {}
    if _is_prism_variant(variant) and variant != 'no_safe_stack':
        fusion_weights = fit_safe_stacking(cal, local)
        model.posthoc_fusion_weights = fusion_weights
        cal, cal_emb, cal_domains = predict(model, ca, device)
        state['posthoc_fusion_weights'] = fusion_weights
        torch.save(state, outdir / 'best_model.pt')
    test, emb, domains = predict(model, te, device)
    evidence_summary = {}
    if _is_prism_variant(variant):
        from .evidence import fit_source_evidence, refit_source_evidence, apply_source_evidence
        disable_projection = (variant == 'no_nuisance_projection')
        estate = fit_source_evidence(train_emb, train_pred, val_emb, val_pred, local, disable_projection=disable_projection)
        if variant in {'no_prototype', 'no_taxonomy_prototypes'}:
            estate['blend'] = 0.0
        fit_emb = np.concatenate([train_emb, val_emb], axis=0)
        fit_pred = pd.concat([train_pred, val_pred], ignore_index=True)
        estate = refit_source_evidence(estate, fit_emb, fit_pred, local)
        cal, cal_emb = apply_source_evidence(estate, cal_emb, cal, local)
        test, emb = apply_source_evidence(estate, emb, test, local)
        evidence_summary = {k: v for k, v in estate.items() if k not in {'basis', 'prototypes'}}
        np.savez_compressed(
            outdir / 'source_evidence_projection.npz',
            basis=np.asarray(estate.get('basis')),
            strength=float(estate.get('strength', 0)),
            blend=float(estate.get('blend', 0)),
            metric_mean=float(estate.get('metric_mean', 0)),
            metric_std=float(estate.get('metric_std', 1)),
            **{f'prototype_{k}': np.asarray(v) for k, v in estate.get('prototypes', {}).items()}
        )
        write_json(outdir / 'source_evidence_summary.json', evidence_summary)
    calnet_summary = {}
    if _is_prism_variant(variant):
        cstate = fit_prism_calnet(
            train_pred, train_emb, val_pred, val_emb, cal, cal_emb, local,
            disable=(variant == 'no_calnet'),
            disable_hardneg=(variant == 'no_hard_negative_calibration'),
            disable_reliability=(variant == 'no_reliability')
        )
        cal = apply_prism_calnet(cstate, cal, cal_emb, local)
        test = apply_prism_calnet(cstate, test, emb, local)
        calnet_summary = state_to_jsonable(cstate)
        write_json(outdir / 'prism_calnet_state.json', calnet_summary)

    temp = fit_robust_temperature(cal, 'origin_dataset', int(local['evaluation'].get('robust_calibration_temperatures', 161)))
    temp = float(np.clip(temp, 0.5, 2.5))
    cal['p_raw'] = cal.probability
    cal['probability'] = apply_temperature(cal.p_raw, temp)
    cal['uncertainty'] = uncertainty_score(cal)
    if 'measurement_reliability' in cal:
        cal['uncertainty'] = np.clip(0.65 * cal['uncertainty'] + 0.35 * (1 - cal['measurement_reliability']), 0, 1)
    th = transport_threshold(cal, local, 'probability', 'origin_dataset')
    uth = source_uncertainty_threshold(cal.uncertainty, float(local['evaluation'].get('source_coverage_target', 0.85)))
    cal['accepted'] = cal.uncertainty <= uth
    cal['probability_moment_aligned'] = _moment_align_probabilities(cal.probability, cal.probability, th)
    cal.to_csv(outdir / 'calibration_predictions.csv', index=False)

    test['p_raw'] = test.probability
    test['probability'] = apply_temperature(test.p_raw, temp)
    test['uncertainty'] = uncertainty_score(test)
    if 'measurement_reliability' in test:
        test['uncertainty'] = np.clip(0.65 * test['uncertainty'] + 0.35 * (1 - test['measurement_reliability']), 0, 1)
    test['accepted'] = test.uncertainty <= uth
    style_th, style_weights = style_adapted_threshold(cal, test, local, th)
    test['probability_moment_aligned'] = _moment_align_probabilities(cal.probability, test.probability, th)
    adapt_cfg = local['evaluation'].get('label_free_prior_adaptation', {})
    prior_target = np.nan
    if adapt_cfg.get('enabled', True):
        test['probability_prior_adapted'], prior_target = _prior_shift_em(
            test.probability, float(cal.y.mean()),
            int(adapt_cfg.get('iterations', 50)), float(adapt_cfg.get('eps', 1e-6))
        )
    test.to_csv(outdir / 'test_predictions.csv', index=False)

    np.savez_compressed(
        outdir / 'test_embeddings.npz',
        disease=emb, origin=domains,
        person_id=test.person_id.to_numpy(dtype=str),
        y=test.y.to_numpy(int),
        taxonomy=test.taxonomy.to_numpy(int),
        logical_dataset=test.logical_dataset.to_numpy(dtype=str)
    )
    np.savez_compressed(
        outdir / 'calibration_embeddings.npz',
        disease=cal_emb, origin=cal_domains,
        person_id=cal.person_id.to_numpy(dtype=str),
        y=cal.y.to_numpy(int),
        taxonomy=cal.taxonomy.to_numpy(int),
        logical_dataset=cal.logical_dataset.to_numpy(dtype=str)
    )
    risk_curve(test.y, test.probability, test.uncertainty, th).to_csv(outdir / 'risk_coverage.csv', index=False)
    _save_calibration_tables(test, outdir, int(local['evaluation']['calibration_bins']))

    bins = int(local['evaluation']['calibration_bins'])
    fixed = binary_metrics(test.y, test.probability, 0.5, bins, u=test.uncertainty)
    robust = binary_metrics(test.y, test.probability, th, bins, u=test.uncertainty)
    stylem = binary_metrics(test.y, test.probability, style_th, bins, u=test.uncertainty)
    aligned = binary_metrics(test.y, test.probability_moment_aligned, th, bins, u=test.uncertainty)
    adapted = binary_metrics(test.y, test.probability_prior_adapted, style_th, bins, u=test.uncertainty) if 'probability_prior_adapted' in test else aligned
    acc = test[test.accepted]
    selective = binary_metrics(acc.y, acc.probability, style_th, bins, u=acc.uncertainty) if len(acc) else {'n': 0, 'status': 'empty'}

    abn_metrics = binary_metrics(test.abnormal_label, test.abnormality_probability, 0.5, bins) if 'abnormal_label' in test else {'n': 0, 'status': 'empty'}
    abn_sub = test[test.copd_attr_label >= 0] if 'copd_attr_label' in test else pd.DataFrame()
    attr_metrics = binary_metrics(abn_sub.copd_attr_label, abn_sub.copd_attr_probability, 0.5, bins) if len(abn_sub) else {'n': 0, 'status': 'empty'}

    metrics = dict(
        pipeline_version='11.2.0',
        fixed_0_5=fixed,
        robust_source_threshold=robust,
        style_adapted_threshold=stylem,
        label_free_prior_adapted=adapted,
        label_free_moment_aligned=aligned,
        selective_robust_threshold=selective,
        abnormality=abn_metrics,
        copd_attribution=attr_metrics,
        coverage=float(test.accepted.mean()),
        temperature=temp,
        abnormality_temperature=1.0,
        copd_attr_temperature=1.0,
        disease_threshold=th,
        abnormality_threshold=0.5,
        copd_attr_threshold=0.5,
        style_threshold=style_th,
        style_source_weights=style_weights,
        fusion_weights=fusion_weights,
        uncertainty_threshold=uth,
        target_prior_estimate=float(prior_target) if np.isfinite(prior_target) else None,
        best_epoch=best_epoch,
        variant=variant,
        seed=seed,
        protocol_note=protocol_note,
        embedding_domain_probe_bacc=None,
        embedding_analysis_status='pending_cpu_postprocess',
        parameters=sum(p.numel() for p in model.parameters()),
        training_seconds=float(time.perf_counter() - start),
        source_domains=sorted(dm),
        subgroup_cvar=bool(use_cvar),
        quality_aware=bool(hasattr(model, 'expert_names')),
        source_evidence_projection=evidence_summary,
        prism_calnet=calnet_summary,
    )
    write_json(outdir / 'metrics.json', metrics)
    write_json(outdir / 'split.json', dict(
        train=sorted(train_df.person_id.unique()),
        validation=sorted(val_df.person_id.unique()),
        calibration=sorted(cal_df.person_id.unique()),
        test=sorted(test_df.person_id.unique())
    ))
    return metrics


def fit_external(source_frame, target_frame, cfg, variant, seed, outdir, protocol_note='External target labels are never used for fitting, model selection, calibration, or threshold selection'):
    p = patient_table(source_frame)
    train, val, cal = train_val_cal_split(p, float(cfg['training']['validation_fraction']), float(cfg['training']['calibration_fraction']), seed)
    test = target_frame.person_id.unique().tolist()
    assert_disjoint(train, val, cal, test)
    return fit_model(subset(source_frame, train), subset(source_frame, val), subset(source_frame, cal), target_frame, cfg, variant, seed, outdir, protocol_note)
