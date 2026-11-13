from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from .utils import ensure_dir, write_json
from .metrics import bootstrap_metrics, holm, paired_auroc_comparison, roc_auc_np, binary_metrics, robust_domain_threshold


def _flatten_metrics(path):
    m = json.loads(Path(path).read_text())
    row = {k: v for k, v in m.items() if not isinstance(v, dict)}
    for group in ['fixed_0_5', 'robust_source_threshold', 'style_adapted_threshold', 'label_free_prior_adapted', 'label_free_moment_aligned', 'selective_robust_threshold', 'abnormality', 'copd_attribution']:
        if isinstance(m.get(group), dict):
            for k, v in m[group].items():
                row[f'{group}_{k}'] = v
    row['metrics_path'] = str(path)
    return row


def _collect(root, protocol):
    base = Path(root) / 'runs' / protocol
    rows = []
    if base.exists():
        for p in base.rglob('metrics.json'):
            r = _flatten_metrics(p)
            rel = p.relative_to(base).parts
            r['run_key'] = '/'.join(rel[:-1])
            r['model'] = r.get('variant', '')
            if protocol in {'lodo', 'crossdb', 'external', 'ablations'} and rel:
                r['target_or_setting'] = rel[0]
            rows.append(r)
    return pd.DataFrame(rows)


def _ensemble_one(run_dirs, cfg):
    tests = []
    cals = []
    for rd in run_dirs:
        tp = rd / 'test_predictions.csv'
        cp = rd / 'calibration_predictions.csv'
        if tp.exists():
            tests.append(pd.read_csv(tp))
        if cp.exists():
            cals.append(pd.read_csv(cp))
    if not tests or not cals:
        return None, None, None
    base_keys = ['person_id', 'y', 'logical_dataset', 'origin_dataset', 'category']
    alltest = pd.concat(tests, ignore_index=True)
    allcal = pd.concat(cals, ignore_index=True)
    keys = [k for k in ['person_id', 'y', 'abnormal_label', 'copd_attr_label', 'logical_dataset', 'origin_dataset', 'category'] if k in alltest.columns]
    keys_c = [k for k in ['person_id', 'y', 'abnormal_label', 'copd_attr_label', 'logical_dataset', 'origin_dataset', 'category'] if k in allcal.columns]
    extra = [c for c in alltest.columns if c.startswith('expert_weight_') or c.startswith('quality_summary_') or c in {'quality_index', 'abnormality_probability', 'copd_attr_probability', 'measurement_reliability', 'probability_moment_aligned'}]
    agg = {'probability': 'mean', 'uncertainty': 'mean', **{c: 'mean' for c in extra}}
    test = alltest.groupby(keys if keys else base_keys, as_index=False).agg(agg)
    extra_c = [c for c in allcal.columns if c.startswith('expert_weight_') or c.startswith('quality_summary_') or c in {'quality_index', 'abnormality_probability', 'copd_attr_probability', 'measurement_reliability', 'probability_moment_aligned'}]
    agg_c = {'probability': 'mean', 'uncertainty': 'mean', **{c: 'mean' for c in extra_c}}
    cal = allcal.groupby(keys_c if keys_c else base_keys, as_index=False).agg(agg_c)
    source_th = robust_domain_threshold(cal, 'probability', 'origin_dataset', int(cfg['evaluation'].get('threshold_grid_points', 501)))
    th = source_th
    st = cfg['evaluation'].get('style_threshold_adaptation', {})
    qcols = [c for c in cal.columns if c.startswith('quality_summary_')]
    if st.get('enabled', True) and qcols:
        tc = np.nanmedian(test[qcols].to_numpy(float), axis=0)
        rows = []
        for origin, g in cal.groupby('origin_dataset'):
            if g.y.nunique() < 2:
                continue
            oth = robust_domain_threshold(g, 'probability', 'origin_dataset', int(cfg['evaluation'].get('threshold_grid_points', 501)))
            cen = np.nanmedian(g[qcols].to_numpy(float), axis=0)
            dist = float(np.linalg.norm(tc - cen) / np.sqrt(len(qcols)))
            rows.append((oth, dist))
        if rows:
            temp = max(float(st.get('temperature', 0.35)), 1e-6)
            ds = np.asarray([r[1] for r in rows])
            ww = np.exp(-(ds - ds.min()) / temp)
            ww = ww / ww.sum()
            th = float(sum(ww[i] * rows[i][0] for i in range(len(rows))))
    bins = int(cfg['evaluation']['calibration_bins'])
    met = binary_metrics(test.y, test.probability, th, bins, u=test.uncertainty)
    met['source_only_threshold'] = float(source_th)
    met['style_adapted_threshold'] = float(th)
    if 'abnormal_label' in test and 'abnormality_probability' in test and test.abnormal_label.nunique() == 2:
        abn_m = binary_metrics(test.abnormal_label, test.abnormality_probability, 0.5, bins)
        met['abnormality_auroc'] = float(abn_m['auroc'])
        met['abnormality_bacc'] = float(abn_m['balanced_accuracy'])
    else:
        met['abnormality_auroc'] = float('nan')
        met['abnormality_bacc'] = float('nan')
    if 'copd_attr_label' in test and 'copd_attr_probability' in test:
        sub = test[test.copd_attr_label >= 0]
        if len(sub) and sub.copd_attr_label.nunique() == 2:
            attr_m = binary_metrics(sub.copd_attr_label, sub.copd_attr_probability, 0.5, bins)
            met['copd_attr_auroc'] = float(attr_m['auroc'])
            met['copd_attr_bacc'] = float(attr_m['balanced_accuracy'])
        else:
            met['copd_attr_auroc'] = float('nan')
            met['copd_attr_bacc'] = float('nan')
    else:
        met['copd_attr_auroc'] = float('nan')
        met['copd_attr_bacc'] = float('nan')
    return test, cal, (th, met)


def _build_ensembles(out, cfg, tables):
    rows = []
    predroot = ensure_dir(tables / 'ensemble_predictions')
    for protocol in ['lodo', 'external', 'crossdb', 'ablations']:
        base = Path(out) / 'runs' / protocol
        if not base.exists():
            continue
        for target_dir in [p for p in base.iterdir() if p.is_dir()]:
            for model_dir in [p for p in target_dir.iterdir() if p.is_dir()]:
                runs = [p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith('seed_')]
                test, cal, res = _ensemble_one(runs, cfg)
                if res is None:
                    continue
                th, met = res
                if protocol in {'lodo', 'external', 'crossdb'}:
                    rows.append(dict(protocol=protocol, target_or_setting=target_dir.name, model=model_dir.name, seeds=len(runs), threshold=th, **met))
                test.to_csv(predroot / f'{protocol}__{target_dir.name}__{model_dir.name}.csv', index=False)
    if rows:
        q = pd.DataFrame(rows)
        q.to_csv(tables / 'Table15_seed_ensemble_metrics.csv', index=False)
        l = q[q.protocol.eq('lodo')].copy()
        l.to_csv(tables / 'Table03b_primary_LODO_ensemble.csv', index=False)
        q[q.protocol.eq('external')].to_csv(tables / 'Table05c_locked_external_ensemble.csv', index=False)
        q[q.protocol.eq('crossdb')].to_csv(tables / 'Table04b_pairwise_transfer_ensemble.csv', index=False)
        if len(l):
            l.groupby('model', as_index=False).agg(
                macro_auroc=('auroc', 'mean'),
                worst_auroc=('auroc', 'min'),
                abnormality_macro_auroc=('abnormality_auroc', 'mean'),
                copd_attr_macro_auroc=('copd_attr_auroc', 'mean'),
                macro_balanced_accuracy=('balanced_accuracy', 'mean'),
                worst_balanced_accuracy=('balanced_accuracy', 'min'),
                macro_sensitivity=('sensitivity', 'mean'),
                macro_specificity=('specificity', 'mean'),
                macro_ece=('ece', 'mean')
            ).to_csv(tables / 'Table03c_primary_LODO_macro.csv', index=False)
        return q
    return pd.DataFrame()


def _lodo_ci_ensemble(ensemble, cfg, tables):
    if ensemble.empty:
        return
    rows = []
    for target in cfg['experiments']['lodo_targets']:
        m = ensemble[(ensemble.protocol == 'lodo') & (ensemble.target_or_setting == target) & (ensemble.model == 'unimetricnet')]
        p = tables / 'ensemble_predictions' / f'lodo__{target}__unimetricnet.csv'
        if m.empty or not p.exists():
            continue
        pred = pd.read_csv(p)
        th = float(m.iloc[0].threshold)
        ci = bootstrap_metrics(pred.y, pred.probability, pred.uncertainty, repeats=int(cfg['evaluation']['bootstrap_resamples']), seed=int(cfg['evaluation']['bootstrap_seed']), bins=int(cfg['evaluation']['calibration_bins']), threshold=th)
        for k, v in ci.items():
            rows.append(dict(target=target, metric=k, **v))
    if rows:
        pd.DataFrame(rows).to_csv(tables / 'Table03d_primary_LODO_bootstrap_CI.csv', index=False)


def _external_ci_ensemble(ensemble, cfg, tables):
    if ensemble.empty:
        return
    row = ensemble[(ensemble.protocol == 'external') & (ensemble.model == 'unimetricnet')]
    if row.empty:
        return
    p = tables / 'ensemble_predictions' / 'external__BRACETS__unimetricnet.csv'
    pred = pd.read_csv(p)
    th = float(row.iloc[0].threshold)
    ci = bootstrap_metrics(pred.y, pred.probability, pred.uncertainty, repeats=int(cfg['evaluation']['bootstrap_resamples']), seed=int(cfg['evaluation']['bootstrap_seed']), bins=int(cfg['evaluation']['calibration_bins']), threshold=th)
    pd.DataFrame([dict(metric=k, **v) for k, v in ci.items()]).to_csv(tables / 'Table05d_locked_external_ensemble_bootstrap_CI.csv', index=False)


def _domain_tables(out, tables):
    rows = []
    geom = []
    for p in (Path(out) / 'runs').rglob('metrics.json'):
        m = json.loads(p.read_text())
        rows.append(dict(
            protocol=str(p.parent), variant=m.get('variant', ''), seed=m.get('seed'),
            chance_bacc=m.get('domain_probe_chance_bacc'),
            disease_domain_probe_bacc=m.get('disease_embedding_domain_probe_bacc'),
            disease_conditional_domain_probe_bacc=m.get('disease_embedding_conditional_domain_probe_bacc')
        ))
        q = p.parent / 'disease_domain_geometry.csv'
        if q.exists() and q.stat().st_size > 1:
            try:
                z = pd.read_csv(q)
            except pd.errors.EmptyDataError:
                continue
            if len(z):
                z['run'] = str(p.parent)
                z['variant'] = m.get('variant', '')
                z['seed'] = m.get('seed')
                geom.append(z)
    if rows:
        pd.DataFrame(rows).to_csv(tables / 'Table09_domain_leakage_diagnostic.csv', index=False)
    if geom:
        pd.concat(geom, ignore_index=True).to_csv(tables / 'Table10_domain_geometry.csv', index=False)


def _primary_paired_ensemble(ensemble, cfg, tables):
    if ensemble.empty:
        return
    predroot = tables / 'ensemble_predictions'
    rows = []
    for target in cfg['experiments']['lodo_targets']:
        prop = predroot / f'lodo__{target}__unimetricnet.csv'
        if not prop.exists():
            continue
        a = pd.read_csv(prop).rename(columns={'probability': 'unimetricnet'})
        for p in predroot.glob(f'lodo__{target}__*.csv'):
            name = p.stem.split('__')[-1]
            if name == 'unimetricnet':
                continue
            b = pd.read_csv(p)[['person_id', 'y', 'probability']].rename(columns={'probability': name})
            m = a[['person_id', 'y', 'unimetricnet']].merge(b, on=['person_id', 'y'])
            r = paired_auroc_comparison(m.y, m.unimetricnet, m[name], int(cfg['evaluation']['bootstrap_resamples']), int(cfg['evaluation']['bootstrap_seed']))
            r.update(target=target, baseline=name, baseline_auroc=float(roc_auc_np(m.y, m[name])), unimetricnet_auroc=float(roc_auc_np(m.y, m.unimetricnet)))
            rows.append(r)
    if rows:
        q = pd.DataFrame(rows)
        q['holm_p'] = holm(q.p_value.to_numpy())
        q.to_csv(tables / 'Table12_primary_LODO_paired_AUROC.csv', index=False)


def _ablation_stats(out, tables):
    ab = _collect(out, 'ablations')
    if ab.empty or 'robust_source_threshold_auroc' not in ab:
        return None
    ab['pair'] = ab.apply(lambda r: f"{r.get('target_or_setting', '')}|{int(r['seed'])}", axis=1)
    full = ab[ab.model.eq('unimetricnet')].set_index('pair')['robust_source_threshold_auroc']
    tests = []
    for v, g in ab[~ab.model.eq('unimetricnet')].groupby('model'):
        x = g.set_index('pair')['robust_source_threshold_auroc']
        idx = full.index.intersection(x.index)
        if len(idx) >= 3:
            d = full.loc[idx] - x.loc[idx]
            try:
                stat, pv = wilcoxon(d)
            except Exception:
                stat, pv = float('nan'), float('nan')
            tests.append(dict(
                ablation=v, n=len(idx),
                full_mean_auroc=float(full.loc[idx].mean()),
                ablated_mean_auroc=float(x.loc[idx].mean()),
                mean_delta_auroc=float(d.mean()),
                positive_pairs=int((d > 0).sum()),
                wilcoxon_stat=stat, p_value=pv
            ))
    if tests:
        t = pd.DataFrame(tests).sort_values('p_value').reset_index(drop=True)
        t['holm_p'] = holm(t.p_value.to_numpy())
        t.to_csv(tables / 'TableS01_ablation_statistics.csv', index=False)
        t.to_csv(tables / 'Table_AnalysisC_Module_and_Mechanism_Ablations.csv', index=False)
        return t
    return None


def _error_taxonomy(ensemble, cfg, tables):
    if ensemble.empty:
        return
    predroot = tables / 'ensemble_predictions'
    rows = []
    for target in cfg['experiments']['lodo_targets']:
        erow = ensemble[(ensemble.protocol == 'lodo') & (ensemble.target_or_setting == target) & (ensemble.model == 'unimetricnet')]
        p = predroot / f'lodo__{target}__unimetricnet.csv'
        if erow.empty or not p.exists():
            continue
        z = pd.read_csv(p)
        th = float(erow.iloc[0].threshold)
        if 'category' not in z.columns:
            continue
        for cat, g in z.groupby('category', dropna=False):
            if g.empty:
                continue
            yy = int(round(float(g.y.mean())))
            pred = (g.probability.to_numpy(float) >= th).astype(int)
            error = float(np.mean(pred != g.y.to_numpy(int)))
            rows.append(dict(target=target, category=str(cat), n=int(len(g)), label=yy, mean_probability=float(g.probability.mean()), median_probability=float(g.probability.median()), error_rate=error, threshold=th))
    if rows:
        pd.DataFrame(rows).to_csv(tables / 'Table16_category_error_taxonomy.csv', index=False)


def _submission_gates(out, cfg, tables, abstats, ensemble):
    gates = []
    g = cfg['submission_gates']
    lodo = ensemble[ensemble.protocol.eq('lodo')] if not ensemble.empty else pd.DataFrame()
    if len(lodo):
        p = lodo[lodo.model.eq('unimetricnet')].copy()
        macro = float(p.auroc.mean())
        worst = float(p.auroc.min())
        abn_macro = float(p.abnormality_auroc.mean()) if 'abnormality_auroc' in p else float('nan')
        attr_macro = float(p.copd_attr_auroc.mean()) if 'copd_attr_auroc' in p else float('nan')
        target = float(g['macro_lodo_auroc_min'])
        gates.append(dict(gate='1. Macro strict-LODO ensemble AUROC (COPD)', value=macro, target=f'>={target:.2f}', passed=macro >= target))
        gates.append(dict(gate='2. Worst-target strict-LODO AUROC (COPD)', value=worst, target=f">={float(g['worst_lodo_auroc_min']):.2f}", passed=worst >= float(g['worst_lodo_auroc_min'])))
        gates.append(dict(gate='3. Hierarchical Abnormality Detection macro LODO AUROC', value=abn_macro, target=f">={float(g.get('abnormality_macro_lodo_auroc_min', 0.72)):.2f}", passed=bool(np.isfinite(abn_macro) and abn_macro >= float(g.get('abnormality_macro_lodo_auroc_min', 0.72)))))
        gates.append(dict(gate='4. Conditional COPD Attribution macro LODO AUROC', value=attr_macro, target=f">={float(g.get('copd_attr_macro_lodo_auroc_min', 0.64)):.2f}", passed=bool(np.isfinite(attr_macro) and attr_macro >= float(g.get('copd_attr_macro_lodo_auroc_min', 0.64)))))
        base = lodo[~lodo.model.eq('unimetricnet')].groupby('model').auroc.mean()
        best = float(base.max()) if len(base) else np.nan
        delta = macro - best if np.isfinite(best) else np.nan
        gates.append(dict(gate='5. Macro LODO AUROC gain vs strongest matched baseline (Analysis B)', value=delta, target=f">={float(g['proposed_minus_best_baseline_macro_lodo_min']):.3f}", passed=bool(np.isfinite(delta) and delta >= float(g['proposed_minus_best_baseline_macro_lodo_min']))))
        deltas = []
        for tgt in p.target_or_setting:
            pv = float(p[p.target_or_setting.eq(tgt)].auroc.iloc[0])
            b = lodo[(lodo.target_or_setting.eq(tgt)) & (~lodo.model.eq('unimetricnet'))].auroc
            deltas.append(pv - float(b.max()) if len(b) else np.nan)
        mind = float(np.nanmin(deltas))
        gates.append(dict(gate='6. No unseen database materially worse than strongest baseline', value=mind, target=f">={float(g['no_target_materially_worse_margin']):.2f}", passed=mind >= float(g['no_target_materially_worse_margin'])))
        ms = float(p.sensitivity.mean())
        mp = float(p.specificity.mean())
        wb = float(p.balanced_accuracy.min())
        me = float(p.ece.mean())
        gates += [
            dict(gate='7. Macro LODO sensitivity', value=ms, target=f">={float(g['macro_lodo_sensitivity_min']):.2f}", passed=ms >= float(g['macro_lodo_sensitivity_min'])),
            dict(gate='8. Macro LODO specificity', value=mp, target=f">={float(g['macro_lodo_specificity_min']):.2f}", passed=mp >= float(g['macro_lodo_specificity_min'])),
            dict(gate='9. Worst-target LODO balanced accuracy', value=wb, target=f">={float(g['worst_lodo_balanced_accuracy_min']):.2f}", passed=wb >= float(g['worst_lodo_balanced_accuracy_min'])),
            dict(gate='10. Macro LODO ECE', value=me, target=f"<={float(g['macro_lodo_ece_max']):.2f}", passed=me <= float(g['macro_lodo_ece_max']))
        ]
    fs = Path(out) / 'filter_test' / 'filter_stability.csv'
    if fs.exists():
        z = pd.read_csv(fs)
        p = float(z[z.model.eq('unimetricnet')].mean_probability_range.mean())
        b = float(z[z.model.eq('resnet_spec')].mean_probability_range.mean())
        imp = (b - p) / b if b > 0 else np.nan
        gates.append(dict(gate='11. KAUH B/D/E stability relative improvement vs ResNet', value=imp, target=f">={float(g['filter_stability_relative_improvement_min']):.2f}", passed=bool(np.isfinite(imp) and imp >= float(g['filter_stability_relative_improvement_min']))))
        gates.append(dict(gate='12. KAUH B/D/E absolute mean probability range', value=p, target=f"<={float(g['filter_stability_absolute_range_max']):.2f}", passed=p <= float(g['filter_stability_absolute_range_max'])))
    st = Path(out) / 'stress' / 'external_stress.csv'
    if st.exists():
        z = pd.read_csv(st)
        z = z[z.model.eq('unimetricnet')]
        clean = float(z[z.condition.eq('clean')].auroc.mean())
        stressed = z[~z.condition.eq('clean')].groupby('condition').auroc.mean()
        ret = float((stressed / clean).min()) if clean > 0 and len(stressed) else np.nan
        gates.append(dict(gate='13. Worst acoustic-stress AUROC retention', value=ret, target=f">={float(g['stress_auroc_retention_min']):.2f}", passed=bool(np.isfinite(ret) and ret >= float(g['stress_auroc_retention_min']))))
    if abstats is not None:
        positive = int((abstats.mean_delta_auroc > 0).sum())
        sig = int((abstats.holm_p < 0.05).sum())
        gates.append(dict(gate='14. Analysis C ablations with positive matched AUROC contribution', value=positive, target=f">={int(g['central_ablation_positive_count_min'])}", passed=positive >= int(g['central_ablation_positive_count_min'])))
        gates.append(dict(gate='15. Holm-significant Analysis C mechanisms', value=sig, target=f">={int(g['central_ablation_holm_significant_count_min'])}", passed=sig >= int(g['central_ablation_holm_significant_count_min'])))
    if gates:
        q = pd.DataFrame(gates)
        q.to_csv(tables / 'Table13_submission_evidence_gates.csv', index=False)
        write_json(Path(out) / 'submission_gate_summary.json', dict(passed=int(q.passed.sum()), total=int(len(q)), all_passed=bool(q.passed.all()), gates=q.to_dict('records')))


def _measurement_performance_summary(out, tables, ensemble):
    rows = []
    if ensemble is None or ensemble.empty:
        return
    p = ensemble[(ensemble.protocol == 'lodo') & (ensemble.model == 'unimetricnet')].copy()
    if p.empty:
        return
    row = dict(
        model='UniMETRICNet',
        macro_lodo_auroc=float(p.auroc.mean()),
        worst_lodo_auroc=float(p.auroc.min()),
        abnormality_macro_lodo_auroc=float(p.abnormality_auroc.mean()) if 'abnormality_auroc' in p else float('nan'),
        copd_attr_macro_lodo_auroc=float(p.copd_attr_auroc.mean()) if 'copd_attr_auroc' in p else float('nan'),
        macro_balanced_accuracy=float(p.balanced_accuracy.mean()),
        worst_balanced_accuracy=float(p.balanced_accuracy.min()),
        macro_sensitivity=float(p.sensitivity.mean()),
        macro_specificity=float(p.specificity.mean()),
        macro_ece=float(p.ece.mean())
    )
    cv = tables / 'Table02_internal_cv.csv'
    if cv.exists():
        z = pd.read_csv(cv)
        z = z[z.model.eq('unimetricnet')] if 'model' in z else z
        col = 'robust_source_threshold_auroc' if 'robust_source_threshold_auroc' in z else ('auroc' if 'auroc' in z else None)
        if col and len(z):
            internal = float(z[col].mean())
            row['supportive_internal_cv_auroc'] = internal
            row['cross_domain_discrimination_retention'] = float(row['macro_lodo_auroc'] / internal) if internal > 0 else np.nan
    fs = Path(out) / 'filter_test' / 'filter_stability.csv'
    if fs.exists():
        z = pd.read_csv(fs)
        a = z[z.model.eq('unimetricnet')]
        b = z[z.model.eq('resnet_spec')]
        if len(a):
            row['KAUH_same_patient_probability_range'] = float(a.mean_probability_range.mean())
        if len(a) and len(b):
            br = float(b.mean_probability_range.mean())
            ar = float(a.mean_probability_range.mean())
            row['KAUH_stability_improvement_vs_resnet'] = float((br - ar) / br) if br > 0 else np.nan
    st = Path(out) / 'stress' / 'external_stress.csv'
    if st.exists():
        z = pd.read_csv(st)
        z = z[z.model.eq('unimetricnet')]
        if len(z):
            clean = float(z[z.condition.eq('clean')].auroc.mean())
            zz = z[~z.condition.eq('clean')].groupby('condition').auroc.mean()
            row['worst_stress_auroc_retention'] = float((zz / clean).min()) if clean > 0 and len(zz) else np.nan
    pr = Path(out) / 'profile' / 'computational_profile.csv'
    if pr.exists():
        z = pd.read_csv(pr)
        z = z[z.model.eq('unimetricnet')]
        if len(z):
            row['parameters'] = float(z.parameters.mean())
            row['latency_ms_mean'] = float(z.latency_ms_mean.mean())
            row['checkpoint_mb'] = float(z.checkpoint_mb.mean())
    rows.append(row)
    pd.DataFrame(rows).to_csv(tables / 'Table17_TIM_measurement_performance_summary.csv', index=False)


def _source_evidence_tables(out, cfg, tables, ensemble):
    rows = []
    for target in cfg['experiments']['lodo_targets']:
        for seed in cfg['training']['seeds']:
            p = Path(out) / 'runs' / 'lodo' / target / 'unimetricnet' / f'seed_{seed}' / 'source_evidence_summary.json'
            if not p.exists():
                continue
            try:
                d = json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                continue
            va = d.get('validation_anchor', {})
            vs = d.get('validation_selected', {})
            rows.append(dict(
                target=target, seed=seed,
                strength=d.get('strength', np.nan), blend=d.get('blend', np.nan),
                validation_gain=d.get('validation_gain', np.nan),
                anchor_worst=va.get('worst', np.nan), selected_worst=vs.get('worst', np.nan),
                anchor_mean=va.get('mean', np.nan), selected_mean=vs.get('mean', np.nan),
                anchor_pooled=va.get('pooled', np.nan), selected_pooled=vs.get('pooled', np.nan)
            ))
    if rows:
        pd.DataFrame(rows).to_csv(tables / 'Table18_source_evidence_projection.csv', index=False)
    if ensemble is None or ensemble.empty:
        return
    predroot = tables / 'ensemble_predictions'
    h = []
    for target in cfg['experiments']['lodo_targets']:
        p = predroot / f'lodo__{target}__unimetricnet.csv'
        if not p.exists():
            continue
        z = pd.read_csv(p)
        if z.empty or 'category' not in z:
            continue
        cp = z[z.y.eq(1)].probability
        if cp.empty:
            continue
        cmean = float(cp.mean())
        for cat, g in z[z.y.eq(0)].groupby('category', dropna=False):
            if g.empty:
                continue
            h.append(dict(
                target=target, negative_category=str(cat), n=int(len(g)),
                mean_COPD_probability=cmean, mean_negative_probability=float(g.probability.mean()),
                probability_margin=float(cmean - g.probability.mean())
            ))
    if h:
        pd.DataFrame(h).to_csv(tables / 'Table19_hard_negative_probability_margins.csv', index=False)


def _prismcal_reliability_tables(out, cfg, tables):
    rows = []
    rel = []
    for target in cfg['experiments']['lodo_targets']:
        for seed in cfg['training']['seeds']:
            d = Path(out) / 'runs' / 'lodo' / target / 'unimetricnet' / f'seed_{seed}'
            st = d / 'prism_calnet_state.json'
            tp = d / 'test_predictions.csv'
            if st.exists():
                try:
                    x = json.loads(st.read_text(encoding='utf-8'))
                    rows.append(dict(
                        target=target, seed=seed, active=x.get('active', False), reason=x.get('reason', ''),
                        validation_gain=x.get('validation_gain', np.nan), base_score=x.get('base_score', np.nan),
                        selected_score=x.get('selected_score', np.nan), max_origin_auc_drop=x.get('max_origin_auc_drop', np.nan),
                        l2=x.get('l2', np.nan)
                    ))
                except Exception:
                    pass
            if tp.exists():
                z = pd.read_csv(tp)
                if 'measurement_reliability' in z:
                    for c, g in z.groupby('y'):
                        rel.append(dict(
                            target=target, seed=seed, y=int(c), n=len(g),
                            reliability_mean=float(g.measurement_reliability.mean()),
                            reliability_std=float(g.measurement_reliability.std(ddof=1)) if len(g) > 1 else 0.0,
                            uncertainty_mean=float(g.uncertainty.mean()) if 'uncertainty' in g else np.nan
                        ))
    if rows:
        pd.DataFrame(rows).to_csv(tables / 'Table20_PRISMCal_source_only_calibration.csv', index=False)
    if rel:
        pd.DataFrame(rel).to_csv(tables / 'Table21_measurement_reliability_summary.csv', index=False)


def summarize(cfg):
    from .validation import validate_results
    is_quick = bool(cfg.get('project', {}).get('quick_mode', False)) or bool(cfg.get('project', {}).get('synthetic', False))
    if not is_quick:
        validate_results(cfg, strict=True)
    out = Path(cfg['project']['output_dir'])
    tables = ensure_dir(out / 'tables')
    summary = {}
    mapping = [
        ('cv', 'Table02_internal_cv.csv'),
        ('lodo', 'Table03_primary_LODO.csv'),
        ('crossdb', 'Table04_pairwise_transfer.csv'),
        ('external', 'Table05_secondary_locked_external_BRACETS.csv'),
        ('ablations', 'Table07_ablation_runs.csv'),
    ]
    for protocol, name in mapping:
        df = _collect(out, protocol)
        if len(df):
            df.to_csv(tables / name, index=False)
            summary[protocol] = len(df)
            metrics = [c for c in df if c.startswith('robust_source_threshold_') and pd.api.types.is_numeric_dtype(df[c])]
            groupcols = [c for c in ['target_or_setting', 'model'] if c in df]
            if groupcols and metrics:
                df.groupby(groupcols)[metrics].agg(['mean', 'std']).to_csv(tables / name.replace('.csv', '_summary.csv'))
    for src, name in [
        (out / 'filter_test' / 'filter_stability.csv', 'Table08_KAUH_filter_intervention.csv'),
        (out / 'stress' / 'external_stress.csv', 'Table11_external_stress.csv'),
        (out / 'profile' / 'computational_profile.csv', 'Table14_computational_profile.csv'),
    ]:
        if src.exists():
            pd.read_csv(src).to_csv(tables / name, index=False)
    ensemble = _build_ensembles(out, cfg, tables)
    _lodo_ci_ensemble(ensemble, cfg, tables)
    _external_ci_ensemble(ensemble, cfg, tables)
    _domain_tables(out, tables)
    _primary_paired_ensemble(ensemble, cfg, tables)
    _error_taxonomy(ensemble, cfg, tables)
    _measurement_performance_summary(out, tables, ensemble)
    _source_evidence_tables(out, cfg, tables, ensemble)
    _prismcal_reliability_tables(out, cfg, tables)
    abstats = _ablation_stats(out, tables)
    _submission_gates(out, cfg, tables, abstats, ensemble)
    write_json(out / 'final_summary.json', summary)
    if not is_quick:
        try:
            from .fast_upgrade import run_fast_upgrade
            run_fast_upgrade(cfg)
        except Exception as exc:
            print(f'Note: fast_upgrade skipped during partial summarize ({exc})')
    print('Summary tables generated at:', tables)
    return summary
