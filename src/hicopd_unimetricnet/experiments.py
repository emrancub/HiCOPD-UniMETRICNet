from __future__ import annotations
import json, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .features import load_feature_frame
from .splits import patient_cv, patient_table, subset, train_val_cal_split, assert_disjoint
from .train import fit_model, fit_external, device_for, predict
from .models import build_model, estimate_patient_macs
from .dataset import make_loader
from .utils import ensure_dir


def clean_main(cfg, condition='clean'):
    f = load_feature_frame(cfg, condition)
    return f[f.main_role.astype(int).eq(1)].copy()


def three_database_frame(cfg, condition='clean'):
    f = clean_main(cfg, condition)
    return f[f.logical_dataset.isin(cfg['data']['logical_datasets'])].copy()


def dataset_frame(cfg, name, condition='clean'):
    f = clean_main(cfg, condition)
    return f[f.logical_dataset.eq(name)].copy()


def run_cv(cfg, models=None):
    frame = three_database_frame(cfg)
    models = models or cfg['experiments']['core_models']
    for seed in cfg['training']['seeds']:
        for sp in patient_cv(frame, int(cfg['training']['folds']), int(seed)):
            train_outer = subset(frame, sp['train_people'])
            test = subset(frame, sp['test_people'])
            p = patient_table(train_outer)
            tr, va, ca = train_val_cal_split(p, float(cfg['training']['validation_fraction']), float(cfg['training']['calibration_fraction']), int(seed) + sp['fold'])
            assert_disjoint(tr, va, ca, sp['test_people'])
            for model in models:
                out = Path(cfg['project']['output_dir']) / 'runs' / 'cv' / model / f'seed_{seed}' / f'fold_{sp["fold"]}'
                fit_model(subset(frame, tr), subset(frame, va), subset(frame, ca), test, cfg, model, int(seed), out, 'Supportive pooled patient-disjoint CV across the three logical databases')


def run_lodo(cfg, models=None):
    frame = three_database_frame(cfg)
    models = models or cfg['experiments']['core_models']
    for target_name in cfg['experiments']['lodo_targets']:
        source = frame[~frame.logical_dataset.eq(target_name)].copy()
        target = frame[frame.logical_dataset.eq(target_name)].copy()
        if source.logical_dataset.nunique() < 2:
            raise RuntimeError(f'LODO {target_name}: need two source logical databases')
        for seed in cfg['training']['seeds']:
            for model in models:
                out = Path(cfg['project']['output_dir']) / 'runs' / 'lodo' / target_name / model / f'seed_{seed}'
                fit_external(source, target, cfg, model, int(seed), out, protocol_note=f'Strict whole-database LODO: unseen target={target_name}; sources={sorted(source.logical_dataset.unique())}; target labels unused for fitting/calibration/threshold')


def run_crossdb(cfg, models=None):
    frame = three_database_frame(cfg)
    models = models or ['spectral_cnn', 'resnet_spec', 'mobilenet_spec', 'unimetricnet']
    for train_db, test_db in [('ICBHI', 'HYBRID'), ('HYBRID', 'ICBHI')]:
        source = frame[frame.logical_dataset.eq(train_db)]
        target = frame[frame.logical_dataset.eq(test_db)]
        for seed in cfg['training']['seeds']:
            for model in models:
                out = Path(cfg['project']['output_dir']) / 'runs' / 'crossdb' / f'{train_db}_to_{test_db}' / model / f'seed_{seed}'
                fit_external(source, target, cfg, model, int(seed), out, protocol_note=f'Strict transfer {train_db}->{test_db}; complete target held out')


def run_external(cfg, models=None):
    frame = three_database_frame(cfg)
    source = frame[frame.logical_dataset.isin(cfg['experiments']['locked_external_source_datasets'])].copy()
    target = frame[frame.logical_dataset.eq(cfg['experiments']['locked_external_target'])].copy()
    models = models or cfg['experiments']['core_models']
    for seed in cfg['training']['seeds']:
        for model in models:
            out = Path(cfg['project']['output_dir']) / 'runs' / 'external' / cfg['experiments']['locked_external_target'] / model / f'seed_{seed}'
            lodo = Path(cfg['project']['output_dir']) / 'runs' / 'lodo' / cfg['experiments']['locked_external_target'] / model / f'seed_{seed}'
            if lodo.exists() and (lodo / 'best_model.pt').exists():
                if out.exists():
                    shutil.rmtree(out)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(lodo, out)
                m = json.loads((out / 'metrics.json').read_text())
                m['protocol_note'] = 'Secondary locked BRACETS deployment analysis copied from strict BRACETS LODO without refitting'
                (out / 'metrics.json').write_text(json.dumps(m, indent=2), encoding='utf-8')
            else:
                fit_external(source, target, cfg, model, int(seed), out, protocol_note='Secondary locked BRACETS: sources ICBHI+HYBRID; no BRACETS labels used during development')


def run_hybrid_compare(cfg, models=None):
    return run_crossdb(cfg, models)


def run_ablations(cfg):
    frame = three_database_frame(cfg)
    variants = ['unimetricnet'] + list(cfg['experiments']['ablations'])
    for target_name in cfg['experiments']['lodo_targets']:
        source = frame[~frame.logical_dataset.eq(target_name)].copy()
        target = frame[frame.logical_dataset.eq(target_name)].copy()
        for seed in cfg['training']['seeds']:
            for v in variants:
                out = Path(cfg['project']['output_dir']) / 'runs' / 'ablations' / target_name / v / f'seed_{seed}'
                lodo_u = Path(cfg['project']['output_dir']) / 'runs' / 'lodo' / target_name / 'unimetricnet' / f'seed_{seed}'
                if v == 'unimetricnet' and lodo_u.exists() and (lodo_u / 'best_model.pt').exists():
                    if out.exists():
                        shutil.rmtree(out)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(lodo_u, out)
                else:
                    fit_external(source, target, cfg, v, int(seed), out, protocol_note=f'Controlled strict-LODO ablation target={target_name}')


def _load_checkpoint(path, cfg):
    state = torch.load(path, map_location='cpu', weights_only=False)
    device = device_for(cfg)
    model = build_model(state['config'], state['variant']).to(device)
    model.load_state_dict(state['state_dict'])
    model.posthoc_fusion_weights = state.get('posthoc_fusion_weights', {})
    ps = {'median': np.asarray(state['phys_median'], np.float32), 'scale': np.asarray(state['phys_scale'], np.float32)}
    qs = {'median': np.asarray(state['quality_median'], np.float32), 'scale': np.asarray(state['quality_scale'], np.float32)}
    return model, state, device, ps, qs


def _icc_2_1(mat: np.ndarray) -> tuple[float, float, float]:
    n, k = mat.shape
    if n < 2 or k < 2:
        return float('nan'), float('nan'), float('nan')
    grand = float(np.mean(mat))
    row_means = np.mean(mat, axis=1, keepdims=True)
    col_means = np.mean(mat, axis=0, keepdims=True)
    ssr = float(k * np.sum((row_means - grand) ** 2))
    ssc = float(n * np.sum((col_means - grand) ** 2))
    sse = float(np.sum((mat - row_means - col_means + grand) ** 2))
    msr = ssr / max(1, n - 1)
    msc = ssc / max(1, k - 1)
    mse = sse / max(1, (n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + (k * max(0.0, msc - mse)) / max(1, n)
    icc = float(max(0.0, (msr - mse) / denom)) if denom > 1e-12 else 0.0
    wsd = float(np.sqrt(max(0.0, mse)))
    rc95 = float(1.96 * np.sqrt(2.0) * wsd)
    return icc, wsd, rc95


def run_filter_test(cfg):
    """Same-patient KAUH B/D/E intervention across all core models that never saw HYBRID."""
    clean = load_feature_frame(cfg, 'clean')
    filt = clean[(clean.origin_dataset.eq('KAUH')) & clean.filter_intervention_role.astype(int).eq(1)].copy()
    outroot = ensure_dir(Path(cfg['project']['output_dir']) / 'filter_test')
    rows = []
    models = list(cfg['experiments']['core_models'])
    for seed in cfg['training']['seeds']:
        for model_name in models:
            ck = Path(cfg['project']['output_dir']) / 'runs' / 'lodo' / 'HYBRID' / model_name / f'seed_{seed}' / 'best_model.pt'
            if not ck.exists():
                continue
            model, state, device, ps, qs = _load_checkpoint(ck, cfg)
            per = []
            for mode in ['B', 'D', 'E']:
                dl = make_loader(filt, state['domain_map'], cfg, False, int(seed), role='filter', filter_mode=mode, phys_scaler=ps, quality_scaler=qs)
                pred, _, _ = predict(model, dl, device)
                pred['filter_mode'] = mode
                per.append(pred)
            allp = pd.concat(per, ignore_index=True)
            allp.to_csv(outroot / f'predictions_{model_name}_seed_{seed}.csv', index=False)
            piv = allp.pivot_table(index='person_id', columns='filter_mode', values='probability', aggfunc='mean').dropna()
            piv_abn = allp.pivot_table(index='person_id', columns='filter_mode', values='abnormality_probability', aggfunc='mean').dropna() if 'abnormality_probability' in allp else piv
            if len(piv):
                mat = piv.to_numpy(float)
                prange = piv.max(axis=1) - piv.min(axis=1)
                arange = (piv_abn.max(axis=1) - piv_abn.min(axis=1)) if len(piv_abn) else prange
                icc, wsd, rc95 = _icc_2_1(mat)
                rows.append(dict(
                    model=model_name, seed=seed, patients=len(piv),
                    mean_probability_range=float(prange.mean()),
                    median_probability_range=float(prange.median()),
                    p90_probability_range=float(np.quantile(prange, 0.90)),
                    max_probability_range=float(prange.max()),
                    mean_abnormality_range=float(arange.mean()),
                    icc_2_1=icc, within_subject_sd=wsd, repeatability_coefficient_95=rc95
                ))
    pd.DataFrame(rows).to_csv(outroot / 'filter_stability.csv', index=False)
    return rows


def run_stress(cfg):
    outroot = ensure_dir(Path(cfg['project']['output_dir']) / 'stress')
    rows = []
    models = list(cfg['experiments']['core_models'])
    for condition in ['clean'] + list(cfg['experiments']['stress_conditions']):
        t = dataset_frame(cfg, 'BRACETS', condition)
        for seed in cfg['training']['seeds']:
            for model_name in models:
                ck = Path(cfg['project']['output_dir']) / 'runs' / 'external' / 'BRACETS' / model_name / f'seed_{seed}' / 'best_model.pt'
                if not ck.exists():
                    continue
                model, state, device, ps, qs = _load_checkpoint(ck, cfg)
                dl = make_loader(t, state['domain_map'], cfg, False, int(seed), phys_scaler=ps, quality_scaler=qs)
                pred, _, _ = predict(model, dl, device)
                from .calibration import apply_temperature
                mj = json.loads((ck.parent / 'metrics.json').read_text())
                pred['probability'] = apply_temperature(pred.probability, mj['temperature'])
                from .metrics import binary_metrics
                th = float(mj.get('style_threshold', mj['disease_threshold']))
                m = binary_metrics(pred.y, pred.probability, th, int(cfg['evaluation']['calibration_bins']))
                rows.append(dict(
                    model=model_name, condition=condition, seed=seed, threshold=th,
                    aligned_bacc=float(m.get('balanced_accuracy', np.nan)),
                    aligned_sens=float(m.get('sensitivity', np.nan)),
                    aligned_spec=float(m.get('specificity', np.nan)),
                    **m
                ))
    pd.DataFrame(rows).to_csv(outroot / 'external_stress.csv', index=False)
    return rows


def run_profile(cfg, models=None):
    models = models or cfg['experiments']['core_models']
    frame = dataset_frame(cfg, 'ICBHI', 'clean')
    outroot = ensure_dir(Path(cfg['project']['output_dir']) / 'profile')
    rows = []
    for seed in cfg['training']['seeds']:
        for name in models:
            ck = Path(cfg['project']['output_dir']) / 'runs' / 'external' / 'BRACETS' / name / f'seed_{seed}' / 'best_model.pt'
            if not ck.exists():
                continue
            model, state, device, ps, qs = _load_checkpoint(ck, cfg)
            one = frame[frame.person_id.eq(frame.person_id.iloc[0])]
            loader = make_loader(one, state['domain_map'], cfg, False, int(seed), phys_scaler=ps, quality_scaler=qs)
            b = next(iter(loader))
            x = b['x'].to(device)
            ph = b['phys'].to(device)
            qu = b['quality'].to(device)
            mask = b['mask'].to(device)
            model.eval()
            macs = estimate_patient_macs(model, x, ph, qu, mask)
            params = sum(p.numel() for p in model.parameters())
            with torch.inference_mode():
                for _ in range(5):
                    model(x, ph, qu, mask)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                times = []
                for _ in range(20):
                    t0 = time.perf_counter()
                    model(x, ph, qu, mask)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000)
                peak = float(torch.cuda.max_memory_allocated(device) / 1024 ** 2) if device.type == 'cuda' else float('nan')
            rows.append(dict(
                model=name, seed=seed, device=str(device),
                patient_bag_segments=int(mask.sum().item()),
                parameters=params,
                parameters_millions=float(params / 1e6),
                macs=int(macs),
                gmacs=float(macs / 1e9),
                gflops=float(2.0 * macs / 1e9),
                latency_ms_mean=float(np.mean(times)),
                latency_ms_std=float(np.std(times, ddof=1)),
                peak_allocated_mb=peak,
                checkpoint_mb=float(ck.stat().st_size / 1024 ** 2)
            ))
    pd.DataFrame(rows).to_csv(outroot / 'computational_profile.csv', index=False)
    return rows
