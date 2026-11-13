from __future__ import annotations
import argparse, json, sys, subprocess, os
from pathlib import Path
import pandas as pd
from .config import load_config
from .utils import ensure_dir, write_json, project_root


def _torch_info(cfg, gpu_only=False):
    import torch
    report = dict(
        python=sys.version, executable=sys.executable, torch=torch.__version__,
        cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        paths={k: v for k, v in cfg['data'].items() if k.endswith('_root')}
    )
    if str(cfg['training']['device']).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by config but unavailable')
    if torch.cuda.is_available():
        x = torch.randn(256, 256, device='cuda')
        _ = (x @ x.T).mean()
        torch.cuda.synchronize()
    if not gpu_only:
        missing = [f'{k}: {v}' for k, v in report['paths'].items() if not Path(v).exists()]
        if missing and not cfg['project'].get('synthetic', False):
            raise RuntimeError('Missing data roots:\n' + '\n'.join(missing))
    ensure_dir(cfg['project']['output_dir'])
    write_json(Path(cfg['project']['output_dir']) / 'doctor.json', report)
    print(json.dumps(report, indent=2))
    return report


def doctor(cfg, gpu_only=False):
    return _torch_info(cfg, gpu_only)


def smoke(cfg):
    import shutil
    from .synthetic import make_fixture
    base = project_root() / 'software_smoke'
    if base.exists():
        shutil.rmtree(base)
    config_path = make_fixture(base, cfg)
    py = sys.executable
    env = os.environ.copy()
    stages = [
        ['audit'], ['features', '--condition', 'clean'],
        ['lodo', '--models', 'spectral_cnn', 'crnn', 'unimetricnet'],
        ['external', '--models', 'spectral_cnn', 'crnn', 'unimetricnet'],
        ['embedding-analysis'], ['explainability'], ['summarize'], ['figures']
    ]
    for a in stages:
        rc = subprocess.run([py, str(project_root() / 'run.py'), *a, '--config', str(config_path), '--quick'], cwd=project_root(), env=env).returncode
        if rc != 0:
            raise SystemExit(rc)
    print('Synthetic end-to-end smoke test passed. Synthetic metrics are not research evidence.')


def main(argv=None):
    choices = [
        'doctor', 'gpu-check', 'audit', 'features', 'cv', 'lodo', 'crossdb', 'external',
        'hybrid-compare', 'ablations', 'filter-test', 'stress', 'profile',
        'embedding-analysis', 'explainability', 'validate-results', 'deployment',
        'summarize', 'figures', 'gates', 'fast-upgrade', 'all', 'smoke'
    ]
    p = argparse.ArgumentParser(description='HiCOPD-UniMETRICNet v11.2: Quality-Gated Multi-Expert Hierarchical Respiratory Acoustic Measurement')
    p.add_argument('command', choices=choices)
    p.add_argument('--config', default='configs/main.yaml')
    p.add_argument('--condition', default='clean')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--models', nargs='+')
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.quick)
    if args.command == 'doctor':
        return doctor(cfg)
    if args.command == 'gpu-check':
        return doctor(cfg, True)
    if args.command == 'smoke':
        return smoke(cfg)
    if args.command == 'audit':
        from .audit import build_audit
        return build_audit(cfg)
    if args.command == 'features':
        from .features import precompute
        return precompute(cfg, args.condition)
    if args.command in {'cv', 'lodo', 'crossdb', 'external', 'hybrid-compare', 'ablations', 'filter-test', 'stress', 'profile'}:
        from . import experiments as e
        fn = {
            'cv': e.run_cv, 'lodo': e.run_lodo, 'crossdb': e.run_crossdb, 'external': e.run_external,
            'hybrid-compare': e.run_hybrid_compare, 'ablations': e.run_ablations,
            'filter-test': e.run_filter_test, 'stress': e.run_stress, 'profile': e.run_profile
        }[args.command]
        if args.command in {'cv', 'lodo', 'crossdb', 'external', 'hybrid-compare', 'profile'}:
            return fn(cfg, args.models)
        return fn(cfg)
    if args.command == 'embedding-analysis':
        from .embedding_analysis import analyze_all
        return analyze_all(cfg)
    if args.command == 'explainability':
        from .explainability import generate_explanations
        return generate_explanations(cfg)
    if args.command == 'validate-results':
        from .validation import validate_results
        return validate_results(cfg, strict=not bool(cfg['project'].get('quick_mode', False)))
    if args.command == 'deployment':
        from .deployment import run_deployment
        return run_deployment(cfg)
    if args.command == 'summarize':
        from .reporting import summarize
        return summarize(cfg)
    if args.command == 'figures':
        from .figures import generate_all
        return generate_all(cfg)
    if args.command == 'fast-upgrade':
        from .fast_upgrade import run_fast_upgrade
        return run_fast_upgrade(cfg)
    if args.command == 'gates':
        gates_csv = Path(cfg['project']['output_dir']) / 'tables' / 'Table13_submission_evidence_gates.csv'
        if not gates_csv.exists():
            from .reporting import summarize
            summarize(cfg)
        if gates_csv.exists():
            df_g = pd.read_csv(gates_csv)
            print(df_g.to_string(index=False))
            return df_g.to_dict('records')
        return []
