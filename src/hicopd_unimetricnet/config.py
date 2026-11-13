from __future__ import annotations
import copy
from pathlib import Path
import yaml
from .utils import project_root


def load_config(path: str | Path, quick: bool=False) -> dict:
    path=Path(path)
    if not path.is_absolute(): path=project_root()/path
    cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
    cfg=copy.deepcopy(cfg)
    root=project_root()
    if quick:
        q=cfg.get('quick',{})
        for k in ('seeds','folds','epochs','patience','train_segments_per_patient','eval_segments_per_patient','batch_size_patients'):
            if k in q: cfg['training'][k]=q[k]
        if 'lodo_targets' in q: cfg['experiments']['lodo_targets']=q['lodo_targets']
        suffix=str(q.get('output_suffix','quick'))
        cfg['project']['output_dir']=str(cfg['project']['output_dir'])+'_'+suffix
        cfg['project']['cache_dir']=str(cfg['project']['cache_dir'])+'_'+suffix
        cfg['project']['quick_mode']=True
    else:
        cfg['project']['quick_mode']=False
    for key in ('output_dir','cache_dir'):
        p=Path(cfg['project'][key])
        if not p.is_absolute(): cfg['project'][key]=str((root/p).resolve())
    cfg['config_path']=str(path.resolve())
    return cfg
