from __future__ import annotations
"""CPU-only representation diagnostics and t-SNE (no torch import by design)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import domain_probe, conditional_domain_probe, pairwise_domain_geometry, domain_chance_bacc
from .utils import write_json


def _safe_silhouette(x,labels):
    try:
        from sklearn.metrics import silhouette_score
        labels=np.asarray(labels)
        if len(x)<4 or len(np.unique(labels))<2 or min(np.bincount(pd.factorize(labels)[0]))<2: return float('nan')
        return float(silhouette_score(np.asarray(x,float),labels,metric='euclidean'))
    except Exception: return float('nan')


def _write_tsne(run_dir,cal,test,cfg):
    try:
        from sklearn.manifold import TSNE
        xs=[]; rows=[]
        for split,z in [('source_cal',cal),('unseen_test',test)]:
            if z is None: continue
            x=np.asarray(z['disease'],dtype=np.float64); xs.append(x)
            for i in range(len(x)):
                rows.append(dict(split=split,person_id=str(z['person_id'][i]),y=int(z['y'][i]),origin_dataset=str(z['origin'][i]),logical_dataset=str(z['logical_dataset'][i])))
        if not xs: return
        X=np.concatenate(xs,axis=0); n=len(X)
        if n<5: return
        perplex=min(float(cfg['evaluation'].get('tsne_perplexity',20)),max(2.0,(n-1)/3))
        xy=TSNE(n_components=2,perplexity=perplex,learning_rate='auto',init='pca',random_state=int(cfg['evaluation'].get('tsne_seed',2026)),max_iter=1200).fit_transform(X)
        q=pd.DataFrame(rows); q['tsne_1']=xy[:,0]; q['tsne_2']=xy[:,1]; q.to_csv(run_dir/'tsne_coordinates.csv',index=False)
    except Exception as exc:
        (run_dir/'tsne_warning.txt').write_text(str(exc),encoding='utf-8')


def _load_npz(path):
    if not path.exists(): return None
    z=np.load(path,allow_pickle=False)
    return {k:z[k] for k in z.files}


def analyze_one(run_dir: Path, cfg: dict) -> bool:
    test=_load_npz(run_dir/'test_embeddings.npz'); cal=_load_npz(run_dir/'calibration_embeddings.npz'); mp=run_dir/'metrics.json'
    if test is None or not mp.exists(): return False
    metrics=json.loads(mp.read_text(encoding='utf-8')); folds=int(cfg['evaluation'].get('domain_probe_folds',5)); steps=200
    # Domain leakage must be measured on SOURCE calibration embeddings.  A single
    # held-out target database has only one domain and cannot support a domain probe.
    if cal is not None:
        x=np.asarray(cal['disease'],dtype=np.float64); origin=np.asarray(cal['origin']).astype(str); y=np.asarray(cal['y'],dtype=int)
        metrics['disease_embedding_domain_probe_bacc']=domain_probe(x,origin,folds=folds,steps=steps)
        metrics['disease_embedding_conditional_domain_probe_bacc']=conditional_domain_probe(x,origin,y,folds=folds,steps=steps)
        metrics['domain_probe_chance_bacc']=domain_chance_bacc(origin)
        metrics['source_embedding_class_silhouette']=_safe_silhouette(x,y)
        metrics['source_embedding_domain_silhouette']=_safe_silhouette(x,origin)
        pairwise_domain_geometry(x,origin).to_csv(run_dir/'disease_domain_geometry.csv',index=False)
    else:
        metrics['disease_embedding_domain_probe_bacc']=None; metrics['disease_embedding_conditional_domain_probe_bacc']=None; metrics['domain_probe_chance_bacc']=None
    xt=np.asarray(test['disease'],dtype=np.float64); yt=np.asarray(test['y'],dtype=int)
    metrics['unseen_embedding_class_silhouette']=_safe_silhouette(xt,yt)
    metrics['embedding_analysis_status']='complete_cpu_isolated'
    _write_tsne(run_dir,cal,test,cfg); write_json(mp,metrics); return True


def analyze_all(cfg: dict) -> dict:
    runs=Path(cfg['project']['output_dir'])/'runs'; done=skipped=0
    if not runs.exists(): print('No run directory exists yet; embedding analysis skipped.'); return {'processed':0,'skipped':0}
    for mp in runs.rglob('metrics.json'):
        try:
            if analyze_one(mp.parent,cfg): done+=1
            else: skipped+=1
        except Exception as exc: raise RuntimeError(f'Embedding analysis failed for {mp.parent}: {exc}') from exc
    print(f'CPU-isolated embedding analysis complete: processed={done}, skipped={skipped}'); return {'processed':done,'skipped':skipped}
