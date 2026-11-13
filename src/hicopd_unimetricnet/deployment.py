from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from .models import build_model
from .dataset import make_loader
from .features import load_feature_frame
from .utils import ensure_dir


def _bench(model,x,ph,q,mask,n=30):
    model.eval(); times=[]
    with torch.inference_mode():
        for _ in range(5): model(x,ph,q,mask)
        for _ in range(n):
            t=time.perf_counter(); model(x,ph,q,mask); times.append((time.perf_counter()-t)*1000)
    return float(np.mean(times)),float(np.std(times,ddof=1))


def run_deployment(cfg):
    out=ensure_dir(Path(cfg['project']['output_dir'])/'deployment'); seed=int(cfg['training']['seeds'][0]); ck=Path(cfg['project']['output_dir'])/'runs'/'external'/'BRACETS'/'unimetricnet'/f'seed_{seed}'/'best_model.pt'
    if not ck.exists(): raise RuntimeError('Run the primary/external training before deployment benchmarking')
    state=torch.load(ck,map_location='cpu',weights_only=False); model=build_model(state['config'],state['variant']).cpu(); model.load_state_dict(state['state_dict']); ps={'median':np.asarray(state['phys_median'],np.float32),'scale':np.asarray(state['phys_scale'],np.float32)}; qs={'median':np.asarray(state['quality_median'],np.float32),'scale':np.asarray(state['quality_scale'],np.float32)}
    frame=load_feature_frame(cfg,'clean'); frame=frame[(frame.main_role.astype(int).eq(1)) & frame.logical_dataset.eq('ICBHI')]; one=frame[frame.person_id.eq(frame.person_id.iloc[0])]; loader=make_loader(one,state['domain_map'],cfg,False,seed,phys_scaler=ps,quality_scaler=qs); b=next(iter(loader)); x=b['x'].cpu(); ph=b['phys'].cpu(); q=b['quality'].cpu(); mask=b['mask'].cpu()
    fp_m,fp_s=_bench(model,x,ph,q,mask)
    qmodel=torch.ao.quantization.quantize_dynamic(model,{nn.Linear,nn.GRU},dtype=torch.qint8); int_m,int_s=_bench(qmodel,x,ph,q,mask)
    fp_path=out/'unimetricnet_fp32_state.pt'; int_path=out/'unimetricnet_dynamic_int8_model.pt'; torch.save(model.state_dict(),fp_path); torch.save(qmodel,int_path)
    rows=[dict(model='UniMETRICNet-FP32',latency_ms_mean=fp_m,latency_ms_std=fp_s,file_mb=fp_path.stat().st_size/1024**2),dict(model='UniMETRICNet-dynamic-INT8',latency_ms_mean=int_m,latency_ms_std=int_s,file_mb=int_path.stat().st_size/1024**2)]
    pd.DataFrame(rows).to_csv(out/'deployment_profile.csv',index=False); print(pd.DataFrame(rows).to_string(index=False)); return rows
