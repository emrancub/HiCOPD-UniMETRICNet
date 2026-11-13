from __future__ import annotations
"""Post-hoc interpretability utilities.

This module is evaluation-only. It never changes training, model selection,
calibration, or the operating threshold. Ground-truth labels are used only to
choose representative examples after all predictions have been frozen.

Windows/cuDNN note
------------------
cuDNN RNNs (GRU/LSTM) do not keep the reserve buffer required for backward
when they are run in evaluation mode.  Saliency needs gradients, while the
rest of the network must remain in deterministic evaluation mode.  Therefore
only RNNBase modules are temporarily switched to training mode (with recurrent
dropout forced to zero) during the explanation forward/backward pass.  If a
platform still raises the cuDNN reserve-buffer error, the code retries the
single explanation pass with cuDNN disabled, which uses PyTorch's native RNN
implementation.  Model weights are never updated.
"""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .experiments import dataset_frame,_load_checkpoint
from .dataset import make_loader
from .utils import ensure_dir


def _single_patient_loader(frame,person,state,cfg,seed,ps,qs):
    one=frame[frame.person_id.astype(str).eq(str(person))].copy()
    return make_loader(one,state['domain_map'],cfg,False,int(seed),phys_scaler=ps,quality_scaler=qs)


@contextmanager
def _rnn_backward_ready(model):
    """Enable cuDNN RNN backward for saliency without enabling model dropout.

    ``model.eval()`` is intentionally kept for every non-RNN module.  Only
    GRU/LSTM/RNN modules are temporarily marked as training because cuDNN
    requires a training-mode forward pass before RNN backward.  Recurrent
    dropout is temporarily set to 0 so the explanation remains deterministic.
    All states are restored even if backward raises.
    """
    saved=[]
    for module in model.modules():
        if isinstance(module, nn.RNNBase):
            saved.append((module, bool(module.training), float(getattr(module,'dropout',0.0))))
            module.train(True)
            if hasattr(module,'dropout'):
                module.dropout=0.0
    try:
        yield
    finally:
        for module,was_training,dropout in saved:
            if hasattr(module,'dropout'):
                module.dropout=dropout
            module.train(was_training)


def _forward_backward(model,batch,device,disable_cudnn=False):
    """Run exactly one gradient explanation pass and return model output + grad."""
    x=batch['x'].to(device).detach().clone().requires_grad_(True)
    ph=batch['phys'].to(device)
    qu=batch['quality'].to(device)
    mask=batch['mask'].to(device)
    model.zero_grad(set_to_none=True)

    cudnn_ctx = torch.backends.cudnn.flags(enabled=False) if disable_cudnn else nullcontext()
    with torch.enable_grad():
        with cudnn_ctx:
            with _rnn_backward_ready(model):
                out=model(x,ph,qu,mask)
                score=out['logits'][:,1].sum()
                score.backward()
    if x.grad is None:
        raise RuntimeError('Explainability backward completed without input gradients.')
    return x,mask,out,x.grad.detach()


def _explain_patient(model,batch,device):
    # Keep BatchNorm/Dropout/routers in deterministic inference mode.
    model.eval()
    try:
        x,mask,out,grad=_forward_backward(model,batch,device,disable_cudnn=False)
    except RuntimeError as exc:
        msg=str(exc).lower()
        if 'cudnn rnn backward' not in msg and 'rnn backward can only be called in training mode' not in msg:
            raise
        # Defensive fallback for platform-specific cuDNN behaviour.  This is
        # explanation-only and processes only a few representative patients.
        x,mask,out,grad=_forward_backward(model,batch,device,disable_cudnn=True)

    att=None
    if 'segment_attention' in out and 'raw_resnet' in out['segment_attention']:
        att=out['segment_attention']['raw_resnet'][0].detach()
    elif 'segment_attention' in out and len(out['segment_attention']):
        att=next(iter(out['segment_attention'].values()))[0].detach()

    valid=mask[0].nonzero(as_tuple=False).flatten()
    if len(valid)==0:
        raise RuntimeError('Explainability received a patient bag with no valid segments.')
    idx=int(valid[torch.argmax(att[valid])].item()) if att is not None else int(valid[0].item())

    spec=x.detach()[0,idx,0].float().cpu().numpy()
    sal=(grad[0,idx,0].abs()*x.detach()[0,idx,0].abs()).float().cpu().numpy()
    sal=sal/(sal.max()+1e-8)
    prob=float(out['logits'].softmax(-1)[0,1].detach().cpu())
    ew=out.get('expert_weights')
    ew=ew[0].detach().cpu().numpy() if ew is not None else np.asarray([])
    return spec,sal,prob,idx,ew


def generate_explanations(cfg):
    outroot=ensure_dir(Path(cfg['project']['output_dir'])/'explainability')
    rows=[]
    seed=int(cfg['training']['seeds'][0])
    for target in cfg['experiments']['lodo_targets']:
        rd=Path(cfg['project']['output_dir'])/'runs'/'lodo'/target/'unimetricnet'/f'seed_{seed}'
        ck=rd/'best_model.pt'; pp=rd/'test_predictions.csv'
        if not ck.exists() or not pp.exists():
            continue
        pred=pd.read_csv(pp)
        frame=dataset_frame(cfg,target,'clean')
        model,state,device,ps,qs=_load_checkpoint(ck,cfg)
        for y,label in [(0,'non_copd'),(1,'copd')]:
            q=pred[pred.y.eq(y)].copy()
            q['correct_side']=np.where(y==1,q.probability,1-q.probability)
            q=q.sort_values('correct_side',ascending=False)
            if q.empty:
                continue
            person=str(q.iloc[0].person_id)
            loader=_single_patient_loader(frame,person,state,cfg,seed,ps,qs)
            batch=next(iter(loader))
            spec,sal,prob,idx,ew=_explain_patient(model,batch,device)
            dest=outroot/f'{target}_{label}.npz'
            np.savez_compressed(dest,spectrogram=spec,saliency=sal,probability=prob,segment_index=idx,expert_weights=ew)
            rows.append(dict(target=target,class_label=label,person_id=person,probability=prob,segment_index=idx,file=dest.name))
    pd.DataFrame(rows).to_csv(outroot/'explanation_index.csv',index=False)
    print('Explainability assets:',outroot)
    return rows
