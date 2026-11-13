from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import roc_auc_np


def _normalize(z):
    z=np.asarray(z,np.float64); n=np.linalg.norm(z,axis=1,keepdims=True); return z/np.clip(n,1e-8,None)


def _logit(p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6); return np.log(p/(1-p))


def _sigmoid(x):
    x=np.asarray(x,float); return 1/(1+np.exp(-np.clip(x,-30,30)))


def fit_nuisance_basis(z,frame,max_dim=6,min_count=2):
    """Source-only class-conditional acquisition nuisance basis.

    Within each endpoint class, origin-specific mean deviations are stacked and an SVD
    extracts directions that vary with acquisition after class centering.  No target
    observations or target labels are used.
    """
    z=_normalize(z); y=frame.y.to_numpy(int); d=frame.origin_dataset.astype(str).to_numpy(); vec=[]
    for c in [0,1]:
        cm=y==c
        if cm.sum()<max(2,min_count): continue
        mu=z[cm].mean(0)
        for origin in sorted(np.unique(d[cm])):
            m=cm & (d==origin)
            if m.sum()>=int(min_count): vec.append(z[m].mean(0)-mu)
    if len(vec)<2: return np.zeros((z.shape[1],0),np.float64)
    a=np.stack(vec); a=a-a.mean(0,keepdims=True)
    _,s,vt=np.linalg.svd(a,full_matrices=False)
    k=min(int(max_dim),int((s>1e-7).sum()),vt.shape[0])
    return vt[:k].T.copy() if k>0 else np.zeros((z.shape[1],0),np.float64)


def project(z,basis,strength):
    z=_normalize(z)
    if basis is None or basis.size==0 or float(strength)<=0: return z
    comp=(z@basis)@basis.T
    return _normalize(z-float(strength)*comp)


def fit_taxonomy_prototypes(z,frame):
    z=_normalize(z); tax=frame.taxonomy.to_numpy(int); prot={}
    for t in sorted(np.unique(tax)):
        m=tax==t
        if t>=0 and m.any(): prot[int(t)]=_normalize(z[m].mean(0,keepdims=True))[0]
    return prot


def metric_score(z,prototypes,temperature=.20):
    z=_normalize(z)
    if 0 not in prototypes or not any(t in prototypes for t in [1,2,3,4]): return np.zeros(len(z),float)
    pos=z@prototypes[0]
    neg=[]
    for t in [1,2,3,4]:
        if t in prototypes: neg.append(z@prototypes[t])
    n=np.stack(neg,axis=1)/max(float(temperature),1e-6)
    m=n.max(1,keepdims=True); lse=(m[:,0]+np.log(np.exp(n-m).mean(1)+1e-12))*max(float(temperature),1e-6)
    return pos-lse


def _origin_auc_summary(frame,p):
    y=frame.y.to_numpy(int); d=frame.origin_dataset.astype(str).to_numpy(); vals=[]
    for o in np.unique(d):
        m=d==o
        if len(np.unique(y[m]))==2:
            a=roc_auc_np(y[m],p[m])
            if np.isfinite(a): vals.append(float(a))
    pooled=float(roc_auc_np(y,p)) if len(np.unique(y))==2 else .5
    if not vals: vals=[pooled]
    return dict(worst=float(np.min(vals)),mean=float(np.mean(vals)),pooled=pooled)


def fit_source_evidence(train_emb,train_pred,val_emb,val_pred,cfg,disable_projection=False):
    ecfg=cfg['evaluation'].get('source_evidence_projection',{})
    if not ecfg.get('enabled',True): disable_projection=True
    basis0=fit_nuisance_basis(train_emb,train_pred,int(ecfg.get('max_nuisance_dim',6)),int(ecfg.get('min_origin_class_count',2)))
    strengths=[0.0] if disable_projection else [float(x) for x in ecfg.get('strength_grid',[0,.25,.5,.75,1])]
    blends=[float(x) for x in ecfg.get('blend_grid',[0,.1,.2,.3,.4,.5])]
    anchor_col='expert_prob_anchor' if 'expert_prob_anchor' in val_pred else 'probability'
    anchor_val=val_pred[anchor_col].to_numpy(float); anchor_logit=_logit(anchor_val)
    base=_origin_auc_summary(val_pred,anchor_val); base_score=.62*base['worst']+.28*base['mean']+.10*base['pooled']
    max_drop=float(ecfg.get('max_anchor_auc_drop',.005)); best=(base_score,0.0,0.0,base,None,None)
    for strength in strengths:
        trp=project(train_emb,basis0,strength); vap=project(val_emb,basis0,strength); prot=fit_taxonomy_prototypes(trp,train_pred)
        trm=metric_score(trp,prot,float(ecfg.get('hard_negative_temperature',.20))); vm=metric_score(vap,prot,float(ecfg.get('hard_negative_temperature',.20)))
        mu=float(np.mean(trm)); sd=float(np.std(trm)+1e-6); vz=(vm-mu)/sd
        for blend in blends:
            if blend<=0: p=anchor_val.copy()
            else: p=_sigmoid(anchor_logit+blend*vz)
            sm=_origin_auc_summary(val_pred,p)
            if sm['pooled'] < base['pooled']-max_drop: continue
            score=.62*sm['worst']+.28*sm['mean']+.10*sm['pooled']
            if score>best[0]+1e-9: best=(score,strength,blend,sm,prot,(mu,sd))
    gain=best[0]-base_score
    if gain<float(ecfg.get('min_gain_to_use_aux',.002)):
        best=(base_score,0.0,0.0,base,None,None)
    return dict(strength=float(best[1]),blend=float(best[2]),validation_anchor=base,validation_selected=best[3],validation_gain=float(best[0]-base_score),basis=basis0)


def refit_source_evidence(state,fit_emb,fit_pred,cfg):
    ecfg=cfg['evaluation'].get('source_evidence_projection',{}); strength=float(state.get('strength',0)); blend=float(state.get('blend',0))
    basis=fit_nuisance_basis(fit_emb,fit_pred,int(ecfg.get('max_nuisance_dim',6)),int(ecfg.get('min_origin_class_count',2)))
    zp=project(fit_emb,basis,strength); prot=fit_taxonomy_prototypes(zp,fit_pred); ms=metric_score(zp,prot,float(ecfg.get('hard_negative_temperature',.20)))
    state=dict(state); state.update(basis=basis,prototypes=prot,metric_mean=float(ms.mean()),metric_std=float(ms.std()+1e-6)); return state


def apply_source_evidence(state,emb,pred,cfg):
    z=project(emb,state.get('basis'),float(state.get('strength',0))); prot=state.get('prototypes',{}); ms=metric_score(z,prot,float(cfg['evaluation'].get('source_evidence_projection',{}).get('hard_negative_temperature',.20)))
    mz=(ms-float(state.get('metric_mean',0)))/max(float(state.get('metric_std',1)),1e-6)
    anchor_col='expert_prob_anchor' if 'expert_prob_anchor' in pred else 'probability'; anchor=pred[anchor_col].to_numpy(float); score=_logit(anchor)+float(state.get('blend',0))*mz
    out=pred.copy(); out['anchor_probability']=anchor; out['metric_evidence']=ms; out['metric_evidence_z']=mz; out['probability']=_sigmoid(score)
    return out,z
