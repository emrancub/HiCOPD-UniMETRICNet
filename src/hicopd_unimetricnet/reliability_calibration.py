
from __future__ import annotations
import numpy as np
import pandas as pd

def _sigmoid(x):
    x=np.asarray(x,float)
    return 1.0/(1.0+np.exp(-np.clip(x,-35,35)))

def _logit(p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    return np.log(p/(1-p))

def _norm(z):
    z=np.asarray(z,np.float64)
    return z/np.clip(np.linalg.norm(z,axis=1,keepdims=True),1e-8,None)

def _auroc(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float)
    pos=p[y==1]; neg=p[y==0]
    if not len(pos) or not len(neg): return np.nan
    # exact Mann-Whitney formulation without scipy
    return float(((pos[:,None]>neg[None,:]).sum()+0.5*(pos[:,None]==neg[None,:]).sum())/(len(pos)*len(neg)))

def _ece(y,p,bins=10):
    y=np.asarray(y,int); p=np.asarray(p,float); out=0.
    for b in range(int(bins)):
        lo,hi=b/bins,(b+1)/bins
        m=(p>=lo)&((p<=hi) if b==bins-1 else (p<hi))
        if m.any(): out += m.mean()*abs(float(p[m].mean())-float(y[m].mean()))
    return float(out)

def _bacc(y,p,th=.5):
    y=np.asarray(y,int); pr=np.asarray(p,float)>=float(th)
    pos=y==1; neg=y==0
    se=float(pr[pos].mean()) if pos.any() else np.nan
    sp=float((~pr[neg]).mean()) if neg.any() else np.nan
    ba=float(np.nanmean([se,sp]))
    return ba,se,sp

def _taxonomy_prototypes(z,frame):
    z=_norm(z); tax=frame['taxonomy'].to_numpy(int); prot={}
    for t in sorted(np.unique(tax)):
        m=tax==t
        if int(t)>=0 and m.any():
            v=z[m].mean(0); prot[int(t)]=v/max(np.linalg.norm(v),1e-8)
    return prot

def _hard_negative_score(z,prototypes,temp=.25):
    z=_norm(z)
    if 0 not in prototypes: return np.zeros(len(z))
    pos=z@prototypes[0]
    neg=[z@prototypes[t] for t in sorted(prototypes) if t!=0]
    if not neg: return np.zeros(len(z))
    n=np.stack(neg,1)/max(float(temp),1e-6)
    mm=n.max(1,keepdims=True)
    lse=(mm[:,0]+np.log(np.exp(n-mm).mean(1)+1e-12))*max(float(temp),1e-6)
    return pos-lse

def _density_reference(z,frame):
    z=_norm(z); y=frame.y.to_numpy(int); centers={}; radii={}
    for c in [0,1]:
        m=y==c
        if not m.any(): continue
        cen=z[m].mean(0); cen=cen/max(np.linalg.norm(cen),1e-8); centers[c]=cen
        dist=1-(z[m]@cen)
        radii[c]=float(np.quantile(dist,.90)) if len(dist)>1 else max(float(dist.mean()),.05)
    return centers,radii

def _density_reliability(z,centers,radii,temp=.30):
    z=_norm(z)
    if not centers: return np.ones(len(z))*0.5
    ds=[]
    for c,cen in centers.items():
        d=1-(z@cen); scale=max(radii.get(c,.2),.03); ds.append(d/scale)
    d=np.stack(ds,1).min(1)
    return _sigmoid((1.0-d)/max(float(temp),1e-6))

def _feature_matrix(frame,emb,prototypes,centers,radii,cfg):
    pcfg=cfg['evaluation'].get('prism_calnet',{})
    base=frame['probability'].to_numpy(float)
    hard=_hard_negative_score(emb,prototypes,float(cfg['evaluation'].get('source_evidence_projection',{}).get('hard_negative_temperature',.20)))
    density=_density_reliability(emb,centers,radii,float(pcfg.get('density_temperature',.30)))
    quality=frame['quality_index'].to_numpy(float) if 'quality_index' in frame else np.ones(len(frame))*.5
    model_rel=frame['model_reliability'].to_numpy(float) if 'model_reliability' in frame else np.ones(len(frame))*.5
    entropy=frame['entropy'].to_numpy(float) if 'entropy' in frame else -(np.c_[1-base,base]*np.log(np.c_[1-base,base]+1e-8)).sum(1)
    entropy=entropy/np.log(2)
    # Anchor logit is always first; correction features are bounded/standardized later.
    X=np.column_stack([_logit(base),hard,quality,density,model_rel,entropy])
    meta=dict(base=base,hard_negative_score=hard,density_reliability=density,quality=quality,model_reliability=model_rel,entropy=entropy)
    return X,meta

def _fit_logistic(X,y,w,l2,steps,lr,anchor_min,anchor_max):
    X=np.asarray(X,float); y=np.asarray(y,float); w=np.asarray(w,float)
    mu=X[:,1:].mean(0); sd=X[:,1:].std(0)+1e-6
    Z=X.copy(); Z[:,1:]=(Z[:,1:]-mu)/sd
    beta=np.zeros(Z.shape[1]+1,float); beta[1]=1.0
    w=w/np.clip(w.mean(),1e-8,None)
    for t in range(int(steps)):
        eta=float(lr)/(1+0.004*t)
        p=_sigmoid(beta[0]+Z@beta[1:])
        e=(p-y)*w
        g0=e.mean()
        g=(Z.T@e)/len(y)
        g[1:]+=float(l2)*beta[2:]
        beta[0]-=eta*g0; beta[1:]-=eta*g
        beta[1]=np.clip(beta[1],float(anchor_min),float(anchor_max))
    return dict(beta=beta,mu=mu,sd=sd)

def _apply_logistic(state,X,clip=2.0):
    Z=np.asarray(X,float).copy()
    if Z.shape[1]>1: Z[:,1:]=(Z[:,1:]-state['mu'])/state['sd']
    raw=state['beta'][0]+Z@state['beta'][1:]
    # The calibration layer is intentionally bounded around anchor evidence.
    anchor=Z[:,0]*state['beta'][1]+state['beta'][0]
    corr=np.clip(raw-anchor,-float(clip),float(clip))
    return _sigmoid(anchor+corr)

def _domain_weights(frame):
    d=frame.origin_dataset.astype(str).to_numpy(); y=frame.y.to_numpy(int)
    keys=np.array([f'{a}|{b}' for a,b in zip(d,y)],object)
    vals=pd.Series(keys).value_counts().to_dict()
    return np.array([1/max(vals[k],1) for k in keys],float)

def _robust_score(frame,p):
    y=frame.y.to_numpy(int); d=frame.origin_dataset.astype(str).to_numpy()
    aucs=[]; baccs=[]; nlls=[]
    for dom in np.unique(d):
        m=d==dom
        yy=y[m]; pp=np.clip(np.asarray(p)[m],1e-6,1-1e-6)
        nlls.append(float(-np.mean(yy*np.log(pp)+(1-yy)*np.log(1-pp))))
        if len(np.unique(yy))==2:
            aucs.append(_auroc(yy,pp))
            th=_threshold_simple(yy,pp)
            baccs.append(_bacc(yy,pp,th)[0])
    pooled=_auroc(y,p) if len(np.unique(y))==2 else .5
    wa=min(aucs) if aucs else pooled; ma=float(np.mean(aucs)) if aucs else pooled
    wb=min(baccs) if baccs else .5
    e=_ece(y,p)
    return .45*wa+.25*ma+.15*pooled+.10*wb-.05*e

def _threshold_simple(y,p):
    best=(-1,.5)
    for th in np.linspace(.05,.95,181):
        ba,se,sp=_bacc(y,p,th)
        sc=ba-.12*abs(se-sp)
        if sc>best[0]: best=(sc,float(th))
    return best[1]

def fit_prism_calnet(train_pred,train_emb,val_pred,val_emb,cal_pred,cal_emb,cfg,disable=False,disable_hardneg=False,disable_reliability=False):
    pcfg=cfg['evaluation'].get('prism_calnet',{})
    fit_pred=pd.concat([train_pred,val_pred],ignore_index=True)
    fit_emb=np.concatenate([train_emb,val_emb],axis=0)
    prototypes=_taxonomy_prototypes(fit_emb,fit_pred)
    centers,radii=_density_reference(fit_emb,fit_pred)
    if disable_hardneg: prototypes={}
    X,meta=_feature_matrix(cal_pred,cal_emb,prototypes,centers,radii,cfg)
    if disable_reliability:
        X[:,2:5]=0
    y=cal_pred.y.to_numpy(int); w=_domain_weights(cal_pred)
    base=cal_pred.probability.to_numpy(float); base_score=_robust_score(cal_pred,base)
    if disable or len(np.unique(y))<2:
        state=dict(active=False,reason='disabled_or_single_class',model=None,prototypes=prototypes,centers=centers,radii=radii,base_score=base_score,selected_score=base_score)
        return state
    best=None
    for l2 in pcfg.get('l2_grid',[.001,.01,.1,1.]):
        st=_fit_logistic(X,y,w,l2,int(pcfg.get('steps',500)),float(pcfg.get('learning_rate',.04)),float(pcfg.get('anchor_coef_min',.65)),float(pcfg.get('anchor_coef_max',2.5)))
        pp=_apply_logistic(st,X,float(pcfg.get('correction_clip',2.0)))
        sc=_robust_score(cal_pred,pp)
        # Safety: do not sacrifice any mixed-source AUROC too much.
        drops=[]
        for dom,g in cal_pred.groupby('origin_dataset'):
            if g.y.nunique()<2: continue
            idx=g.index.to_numpy()
            # cal_pred has a reset index in normal pipeline; map by positional index robustly
            pos=np.flatnonzero(cal_pred.index.isin(idx))
            if len(pos):
                a0=_auroc(g.y.to_numpy(int),base[pos]); a1=_auroc(g.y.to_numpy(int),pp[pos])
                if np.isfinite(a0) and np.isfinite(a1): drops.append(a0-a1)
        maxdrop=max(drops) if drops else 0.
        key=(sc,-maxdrop,-float(l2))
        if best is None or key>best[0]: best=(key,float(l2),st,pp,maxdrop)
    gain=float(best[0][0]-base_score)
    active=bool(gain>=float(pcfg.get('min_cv_gain',.002)) and best[4]<=float(pcfg.get('max_origin_auc_drop',.02)))
    return dict(active=active,reason='source_validation_gain' if active else 'safe_fallback_to_anchor',
                model=best[2],l2=best[1],prototypes=prototypes,centers=centers,radii=radii,
                base_score=float(base_score),selected_score=float(best[0][0]),validation_gain=gain,max_origin_auc_drop=float(best[4]),
                disable_hardneg=bool(disable_hardneg),disable_reliability=bool(disable_reliability))

def apply_prism_calnet(state,frame,emb,cfg):
    out=frame.copy()
    X,meta=_feature_matrix(out,emb,state.get('prototypes',{}),state.get('centers',{}),state.get('radii',{}),cfg)
    if state.get('disable_reliability',False): X[:,2:5]=0
    p0=out.probability.to_numpy(float)
    if state.get('active',False) and state.get('model') is not None:
        p=_apply_logistic(state['model'],X,float(cfg['evaluation'].get('prism_calnet',{}).get('correction_clip',2.0)))
    else:
        p=p0.copy()
    density=meta['density_reliability']; conf=1-np.minimum(1.0,meta['entropy'])
    modelrel=np.clip(meta['model_reliability'],0,1)
    pcfg=cfg['evaluation'].get('prism_calnet',{})
    a=float(pcfg.get('reliability_confidence_weight',.45)); b=float(pcfg.get('reliability_density_weight',.35)); c=float(pcfg.get('reliability_model_weight',.20))
    rel=np.clip((a*conf+b*density+c*modelrel)/max(a+b+c,1e-6),0,1)
    out['probability_pre_calnet']=p0
    out['probability']=np.clip(p,1e-6,1-1e-6)
    out['hard_negative_score']=meta['hard_negative_score']
    out['density_reliability']=density
    out['measurement_reliability']=rel
    return out

def state_to_jsonable(state):
    out={}
    for k,v in state.items():
        if k in {'prototypes','centers','radii','model'}: continue
        if isinstance(v,(np.floating,np.integer)): v=v.item()
        out[k]=v
    if state.get('model') is not None:
        out['model']={k:(np.asarray(v).tolist() if isinstance(v,np.ndarray) else v) for k,v in state['model'].items()}
    out['radii']={str(k):float(v) for k,v in state.get('radii',{}).items()}
    return out
