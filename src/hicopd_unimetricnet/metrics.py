from __future__ import annotations
import math
import numpy as np
import pandas as pd


def _rank_average(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,float)
    order=np.argsort(x,kind='mergesort')
    ranks=np.empty(len(x),float)
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]: j+=1
        r=(i+1+j)/2.0
        ranks[order[i:j]]=r
        i=j
    return ranks


def roc_auc_np(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float)
    pos=y==1; neg=y==0; np1=int(pos.sum()); nn=int(neg.sum())
    if np1==0 or nn==0: return float('nan')
    ranks=_rank_average(p)
    return float((ranks[pos].sum()-np1*(np1+1)/2)/(np1*nn))


def average_precision_np(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float); npos=int((y==1).sum())
    if npos==0 or npos==len(y): return float('nan') if npos==0 else 1.0
    order=np.argsort(-p,kind='mergesort'); ys=y[order]
    tp=np.cumsum(ys==1); rank=np.arange(1,len(ys)+1)
    return float((tp[ys==1]/rank[ys==1]).mean())


def roc_curve_np(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float)
    pos=(y==1).sum(); neg=(y==0).sum()
    if pos==0 or neg==0: return np.array([0.,1.]),np.array([0.,1.]),np.array([np.inf,-np.inf])
    order=np.argsort(-p,kind='mergesort'); ys=y[order]; ps=p[order]
    distinct=np.r_[True,ps[1:]!=ps[:-1]]
    tp=np.cumsum(ys==1)[distinct]; fp=np.cumsum(ys==0)[distinct]; th=ps[distinct]
    return np.r_[0.,fp/neg,1.],np.r_[0.,tp/pos,1.],np.r_[np.inf,th,-np.inf]


def ece(y,p,bins=10):
    y=np.asarray(y,int); p=np.asarray(p,float); total=0.0
    for b in range(int(bins)):
        lo,hi=b/bins,(b+1)/bins
        m=(p>=lo)&((p<=hi) if b==bins-1 else (p<hi))
        if m.any(): total+=float(m.mean())*abs(float(p[m].mean())-float(y[m].mean()))
    return float(total)


def risk_curve(y,p,u,threshold=.5):
    y=np.asarray(y,int); p=np.asarray(p,float); u=np.asarray(u,float)
    if len(y)==0: return pd.DataFrame(columns=['coverage','risk','uncertainty_threshold'])
    err=((p>=threshold).astype(int)!=y).astype(float)
    order=np.argsort(u,kind='mergesort'); u=u[order]; err=err[order]
    rows=[]; included=0; total_error=0.0
    for score in np.unique(u):
        g=err[u==score]; m=len(g); group_error=float(g.mean())
        for k in range(1,m+1):
            rows.append(dict(coverage=(included+k)/len(y),risk=(total_error+k*group_error)/(included+k),uncertainty_threshold=float(score)))
        included+=m; total_error+=float(g.sum())
    return pd.DataFrame(rows)


def binary_metrics(y,p,threshold=.5,bins=10,u=None):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    if len(y)==0: return {'n':0,'status':'empty'}
    if not np.isin(y,[0,1]).all(): raise ValueError('Labels must be 0/1')
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError('Probabilities must be finite in [0,1]')
    pred=(p>=threshold).astype(int); pos=y==1; neg=y==0
    tp=int(((pred==1)&pos).sum()); fn=int(((pred==0)&pos).sum()); tn=int(((pred==0)&neg).sum()); fp=int(((pred==1)&neg).sum())
    safe=lambda a,b: float(a/b) if b else float('nan')
    sens=safe(tp,tp+fn); spec=safe(tn,tn+fp); ppv=safe(tp,tp+fp); npv=safe(tn,tn+fn)
    acc=float((pred==y).mean()); bacc=(sens+spec)/2 if np.isfinite(sens) and np.isfinite(spec) else float('nan')
    f1=safe(2*tp,2*tp+fp+fn)
    den=math.sqrt(max(0,(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))); mcc=float((tp*tn-fp*fn)/den) if den else float('nan')
    pclip=np.clip(p,1e-7,1-1e-7)
    default_u=1-np.maximum(p,1-p) if u is None else np.asarray(u,float)
    rc=risk_curve(y,p,default_u,threshold)
    return dict(
        n=int(len(y)),positives=int(pos.sum()),negatives=int(neg.sum()),prevalence=float(pos.mean()),
        accuracy=acc,balanced_accuracy=bacc,sensitivity=sens,specificity=spec,precision=ppv,ppv=ppv,npv=npv,f1=f1,mcc=mcc,
        auroc=roc_auc_np(y,p),auprc=average_precision_np(y,p),brier=float(np.mean((p-y)**2)),
        nll=float(-np.mean(y*np.log(pclip)+(1-y)*np.log(1-pclip))),ece=ece(y,p,bins),aurc=float(rc.risk.mean()) if len(rc) else float('nan'),
        tp=tp,fn=fn,tn=tn,fp=fp,status='two_class' if pos.any() and neg.any() else 'single_class_discrimination_undefined')


def source_threshold(y,p,target_sensitivity=.90):
    y=np.asarray(y,int); p=np.asarray(p,float); pos=np.sort(p[y==1])
    if len(pos)==0: raise ValueError('No positive source calibration patients')
    q=max(0,min(len(pos)-1,int(np.floor((1-float(target_sensitivity))*len(pos)))))
    return float(pos[q])


def source_uncertainty_threshold(u,target_coverage=.90):
    u=np.asarray(u,float)
    if len(u)==0: raise ValueError('No source calibration uncertainty values')
    return float(np.quantile(u,np.clip(float(target_coverage),0,1),method='higher'))


def uncertainty_score(frame: pd.DataFrame) -> np.ndarray:
    parts=[]
    if 'evidential_uncertainty' in frame.columns and frame.evidential_uncertainty.notna().all():
        parts.append(np.clip(frame.evidential_uncertainty.to_numpy(float),0,1))
    if 'entropy' in frame.columns and frame.entropy.notna().all():
        parts.append(np.clip(frame.entropy.to_numpy(float)/math.log(2),0,1))
    if 'prototype_distance' in frame.columns and frame.prototype_distance.notna().all():
        parts.append(np.clip(frame.prototype_distance.to_numpy(float)/2,0,1))
    if not parts:
        p=frame.probability.to_numpy(float); parts=[1-np.maximum(p,1-p)]
    return np.stack(parts).mean(0)


def _balanced_accuracy_multiclass(y_true,y_pred):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred); vals=np.unique(y_true); rec=[]
    for c in vals:
        m=y_true==c
        if m.any(): rec.append(float((y_pred[m]==c).mean()))
    return float(np.mean(rec)) if rec else float('nan')


def _stratified_folds(y,folds,seed):
    y=np.asarray(y); rng=np.random.default_rng(seed); bins=[[] for _ in range(int(folds))]
    for c in np.unique(y):
        ids=np.flatnonzero(y==c); rng.shuffle(ids)
        for j,i in enumerate(ids): bins[j%folds].append(int(i))
    return [np.asarray(sorted(x),int) for x in bins]


def _softmax(z):
    z=z-z.max(axis=1,keepdims=True); e=np.exp(np.clip(z,-40,40)); return e/e.sum(axis=1,keepdims=True)


def _fit_linear_softmax(x,y,steps=200,lr=.08,l2=1e-3):
    x=np.asarray(x,float); y=np.asarray(y); classes=np.unique(y); mapper={c:i for i,c in enumerate(classes)}; yi=np.array([mapper[v] for v in y])
    mu=x.mean(0); sd=x.std(0)+1e-6; xs=(x-mu)/sd; n,d=xs.shape; k=len(classes); w=np.zeros((d,k)); b=np.zeros(k); one=np.eye(k)[yi]
    for t in range(int(steps)):
        pr=_softmax(xs@w+b); grad=(pr-one)/n; eta=lr/(1+0.01*t)
        w-=eta*(xs.T@grad+l2*w); b-=eta*grad.sum(0)
    return classes,mu,sd,w,b


def domain_probe(embedding,domain,folds=5,steps=200):
    x=np.asarray(embedding,float); y=np.asarray(domain)
    if len(x)<4 or len(np.unique(y))<2: return float('nan')
    counts=pd.Series(y).value_counts(); nf=min(int(folds),int(counts.min()))
    if nf<2: return float('nan')
    scores=[]
    for test in _stratified_folds(y,nf,123):
        train=np.setdiff1d(np.arange(len(y)),test,assume_unique=False)
        classes,mu,sd,w,b=_fit_linear_softmax(x[train],y[train],steps=steps)
        pred=classes[np.argmax(((x[test]-mu)/sd)@w+b,axis=1)]
        scores.append(_balanced_accuracy_multiclass(y[test],pred))
    return float(np.nanmean(scores))


def aggregate_prediction_views(frame):
    if frame.groupby('person_id').y.nunique().gt(1).any(): raise RuntimeError('Prediction labels disagree within patient')
    first=['subject_id','logical_dataset','origin_dataset','category','y','age','sex']
    mean=['probability','p_raw','prototype_distance','response_energy','entropy','evidential_uncertainty','uncertainty']
    agg={c:'first' for c in first if c in frame.columns}; agg.update({c:'mean' for c in mean if c in frame.columns})
    return frame.groupby('person_id',sort=True).agg(agg).reset_index()


def bootstrap_metrics(y,p,u=None,repeats=1000,seed=2026,bins=10,threshold=.5):
    y=np.asarray(y,int); p=np.asarray(p,float); u=(1-np.maximum(p,1-p)) if u is None else np.asarray(u,float); rng=np.random.default_rng(seed)
    keys=['auroc','auprc','balanced_accuracy','sensitivity','specificity','brier','ece','aurc']; vals={k:[] for k in keys}
    for _ in range(int(repeats)):
        ix=rng.integers(0,len(y),len(y)); m=binary_metrics(y[ix],p[ix],threshold=threshold,u=u[ix],bins=bins)
        for k in keys:
            v=m.get(k,float('nan'))
            if np.isfinite(v): vals[k].append(v)
    return {k:{'ci_low':float(np.quantile(v,.025)) if v else float('nan'),'ci_high':float(np.quantile(v,.975)) if v else float('nan'),'valid_resamples':len(v)} for k,v in vals.items()}


def pairwise_domain_geometry(embedding,domains):
    x=np.asarray(embedding,float); d=np.asarray(domains); rows=[]; names=sorted(np.unique(d).tolist())
    for i,a in enumerate(names):
        xa=x[d==a]
        for b in names[:i]:
            xb=x[d==b]; ma,mb=xa.mean(0),xb.mean(0); ca=np.cov(xa,rowvar=False) if len(xa)>1 else np.zeros((x.shape[1],x.shape[1])); cb=np.cov(xb,rowvar=False) if len(xb)>1 else np.zeros_like(ca)
            centroid=float(np.linalg.norm(ma-mb)); coral=float(np.mean((ca-cb)**2))
            rows.append(dict(domain_a=a,domain_b=b,centroid_distance=centroid,coral_distance=coral,n_a=len(xa),n_b=len(xb)))
    return pd.DataFrame(rows,columns=['domain_a','domain_b','centroid_distance','coral_distance','n_a','n_b'])


def holm(pvalues):
    p=np.asarray(pvalues,float); out=np.full(len(p),np.nan); ids=np.flatnonzero(np.isfinite(p))
    if not len(ids): return out
    order=ids[np.argsort(p[ids])]; running=0.0; m=len(order)
    for rank,i in enumerate(order):
        running=max(running,(m-rank)*p[i]); out[i]=min(1.0,running)
    return out


def paired_auroc_comparison(y,pa,pb,repeats=1000,seed=2026):
    y=np.asarray(y,int); pa=np.asarray(pa,float); pb=np.asarray(pb,float)
    if len(y)==0 or len(np.unique(y))<2: return dict(status='single_class',n=len(y),delta_auroc=float('nan'),ci_low=float('nan'),ci_high=float('nan'),p_value=float('nan'))
    observed=roc_auc_np(y,pa)-roc_auc_np(y,pb); rng=np.random.default_rng(seed); boot=[]; perm=[]
    for _ in range(int(repeats)):
        ix=rng.integers(0,len(y),len(y))
        if len(np.unique(y[ix]))==2: boot.append(roc_auc_np(y[ix],pa[ix])-roc_auc_np(y[ix],pb[ix]))
        swap=rng.random(len(y))<.5; xa=np.where(swap,pb,pa); xb=np.where(swap,pa,pb); perm.append(roc_auc_np(y,xa)-roc_auc_np(y,xb))
    return dict(status='conditional_on_fitted_predictions',n=int(len(y)),delta_auroc=float(observed),
                ci_low=float(np.quantile(boot,.025)) if boot else float('nan'),ci_high=float(np.quantile(boot,.975)) if boot else float('nan'),
                p_value=float((1+np.sum(np.abs(perm)>=abs(observed)))/(1+len(perm))))

def robust_domain_threshold(frame, probability_col='probability', domain_col='logical_dataset', grid_points=401):
    """Source-only operating point maximizing worst-domain balanced accuracy.

    This is safer under domain shift than a pooled threshold that can be
    dominated by the largest source database.
    """
    y=frame.y.to_numpy(int); p=frame[probability_col].to_numpy(float); d=frame[domain_col].astype(str).to_numpy()
    candidates=np.linspace(0.02,0.98,int(grid_points)); best=None
    for th in candidates:
        vals=[]; ss=[]
        for dom in np.unique(d):
            m=d==dom
            if len(np.unique(y[m]))<2: continue
            met=binary_metrics(y[m],p[m],th)
            vals.append(met['balanced_accuracy']); ss.append(min(met['sensitivity'],met['specificity']))
        if not vals: continue
        score=(float(np.min(vals)),float(np.mean(vals)),float(np.min(ss)))
        key=(score[0]+0.15*score[2],score[1],-abs(th-.5))
        if best is None or key>best[0]: best=(key,float(th),score)
    if best is None:
        # Small nested calibration partitions can occasionally leave every
        # individual source domain single-class even though the pooled source
        # calibration set contains both endpoints. Falling back to a pooled
        # SOURCE-ONLY threshold keeps the pipeline well-defined without ever
        # consulting the unseen target. The run metadata records this fallback.
        if len(np.unique(y))<2:
            raise ValueError('Source calibration data contain only one endpoint class')
        pooled_best=None
        for th in candidates:
            met=binary_metrics(y,p,th)
            key=(float(met['balanced_accuracy']),float(min(met['sensitivity'],met['specificity'])),-abs(th-.5))
            if pooled_best is None or key>pooled_best[0]: pooled_best=(key,float(th))
        return pooled_best[1]
    return best[1]


def domain_chance_bacc(domains):
    vals=np.unique(np.asarray(domains)); return 1.0/max(1,len(vals))

def conditional_domain_probe(embedding,domain,y,folds=5,steps=200):
    """Domain separability after conditioning on endpoint class.

    This avoids falsely attributing label-prevalence differences across domains
    to representation leakage. Returns a support-weighted mean over classes.
    """
    x=np.asarray(embedding,float); d=np.asarray(domain); y=np.asarray(y,int); vals=[]; weights=[]
    for c in np.unique(y):
        m=y==c
        if m.sum()<4 or len(np.unique(d[m]))<2: continue
        v=domain_probe(x[m],d[m],folds=folds,steps=steps)
        if np.isfinite(v): vals.append(v); weights.append(int(m.sum()))
    if not vals: return float('nan')
    return float(np.average(vals,weights=weights))
