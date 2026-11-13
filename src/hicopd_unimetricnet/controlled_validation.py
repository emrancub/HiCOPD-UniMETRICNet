
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd

def _auc(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float)
    pos=p[y==1]; neg=p[y==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    return float(((pos[:,None]>neg[None,:]).sum()+0.5*(pos[:,None]==neg[None,:]).sum())/(len(pos)*len(neg)))

def _ece(y,p,bins=10):
    y=np.asarray(y,int); p=np.asarray(p,float); out=0.0
    for b in range(int(bins)):
        lo,hi=b/bins,(b+1)/bins
        m=(p>=lo)&((p<=hi) if b==bins-1 else (p<hi))
        if m.any(): out += float(m.mean())*abs(float(p[m].mean())-float(y[m].mean()))
    return float(out)

def _metrics(y,p,threshold):
    y=np.asarray(y,int); p=np.asarray(p,float); pr=p>=float(threshold)
    pos=y==1; neg=y==0
    tp=int((pr&pos).sum()); fn=int((~pr&pos).sum()); tn=int((~pr&neg).sum()); fp=int((pr&neg).sum())
    sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1); ppv=tp/max(tp+fp,1); f1=2*ppv*sens/max(ppv+sens,1e-12)
    return dict(n=len(y),auroc=_auc(y,p),sensitivity=sens,specificity=spec,balanced_accuracy=(sens+spec)/2,
                precision=ppv,f1=f1,ece=_ece(y,p),tp=tp,fn=fn,tn=tn,fp=fp)

def _bootstrap_auc(y,p,groups,resamples=2000,seed=2026):
    y=np.asarray(y,int); p=np.asarray(p,float); groups=np.asarray(groups,str); ids=np.unique(groups)
    if len(ids)<2: return np.nan,np.nan
    rng=np.random.default_rng(int(seed)); by={g:np.flatnonzero(groups==g) for g in ids}; vals=[]
    for _ in range(int(resamples)):
        draw=rng.choice(ids,size=len(ids),replace=True)
        idx=np.concatenate([by[g] for g in draw]); yy=y[idx]; pp=p[idx]
        if len(np.unique(yy))==2: vals.append(_auc(yy,pp))
    return (float(np.quantile(vals,.025)),float(np.quantile(vals,.975))) if vals else (np.nan,np.nan)

def _within_subject_sd(df):
    residuals=[]
    for _,g in df.groupby("participant_id"):
        x=g.probability.to_numpy(float)
        if len(x)>=2: residuals.extend((x-x.mean()).tolist())
    return float(np.sqrt(np.mean(np.square(residuals)))) if residuals else np.nan

def _icc_2_1(pivot):
    a=pivot.dropna(axis=0,how="any").to_numpy(float)
    if a.ndim!=2 or a.shape[0]<2 or a.shape[1]<2: return np.nan
    n,k=a.shape; gm=a.mean(); row=a.mean(1); col=a.mean(0)
    ssr=k*np.square(row-gm).sum(); ssc=n*np.square(col-gm).sum()
    sse=np.square(a-row[:,None]-col[None,:]+gm).sum()
    msr=ssr/(n-1); msc=ssc/(k-1); mse=sse/((n-1)*(k-1))
    den=msr+(k-1)*mse+k*(msc-mse)/n
    return float((msr-mse)/den) if abs(den)>1e-12 else np.nan

def validate_controlled_predictions(df):
    req={"participant_id","recording_id","probability","measurement_condition"}
    missing=req-set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    out=df.copy()
    out["participant_id"]=out.participant_id.astype(str); out["recording_id"]=out.recording_id.astype(str)
    out["measurement_condition"]=out.measurement_condition.astype(str)
    out["probability"]=pd.to_numeric(out.probability,errors="raise")
    if not np.isfinite(out.probability).all(): raise ValueError("probability contains non-finite values")
    if ((out.probability<0)|(out.probability>1)).any(): raise ValueError("probability must be within [0,1]")
    if out.recording_id.duplicated().any(): raise ValueError("recording_id must be unique")
    if "reference_label" in out.columns:
        y=pd.to_numeric(out.reference_label,errors="coerce")
        vals=set(y.dropna().astype(int).unique())
        if not vals.issubset({0,1}): raise ValueError("reference_label must be 0/1 when present")
        out["reference_label"]=y
    return out

def analyze_controlled_predictions(df,threshold,bootstrap_resamples=2000,seed=2026):
    df=validate_controlled_predictions(df); threshold=float(threshold)
    rows=[]
    for pid,g in df.groupby("participant_id"):
        p=g.probability.to_numpy(float)
        rows.append(dict(participant_id=pid,n_recordings=len(g),mean_probability=float(p.mean()),
                         std_probability=float(p.std(ddof=1)) if len(p)>1 else 0.0,
                         min_probability=float(p.min()),max_probability=float(p.max()),
                         probability_range=float(p.max()-p.min())))
    per=pd.DataFrame(rows)
    sw=_within_subject_sd(df)
    pivot=df.pivot_table(index="participant_id",columns="measurement_condition",values="probability",aggfunc="mean")
    summary=dict(participants=int(df.participant_id.nunique()),recordings=int(len(df)),
                 conditions=int(df.measurement_condition.nunique()),
                 mean_within_participant_range=float(per.probability_range.mean()) if len(per) else np.nan,
                 median_within_participant_range=float(per.probability_range.median()) if len(per) else np.nan,
                 within_subject_sd=sw,
                 repeatability_coefficient_95=float(1.96*math.sqrt(2)*sw) if np.isfinite(sw) else np.nan,
                 icc_2_1_absolute_agreement=_icc_2_1(pivot),frozen_threshold=threshold)
    cond=[]
    for name,g in df.groupby("measurement_condition"):
        row=dict(measurement_condition=name,participants=int(g.participant_id.nunique()),recordings=int(len(g)),
                 mean_probability=float(g.probability.mean()),
                 std_probability=float(g.probability.std(ddof=1)) if len(g)>1 else 0.0)
        if "reference_label" in g and g.reference_label.notna().all() and g.reference_label.nunique()==2:
            row.update(_metrics(g.reference_label.astype(int),g.probability,threshold))
        cond.append(row)
    overall=None
    if "reference_label" in df and df.reference_label.notna().all() and df.reference_label.nunique()==2:
        patient=df.groupby("participant_id",as_index=False).agg(reference_label=("reference_label","first"),probability=("probability","mean"))
        overall=_metrics(patient.reference_label.astype(int),patient.probability,threshold)
        lo,hi=_bootstrap_auc(patient.reference_label.astype(int),patient.probability,patient.participant_id,bootstrap_resamples,seed)
        overall.update(auroc_ci_low=lo,auroc_ci_high=hi,analysis_level="participant")
    return df,per,pd.DataFrame(cond),summary,overall

def _save_figures(df,per,outdir,dpi=600):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered=per.sort_values("mean_probability").reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(7.16,4.2))
    ax.vlines(np.arange(len(ordered)),ordered.min_probability,ordered.max_probability,linewidth=.8)
    ax.scatter(np.arange(len(ordered)),ordered.mean_probability,s=12)
    ax.set_xlabel("Participant ordered by mean probability"); ax.set_ylabel("COPD probability")
    ax.set_title("Controlled-acquisition within-participant repeatability"); fig.tight_layout()
    for ext in ("png","pdf","tiff"): fig.savefig(outdir/f"Fig_CV01_within_participant_repeatability.{ext}",dpi=dpi,bbox_inches="tight")
    plt.close(fig)
    labels=[]; groups=[]
    for name,g in df.groupby("measurement_condition"):
        labels.append(str(name)); groups.append(g.probability.to_numpy(float))
    fig,ax=plt.subplots(figsize=(7.16,4.2)); ax.boxplot(groups,labels=labels,showfliers=False)
    ax.set_ylabel("COPD probability"); ax.set_xlabel("Measurement condition")
    ax.set_title("Prediction distribution across acquisition conditions"); ax.tick_params(axis="x",rotation=30); fig.tight_layout()
    for ext in ("png","pdf","tiff"): fig.savefig(outdir/f"Fig_CV02_condition_effects.{ext}",dpi=dpi,bbox_inches="tight")
    plt.close(fig)

def write_controlled_report(predictions_csv,outdir,threshold,bootstrap_resamples=2000,seed=2026,dpi=600):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(predictions_csv)
    df,per,cond,summary,overall=analyze_controlled_predictions(df,threshold,bootstrap_resamples,seed)
    per.to_csv(outdir/"Table_CV01_participant_repeatability.csv",index=False)
    cond.to_csv(outdir/"Table_CV02_condition_effects.csv",index=False)
    pd.DataFrame([summary]).to_csv(outdir/"Table_CV03_measurement_repeatability_summary.csv",index=False)
    if overall is not None: pd.DataFrame([overall]).to_csv(outdir/"Table_CV04_locked_diagnostic_performance.csv",index=False)
    _save_figures(df,per,outdir,int(dpi))
    return summary,overall
