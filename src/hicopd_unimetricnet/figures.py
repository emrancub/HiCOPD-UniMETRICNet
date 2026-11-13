from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from .utils import ensure_dir
from .metrics import roc_curve_np

PALETTE={'unimetricnet':'#1f4e79','resnet_spec':'#6f6f6f','spectral_cnn':'#aaaaaa','crnn':'#8f9fb3','ast_lite':'#9c755f','physio_mlp':'#59a14f'}
LABEL={'unimetricnet':'UniMETRICNet','resnet_spec':'ResNet-Spec','spectral_cnn':'Spectral-CNN','crnn':'PCEN-CRNN','ast_lite':'AST-lite','physio_mlp':'Physio-Quality MLP'}


def _style(cfg):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':cfg['publication'].get('font_size',9),'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'xtick.major.width':.7,'ytick.major.width':.7,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42})


def _save(fig,path,cfg):
    path=Path(path); ensure_dir(path.parent); dpi=int(cfg['publication'].get('dpi',600))
    for ext in cfg['publication'].get('figure_formats',['png','pdf','tiff']): fig.savefig(path.with_suffix('.'+ext),dpi=dpi,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)


def _box(ax,xy,w,h,text,fc='#f7f7f7',ec='#333333',fontsize=8):
    p=FancyBboxPatch(xy,w,h,boxstyle='round,pad=0.012,rounding_size=0.015',facecolor=fc,edgecolor=ec,linewidth=.9); ax.add_patch(p); ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=fontsize); return p


def _arrow(ax,a,b,style='-|>',color='#555555',ls='-'): ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=10,linewidth=.8,color=color,linestyle=ls))

def _read(path): return pd.read_csv(path) if Path(path).exists() else None


def fig_protocol(cfg,out):
    fig,ax=plt.subplots(figsize=(7.16,3.75)); ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.text(.5,.97,'Three-database whole-domain external validation protocol',ha='center',va='top',weight='bold',fontsize=10)
    items=[(.03,'ICBHI','6 retained classes\npatient-level labels'),(.365,'HYBRID','KAUH + RDTR\nCOPD-1–4 → COPD\nCOPD-0 excluded'),(.70,'BRACETS','5 retained classes\nindependent acquisition')]
    for x,t,sub in items: _box(ax,(x,.70),.27,.17,f'{t}\n{sub}','#f7f7f7','#333333',7.5)
    _box(ax,(.07,.39),.25,.14,'LODO-A\nHYBRID + BRACETS\n→ unseen ICBHI','#ffffff','#333333',7.2); _box(ax,(.375,.39),.25,.14,'LODO-B\nICBHI + BRACETS\n→ unseen HYBRID','#ffffff','#333333',7.2); _box(ax,(.68,.39),.25,.14,'LODO-C\nICBHI + HYBRID\n→ unseen BRACETS','#ffffff','#333333',7.2)
    for x in [.165,.50,.835]: _arrow(ax,(x,.69),(x,.54))
    _box(ax,(.10,.10),.80,.16,'PRIMARY ENDPOINTS\nMacro & worst-domain AUROC • balanced accuracy • source-only robust threshold transport • calibration\nsame-patient KAUH B/D/E intervention • acoustic stress • bootstrap CIs • matched baselines','#fafafa','#222222',7.2)
    for x in [.195,.50,.805]: _arrow(ax,(x,.38),(x,.27))
    ax.text(.5,.035,'No target labels are used for model selection, calibration, threshold selection, or ensemble weighting.',ha='center',fontsize=7.2,color='#444444')
    _save(fig,out/'Fig01_three_database_protocol',cfg)


def fig_architecture(cfg,out):
    fig,ax=plt.subplots(figsize=(7.16,4.6)); ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.text(.5,.975,'UniMETRICNet v9.0: transfer-anchored COPD measurement under acquisition shift',ha='center',va='top',weight='bold',fontsize=9.2)
    ax.text(.13,.90,'SIGNAL PREPARATION',ha='center',fontsize=6.5,weight='bold',color='#555555')
    ax.text(.48,.90,'PATIENT EVIDENCE',ha='center',fontsize=6.5,weight='bold',color='#555555')
    ax.text(.83,.90,'TRANSPORTABLE DECISION',ha='center',fontsize=6.5,weight='bold',color='#555555')
    # left pipeline
    _box(ax,(.035,.77),.19,.075,'Respiratory recording','#ffffff','#333333',6.7)
    _box(ax,(.035,.655),.19,.075,'4 kHz · 60–1800 Hz\nsoft denoise · edge trim','#ffffff','#333333',6.0)
    _box(ax,(.035,.54),.19,.075,'4-s windows · 2-s stride','#ffffff','#333333',6.2)
    _arrow(ax,(.13,.77),(.13,.73)); _arrow(ax,(.13,.655),(.13,.615))
    # representations
    _box(ax,(.29,.75),.17,.09,'Raw log-Mel\ntransfer anchor','#eef4f8','#1f4e79',6.4)
    _box(ax,(.29,.61),.17,.09,'RISN + physiology\nquality residuals','#f2f7ef','#4f772d',6.1)
    _arrow(ax,(.225,.58),(.29,.795)); _arrow(ax,(.225,.58),(.29,.655))
    _box(ax,(.51,.75),.18,.09,'Raw ResNet +\nquality-bounded MIL','#ffffff','#1f4e79',6.2)
    _box(ax,(.51,.61),.18,.09,'Bounded residual\nencoders','#ffffff','#4f772d',6.2)
    _arrow(ax,(.46,.795),(.51,.795)); _arrow(ax,(.46,.655),(.51,.655))
    _box(ax,(.745,.73),.20,.105,'Anchor score +\nhard-negative prototype margin\n+ bounded auxiliary evidence','#ffffff','#333333',6.1)
    _arrow(ax,(.69,.795),(.745,.785)); _arrow(ax,(.69,.655),(.745,.755))
    _box(ax,(.745,.545),.20,.10,'Source-only class-conditional\nacquisition nuisance projection','#eef4f8','#1f4e79',6.0)
    _arrow(ax,(.845,.73),(.845,.645))
    _box(ax,(.745,.405),.20,.085,'Robust source calibration\n+ consensus threshold','#eef4f8','#1f4e79',6.0)
    _arrow(ax,(.845,.545),(.845,.49))
    _box(ax,(.79,.30),.11,.06,'COPD score','#eef4f8','#1f4e79',6.7); _arrow(ax,(.845,.405),(.845,.36))
    # training-only controls
    ax.text(.36,.39,'TRAINING-ONLY ROBUSTNESS',ha='center',fontsize=6.5,weight='bold',color='#7a6045')
    controls=[(.035,.25,.14,'Instrument-response\nrandomization'),(.185,.25,.14,'Hard-negative\npairwise ranking'),(.335,.25,.14,'5-way clinical\nprototypes'),(.485,.25,.14,'Cross-origin\ncontrastive'),(.635,.25,.14,'KAUH B/D/E\nconsistency')]
    for x,y,w,t in controls: _box(ax,(x,y),w,.075,t,'#faf5ef','#9c755f',5.7)
    ax.text(.5,.13,'Primary evidence: strict 3-way LODO · hard-negative separation · repeatability · threshold transport · calibration · stress',ha='center',fontsize=6.4,color='#555555')
    ax.text(.5,.078,'Interpretability: t-SNE · saliency · acquisition leakage probe · clinical error taxonomy · quality strata',ha='center',fontsize=6.3,color='#555555')
    ax.text(.5,.028,'Held-out target labels are never used for training, projection selection, calibration, thresholding, or model selection.',ha='center',fontsize=6.1,color='#444444')
    _save(fig,out/'Fig02_UniMETRICNet_architecture',cfg)


def fig_preprocessing(cfg,out):
    fig,ax=plt.subplots(figsize=(7.16,2.65)); ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.text(.5,.94,'Measurement-aware signal preparation and representation',ha='center',weight='bold',fontsize=9.7)
    labels=['Raw recording','Resample\n4 kHz','60–1800 Hz\nband-pass','Soft stationary\nnoise reduction','Edge-silence\ntrim','4-s / 2-s\nsegmentation','Log-Mel + RISN\n(PCEN diagnostic)','Quality +\nphysiology']
    xs=np.linspace(.02,.86,len(labels))
    for i,(x,t) in enumerate(zip(xs,labels)):
        _box(ax,(x,.42),.105,.20,t,'#ffffff','#444444',6.6)
        if i<len(labels)-1: _arrow(ax,(x+.105,.52),(xs[i+1],.52))
    ax.text(.5,.17,'Conservative preprocessing is fixed before train/test splitting; augmentation is training-only and measurement-specific.',ha='center',fontsize=7.2,color='#444444')
    _save(fig,out/'Fig03_preprocessing_pipeline',cfg)


def fig_dataset_composition(cfg,out):
    p=Path(cfg['project']['cache_dir'])/'Table02_three_database_final.csv'; q=_read(p)
    if q is None or q.empty: return
    if 'logical_dataset' not in q: return
    # Prefer patient counts where available.
    value='number_people' if 'number_people' in q else ('people' if 'people' in q else None)
    if value is None: return
    piv=q.pivot_table(index='category',columns='logical_dataset',values=value,aggfunc='sum',fill_value=0); fig,ax=plt.subplots(figsize=(7.16,3.4)); bottom=np.zeros(len(piv.columns)); x=np.arange(len(piv.columns))
    for cat,row in piv.iterrows(): ax.bar(x,row.to_numpy(float),bottom=bottom,label=str(cat)); bottom+=row.to_numpy(float)
    ax.set_xticks(x,piv.columns); ax.set_ylabel('Patients'); ax.set_title('Final patient composition after integrity controls'); ax.legend(ncol=3,fontsize=6.8,loc='upper center',bbox_to_anchor=(.5,-.12)); _save(fig,out/'Fig04_dataset_composition',cfg)


def fig_lodo(cfg,out,tables):
    q=_read(tables/'Table03b_primary_LODO_ensemble.csv')
    if q is None or q.empty: return
    targets=['ICBHI','HYBRID','BRACETS']; models=[m for m in ['spectral_cnn','crnn','resnet_spec','ast_lite','physio_mlp','unimetricnet'] if m in set(q.model)]; fig,ax=plt.subplots(figsize=(7.16,3.65)); x=np.arange(len(targets)); width=.82/max(1,len(models))
    for j,m in enumerate(models):
        vals=[q[(q.target_or_setting==t)&(q.model==m)].auroc.mean() for t in targets]; ax.bar(x-.41+width/2+j*width,vals,width,label=LABEL.get(m,m),color=PALETTE.get(m,'#777777'))
    ax.axhline(.5,color='#777777',lw=.8,ls='--'); ax.set_xticks(x,targets); ax.set_ylim(.35,1.0); ax.set_ylabel('Patient-level AUROC'); ax.set_title('Primary strict whole-database LODO generalization'); ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.13),fontsize=7.2); _save(fig,out/'Fig05_primary_LODO_AUROC',cfg)


def fig_lodo_macro(cfg,out,tables):
    q=_read(tables/'Table03c_primary_LODO_macro.csv')
    if q is None or q.empty: return
    q=q.sort_values('macro_auroc'); y=np.arange(len(q)); fig,ax=plt.subplots(figsize=(5.0,3.55)); ax.barh(y,q.macro_auroc,color=[PALETTE.get(m,'#888888') for m in q.model]); ax.scatter(q.worst_auroc,y,marker='D',s=24,color='#222222',label='Worst-domain AUROC'); ax.axvline(.5,ls='--',lw=.8,color='#999999'); ax.set_yticks(y,[LABEL.get(m,m) for m in q.model]); ax.set_xlabel('AUROC'); ax.set_xlim(.35,1); ax.set_title('Macro and worst-domain external generalization'); ax.legend(fontsize=7.3); _save(fig,out/'Fig06_macro_worst_LODO',cfg)


def fig_lodo_rocs(cfg,out,tables):
    pred=tables/'ensemble_predictions'
    for target in cfg['experiments']['lodo_targets']:
        fig,ax=plt.subplots(figsize=(4.35,3.7)); found=False
        for m in ['resnet_spec','crnn','spectral_cnn','ast_lite','physio_mlp','unimetricnet']:
            p=pred/f'lodo__{target}__{m}.csv'
            if not p.exists(): continue
            z=pd.read_csv(p); fpr,tpr,_=roc_curve_np(z.y,z.probability); auc=float(np.trapz(tpr,fpr)); ax.plot(fpr,tpr,lw=1.5,label=f'{LABEL.get(m,m)} {auc:.3f}',color=PALETTE.get(m,'#777777')); found=True
        if not found: plt.close(fig); continue
        ax.plot([0,1],[0,1],ls='--',lw=.8,color='#999999'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate'); ax.set_title(f'Unseen {target}: patient-level ROC'); ax.legend(loc='lower right',fontsize=6.8); _save(fig,out/f'Fig07_ROC_unseen_{target}',cfg)


def fig_threshold_transport(cfg,out,tables):
    q=_read(tables/'Table03b_primary_LODO_ensemble.csv')
    if q is None or q.empty: return
    p=q[q.model.eq('unimetricnet')].set_index('target_or_setting'); targets=[t for t in ['ICBHI','HYBRID','BRACETS'] if t in p.index]; x=np.arange(len(targets)); w=.27; fig,ax=plt.subplots(figsize=(5.2,3.45)); ax.bar(x-w/2,[p.loc[t,'sensitivity'] for t in targets],w,label='Sensitivity'); ax.bar(x+w/2,[p.loc[t,'specificity'] for t in targets],w,label='Specificity'); ax.plot(x,[p.loc[t,'balanced_accuracy'] for t in targets],marker='o',lw=1.2,label='Balanced accuracy'); ax.axhline(.60,ls='--',lw=.8,color='#888888'); ax.set_xticks(x,targets); ax.set_ylim(0,1); ax.set_ylabel('Rate'); ax.set_title('Label-free measurement-style threshold transport'); ax.legend(fontsize=7.4); _save(fig,out/'Fig08_threshold_transport',cfg)


def fig_calibration(cfg,out,tables):
    pred=tables/'ensemble_predictions'; fig,ax=plt.subplots(figsize=(4.6,3.65)); found=False
    for target in cfg['experiments']['lodo_targets']:
        p=pred/f'lodo__{target}__unimetricnet.csv'
        if not p.exists(): continue
        z=pd.read_csv(p); bins=np.linspace(0,1,8); xx=[]; yy=[]
        for i in range(len(bins)-1):
            m=(z.probability>=bins[i])&(z.probability<(bins[i+1] if i<len(bins)-2 else bins[i+1]+1e-9))
            if m.any(): xx.append(z.loc[m,'probability'].mean()); yy.append(z.loc[m,'y'].mean())
        if xx: ax.plot(xx,yy,marker='o',lw=1.3,label=target); found=True
    if not found: plt.close(fig); return
    ax.plot([0,1],[0,1],ls='--',lw=.8,color='#999999'); ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed COPD frequency'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_title('LODO calibration transport'); ax.legend(fontsize=7.5); _save(fig,out/'Fig09_LODO_calibration',cfg)


def fig_filter(cfg,out,tables):
    p=Path(cfg['project']['output_dir'])/'filter_test'; files=list(p.glob('predictions_*_seed_*.csv'))
    if not files: return
    rows=[]
    for f in files:
        model=f.name.split('_seed_')[0].replace('predictions_',''); z=pd.read_csv(f); piv=z.pivot_table(index='person_id',columns='filter_mode',values='probability',aggfunc='mean').dropna(); r=piv.max(axis=1)-piv.min(axis=1)
        for v in r: rows.append((model,float(v)))
    q=pd.DataFrame(rows,columns=['model','range']); fig,ax=plt.subplots(figsize=(4.5,3.45)); data=[q[q.model==m].range.to_numpy() for m in ['resnet_spec','unimetricnet']]; ax.boxplot(data,labels=['ResNet-Spec','UniMETRICNet'],showfliers=False); ax.scatter(np.ones(len(data[0]))*.96,data[0],s=8,alpha=.25); ax.scatter(np.ones(len(data[1]))*2.04,data[1],s=8,alpha=.25); ax.axhline(.20,ls='--',lw=.8,color='#888888'); ax.set_ylabel('Within-patient probability range'); ax.set_title('KAUH same-patient B/D/E measurement intervention'); _save(fig,out/'Fig10_KAUH_intervention_stability',cfg)


def fig_stress(cfg,out,tables):
    q=_read(tables/'Table11_external_stress.csv')
    if q is None or q.empty: return
    fig,ax=plt.subplots(figsize=(5.4,3.5)); conditions=['clean','noise_20db','noise_10db','bandlimit_800','clipping_050']; x=np.arange(len(conditions))
    for m in ['resnet_spec','unimetricnet']:
        g=q[q.model==m].groupby('condition').auroc.mean(); vals=[g.get(c,np.nan) for c in conditions]; ax.plot(x,vals,marker='o',lw=1.6,label=LABEL[m],color=PALETTE[m])
    ax.axhline(.5,ls='--',lw=.8,color='#999999'); ax.set_xticks(x,['Clean','20 dB','10 dB','≤800 Hz','Clip 50%']); ax.set_ylabel('AUROC'); ax.set_title('External acoustic stress robustness'); ax.legend(); _save(fig,out/'Fig11_external_stress',cfg)


def fig_expert_weights(cfg,out,tables):
    pred=tables/'ensemble_predictions'; rows=[]
    for target in cfg['experiments']['lodo_targets']:
        p=pred/f'lodo__{target}__unimetricnet.csv'
        if not p.exists(): continue
        z=pd.read_csv(p); cols=[c for c in z if c.startswith('expert_weight_')]
        for c in cols: rows.append(dict(target=target,expert=c.replace('expert_weight_',''),weight=float(z[c].mean())))
    if not rows: return
    q=pd.DataFrame(rows); experts=sorted(q.expert.unique()); targets=cfg['experiments']['lodo_targets']; x=np.arange(len(targets)); w=.8/max(1,len(experts)); fig,ax=plt.subplots(figsize=(5.8,3.45))
    for j,e in enumerate(experts): ax.bar(x-.4+w/2+j*w,[q[(q.target==t)&(q.expert==e)].weight.mean() for t in targets],w,label=e.replace('_',' '))
    ax.set_xticks(x,targets); ax.set_ylim(0,1); ax.set_ylabel('Mean expert weight'); ax.set_title('Quality-conditioned expert utilization on unseen databases'); ax.legend(fontsize=7.1,ncol=len(experts)); _save(fig,out/'Fig12_expert_gate_weights',cfg)


def fig_domain(cfg,out,tables):
    q=_read(tables/'Table09_domain_leakage_diagnostic.csv')
    if q is None or q.empty: return
    q=q[q.variant.isin(['resnet_spec','unimetricnet'])]; vals=q.groupby('variant').disease_conditional_domain_probe_bacc.mean(); names=[m for m in ['resnet_spec','unimetricnet'] if m in vals.index]
    if not names: return
    fig,ax=plt.subplots(figsize=(4.2,3.3)); ax.bar(np.arange(len(names)),[vals[m] for m in names],color=[PALETTE[m] for m in names]); ax.set_xticks(np.arange(len(names)),[LABEL[m] for m in names]); ax.set_ylabel('Conditional domain-probe BAcc'); ax.set_ylim(0,1); ax.set_title('Diagnostic acquisition-domain leakage'); _save(fig,out/'Fig13_domain_leakage_diagnostic',cfg)


def fig_ablation(cfg,out,tables):
    q=_read(tables/'TableS01_ablation_statistics.csv')
    if q is None or q.empty: return
    q=q.sort_values('mean_delta_auroc'); fig,ax=plt.subplots(figsize=(5.8,3.9)); y=np.arange(len(q)); ax.barh(y,q.mean_delta_auroc,color=['#a0a0a0' if x<0 else PALETTE['unimetricnet'] for x in q.mean_delta_auroc]); ax.axvline(0,color='#555555',lw=.8); labs=[x.replace('no_','− ').replace('_',' ') for x in q.ablation]; ax.set_yticks(y,labs); ax.set_xlabel('Full UniMETRICNet AUROC − ablated AUROC'); ax.set_title('Matched strict-LODO ablation effects'); _save(fig,out/'Fig14_ablation_effects',cfg)


def fig_complexity(cfg,out,tables):
    q=_read(tables/'Table14_computational_profile.csv'); l=_read(tables/'Table03c_primary_LODO_macro.csv')
    if q is None or l is None or q.empty or l.empty: return
    perf=l.set_index('model').macro_auroc; prof=q.groupby('model').agg(parameters=('parameters','mean'),latency=('latency_ms_mean','mean')).reset_index(); fig,ax=plt.subplots(figsize=(4.9,3.6))
    for r in prof.itertuples():
        if r.model not in perf: continue
        ax.scatter(r.parameters/1e6,perf[r.model],s=40,color=PALETTE.get(r.model,'#777777')); ax.annotate(LABEL.get(r.model,r.model),(r.parameters/1e6,perf[r.model]),xytext=(4,3),textcoords='offset points',fontsize=7.2)
    ax.set_xlabel('Parameters (million)'); ax.set_ylabel('Macro strict-LODO AUROC'); ax.set_title('Generalization–complexity trade-off'); _save(fig,out/'Fig15_generalization_complexity',cfg)




def fig_measurement_summary(cfg,out,tables):
    p=tables/'Table17_TIM_measurement_performance_summary.csv'
    if not p.exists(): return
    q=pd.read_csv(p)
    if q.empty: return
    r=q.iloc[0]; fig,axs=plt.subplots(1,3,figsize=(7.16,2.65),constrained_layout=True)
    vals=[float(r.get('macro_lodo_auroc',np.nan)),float(r.get('worst_lodo_auroc',np.nan))]
    axs[0].bar([0,1],vals,color=[PALETTE['unimetricnet'],'#777777']); axs[0].axhline(.5,ls='--',lw=.8,color='#999999'); axs[0].set_xticks([0,1],['Macro','Worst']); axs[0].set_ylim(.3,1); axs[0].set_ylabel('AUROC'); axs[0].set_title('External discrimination')
    rng=float(r.get('KAUH_same_patient_probability_range',np.nan)); imp=float(r.get('KAUH_stability_improvement_vs_resnet',np.nan)); axs[1].bar([0],[rng],color=PALETTE['unimetricnet']); axs[1].axhline(.15,ls='--',lw=.8,color='#999999'); axs[1].set_xticks([0],['UniMETRICNet']); axs[1].set_ylim(0,max(.35,rng*1.25 if np.isfinite(rng) else .35)); axs[1].set_ylabel('Mean probability range'); axs[1].set_title(f'KAUH repeatability\nΔ vs ResNet={imp*100:.0f}%' if np.isfinite(imp) else 'KAUH repeatability')
    ret=float(r.get('worst_stress_auroc_retention',np.nan)); ece=float(r.get('macro_ece',np.nan)); axs[2].bar([0,1],[ret,ece],color=['#4f772d','#9c755f']); axs[2].axhline(.85,ls='--',lw=.7,color='#777777'); axs[2].set_xticks([0,1],['Stress\nretention','Macro\nECE']); axs[2].set_ylim(0,1); axs[2].set_title('Reliability characteristics')
    fig.suptitle('TIM-oriented measurement-performance summary',fontsize=9.5,weight='bold'); _save(fig,out/'Fig22_TIM_measurement_performance_summary',cfg)

def generate_all(cfg):
    _style(cfg); out=ensure_dir(Path(cfg['project']['output_dir'])/'figures'); tables=Path(cfg['project']['output_dir'])/'tables'; fig_protocol(cfg,out); fig_architecture(cfg,out); fig_preprocessing(cfg,out); fig_dataset_composition(cfg,out); fig_lodo(cfg,out,tables); fig_lodo_macro(cfg,out,tables); fig_lodo_rocs(cfg,out,tables); fig_threshold_transport(cfg,out,tables); fig_calibration(cfg,out,tables); fig_filter(cfg,out,tables); fig_stress(cfg,out,tables); fig_expert_weights(cfg,out,tables); fig_domain(cfg,out,tables); fig_ablation(cfg,out,tables); fig_complexity(cfg,out,tables); fig_lodo_forest(cfg,out,tables); fig_tsne(cfg,out,tables); fig_explainability(cfg,out,tables); fig_quality_strata(cfg,out,tables); fig_risk_coverage(cfg,out,tables); fig_error_taxonomy(cfg,out,tables); fig_measurement_summary(cfg,out,tables); fig_source_evidence(cfg,out,tables); fig_hard_negative_margins(cfg,out,tables); print('Publication figures:',out); return out

# ---- v8.0 additional TIM figures -------------------------------------------------
def fig_lodo_forest(cfg,out,tables):
    ci=_read(tables/'Table03d_primary_LODO_bootstrap_CI.csv'); ens=_read(tables/'Table03b_primary_LODO_ensemble.csv')
    if ci is None or ens is None or ci.empty or ens.empty: return
    q=ci[ci.metric.eq('auroc')].copy()
    if q.empty: return
    targets=[t for t in cfg['experiments']['lodo_targets'] if t in set(q.target)]
    fig,ax=plt.subplots(figsize=(5.8,3.25)); y=np.arange(len(targets))
    for i,t in enumerate(targets):
        r=q[q.target.eq(t)].iloc[0]; point=float(ens[(ens.target_or_setting==t)&(ens.model=='unimetricnet')].auroc.iloc[0]); ax.errorbar(point,i,xerr=[[point-float(r.ci_low)],[float(r.ci_high)-point]],fmt='o',capsize=3,color=PALETTE['unimetricnet'],label='UniMETRICNet' if i==0 else None)
        b=ens[(ens.target_or_setting==t)&(~ens.model.eq('unimetricnet'))]
        if len(b):
            br=b.loc[b.auroc.idxmax()]; ax.scatter(float(br.auroc),i,marker='s',s=28,color='#777777',label='Strongest matched baseline' if i==0 else None)
    ax.axvline(.5,ls='--',lw=.8,color='#999999'); ax.set_yticks(y,targets); ax.set_xlabel('Patient-level AUROC (95% bootstrap CI)'); ax.set_xlim(.25,1.0); ax.set_title('Strict LODO discrimination with uncertainty'); ax.legend(fontsize=7.4,loc='lower right'); _save(fig,out/'Fig16_LODO_forest_CI',cfg)


def fig_tsne(cfg,out,tables):
    seed=int(cfg['training']['seeds'][0])
    for target in cfg['experiments']['lodo_targets']:
        p=Path(cfg['project']['output_dir'])/'runs'/'lodo'/target/'unimetricnet'/f'seed_{seed}'/'tsne_coordinates.csv'
        if not p.exists(): continue
        q=pd.read_csv(p)
        if q.empty: continue
        fig,axs=plt.subplots(1,2,figsize=(7.16,3.15))
        # Disease separation
        for y,label,marker in [(0,'Non-COPD','o'),(1,'COPD','^')]:
            g=q[q.y.eq(y)]; axs[0].scatter(g.tsne_1,g.tsne_2,s=18,alpha=.68,label=label,marker=marker)
        axs[0].set_title(f'{target}: disease structure'); axs[0].set_xticks([]); axs[0].set_yticks([]); axs[0].legend(fontsize=7)
        # Acquisition structure
        for origin,g in q.groupby('origin_dataset'): axs[1].scatter(g.tsne_1,g.tsne_2,s=18,alpha=.68,label=str(origin))
        axs[1].set_title(f'{target}: acquisition structure'); axs[1].set_xticks([]); axs[1].set_yticks([]); axs[1].legend(fontsize=6.5)
        fig.suptitle('Post-hoc t-SNE of frozen patient embeddings',fontsize=9.5,weight='bold'); _save(fig,out/f'Fig17_tSNE_{target}',cfg)


def fig_explainability(cfg,out,tables):
    idx=Path(cfg['project']['output_dir'])/'explainability'/'explanation_index.csv'
    if not idx.exists(): return
    q=pd.read_csv(idx)
    if q.empty: return
    targets=[t for t in cfg['experiments']['lodo_targets'] if t in set(q.target)]
    fig,axs=plt.subplots(len(targets),2,figsize=(7.16,max(4.1,2.25*len(targets))),squeeze=False,constrained_layout=True)
    for i,t in enumerate(targets):
        for j,(lab,title) in enumerate([('non_copd','Non-COPD'),('copd','COPD')]):
            r=q[(q.target==t)&(q.class_label==lab)]; ax=axs[i,j]
            if r.empty: ax.axis('off'); continue
            z=np.load(Path(cfg['project']['output_dir'])/'explainability'/str(r.iloc[0]['file'])); spec=z['spectrogram']; sal=z['saliency']
            ax.imshow(spec,origin='lower',aspect='auto',cmap='gray_r'); ax.imshow(sal,origin='lower',aspect='auto',cmap='magma',alpha=.45,vmin=0,vmax=1)
            ax.text(.02,.96,f'{t} | {title}',transform=ax.transAxes,ha='left',va='top',fontsize=7.0,weight='bold',bbox=dict(facecolor='white',alpha=.78,edgecolor='none',pad=1.2))
            ax.text(.98,.96,f'p={float(z["probability"]):.2f}',transform=ax.transAxes,ha='right',va='top',fontsize=6.7,bbox=dict(facecolor='white',alpha=.78,edgecolor='none',pad=1.2))
            ax.set_xlabel('Time frame'); ax.set_ylabel('Mel bin')
    fig.suptitle('Frozen unseen-database acoustic saliency (input × gradient)',fontsize=9.4,weight='bold')
    _save(fig,out/'Fig18_explainability_saliency',cfg)


def fig_quality_strata(cfg,out,tables):
    pred=tables/'ensemble_predictions'; rows=[]
    from .metrics import roc_auc_np
    for target in cfg['experiments']['lodo_targets']:
        p=pred/f'lodo__{target}__unimetricnet.csv'
        if not p.exists(): continue
        z=pd.read_csv(p)
        if 'quality_index' not in z: continue
        try: z['quartile']=pd.qcut(z.quality_index,4,labels=['Q1 low','Q2','Q3','Q4 high'],duplicates='drop')
        except Exception: continue
        for k,g in z.groupby('quartile',observed=True):
            if g.y.nunique()<2: continue
            rows.append(dict(target=target,quartile=str(k),auroc=roc_auc_np(g.y,g.probability),n=len(g)))
    if not rows: return
    q=pd.DataFrame(rows); order=['Q1 low','Q2','Q3','Q4 high']; fig,ax=plt.subplots(figsize=(5.3,3.35)); x=np.arange(4)
    for t,g in q.groupby('target'):
        vals=[g[g.quartile==o].auroc.mean() for o in order]; ax.plot(x,vals,marker='o',lw=1.4,label=t)
    ax.axhline(.5,ls='--',lw=.8,color='#999999'); ax.set_xticks(x,order); ax.set_ylim(.3,1); ax.set_ylabel('AUROC'); ax.set_title('Performance across signal-quality strata'); ax.legend(fontsize=7); _save(fig,out/'Fig19_quality_stratified_AUROC',cfg)


def fig_risk_coverage(cfg,out,tables):
    rows=[]
    for target in cfg['experiments']['lodo_targets']:
        for seed in cfg['training']['seeds']:
            p=Path(cfg['project']['output_dir'])/'runs'/'lodo'/target/'unimetricnet'/f'seed_{seed}'/'risk_coverage.csv'
            if not p.exists(): continue
            z=pd.read_csv(p); z['target']=target; z['seed']=seed; rows.append(z)
    if not rows: return
    q=pd.concat(rows,ignore_index=True); fig,ax=plt.subplots(figsize=(4.8,3.4))
    errcol='error_rate' if 'error_rate' in q.columns else ('risk' if 'risk' in q.columns else None)
    if errcol is None: return
    for t,g in q.groupby('target'):
        m=g.groupby('coverage')[errcol].mean().sort_index(); ax.plot(m.index,m.values,marker='o',lw=1.4,label=t)
    ax.set_xlabel('Coverage retained'); ax.set_ylabel('Selective error / risk'); ax.set_title('Uncertainty-based risk–coverage behavior'); ax.legend(fontsize=7); _save(fig,out/'Fig20_risk_coverage',cfg)


def fig_error_taxonomy(cfg,out,tables):
    p=tables/'Table16_category_error_taxonomy.csv'
    if not p.exists(): return
    q=pd.read_csv(p)
    if q.empty: return
    targets=[t for t in cfg['experiments']['lodo_targets'] if t in set(q.target)]
    cats=sorted(q.category.astype(str).unique())
    mat=np.full((len(cats),len(targets)),np.nan)
    for i,c in enumerate(cats):
        for j,t in enumerate(targets):
            g=q[(q.category.astype(str)==c)&(q.target==t)]
            if len(g): mat[i,j]=float(g.error_rate.mean())
    fig,ax=plt.subplots(figsize=(5.8,max(3.2,.30*len(cats)+1.2)))
    im=ax.imshow(mat,aspect='auto',vmin=0,vmax=1,cmap='magma_r')
    ax.set_xticks(np.arange(len(targets)),targets); ax.set_yticks(np.arange(len(cats)),cats)
    ax.set_title('Unseen-database error taxonomy by clinical category'); ax.set_xlabel('Held-out database'); ax.set_ylabel('Clinical category')
    cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.03); cb.set_label('Patient-level error rate')
    for i in range(len(cats)):
        for j in range(len(targets)):
            if np.isfinite(mat[i,j]): ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=6.5,color='black' if mat[i,j]<.65 else 'white')
    _save(fig,out/'Fig21_category_error_taxonomy',cfg)


def fig_source_evidence(cfg,out,tables):
    p=tables/'Table18_source_evidence_projection.csv'
    if not p.exists(): return
    q=pd.read_csv(p)
    if q.empty: return
    fig,axs=plt.subplots(1,2,figsize=(7.16,3.0),constrained_layout=True)
    targets=[t for t in cfg['experiments']['lodo_targets'] if t in set(q.target)]
    x=np.arange(len(targets)); gain=[q[q.target.eq(t)].validation_gain.mean() for t in targets]; strength=[q[q.target.eq(t)].strength.mean() for t in targets]; blend=[q[q.target.eq(t)].blend.mean() for t in targets]
    axs[0].bar(x,gain,color=PALETTE['unimetricnet']); axs[0].axhline(0,color='#777777',lw=.8); axs[0].set_xticks(x,targets); axs[0].set_ylabel('Source-validation robust-score gain'); axs[0].set_title('Source-only evidence selection')
    w=.34; axs[1].bar(x-w/2,strength,w,label='Nuisance projection',color='#6f6f6f'); axs[1].bar(x+w/2,blend,w,label='Metric-evidence blend',color=PALETTE['unimetricnet']); axs[1].set_xticks(x,targets); axs[1].set_ylim(0,1); axs[1].set_title('Frozen source-selected controls'); axs[1].legend(fontsize=7)
    _save(fig,out/'Fig23_source_evidence_projection',cfg)


def fig_hard_negative_margins(cfg,out,tables):
    p=tables/'Table19_hard_negative_probability_margins.csv'
    if not p.exists(): return
    q=pd.read_csv(p)
    if q.empty: return
    piv=q.pivot_table(index='negative_category',columns='target',values='probability_margin',aggfunc='mean')
    fig,ax=plt.subplots(figsize=(6.4,max(3.0,.32*len(piv)+1.1))); im=ax.imshow(piv.to_numpy(float),aspect='auto',cmap='coolwarm',vmin=-.25,vmax=.25); ax.set_xticks(np.arange(len(piv.columns)),piv.columns); ax.set_yticks(np.arange(len(piv.index)),piv.index); ax.set_title('COPD probability margin versus hard-negative clinical groups'); ax.set_xlabel('Held-out database'); ax.set_ylabel('Negative clinical group'); cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.03); cb.set_label('Mean p(COPD) margin')
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v=piv.iloc[i,j]
            if np.isfinite(v): ax.text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=6.4,color='white' if abs(v)>.13 else 'black')
    _save(fig,out/'Fig24_hard_negative_separation',cfg)
