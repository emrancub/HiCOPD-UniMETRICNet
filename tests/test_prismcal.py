
import numpy as np
import pandas as pd
from hicopd_unimetricnet.reliability_calibration import fit_prism_calnet,apply_prism_calnet

def _cfg():
    return {'evaluation':{'prism_calnet':{
        'enabled':True,'l2_grid':[.01,.1],'learning_rate':.03,'steps':80,
        'anchor_coef_min':.65,'anchor_coef_max':2.5,'correction_clip':1.5,
        'min_cv_gain':-1.0,'max_origin_auc_drop':1.0,'density_temperature':.3,
        'reliability_confidence_weight':.45,'reliability_density_weight':.35,'reliability_model_weight':.2},
        'source_evidence_projection':{'hard_negative_temperature':.2}}}

def _frame(n=24):
    y=np.array([0,1]*(n//2))
    neg=np.resize(np.array([1,2,3,4]),n); tax=np.where(y==1,0,neg)
    p=np.where(y==1,.68,.32)
    return pd.DataFrame({'person_id':[f'p{i}' for i in range(n)],'y':y,'taxonomy':tax,
        'origin_dataset':np.where(np.arange(n)%4<2,'A','B'),'probability':p,
        'quality_index':np.linspace(.4,.9,n),'entropy':np.ones(n)*.55,'model_reliability':np.linspace(.5,.9,n)})

def test_prismcal_source_only_and_finite():
    rng=np.random.default_rng(3); tr=_frame(24); va=_frame(24); ca=_frame(24); te=_frame(24)
    emb=lambda: rng.normal(size=(24,16))
    st=fit_prism_calnet(tr,emb(),va,emb(),ca,emb(),_cfg())
    out=apply_prism_calnet(st,te,emb(),_cfg())
    assert np.isfinite(out.probability).all()
    assert ((out.measurement_reliability>=0)&(out.measurement_reliability<=1)).all()

def test_prismcal_fallback_is_identity():
    rng=np.random.default_rng(4); tr=_frame(24); va=_frame(24); ca=_frame(24); te=_frame(24)
    emb=lambda: rng.normal(size=(24,12))
    st=fit_prism_calnet(tr,emb(),va,emb(),ca,emb(),_cfg(),disable=True)
    base=te.probability.to_numpy().copy()
    out=apply_prism_calnet(st,te,emb(),_cfg())
    assert np.allclose(base,out.probability.to_numpy())
