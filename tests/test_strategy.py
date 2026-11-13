import numpy as np
import pandas as pd
from hicopd_unimetricnet.evidence import fit_nuisance_basis, project, fit_taxonomy_prototypes, metric_score, fit_source_evidence, refit_source_evidence, apply_source_evidence


def _frame(n=24):
    y=np.array([0,1]*(n//2),int)
    origins=np.array(['A','B','C']*(n//3),object)[:n]
    tax=np.where(y==1,0,np.resize(np.array([1,2,3,4]),n))
    return pd.DataFrame(dict(y=y,origin_dataset=origins,taxonomy=tax,probability=np.linspace(.2,.8,n),expert_prob_anchor=np.linspace(.2,.8,n)))


def test_class_conditional_nuisance_basis_is_finite():
    rng=np.random.default_rng(3); f=_frame(24); z=rng.normal(size=(24,12))
    # add a strong acquisition direction independent of label
    for i,o in enumerate(f.origin_dataset): z[i,0]+={'A':-2,'B':0,'C':2}[o]
    b=fit_nuisance_basis(z,f,max_dim=4,min_count=2)
    assert b.shape[0]==12 and b.shape[1]<=4
    zp=project(z,b,.75); assert np.isfinite(zp).all()
    assert np.allclose(np.linalg.norm(zp,axis=1),1,atol=1e-5)


def test_hard_negative_metric_score_runs():
    rng=np.random.default_rng(4); f=_frame(24); z=rng.normal(size=(24,10)); z[f.y.to_numpy()==1,1]+=1.5
    p=fit_taxonomy_prototypes(z,f); s=metric_score(z,p,.2)
    assert s.shape==(24,) and np.isfinite(s).all()


def test_source_evidence_never_requires_target_labels():
    rng=np.random.default_rng(5); tr=_frame(30); va=_frame(18); te=_frame(12)
    ztr=rng.normal(size=(30,14)); zva=rng.normal(size=(18,14)); zte=rng.normal(size=(12,14))
    ztr[tr.y.to_numpy()==1,2]+=1; zva[va.y.to_numpy()==1,2]+=1
    cfg={'evaluation':{'source_evidence_projection':{'enabled':True,'max_nuisance_dim':3,'min_origin_class_count':2,'strength_grid':[0,.5,1.0],'blend_grid':[0,.2,.4],'max_anchor_auc_drop':.1,'min_gain_to_use_aux':0,'hard_negative_temperature':.2}}}
    state=fit_source_evidence(ztr,tr,zva,va,cfg)
    state=refit_source_evidence(state,np.r_[ztr,zva],pd.concat([tr,va],ignore_index=True),cfg)
    # erase target labels to demonstrate application is label-free
    te2=te.copy(); te2['y']=0
    out,z=apply_source_evidence(state,zte,te2,cfg)
    assert len(out)==12 and z.shape==zte.shape and np.isfinite(out.probability).all()
