from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import DataIntegrityError


def patient_table(frame):
    p=frame[frame.main_role.astype(int).eq(1)].drop_duplicates('person_id').copy()
    for sid,g in frame[frame.main_role.astype(int).eq(1)].groupby('person_id'):
        if g.copd_label.nunique()!=1: raise DataIntegrityError(f'Patient label conflict: {sid}')
    return p.reset_index(drop=True)


def _strata(p,min_count=2):
    s=p.logical_dataset.astype(str)+'|'+p.copd_label.astype(str)
    counts=s.value_counts()
    return s.where(s.map(counts)>=int(min_count),p.copd_label.astype(str))


def _round_robin_folds(labels,nfolds,seed):
    labels=np.asarray(labels,dtype=str); rng=np.random.default_rng(seed); bins=[[] for _ in range(nfolds)]
    # Large strata are spread evenly. Assignment is rotated by stratum to reduce fold-size drift.
    for gi,label in enumerate(sorted(np.unique(labels))):
        ids=np.flatnonzero(labels==label); rng.shuffle(ids); offset=gi % nfolds
        for j,idx in enumerate(ids): bins[(offset+j)%nfolds].append(int(idx))
    return [np.asarray(sorted(x),dtype=int) for x in bins]


def patient_cv(frame,folds,seed):
    p=patient_table(frame); s=_strata(p,min_count=2)
    label_counts=p.copd_label.value_counts()
    n=min(int(folds),int(label_counts.min())) if len(label_counts)>=2 else 0
    if n<2: raise DataIntegrityError('Insufficient subjects for patient-level CV')
    bins=_round_robin_folds(s.to_numpy(),n,seed)
    # Ensure every test fold has at least one case and one non-case; fall back to label-only stratification if needed.
    ok=all(p.iloc[ix].copd_label.nunique()==2 for ix in bins if len(ix))
    if not ok: bins=_round_robin_folds(p.copd_label.astype(str).to_numpy(),n,seed)
    out=[]; allidx=np.arange(len(p))
    for fold,te in enumerate(bins):
        tr=np.setdiff1d(allidx,te,assume_unique=False)
        out.append(dict(fold=fold,train_people=p.iloc[tr].person_id.tolist(),test_people=p.iloc[te].person_id.tolist()))
    return out


def _allocate_group(ids,fractions,rng):
    ids=np.asarray(ids,int).copy(); rng.shuffle(ids); n=len(ids)
    raw=np.asarray(fractions,float)*n; counts=np.floor(raw).astype(int); rem=n-counts.sum()
    for i in np.argsort(-(raw-counts))[:rem]: counts[i]+=1
    # Training partition (index 0) must not become empty for a nonempty stratum.
    if n and counts[0]==0:
        donor=int(np.argmax(counts[1:])+1) if len(counts)>1 else 0
        if counts[donor]>0: counts[donor]-=1; counts[0]+=1
    out=[]; s=0
    for c in counts: out.append(ids[s:s+c]); s+=c
    return out


def _partition(p,fractions,seed,use_domain=True):
    rng=np.random.default_rng(seed); labels=_strata(p,3).to_numpy() if use_domain else p.copd_label.astype(str).to_numpy(); parts=[[] for _ in fractions]
    for label in sorted(np.unique(labels)):
        ids=np.flatnonzero(labels==label); alloc=_allocate_group(ids,fractions,rng)
        for j,a in enumerate(alloc): parts[j].extend(a.tolist())
    return [np.asarray(sorted(x),int) for x in parts]


def train_val_cal_split(people_frame,val_fraction,cal_fraction,seed):
    p=people_frame.reset_index(drop=True).copy(); vf=float(val_fraction); cf=float(cal_fraction); tf=1-vf-cf
    if tf<=0 or vf<=0 or cf<=0: raise ValueError('train/validation/calibration fractions must all be positive')
    for use_domain in (True,False):
        tr,va,ca=_partition(p,[tf,vf,cf],seed,use_domain)
        if min(len(tr),len(va),len(ca))==0: continue
        if all(p.iloc[ix].copd_label.nunique()==2 for ix in [tr,va,ca]):
            return p.iloc[tr].person_id.tolist(),p.iloc[va].person_id.tolist(),p.iloc[ca].person_id.tolist()
    raise DataIntegrityError('Could not construct train/validation/calibration partitions containing both endpoint classes')


def subset(frame,people): return frame[frame.person_id.isin(set(people))].copy()


def assert_disjoint(*groups):
    sets=[set(g) for g in groups]
    for i in range(len(sets)):
        for j in range(i):
            overlap=sets[i]&sets[j]
            if overlap: raise DataIntegrityError(f'Patient leakage across partitions: {sorted(overlap)[:5]}')
