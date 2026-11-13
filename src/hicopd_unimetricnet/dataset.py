from __future__ import annotations
from functools import lru_cache
import math
import numpy as np
try:
    import torch
    from torch.utils.data import Dataset, DataLoader, Sampler
except ImportError:
    torch = None; Dataset = object; DataLoader = object; Sampler = object


TAXONOMY_NAMES = ['COPD','Healthy','ObstructiveOther','InfectionInflammatory','OtherChronic']

def taxonomy_id(category):
    c=str(category)
    if c=='COPD': return 0
    if c=='Healthy': return 1
    if c in {'Asthma','Bronchiectasis'}: return 2
    if c in {'Pneumonia','URTI','Bronchiolitis','Pulmonary Infection'}: return 3
    if c in {'Heart_Failure','ILD'}: return 4
    return -1


@lru_cache(maxsize=8192)
def load_bundle(path):
    with np.load(path, allow_pickle=False) as z:
        return z['specs'].copy(), z['phys'].copy(), z['quality'].copy()


def _fit_robust_scaler(frame, key, max_segments_per_recording=64):
    vals=[]
    for p in frame.feature_path.unique():
        s,ph,q=load_bundle(str(p)); x=ph if key=='phys' else q
        vals.append(x[:min(len(x),max_segments_per_recording)])
    dim=22 if key=='phys' else 8
    x=np.concatenate(vals,axis=0) if vals else np.zeros((1,dim),dtype=np.float32)
    med=np.median(x,axis=0).astype(np.float32); q1=np.quantile(x,.25,axis=0).astype(np.float32); q3=np.quantile(x,.75,axis=0).astype(np.float32)
    scale=(q3-q1).astype(np.float32); scale[scale<1e-3]=1.0
    return {'median':med,'scale':scale}


def fit_phys_scaler(frame,max_segments_per_recording=64):
    return _fit_robust_scaler(frame,'phys',max_segments_per_recording)


def fit_quality_scaler(frame,max_segments_per_recording=64):
    return _fit_robust_scaler(frame,'quality',max_segments_per_recording)


def apply_scaler(x,scaler):
    if scaler is None: return x.astype(np.float32)
    return np.clip((x-scaler['median'])/scaler['scale'],-8,8).astype(np.float32)


def _smooth_curve(rng,f_bins,knots,std):
    xp=np.linspace(0,f_bins-1,max(2,int(knots))); vals=rng.normal(0,float(std),len(xp)); return np.interp(np.arange(f_bins),xp,vals).astype(np.float32)


def instrument_randomize_views(views,rng,cfg):
    """Training-only device/noise perturbation in representation space.

    Channel 0=log-Mel is most affected; PCEN/RISN receive attenuated transfer-function
    perturbations because these normalizers should be less channel-sensitive by design.
    """
    ac=cfg['augmentation']; ir=ac.get('instrument_response',{})
    if not ac.get('enabled',True) or rng.random()>float(ac.get('probability',1.0)): return views.copy()
    y=views.copy().astype(np.float32); f=y.shape[-2]
    if rng.random()<float(ir.get('probability',.8)):
        curve=_smooth_curve(rng,f,ir.get('eq_knots',8),ir.get('eq_std',.28)); tilt=np.linspace(-1,1,f,dtype=np.float32)*rng.normal(0,float(ir.get('spectral_tilt_std',.20))); curve=curve+tilt
        gains=[1.0,.35,.15]
        for c,g in enumerate(gains[:y.shape[1]]): y[:,c,:,:]+=g*curve[None,:,None]
        if rng.random()<float(ir.get('lowpass_probability',.25)):
            cut=int(rng.uniform(float(ir.get('lowpass_min_fraction',.65)),.98)*f)
            for c,g in enumerate(gains[:y.shape[1]]): y[:,c,cut:,:]+=g*rng.uniform(-.8,-.20)
        if rng.random()<float(ir.get('highpass_probability',.15)):
            cut=int(rng.uniform(.03,float(ir.get('highpass_max_fraction',.12)))*f)
            for c,g in enumerate(gains[:y.shape[1]]): y[:,c,:cut,:]+=g*rng.uniform(-.6,-.15)
        if rng.random()<float(ir.get('dynamic_range_probability',.18)):
            s=float(ir.get('dynamic_range_strength',.15)); expo=1+rng.uniform(-s,s); y=np.sign(y)*(np.abs(y)**expo)
    if rng.random()<float(ac.get('additive_noise_probability',.30)):
        lo,hi=ac.get('additive_noise_snr_db',[15,30]); snr=float(rng.uniform(lo,hi)); std=float(np.std(y[:,0])+1e-6)/(10**(snr/20)); y+=rng.normal(0,std,y.shape).astype(np.float32)
    if rng.random()<float(ac.get('gain_probability',.25)):
        lo,hi=ac.get('gain_db',[-4,4]); gain=float(rng.uniform(lo,hi))/20.0; y[:,0]+=gain
    for _ in range(int(ac.get('frequency_masks',0))):
        w=int(rng.integers(0,min(int(ac.get('max_frequency_bins',0)),f)+1)); st=int(rng.integers(0,max(1,f-w+1))); y[:,:,st:st+w,:]=0
    t=y.shape[-1]
    for _ in range(int(ac.get('time_masks',0))):
        w=int(rng.integers(0,min(int(ac.get('max_time_frames',0)),t)+1)); st=int(rng.integers(0,max(1,t-w+1))); y[:,:,:,st:st+w]=0
    std=float(ac.get('feature_jitter_std',0))
    if std>0: y+=rng.normal(0,std,y.shape).astype(np.float32)
    return y.astype(np.float32)


def _recording_round_robin(g,rng,count,training):
    recs=g.to_dict('records'); candidates=[]
    for r in recs:
        n=int(r['n_segments']) if ('n_segments' in r and str(r['n_segments']).isdigit() and int(r['n_segments'])>0) else len(load_bundle(str(r['feature_path']))[0]); idx=rng.permutation(n) if training else np.unique(np.linspace(0,n-1,min(n,count),dtype=int)); candidates.append([(r,int(i)) for i in idx])
    choices=[]
    if not candidates: return choices
    for k in range(max(len(x) for x in candidates)):
        for seq in candidates:
            if k<len(seq): choices.append(seq[k])
            if len(choices)>=count: break
        if len(choices)>=count: break
    return choices


class PatientBags(Dataset):
    def __init__(self,frame,domain_map,cfg,training=False,seed=42,role='main',filter_mode=None,phys_scaler=None,quality_scaler=None):
        self.cfg=cfg; self.training=training; self.seed=seed; self.epoch=0; self.domain_map=domain_map; self.phys_scaler=phys_scaler; self.quality_scaler=quality_scaler
        f=frame.copy()
        if role=='main': f=f[f.main_role.astype(int).eq(1)]
        elif role=='filter': f=f[f.filter_intervention_role.astype(int).eq(1)]
        if filter_mode is not None: f=f[f.filter_mode.astype(str).eq(str(filter_mode))].copy()
        self.groups={sid:g.sort_values('recording_id') for sid,g in f.groupby('person_id',sort=True)}; self.items=sorted(self.groups)
    def __len__(self): return len(self.items)
    def _bag(self,g,rng,count):
        choices=_recording_round_robin(g,rng,count,self.training); sp=[]; ph=[]; qu=[]
        for r,j in choices:
            specs,phys,quality=load_bundle(str(r['feature_path'])); sp.append(specs[j]); ph.append(phys[j]); qu.append(quality[j])
        return (np.stack(sp).astype(np.float32), apply_scaler(np.stack(ph).astype(np.float32),self.phys_scaler),
                apply_scaler(np.stack(qu).astype(np.float32),self.quality_scaler))
    def __getitem__(self,i):
        sid=self.items[i]; g=self.groups[sid]; rng=np.random.default_rng(self.seed+self.epoch*1000003+i); count=int(self.cfg['training']['train_segments_per_patient'] if self.training else self.cfg['training']['eval_segments_per_patient'])
        spec,phys,quality=self._bag(g,rng,count)
        if self.training:
            x1=instrument_randomize_views(spec,rng,self.cfg); x2=instrument_randomize_views(spec,rng,self.cfg)
        else: x1=spec; x2=spec.copy()
        r=g.iloc[0]; measurement_domain=str(r.origin_dataset)
        return dict(x=torch.from_numpy(x1),x2=torch.from_numpy(x2),phys=torch.from_numpy(phys),quality=torch.from_numpy(quality),y=int(r.copd_label),taxonomy=taxonomy_id(r.category),domain=int(self.domain_map.get(measurement_domain,-1)),
            person_id=sid,subject_id=str(r.subject_id),logical_dataset=str(r.logical_dataset),origin_dataset=str(r.origin_dataset),category=str(r.category),age=float(r.age) if str(r.age) not in {'','nan'} else float('nan'),sex=str(r.sex),filter_mode=str(r.filter_mode))


def collate(items):
    b=len(items); n=max(len(z['x']) for z in items); c,f,t=items[0]['x'].shape[-3:]; pdim=items[0]['phys'].shape[-1]; qdim=items[0]['quality'].shape[-1]
    x=torch.zeros(b,n,c,f,t); x2=torch.zeros_like(x); phys=torch.zeros(b,n,pdim); quality=torch.zeros(b,n,qdim); mask=torch.zeros(b,n,dtype=torch.bool)
    for i,z in enumerate(items):
        k=len(z['x']); x[i,:k]=z['x']; x2[i,:k]=z['x2']; phys[i,:k]=z['phys']; quality[i,:k]=z['quality']; mask[i,:k]=True
    out=dict(x=x,x2=x2,phys=phys,quality=quality,mask=mask,y=torch.tensor([z['y'] for z in items],dtype=torch.long),taxonomy=torch.tensor([z['taxonomy'] for z in items],dtype=torch.long),domain=torch.tensor([z['domain'] for z in items],dtype=torch.long))
    for k in ['person_id','subject_id','logical_dataset','origin_dataset','category','age','sex','filter_mode']: out[k]=[z[k] for z in items]
    return out


class BalancedPatientBatchSampler(Sampler):
    """Balance logical database x endpoint class and KAUH/RDTR within HYBRID COPD."""
    def __init__(self,ds,batch_size,seed=42):
        self.ds=ds; self.batch_size=int(batch_size); self.seed=int(seed); self.epoch=0; cells={}; origin_balance=bool(ds.cfg['data'].get('hybrid_origin_balance',True))
        for i,sid in enumerate(ds.items):
            r=ds.groups[sid].iloc[0]; logical=str(r.logical_dataset); y=int(r.copd_label); origin=str(r.origin_dataset); tax=taxonomy_id(r.category);
            if origin_balance and logical=='HYBRID' and y==1: sub=origin
            elif y==0: sub=f'TAX{tax}'
            else: sub='ALL'
            cells.setdefault((logical,y,sub),[]).append(i)
        self.cells={k:v for k,v in cells.items() if v}; self.keys=sorted(self.cells); self.n=max(1,math.ceil(len(ds)/max(1,self.batch_size)))
    def __len__(self): return self.n
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch*1009); self.epoch+=1; pools={k:list(rng.permutation(v)) for k,v in self.cells.items()}; ptr={k:0 for k in self.keys}
        for _ in range(self.n):
            batch=[]
            while len(batch)<self.batch_size:
                progressed=False
                for k in self.keys:
                    if len(batch)>=self.batch_size: break
                    if ptr[k]>=len(pools[k]): pools[k]=list(rng.permutation(self.cells[k])); ptr[k]=0
                    idx=int(pools[k][ptr[k]]); ptr[k]+=1
                    if idx not in batch: batch.append(idx); progressed=True
                if not progressed: break
            yield batch


def make_loader(frame,domain_map,cfg,training,seed,role='main',filter_mode=None,phys_scaler=None,quality_scaler=None):
    ds=PatientBags(frame,domain_map,cfg,training,seed,role,filter_mode,phys_scaler,quality_scaler)
    if training and cfg['training'].get('class_domain_balanced_sampler',True) and len(ds):
        bs=BalancedPatientBatchSampler(ds,int(cfg['training']['batch_size_patients']),seed); return DataLoader(ds,batch_sampler=bs,num_workers=int(cfg['training']['num_workers']),collate_fn=collate,pin_memory=False)
    return DataLoader(ds,batch_size=int(cfg['training']['batch_size_patients']),shuffle=training,num_workers=int(cfg['training']['num_workers']),collate_fn=collate,pin_memory=False,generator=torch.Generator().manual_seed(seed))


class FilterPairDataset(Dataset):
    """Same-patient KAUH B/D/E filter pairs used only when that patient is in training."""
    def __init__(self,frame,people,cfg,seed,phys_scaler,quality_scaler):
        f=frame[(frame.origin_dataset.eq('KAUH')) & frame.person_id.isin(set(people)) & frame.filter_intervention_role.astype(int).eq(1)].copy(); self.cfg=cfg; self.seed=seed; self.phys_scaler=phys_scaler; self.quality_scaler=quality_scaler; self.epoch=0
        self.groups={}
        for sid,g in f.groupby('person_id'):
            modes={m:q for m,q in g.groupby('filter_mode') if len(q)}
            if len(modes)>=2: self.groups[sid]=modes
        self.items=sorted(self.groups)
    def __len__(self): return len(self.items)
    def _mode_bag(self,g,rng,count):
        choices=_recording_round_robin(g,rng,count,True); sp=[]; ph=[]; qu=[]
        for r,j in choices:
            s,p,q=load_bundle(str(r['feature_path'])); sp.append(s[j]); ph.append(p[j]); qu.append(q[j])
        return (np.stack(sp).astype(np.float32),apply_scaler(np.stack(ph).astype(np.float32),self.phys_scaler),apply_scaler(np.stack(qu).astype(np.float32),self.quality_scaler))
    def __getitem__(self,i):
        sid=self.items[i]; modes=self.groups[sid]; keys=sorted(modes); rng=np.random.default_rng(self.seed+self.epoch*100003+i); pick=rng.choice(len(keys),2,replace=False); a,b=keys[int(pick[0])],keys[int(pick[1])]; count=max(2,int(self.cfg['training']['train_segments_per_patient']//2)); xa,pa,qa=self._mode_bag(modes[a],rng,count); xb,pb,qb=self._mode_bag(modes[b],rng,count)
        return dict(xa=torch.from_numpy(xa),pa=torch.from_numpy(pa),qa=torch.from_numpy(qa),xb=torch.from_numpy(xb),pb=torch.from_numpy(pb),qb=torch.from_numpy(qb),person_id=sid,mode_a=a,mode_b=b)


def collate_filter(items):
    def pack(keyx,keyp,keyq):
        n=max(len(z[keyx]) for z in items); c,f,t=items[0][keyx].shape[-3:]; pdim=items[0][keyp].shape[-1]; qdim=items[0][keyq].shape[-1]; x=torch.zeros(len(items),n,c,f,t); p=torch.zeros(len(items),n,pdim); q=torch.zeros(len(items),n,qdim); m=torch.zeros(len(items),n,dtype=torch.bool)
        for i,z in enumerate(items): k=len(z[keyx]); x[i,:k]=z[keyx]; p[i,:k]=z[keyp]; q[i,:k]=z[keyq]; m[i,:k]=True
        return x,p,q,m
    xa,pa,qa,ma=pack('xa','pa','qa'); xb,pb,qb,mb=pack('xb','pb','qb'); return dict(xa=xa,pa=pa,qa=qa,ma=ma,xb=xb,pb=pb,qb=qb,mb=mb,person_id=[z['person_id'] for z in items])


def make_filter_pair_loader(frame,people,cfg,seed,phys_scaler,quality_scaler):
    ds=FilterPairDataset(frame,people,cfg,seed,phys_scaler,quality_scaler)
    if not len(ds): return None
    bs=min(int(cfg['training'].get('paired_filter_patients_per_batch',4)),len(ds)); return DataLoader(ds,batch_size=max(1,bs),shuffle=True,num_workers=0,collate_fn=collate_filter,generator=torch.Generator().manual_seed(seed))
