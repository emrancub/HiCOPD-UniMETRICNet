from __future__ import annotations
from pathlib import Path
import copy
import numpy as np
import pandas as pd
try:
    import soundfile as sf
except ImportError:
    sf = None
import yaml
from .config import load_config
from .utils import ensure_dir

def make_fixture(base,cfg):
    base=Path(base); rng=np.random.default_rng(7)
    def wav(path,f):
        ensure_dir(path.parent); sr=4000; t=np.arange(sr*4)/sr; x=.1*np.sin(2*np.pi*f*t)+rng.normal(0,.01,len(t)); sf.write(path,x,sr,subtype='PCM_16')
    c=copy.deepcopy(cfg); c['project'].update(cache_dir=str(base/'cache'),output_dir=str(base/'out'),synthetic=True); c['publication']['dpi']=120; c['publication']['figure_formats']=['png']; c['data']['strict_reference_counts']=False; c['integrity']['compute_pcm_hash']=True; c['training']['device']='cpu'; c['training']['amp']=False; c['training']['seeds']=[7]; c['training']['epochs']=1; c['training']['patience']=1
    ic=base/'ICBHI'; c['data']['icbhi_root']=str(ic); ensure_dir(ic/'audio_and_txt_files'); dx=[]; dm=[]
    cats=['Healthy','COPD']*6
    for i,cat in enumerate(cats,101): wav(ic/'audio_and_txt_files'/f'{i}_1b1_Al_sc_Litt3200.wav',150+(cat=='COPD')*150); dx.append([i,cat]); dm.append(f'{i}\t{40+i%20}\tM\t24\tNA\tNA')
    pd.DataFrame(dx).to_csv(ic/'patient_diagnosis.csv',header=False,index=False); (ic/'demographic_info.txt').write_text('\n'.join(dm))
    ka=base/'KAUH'; c['data']['kauh_root']=str(ka); ensure_dir(ka/'Audio Files')
    for i in range(12):
        cat='COPD' if i%2 else 'N'
        for m in 'BDE': wav(ka/'Audio Files'/f'{m}P{i+1}_{cat},I E N,{45+i},M.wav',170+(i%2)*140+ord(m)%5)
    rd=base/'RDTR'; c['data']['rdtr_root']=str(rd); ensure_dir(rd/'audio'); labels=[]
    for i in range(6):
        sid=f'H{i+1:03d}'; sev=1+i%4; labels.append([sid,sev]);
        for site in ['L1','R1']: wav(rd/'audio'/f'{sid}_{site}.wav',300+i*5)
    pd.DataFrame(labels,columns=['SubjectID','Severity']).to_excel(rd/'Labels.xlsx',index=False)
    br=base/'BRACETS'; c['data']['bracets_root']=str(br); md=ensure_dir(br/'Metadata'); metadata=[]; pairs=[]
    for i in range(12):
        sid=f'{i+1:02d}'; cat='COPD' if i%2 else ['Healthy','Asthma','ILD','Infection'][(i//2)%4]; place='Aveiro'; metadata.append([sid,50+i,'M',cat,place])
        name=f'{sid}_01_PBL.wav'; pairs.append([sid,place,f'{sid}.eit',name,'','TbDb'])
        for mode in ['BELL','DIAP','EXTD']: wav(br/'Data'/sid/'Sound'/mode/name,160+(i%2)*150)
    pd.DataFrame(metadata,columns=['SubjectID','Age','Gender','Diagnosis','Place']).to_csv(md/'Metadata.txt',index=False)
    pd.DataFrame(pairs,columns=['SubjectID','Local','EITFilename','AudioFilename1','AudioFilename2','AcquistionType']).to_csv(md/'FilePairs.txt',index=False)
    path=base/'synthetic.yaml'; path.write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8'); return path
