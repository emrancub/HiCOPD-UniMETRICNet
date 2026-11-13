from __future__ import annotations
import csv, re
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from .utils import DataIntegrityError

ICBHI_MAP={
    'healthy':'Healthy','copd':'COPD','pneumonia':'Pneumonia','urti':'URTI','lrti':'LRTI',
    'bronchiectasis':'Bronchiectasis','bronchiolitis':'Bronchiolitis','asthma':'Asthma'}
KAUH_MAP={
    'n':'Healthy','normal':'Healthy','healthy':'Healthy','asthma':'Asthma','copd':'COPD',
    'pneumonia':'Pneumonia','bron':'Bronchitis','bronchitis':'Bronchitis',
    'heart failure':'Heart_Failure','heart_failure':'Heart_Failure',
    'lung fibrosis':'Lung_Fibrosis','pulmonary fibrosis':'Lung_Fibrosis',
    'pleural effusion':'Pleural_Effusion','plueral effusion':'Pleural_Effusion'}
BRACETS_MAP={
    'healthy':'Healthy','normal':'Healthy','control':'Healthy','copd':'COPD','asthma':'Asthma',
    'ild':'ILD','infection':'Pulmonary Infection','pulmonary infection':'Pulmonary Infection'}
MISSING={'','na','n/a','nan','none','null','unknown','unk','-'}

def norm_text(x) -> str:
    return re.sub(r'\s+',' ',str(x).strip()).strip()

def optional_text(x) -> str:
    if pd.isna(x): return ''
    s=norm_text(x)
    return '' if s.casefold() in MISSING else s

def normalize_sex(x) -> str:
    s=optional_text(x).casefold()
    return 'M' if s in {'m','male'} else 'F' if s in {'f','female'} else 'UNK'

def normalize_age(x):
    s=optional_text(x)
    if not s: return np.nan
    try: v=float(s)
    except Exception: return np.nan
    return v if np.isfinite(v) and 0<=v<120 else np.nan

def read_flexible_table(path: str | Path) -> pd.DataFrame:
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(p)
    if p.suffix.lower() in {'.xls','.xlsx'}: return pd.read_excel(p, dtype=str).fillna('')
    text=p.read_text(encoding='utf-8-sig', errors='replace')
    if not text.strip(): raise DataIntegrityError(f'Empty table: {p}')
    first=text.splitlines()[0]
    sep='\t' if '\t' in first else ',' if ',' in first else None
    if sep: return pd.read_csv(p, sep=sep, dtype=str, keep_default_na=False)
    return pd.read_csv(p, sep=r'\s+', engine='python', dtype=str, keep_default_na=False)

def parse_icbhi_diagnoses(path: str | Path) -> Dict[str,str]:
    df=pd.read_csv(path, header=None, names=['subject','diagnosis'])
    out={}
    for r in df.itertuples(index=False):
        sid=norm_text(r.subject); raw=norm_text(r.diagnosis).casefold()
        if raw not in ICBHI_MAP: raise DataIntegrityError(f'Unrecognized ICBHI diagnosis {r.diagnosis!r} for {sid}')
        out[sid]=ICBHI_MAP[raw]
    return out

def parse_icbhi_filename(path: str | Path):
    parts=Path(path).stem.split('_')
    if len(parts)<5: raise DataIntegrityError(f'Unexpected ICBHI filename: {Path(path).name}')
    return dict(subject_raw=parts[0],recording_index=parts[1],site=parts[2],mode=parts[3],device='_'.join(parts[4:]))

def parse_icbhi_demographics(path: str | Path) -> Dict[str,dict]:
    p=Path(path); out={}
    if not p.exists(): return out
    for no,raw in enumerate(p.read_text(encoding='utf-8-sig',errors='replace').splitlines(),1):
        if not raw.strip(): continue
        if '\t' in raw: f=[x.strip() for x in raw.split('\t')]
        elif ',' in raw: f=[x.strip() for x in next(csv.reader([raw]))]
        else: f=re.split(r'\s+',raw.strip())
        if not f or not re.fullmatch(r'\d+(?:\.0)?',f[0]): continue
        sid=str(int(float(f[0])))
        if sid in out: raise DataIntegrityError(f'Duplicate ICBHI demographic ID: {sid}')
        age=normalize_age(f[1] if len(f)>1 else '')
        sex=normalize_sex(f[2] if len(f)>2 else '')
        if sex=='UNK' and normalize_sex(f[1] if len(f)>1 else '')!='UNK': sex=normalize_sex(f[1])
        out[sid]=dict(age=age,sex=sex)
    return out

def parse_kauh_labels(raw: str) -> List[str]:
    s=norm_text(raw)
    direct=KAUH_MAP.get(s.casefold().replace('_',' '))
    if direct: return [direct]
    toks=re.split(r'\s*(?:\+|&|\band\b)\s*',s,flags=re.I)
    out=[]
    for tok in toks:
        if not tok: continue
        v=KAUH_MAP.get(norm_text(tok).casefold().replace('_',' '))
        if not v: raise DataIntegrityError(f'Unrecognized KAUH diagnosis token {tok!r} in {raw!r}')
        if v not in out: out.append(v)
    if not out: raise DataIntegrityError(f'Could not parse KAUH diagnosis {raw!r}')
    return out

def parse_kauh_filename(path: str | Path):
    stem=Path(path).stem
    if '_' not in stem: raise DataIntegrityError(f'Unexpected KAUH filename: {Path(path).name}')
    prefix,rest=stem.split('_',1)
    m=re.match(r'^([BDE])P(\d+)$',prefix,re.I)
    if not m: raise DataIntegrityError(f'Unexpected KAUH prefix {prefix!r}')
    mode=m.group(1).upper(); no=int(m.group(2)); parts=[x.strip() for x in rest.split(',')]
    labels=parse_kauh_labels(parts[0])
    age=np.nan; sex='UNK'
    for x in reversed(parts):
        if x.upper() in {'M','F'}: sex=x.upper(); break
    for x in reversed(parts):
        try: v=float(x)
        except Exception: continue
        if 0<v<120: age=v; break
    return dict(subject_id=f'KAUH_P{no:03d}',patient_no=no,filter_mode=mode,
                labels=labels,category=labels[0] if len(labels)==1 else 'MULTI',age=age,sex=sex)

def _id_aliases(x) -> List[str]:
    if pd.isna(x): return []
    s=norm_text(x).upper()
    if not s: return []
    out={s,Path(s).stem}; m=re.search(r'([A-Z]*)(\d+)',s)
    if m:
        prefix,digits=m.group(1),m.group(2); n=int(digits)
        out.update({str(n),f'{n:03d}',f'H{n:03d}',f'P{n}',f'P{n:03d}'})
        if prefix: out.update({f'{prefix}{n}',f'{prefix}{n:03d}'})
    return sorted(out)

def _severity(x) -> int:
    if pd.isna(x): return -1
    s=norm_text(x).upper()
    if re.fullmatch(r'[0-4](?:\.0)?',s): return int(float(s))
    m=re.search(r'COPD\s*[-_ ]?\s*([0-4])',s)
    if m: return int(m.group(1))
    return {'I':1,'II':2,'III':3,'IV':4}.get(s,-1)

def load_rdtr_severity_map(path: str | Path) -> Dict[str,int]:
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(p)
    xls=pd.ExcelFile(p); out={}
    def add(k,v):
        if v<0: return
        for a in _id_aliases(k):
            if a in out and out[a]!=v: raise DataIntegrityError(f'Conflicting RDTR severity for {a}: {out[a]} vs {v}')
            out[a]=v
    for sheet in xls.sheet_names:
        df=pd.read_excel(p,sheet_name=sheet)
        if df.empty: continue
        df.columns=[norm_text(c) for c in df.columns]
        for c in df.columns:
            m=re.search(r'COPD\s*[-_ ]?\s*([0-4])',c.upper())
            if m:
                for v in df[c].dropna(): add(v,int(m.group(1)))
        idcols=[c for c in df if any(k in c.casefold() for k in ['patient','subject','file','record','name','id','no'])]
        sevcols=[c for c in df if any(k in c.casefold() for k in ['severity','stage','grade','gold','copd','class','label'])]
        for ic in (idcols or list(df.columns[:1]))[:3]:
            for sc in (sevcols or list(df.columns[1:2]))[:3]:
                if ic==sc: continue
                for _,r in df[[ic,sc]].iterrows(): add(r[ic],_severity(r[sc]))
    if not out: raise DataIntegrityError(f'Could not infer RDTR severity labels from {p}')
    return out

def parse_rdtr_filename(path: str | Path):
    stem=Path(path).stem.upper(); m=re.match(r'^([A-Z]+\d+)[_-]([LR][1-6])(?:[_-].*)?$',stem)
    if m: raw,site=m.group(1),m.group(2)
    else:
        m2=re.match(r'^([A-Z]*\d+)',stem)
        if not m2: raise DataIntegrityError(f'Unexpected RDTR filename: {Path(path).name}')
        raw=m2.group(1); sm=re.search(r'(?:^|[_-])([LR][1-6])(?:$|[_-])',stem); site=sm.group(1) if sm else 'UNK'
    return dict(subject_raw=raw,subject_id=f'RDTR_{raw}',site=site)

def lookup_rdtr_severity(raw: str, stem: str, mapping: Dict[str,int]) -> int:
    for x in [raw,stem,Path(stem).stem]:
        for a in _id_aliases(x):
            if a in mapping: return int(mapping[a])
    return -1

def norm_bracets_id(x) -> str:
    s=norm_text(x)
    if not re.fullmatch(r'\d+(?:\.0)?',s): raise DataIntegrityError(f'Invalid BRACETS SubjectID: {s!r}')
    return f'{int(float(s)):02d}'

def canonicalize_bracets_metadata(path: str | Path) -> pd.DataFrame:
    df=read_flexible_table(path)
    norm={re.sub(r'[ _]','',c).casefold():c for c in df.columns}
    req={'subjectid':'SubjectID','age':'Age','gender':'Gender','diagnosis':'Diagnosis','place':'Place'}
    missing=[k for k in req if k not in norm]
    if missing: raise DataIntegrityError(f'BRACETS metadata missing {missing}; columns={list(df.columns)}')
    out=pd.DataFrame({canon:df[norm[k]] for k,canon in req.items()})
    out['SubjectID']=out.SubjectID.map(norm_bracets_id)
    def dx(x):
        s=norm_text(x).casefold()
        if s not in BRACETS_MAP: raise DataIntegrityError(f'Unmapped BRACETS diagnosis: {x!r}')
        return BRACETS_MAP[s]
    out['Diagnosis']=out.Diagnosis.map(dx)
    if out.SubjectID.duplicated().any(): raise DataIntegrityError('Duplicate BRACETS SubjectID in metadata')
    return out

def canonicalize_bracets_pairs(path: str | Path) -> pd.DataFrame:
    df=read_flexible_table(path)
    def n(c): return re.sub(r'[ _]','',c).casefold()
    groups={}
    for c in df.columns: groups.setdefault(n(c),[]).append(c)
    aliases={
        'SubjectID':['subjectid'],'Local':['local'],'EITFilename':['eitfilename'],
        'AudioFilename1':['audiofilename1'],'AudioFilename2':['audiofilename2'],
        'AcquisitionType':['acquisitiontype','acquistiontype']}
    out={}
    for canon,opts in aliases.items():
        matches=[c for key in opts for c in groups.get(key,[])]
        if len(matches)!=1: raise DataIntegrityError(f'BRACETS FilePairs field {canon} has matches {matches}; columns={list(df.columns)}')
        out[canon]=df[matches[0]]
    z=pd.DataFrame(out); z['SubjectID']=z.SubjectID.map(norm_bracets_id)
    return z
