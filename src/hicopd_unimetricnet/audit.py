from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import soundfile as sf
except ImportError:
    sf = None
from .audio import read_mono
from .parsers import *
from .utils import DataIntegrityError, ensure_dir, sha256_file, write_json


def collect_icbhi(cfg):
    c=cfg['data']; root=Path(c['icbhi_root']); audio=root/c['icbhi_audio_dir']
    dx=parse_icbhi_diagnoses(root/c['icbhi_diagnosis_file']); dm=parse_icbhi_demographics(root/c['icbhi_demographic_file'])
    rows=[]
    for wav in sorted(audio.glob('*.wav')):
        m=parse_icbhi_filename(wav); sid=m['subject_raw']
        if sid not in dx: raise DataIntegrityError(f'No ICBHI diagnosis for {sid}')
        cat=dx[sid]; d=dm.get(sid,{}); keep=cat in set(c['icbhi_keep'])
        rows.append(dict(origin_dataset='ICBHI',logical_dataset='ICBHI',subject_id=f'ICBHI_{sid}',person_id=f'ICBHI_{sid}',
            recording_id=f'ICBHI_{wav.stem}',audio_path=str(wav.resolve()),category=cat,copd_label=int(cat=='COPD') if keep else -1,
            severity=-1,site=m['site'],view=m['device'],filter_mode='NA',acquisition=m['mode'],age=d.get('age',np.nan),sex=d.get('sex','UNK'),
            selection_reason='selected_reference_classes' if keep else 'excluded_low_support_class',main_role=int(keep),filter_intervention_role=0))
    if not rows: raise DataIntegrityError(f'No ICBHI wav files under {audio}')
    return rows


def collect_kauh(cfg):
    """Collect KAUH with exactly one primary recording per selected patient.

    The reference paper reports one KAUH audio file per person.  We therefore
    prefer the diaphragm/D export when present and fall back deterministically
    to B then E for a patient whose D export is absent.  All available B/D/E
    exports for the retained classes are still preserved for the same-patient
    measurement/filter intervention experiment, but only the selected primary
    export receives ``main_role=1``.  This avoids pseudo-replication while
    recovering the complete selected-patient cohort described in the
    reference table.
    """
    c=cfg['data']; root=Path(c['kauh_root']); audio=root/c['kauh_audio_dir']; keepcats=set(c['kauh_keep']); preferred=str(c['kauh_main_filter']).upper()
    parsed=[]
    for wav in sorted(audio.glob('*.wav')):
        m=parse_kauh_filename(wav); cat=m['category']; keep=cat in keepcats and cat!='MULTI'
        parsed.append(dict(wav=wav,m=m,cat=cat,keep=keep))
    if not parsed: raise DataIntegrityError(f'No KAUH wav files under {audio}')

    # Select one primary recording per physical patient.  D is preferred to
    # match the intended diaphragm measurement, but a B/E fallback is used
    # only when the D export is absent.  This is deterministic and label-free.
    priority=[preferred]+[x for x in ['D','B','E'] if x!=preferred]
    primary={}
    by_subject={}
    for z in parsed:
        if z['keep']: by_subject.setdefault(z['m']['subject_id'],[]).append(z)
    for sid,items in by_subject.items():
        chosen=None
        for mode in priority:
            cand=[z for z in items if z['m']['filter_mode']==mode]
            if cand:
                chosen=sorted(cand,key=lambda z:z['wav'].name.casefold())[0]; break
        if chosen is None: chosen=sorted(items,key=lambda z:z['wav'].name.casefold())[0]
        primary[sid]=str(chosen['wav'].resolve())

    rows=[]
    for z in parsed:
        wav,m,cat,keep=z['wav'],z['m'],z['cat'],z['keep']; selected=keep and primary.get(m['subject_id'])==str(wav.resolve())
        fallback=selected and m['filter_mode']!=preferred
        rows.append(dict(origin_dataset='KAUH',logical_dataset='HYBRID',subject_id=m['subject_id'],person_id=m['subject_id'],
            recording_id=f'KAUH_{wav.stem}',audio_path=str(wav.resolve()),category=cat,copd_label=int(cat=='COPD') if keep else -1,
            severity=-1,site='UNK',view=m['filter_mode'],filter_mode=m['filter_mode'],acquisition='Littmann3200_filter',age=m['age'],sex=m['sex'],
            selection_reason=('selected_HYBRID_primary_'+m['filter_mode']+('_fallback' if fallback else '')) if selected else 'HYBRID_filter_intervention_only' if keep else 'excluded_low_support_or_multi_class',
            main_role=int(selected),filter_intervention_role=int(keep)))
    return rows


def collect_rdtr(cfg):
    c=cfg['data']; root=Path(c['rdtr_root']); audio=root/c['rdtr_audio_dir']; smap=load_rdtr_severity_map(root/c['rdtr_labels_file']); keepsev=set(map(int,c['rdtr_keep_severity']))
    rows=[]
    for wav in sorted(audio.glob('*.wav')):
        m=parse_rdtr_filename(wav); sev=lookup_rdtr_severity(m['subject_raw'],wav.stem,smap)
        if sev<0:
            if str(c.get('missing_rdtr_label_policy','error')).casefold()=='exclude': continue
            raise DataIntegrityError(f'No RDTR severity for {wav.name}')
        keep=sev in keepsev; cat='COPD' if sev>=1 else 'COPD-0'
        rows.append(dict(origin_dataset='RDTR',logical_dataset='HYBRID',subject_id=m['subject_id'],person_id=m['subject_id'],
            recording_id=f'RDTR_{wav.stem}',audio_path=str(wav.resolve()),category=cat,copd_label=1 if keep else -1,
            severity=sev,site=m['site'],view=m['site'],filter_mode='NA',acquisition='Littmann3200',age=np.nan,sex='UNK',
            selection_reason='selected_HYBRID_COPD1_to_4' if keep else 'excluded_COPD0',main_role=int(keep),filter_intervention_role=0))
    if not rows: raise DataIntegrityError(f'No RDTR wav files under {audio}')
    return rows


def collect_bracets(cfg):
    c=cfg['data']; root=Path(c['bracets_root']); md=root/c['bracets_metadata_dir']
    meta=canonicalize_bracets_metadata(md/c['bracets_metadata_file']).set_index('SubjectID')
    pairs=canonicalize_bracets_pairs(md/c['bracets_pairs_file']); mapping={}
    for r in pairs.itertuples(index=False):
        sid=norm_bracets_id(r.SubjectID)
        for name in [r.AudioFilename1,r.AudioFilename2]:
            if not str(name).strip(): continue
            key=(sid,Path(str(name).replace('\\','/')).name.casefold()); val=dict(acquisition=str(r.AcquisitionType).strip(),place=str(r.Local).strip())
            if key in mapping and mapping[key]!=val: raise DataIntegrityError(f'Conflicting BRACETS FilePairs mapping {key}')
            mapping[key]=val
    data=root/c['bracets_data_dir']; rows=[]; seen=set(); keepcats=set(c['bracets_keep']); primary=str(c['bracets_primary_acquisition']).casefold()
    for sub in sorted(data.iterdir() if data.exists() else []):
        if not sub.is_dir() or not sub.name.isdigit(): continue
        raw=norm_bracets_id(sub.name); seen.add(raw)
        if raw not in meta.index: raise DataIntegrityError(f'No BRACETS metadata for {raw}')
        mr=meta.loc[raw]; cat=mr.Diagnosis
        for mode in ['BELL','DIAP','EXTD']:
            folder=sub/'Sound'/mode
            for wav in sorted(folder.rglob('*.wav') if folder.exists() else []):
                key=(raw,wav.name.casefold())
                if key not in mapping: raise DataIntegrityError(f'No BRACETS FilePairs row for {wav}')
                pr=mapping[key]; parts=wav.stem.split('_'); site=parts[-1].upper() if parts else 'UNK'
                keep=cat in keepcats and pr['acquisition'].casefold()==primary and (not c.get('bracets_exclude_tracheal',True) or site!='T')
                sid=f'BRACETS_{raw}'
                rows.append(dict(origin_dataset='BRACETS',logical_dataset='BRACETS',subject_id=sid,person_id=sid,
                    recording_id=f'BRACETS_{raw}_{mode}_{wav.stem}',audio_path=str(wav.resolve()),category=cat,copd_label=int(cat=='COPD') if keep else -1,
                    severity=-1,site=site,view=mode,filter_mode=mode,acquisition=pr['acquisition'],age=normalize_age(mr.Age),sex=normalize_sex(mr.Gender),
                    selection_reason='selected_primary_acquisition' if keep else 'excluded_acquisition_site_or_class',main_role=int(keep),filter_intervention_role=0))
    if not rows: raise DataIntegrityError(f'No BRACETS wav files under {data}')
    if set(meta.index)-seen: raise DataIntegrityError(f'BRACETS metadata subjects missing from Data: {sorted(set(meta.index)-seen)[:10]}')
    return rows


def reference_design_table():
    rows=[]
    ref={
      'ICBHI': [('Healthy',26,35,700,170),('COPD',64,793,17234,4261),('Pneumonia',6,37,740,185),('URTI',14,23,460,113),('Bronchiolitis',6,13,260,65),('Bronchiectasis',7,16,320,80)],
      'KAUH component of HYBRID': [('Healthy',35,35,625,139),('COPD',9,9,158,34),('Asthma',32,32,544,119),('Heart_Failure',18,18,280,61),('Pneumonia',5,5,98,22)],
      'RDTR component of HYBRID': [('COPD-1',5,60,1309,295),('COPD-2',7,84,1776,402),('COPD-3',7,84,1911,440),('COPD-4',17,204,4374,997)],
    }
    for db,vals in ref.items():
        for cat,p,a,t,s in vals: rows.append(dict(database=db,category=cat,number_people=p,number_audio_files=a,recording_time_s=t,number_segments=s))
    return pd.DataFrame(rows)


def _reference_count_checks(df,cfg):
    if not cfg['data'].get('strict_reference_counts',True): return []
    errors=[]; main=df[df.main_role.eq(1)]
    checks=[
      ('ICBHI','Healthy',26,35),('ICBHI','COPD',64,793),('ICBHI','Pneumonia',6,37),('ICBHI','URTI',14,23),('ICBHI','Bronchiolitis',6,13),('ICBHI','Bronchiectasis',7,16),
      ('KAUH','Healthy',35,35),('KAUH','COPD',9,9),('KAUH','Asthma',32,32),('KAUH','Heart_Failure',18,18),('KAUH','Pneumonia',5,5),
    ]
    for origin,cat,ep,ea in checks:
        g=main[(main.origin_dataset.eq(origin))&(main.category.eq(cat))]; got=(g.person_id.nunique(),len(g))
        if got!=(ep,ea): errors.append(f'{origin}/{cat}: selected subjects/files {got} != expected {(ep,ea)}')
    for sev,ep,ea in [(1,5,60),(2,7,84),(3,7,84),(4,17,204)]:
        g=main[(main.origin_dataset.eq('RDTR'))&(main.severity.eq(sev))]; got=(g.person_id.nunique(),len(g))
        if got!=(ep,ea): errors.append(f'RDTR/COPD-{sev}: selected subjects/files {got} != expected {(ep,ea)}')
    return errors


def _quality_hash(df,cfg):
    rows=[]; sr=int(cfg['audio']['sample_rate']); pcm=bool(cfg['integrity'].get('compute_pcm_hash',True)); minsec=float(cfg['integrity']['min_audio_seconds'])
    for r in df.itertuples(index=False):
        info=sf.info(r.audio_path)
        if info.duration<minsec: raise DataIntegrityError(f'Audio shorter than {minsec}s: {r.audio_path}')
        raw=sha256_file(r.audio_path); ph=''
        if pcm:
            x=read_mono(r.audio_path,sr); ph=hashlib.sha256(x.tobytes()).hexdigest()
        rows.append(dict(recording_id=r.recording_id,duration_s=float(info.duration),source_sr=int(info.samplerate),channels=int(info.channels),raw_sha256=raw,pcm_sha256=ph))
    return pd.DataFrame(rows)


def _quarantine_duplicates(df,q,cfg):
    x=df.merge(q,on='recording_id',how='left'); affected=set(); duprows=[]
    for basis in ['raw_sha256','pcm_sha256']:
        if basis=='pcm_sha256' and not cfg['integrity'].get('compute_pcm_hash',True): continue
        for digest,g in x[x[basis].astype(str).ne('')].groupby(basis):
            people=sorted(g.person_id.unique())
            if len(people)>1:
                affected.update(people)
                for r in g.itertuples(index=False): duprows.append(dict(basis=basis,sha256=digest,person_id=r.person_id,subject_id=r.subject_id,origin_dataset=r.origin_dataset,recording_id=r.recording_id,audio_path=r.audio_path,colliding_people=';'.join(people)))
    x['integrity_exclusion_reason']=''
    if affected: x.loc[x.person_id.isin(affected),'integrity_exclusion_reason']='cross_person_audio_duplicate'
    return x,pd.DataFrame(duprows),affected


def _composition(main,cfg):
    seg=float(cfg['audio']['segment_seconds']); z=main.copy(); z['number_segments']=(z.duration_s/seg).astype(int).clip(lower=0)
    cats=(z.groupby(['logical_dataset','category'],dropna=False)
        .agg(number_people=('person_id','nunique'),number_audio_files=('recording_id','nunique'),recording_time_s=('duration_s','sum'),number_segments=('number_segments','sum')).reset_index())
    order=['ICBHI','HYBRID','BRACETS']; cats['__o']=cats.logical_dataset.map({k:i for i,k in enumerate(order)}).fillna(99)
    return cats.sort_values(['__o','category']).drop(columns='__o').reset_index(drop=True)


def build_audit(cfg):
    cache=ensure_dir(cfg['project']['cache_dir'])
    rows=collect_icbhi(cfg)+collect_kauh(cfg)+collect_rdtr(cfg)+collect_bracets(cfg); df=pd.DataFrame(rows)
    if df.recording_id.duplicated().any() or df.audio_path.duplicated().any(): raise DataIntegrityError('Duplicate recording ID or audio path in inventory')
    ref=reference_design_table(); ref.to_csv(cache/'Table00_reference_design_requested.csv',index=False)
    ref_errors=_reference_count_checks(df,cfg)
    if ref_errors: raise DataIntegrityError('Reference selection mismatch:\n'+'\n'.join(ref_errors))
    q=_quality_hash(df,cfg); merged,dups,affected=_quarantine_duplicates(df,q,cfg)
    merged.to_csv(cache/'recordings_all.csv',index=False); q.to_csv(cache/'audio_quality.csv',index=False); dups.to_csv(cache/'cross_person_duplicates.csv',index=False)
    selected_before=merged[merged.main_role.eq(1)].copy(); _composition(selected_before,cfg).to_csv(cache/'Table01_three_database_before_integrity.csv',index=False)
    main=merged[(merged.main_role.eq(1)) & merged.integrity_exclusion_reason.eq('')].copy(); filt=merged[(merged.filter_intervention_role.eq(1)) & merged.integrity_exclusion_reason.eq('')].copy()
    main.to_csv(cache/'analysis_recordings.csv',index=False); filt.to_csv(cache/'filter_intervention_recordings.csv',index=False)
    comp=_composition(main,cfg); comp.to_csv(cache/'Table02_three_database_final.csv',index=False)
    origin=(main.groupby(['logical_dataset','origin_dataset','category']).agg(number_people=('person_id','nunique'),number_audio_files=('recording_id','nunique')).reset_index()); origin.to_csv(cache/'Table02b_origin_breakdown.csv',index=False)
    report=dict(total_audio_exports=int(len(merged)),total_people=int(merged.person_id.nunique()),analysis_audio_files=int(len(main)),analysis_people=int(main.person_id.nunique()),
        logical_database_people={k:int(v) for k,v in main.groupby('logical_dataset').person_id.nunique().items()},excluded_duplicate_people=sorted(affected),
        design='Exactly three logical databases: ICBHI, HYBRID=(KAUH+RDTR COPD1-4; COPD0 excluded), and BRACETS. Low-support ICBHI/KAUH classes are excluded exactly as specified. KAUH uses one primary recording per patient (D preferred; B/E fallback only when D is unavailable); all B/D/E exports are reserved for same-patient measurement intervention.',
        reference_table=str((cache/'Table00_reference_design_requested.csv').resolve()),final_table=str((cache/'Table02_three_database_final.csv').resolve()))
    write_json(cache/'audit_report.json',report)
    print(comp.to_string(index=False)); print(f'\nAnalysis people: {main.person_id.nunique()}; duplicate-quarantined people: {len(affected)}')
    return main
