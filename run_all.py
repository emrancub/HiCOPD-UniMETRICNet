"""Full isolated-process launcher for HiCOPD-UniMETRICNet v8.0."""
from pathlib import Path
import os,subprocess,sys,yaml
ROOT=Path(__file__).resolve().parent
for k,v in {'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MKL_THREADING_LAYER':'SEQUENTIAL','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONHASHSEED':'0'}.items(): os.environ.setdefault(k,v)
os.environ.pop('KMP_DUPLICATE_LIB_OK',None)
cfg=yaml.safe_load((ROOT/'configs/main.yaml').read_text(encoding='utf-8'))
stages=[['audit'],['features','--condition','clean'],['cv','--models','spectral_cnn','crnn','resnet_spec','unimetricnet'],['lodo'],['external'],['crossdb','--models','spectral_cnn','crnn','resnet_spec','unimetricnet'],['ablations'],['filter-test']]
for c in cfg['experiments']['stress_conditions']: stages.append(['features','--condition',c])
stages += [['stress'],['profile'],['embedding-analysis'],['explainability'],['validate-results'],['summarize'],['figures']]
for a in stages:
    print('\n===',' '.join(a),'===',flush=True); rc=subprocess.run([sys.executable,str(ROOT/'run.py'),*a],cwd=ROOT,env=os.environ.copy()).returncode
    if rc: raise SystemExit(rc)
print('\nComplete UniMETRICNet publication pipeline finished successfully.')
