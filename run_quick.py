"""Fast end-to-end sanity launcher using the real data paths."""
from pathlib import Path
import os,subprocess,sys
ROOT=Path(__file__).resolve().parent
for k,v in {'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MKL_THREADING_LAYER':'SEQUENTIAL','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONHASHSEED':'0'}.items(): os.environ.setdefault(k,v)
os.environ.pop('KMP_DUPLICATE_LIB_OK',None)
stages=[['audit','--quick'],['features','--condition','clean','--quick'],['lodo','--quick','--models','spectral_cnn','crnn','unimetricnet'],['external','--quick','--models','spectral_cnn','crnn','unimetricnet'],['embedding-analysis','--quick'],['summarize','--quick'],['figures','--quick']]
for a in stages:
    print('\n===',' '.join(a),'===',flush=True); rc=subprocess.run([sys.executable,str(ROOT/'run.py'),*a],cwd=ROOT,env=os.environ.copy()).returncode
    if rc: raise SystemExit(rc)
print('\nQuick UniMETRICNet pipeline finished successfully.')
