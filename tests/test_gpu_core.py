from pathlib import Path
import os, subprocess, sys


def test_gpu_runtime_in_fresh_process():
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MKL_THREADING_LAYER':'SEQUENTIAL','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    r=subprocess.run([sys.executable,str(root/'tests/runtime_gpu_check.py')],cwd=root,env=env)
    assert r.returncode==0
