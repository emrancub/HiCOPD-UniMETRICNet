from pathlib import Path
import os, subprocess, sys


def test_cpu_runtime_in_fresh_process():
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MKL_THREADING_LAYER':'SEQUENTIAL','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    r=subprocess.run([sys.executable,str(root/'tests/runtime_cpu_check.py')],cwd=root,env=env)
    assert r.returncode==0
