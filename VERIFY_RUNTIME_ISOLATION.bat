@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "MKL_THREADING_LAYER=SEQUENTIAL"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
set "PYTHONHASHSEED=0"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "KMP_DUPLICATE_LIB_OK="
if defined CONDA_PREFIX (set "PY=%CONDA_PREFIX%\python.exe") else (set "PY=python")

echo CPU scientific stack process:
"%PY%" -c "import numpy, scipy, soundfile, pandas, sys; print('CPU stack OK; torch intentionally not imported:', 'torch' not in sys.modules)"
if errorlevel 1 goto :fail

echo CPU audio/spectral process:
"%PY%" tests\runtime_cpu_check.py
if errorlevel 1 goto :fail

echo CPU embedding-analysis process:
"%PY%" -c "from hicopd_unimetricnet.embedding_analysis import analyze_all; import sys; print('Embedding diagnostics import OK; torch loaded:', 'torch' in sys.modules)"
if errorlevel 1 goto :fail

echo GPU/PyTorch process:
"%PY%" tests\runtime_gpu_check.py
if errorlevel 1 goto :fail

echo.
echo Runtime isolation verification PASSED.
popd
exit /b 0

:fail
echo.
echo Runtime isolation verification FAILED.
popd
exit /b 1
