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
echo Using Python: %PY%
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail
"%PY%" -m pip install -r requirements-base.txt
if errorlevel 1 goto :fail
"%PY%" -m pip install -e . --no-deps
if errorlevel 1 goto :fail
"%PY%" run.py gpu-check
if errorlevel 1 goto :fail
echo Environment ready.
popd
exit /b 0
:fail
echo Environment setup failed.
popd
exit /b 1
