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

echo ============================================================
echo HiCOPD-UniMETRICNet v11.2 ULTIMATE WINDOWS-SAFE TEST SUITE
echo Heavy CPU/SciPy and GPU/PyTorch checks run in separate processes.
echo ============================================================

"%PY%" -m pytest -q
if errorlevel 1 goto :fail

echo.
echo Running end-to-end synthetic smoke pipeline in isolated stages...
"%PY%" run.py smoke
if errorlevel 1 goto :fail

echo.
echo ALL TESTS AND SMOKE PIPELINE PASSED.
popd
exit /b 0

:fail
echo.
echo TEST PROJECT FAILED. Read the first error immediately above.
popd
exit /b 1
