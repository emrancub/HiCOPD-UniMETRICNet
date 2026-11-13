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

echo ==================== [1/2] Upgrading Multi-Expert Hierarchical Calibration ^& Tables ====================
"%PY%" run.py fast-upgrade --config configs/main.yaml
if errorlevel 1 goto :fail
echo ==================== [2/2] Printing Submission Evidence Gates (15/15) ====================
"%PY%" run.py gates --config configs/main.yaml
if errorlevel 1 goto :fail
echo.
echo SUCCESS: HiCOPD-UniMETRICNet v11.2 Fast Upgrade completed and all 15 Submission Evidence Gates verified!
popd
exit /b 0
:fail
echo.
echo FAST UPGRADE FAILED. Read the error immediately above.
popd
exit /b 1
