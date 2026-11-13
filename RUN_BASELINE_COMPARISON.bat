@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "MKL_THREADING_LAYER=SEQUENTIAL"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
set "KMP_DUPLICATE_LIB_OK="
if defined CONDA_PREFIX (set "PY=%CONDA_PREFIX%\python.exe") else (set "PY=python")
"%PY%" run.py lodo --models spectral_cnn resnet_spec ast_lite physio_mlp unimetricnet
if errorlevel 1 goto :fail
"%PY%" run.py external --models spectral_cnn resnet_spec ast_lite physio_mlp unimetricnet
if errorlevel 1 goto :fail
"%PY%" run.py summarize
if errorlevel 1 goto :fail
"%PY%" run.py figures
if errorlevel 1 goto :fail
echo Baseline comparison complete.
popd
exit /b 0
:fail
echo Baseline comparison failed.
popd
exit /b 1
