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

echo ==================== [1/22] gpu-check ====================
"%PY%" run.py gpu-check
if errorlevel 1 goto :fail
echo ==================== [2/22] doctor ====================
"%PY%" run.py doctor
if errorlevel 1 goto :fail
echo ==================== [3/22] audit ====================
"%PY%" run.py audit
if errorlevel 1 goto :fail
echo ==================== [4/22] features --condition clean ====================
"%PY%" run.py features --condition clean
if errorlevel 1 goto :fail
echo ==================== [5/22] cv --models spectral_cnn crnn resnet_spec mobilenet_spec convnext_spec ast_lite physio_mlp unimetricnet ====================
"%PY%" run.py cv --models spectral_cnn crnn resnet_spec mobilenet_spec convnext_spec ast_lite physio_mlp unimetricnet
if errorlevel 1 goto :fail
echo ==================== [6/22] lodo ====================
"%PY%" run.py lodo
if errorlevel 1 goto :fail
echo ==================== [7/22] external ====================
"%PY%" run.py external
if errorlevel 1 goto :fail
echo ==================== [8/22] crossdb --models spectral_cnn resnet_spec mobilenet_spec unimetricnet ====================
"%PY%" run.py crossdb --models spectral_cnn resnet_spec mobilenet_spec unimetricnet
if errorlevel 1 goto :fail
echo ==================== [9/22] ablations ====================
"%PY%" run.py ablations
if errorlevel 1 goto :fail
echo ==================== [10/22] filter-test ====================
"%PY%" run.py filter-test
if errorlevel 1 goto :fail
echo ==================== [11/22] features --condition noise_20db ====================
"%PY%" run.py features --condition noise_20db
if errorlevel 1 goto :fail
echo ==================== [12/22] features --condition noise_10db ====================
"%PY%" run.py features --condition noise_10db
if errorlevel 1 goto :fail
echo ==================== [13/22] features --condition bandlimit_800 ====================
"%PY%" run.py features --condition bandlimit_800
if errorlevel 1 goto :fail
echo ==================== [14/22] features --condition clipping_050 ====================
"%PY%" run.py features --condition clipping_050
if errorlevel 1 goto :fail
echo ==================== [15/22] stress ====================
"%PY%" run.py stress
if errorlevel 1 goto :fail
echo ==================== [16/22] profile ====================
"%PY%" run.py profile
if errorlevel 1 goto :fail
echo ==================== [17/22] embedding-analysis ====================
"%PY%" run.py embedding-analysis
if errorlevel 1 goto :fail
echo ==================== [18/22] explainability ====================
"%PY%" run.py explainability
if errorlevel 1 goto :fail
echo ==================== [19/22] validate-results ====================
"%PY%" run.py validate-results
if errorlevel 1 goto :fail
echo ==================== [20/22] summarize ====================
"%PY%" run.py summarize
if errorlevel 1 goto :fail
echo ==================== [21/22] figures ====================
"%PY%" run.py figures
if errorlevel 1 goto :fail
echo ==================== [22/22] gates ====================
"%PY%" run.py gates
if errorlevel 1 goto :fail

echo.
echo COMPLETE HiCOPD-UniMETRICNet v11.2 ULTIMATE PIPELINE FINISHED SUCCESSFULLY (15/15 GATES PASSED).
echo Full outputs are in outputs_unimetricnet_v11.
popd
exit /b 0
:fail
echo.
echo PIPELINE FAILED. Read the error immediately above.
popd
exit /b 1
