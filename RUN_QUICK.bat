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
echo NOTE: QUICK RESULTS GO TO outputs_unimetricnet_v11_quick AND ARE NOT PUBLICATION EVIDENCE.
echo [1/7] Auditing datasets...
"%PY%" run.py audit --quick
if errorlevel 1 goto :fail
echo [2/7] Extracting clean features...
"%PY%" run.py features --condition clean --quick
if errorlevel 1 goto :fail
echo [3/7] Short BRACETS LODO software test...
"%PY%" run.py lodo --quick --models spectral_cnn crnn unimetricnet
if errorlevel 1 goto :fail
echo [4/7] Locked BRACETS materialization...
"%PY%" run.py external --quick --models spectral_cnn crnn unimetricnet
if errorlevel 1 goto :fail
echo [5/7] CPU embedding diagnostics...
"%PY%" run.py embedding-analysis --quick
if errorlevel 1 goto :fail
echo [6/7] Summarizing QUICK-only output...
"%PY%" run.py summarize --quick
if errorlevel 1 goto :fail
echo [7/7] Generating quick figures...
"%PY%" run.py figures --quick
if errorlevel 1 goto :fail
echo.
echo QUICK PIPELINE COMPLETED SUCCESSFULLY.
echo IMPORTANT: Do not upload outputs_unimetricnet_v11_quick as the publication result.
popd
exit /b 0
:fail
echo.
echo QUICK PIPELINE FAILED. Read the error immediately above.
popd
exit /b 1
