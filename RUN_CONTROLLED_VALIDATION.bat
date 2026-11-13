@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0"
if "%~1"=="" (
  echo Usage: RUN_CONTROLLED_VALIDATION.bat FROZEN_SOURCE_THRESHOLD
  echo Example: RUN_CONTROLLED_VALIDATION.bat 0.487
  echo.
  echo The threshold must be frozen from source/public-development data BEFORE controlled-study labels are opened.
  popd
  exit /b 2
)
if not exist "prospective_validation\controlled_predictions.csv" (
  echo ERROR: prospective_validation\controlled_predictions.csv was not found.
  echo Copy controlled_predictions_template.csv and populate it using your FROZEN model.
  popd
  exit /b 3
)
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set MKL_THREADING_LAYER=SEQUENTIAL
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set PYTHONPATH=%CD%\src
python prospective_validation\run_controlled_validation.py --threshold %~1
if errorlevel 1 (
  echo CONTROLLED VALIDATION FAILED.
  popd
  exit /b 1
)
echo Controlled-acquisition validation completed successfully.
popd
