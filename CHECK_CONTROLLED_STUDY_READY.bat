@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0"
if exist "prospective_validation\CONTROLLED_ACQUISITION_PROTOCOL.md" (
  echo Protocol: PRESENT
) else (
  echo Protocol: MISSING
  popd
  exit /b 1
)
if exist "prospective_validation\manifest_template.csv" (
  echo Manifest template: PRESENT
) else (
  echo Manifest template: MISSING
  popd
  exit /b 1
)
if exist "prospective_validation\controlled_predictions.csv" (
  echo Controlled prediction file: PRESENT
) else (
  echo Controlled prediction file: NOT PRESENT YET - optional stage will be skipped.
)
echo Controlled-study infrastructure is ready.
popd
