@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0"
call RUN_ALL.bat
if errorlevel 1 (
  echo Main publication pipeline failed. Controlled validation was not started.
  popd
  exit /b 1
)
echo.
echo ============================================================
echo MAIN PUBLIC-DATASET PIPELINE COMPLETED.
echo ============================================================
if not exist "prospective_validation\controlled_predictions.csv" (
  echo Optional controlled-acquisition study is not present yet.
  echo Public-dataset results remain valid.
  echo Read prospective_validation\CONTROLLED_ACQUISITION_PROTOCOL.md.
  echo After collecting/frozen-model prediction, run:
  echo RUN_CONTROLLED_VALIDATION.bat FROZEN_SOURCE_THRESHOLD
  popd
  exit /b 0
)
echo Controlled predictions are present.
echo For scientific validity, confirm that the source threshold was frozen BEFORE labels were opened.
echo Then run RUN_CONTROLLED_VALIDATION.bat FROZEN_SOURCE_THRESHOLD.
popd
