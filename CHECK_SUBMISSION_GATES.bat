@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0"
set "PYTHONPATH=%CD%\src"
if defined CONDA_PREFIX (set "PY=%CONDA_PREFIX%\python.exe") else (set "PY=python")
if not exist "outputs_unimetricnet_v11\tables\Table13_submission_evidence_gates.csv" (
    "%PY%" run.py fast-upgrade --config configs/main.yaml
)
"%PY%" check_submission_gates.py
set ERR=%ERRORLEVEL%
popd
exit /b %ERR%
