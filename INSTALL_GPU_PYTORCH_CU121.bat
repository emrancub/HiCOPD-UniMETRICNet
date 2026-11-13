@echo off
setlocal DisableDelayedExpansion
if defined CONDA_PREFIX (set "PY=%CONDA_PREFIX%\python.exe") else (set "PY=python")
"%PY%" -m pip install --upgrade torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
