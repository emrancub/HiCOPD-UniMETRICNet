@echo off
setlocal
echo ============================================================================
echo   HiCOPD-UniMETRICNet: Automated GitHub Repository Uploader ^& Updater
echo   Target Repository: https://github.com/emrancub/HiCOPD.git
echo ============================================================================

where git >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Git is not installed or not in your PATH.
    echo Please install Git from https://git-scm.com/download/win and re-run this script.
    pause
    exit /b 1
)

if not exist ".git" (
    echo [1/5] Initializing local Git repository...
    git init
) else (
    echo [1/5] Existing Git repository detected.
)

echo [2/5] Setting branch to main and configuring remote origin...
git branch -M main
git remote remove origin >nul 2>nul
git remote add origin https://github.com/emrancub/HiCOPD.git

echo [3/5] Staging source code, configurations, and scripts...
git add .

echo [4/5] Creating commit...
git commit -m "Release HiCOPD-UniMETRICNet v11.2: Physics-Guided Multi-Expert LODO Framework"

echo [5/5] Pushing to https://github.com/emrancub/HiCOPD ...
git push -u origin main --force

echo ============================================================================
echo   Done! Check https://github.com/emrancub/HiCOPD in your browser.
echo ============================================================================
pause
