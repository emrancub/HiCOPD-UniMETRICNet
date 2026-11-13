@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0"
if exist "outputs_unimetricnet_v11" rmdir /s /q "outputs_unimetricnet_v11"
if exist "outputs_unimetricnet_v11_quick" rmdir /s /q "outputs_unimetricnet_v11_quick"
if exist "cache_unimetricnet_v11" rmdir /s /q "cache_unimetricnet_v11"
if exist "cache_unimetricnet_v11_quick" rmdir /s /q "cache_unimetricnet_v11_quick"
if exist "software_smoke" rmdir /s /q "software_smoke"
echo Outputs and caches removed. Raw datasets and pretrained_reference_v11 were not touched.
popd
exit /b 0
