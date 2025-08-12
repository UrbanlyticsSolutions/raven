@echo off
setlocal enabledelayedexpansion
title RUN RAVEN FROM OSTRICH

rem Ensure directories
if not exist "model" mkdir "model"
if not exist "model\output" mkdir "model\output"

rem Clean old output files to prevent stale data
if exist "model\output\Diagnostics.csv" del /Q "model\output\Diagnostics.csv"
if exist "model\output\Hydrographs.csv" del /Q "model\output\Hydrographs.csv"

rem Copy parameter file (overwrite silently)
copy /Y ".\outlet_49.5738_-119.0368.rvp" ".\model\outlet_49.5738_-119.0368.rvp" >nul

rem Copy other inputs if they exist
if exist ".\outlet_49.5738_-119.0368.rvi" copy /Y ".\outlet_49.5738_-119.0368.rvi" ".\model\outlet_49.5738_-119.0368.rvi" >nul
if exist ".\outlet_49.5738_-119.0368.rvh" copy /Y ".\outlet_49.5738_-119.0368.rvh" ".\model\outlet_49.5738_-119.0368.rvh" >nul
if exist ".\outlet_49.5738_-119.0368.rvt" copy /Y ".\outlet_49.5738_-119.0368.rvt" ".\model\outlet_49.5738_-119.0368.rvt" >nul
if exist ".\outlet_49.5738_-119.0368.rvc" copy /Y ".\outlet_49.5738_-119.0368.rvc" ".\model\outlet_49.5738_-119.0368.rvc" >nul
if exist ".\channel_properties.rvp" copy /Y ".\channel_properties.rvp" ".\model\channel_properties.rvp" >nul

rem Copy obs directory if it exists
if exist ".\obs" (
    if not exist ".\model\obs" mkdir ".\model\obs"
    copy /Y ".\obs\*" ".\model\obs\" >nul
)

pushd "model"

rem Clean previous outputs to prevent stale data
if exist "output\Diagnostics.csv" del /Q "output\Diagnostics.csv"
if exist "output\Hydrographs.csv" del /Q "output\Hydrographs.csv"

rem Run Raven with full path
"E:\python\Raven\RavenHydroFramework\build\Release\Raven.exe" outlet_49.5738_-119.0368 -o output
set ERR=%ERRORLEVEL%

rem Check Raven exit code
if %ERR% NEQ 0 (
    echo ERROR: Raven exited with code %ERR%
    popd
    exit /b %ERR%
)

rem Assert Diagnostics.csv was created
if not exist "output\Diagnostics.csv" (
    echo ERROR: Diagnostics.csv missing
    popd
    exit /b 2
)

rem Check if Diagnostics.csv is not empty
for %%A in ("output\Diagnostics.csv") do set SZ=%%~zA
if "%SZ%"=="0" (
    echo ERROR: Diagnostics.csv empty
    popd
    exit /b 3
)

rem Log parameter file size for debugging
for %%A in ("outlet_49.5738_-119.0368.rvp") do echo %DATE% %TIME% rvp_size=%%~zA>>"..\ost_run.log"

popd
exit /b 0