@echo off
title SAVE BEST SOLUTION
echo Saving best solution artifacts...
if not exist best mkdir best

REM Copy only if file exists to avoid error noise
if exist "model\output\Diagnostics.csv" copy /Y "model\output\Diagnostics.csv" "best\Diagnostics.csv" >nul
if exist "model\output\Hydrographs.csv" copy /Y "model\output\Hydrographs.csv" "best\Hydrographs.csv" >nul
if exist "model\outlet_49.5738_-119.0368.rvp" copy /Y "model\outlet_49.5738_-119.0368.rvp" "best\outlet_49.5738_-119.0368.rvp" >nul

echo Done.
