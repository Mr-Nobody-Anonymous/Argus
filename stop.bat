@echo off
REM Windows: double-click this file to stop Argus.
cd /d "%~dp0"

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY ( where python >nul 2>&1 && set "PY=python" )

if not defined PY (
    echo Python was not found.
    pause
    exit /b 1
)

%PY% argus.py stop
if errorlevel 1 pause
