@echo off
REM Windows: double-click this file to set up and start Argus.
cd /d "%~dp0"

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY ( where python >nul 2>&1 && set "PY=python" )

if not defined PY (
    echo Python 3.9+ is required but was not found.
    echo.
    echo Install it from https://www.python.org/downloads/
    echo IMPORTANT: tick "Add Python to PATH" in the installer.
    echo.
    pause
    exit /b 1
)

%PY% argus.py start %*
if errorlevel 1 pause
