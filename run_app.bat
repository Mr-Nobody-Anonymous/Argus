@echo off
setlocal
echo ===================================
echo Starting Argus AI Platform
echo ===================================

REM Check if we're in the right directory
if not exist "backend\api\main.py" (
    echo [ERROR] Please run this script from the Argus project folder!
    echo Current directory: %CD%
    pause
    exit /b 1
)

REM Check if venv exists and has valid python
if exist "venv\Scripts\python.exe" (
    echo [OK] Virtual environment found
    set PYTHON_CMD=venv\Scripts\python
) else (
    echo [INFO] Using system Python
    set PYTHON_CMD=python
)

REM Install dependencies if needed (minimal install)
echo Checking dependencies...
%PYTHON_CMD% -c "import fastapi" 2>nul
if errorlevel 1 (
    echo Installing minimal dependencies...
    %PYTHON_CMD% -m pip install --quiet fastapi uvicorn[standard] python-multipart opencv-python-headless ultralytics pydantic pydantic-settings httpx
)

echo 1. Starting Backend Server (port 8000)...
start "Argus Backend" cmd /k "%PYTHON_CMD% -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000"

echo 2. Starting Frontend Dashboard (port 3000)...
start "Argus Frontend" cmd /k "cd /d %~dp0frontend && npm install && npm run dev"

echo ===================================
echo System is starting up!
echo Access the dashboard at: http://localhost:3000
echo API docs at: http://localhost:8000/docs
echo ===================================
echo.
echo IMPORTANT: Close this window to stop both servers.
echo.
pause