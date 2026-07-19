@echo off
setlocal enabledelayedexpansion
echo ===================================
echo Starting Argus AI Platform
echo ===================================

REM Check if venv exists
if not exist "venv\Scripts\activate" (
    echo [ERROR] Virtual environment not found!
    echo Please create one first: python -m venv venv
    echo Then install dependencies: pip install -r backend/requirements.txt
    pause
    exit /b 1
)

REM Check if node_modules exists in frontend
if not exist "frontend\node_modules" (
    echo [WARNING] Frontend dependencies not installed!
    echo Running npm install in frontend...
    cd frontend
    call npm install
    cd ..
)

echo 1. Starting Backend Server (port 8000)...
start "Argus Backend" cmd /k "venv\Scripts\activate && python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000"

echo 2. Starting Frontend Dashboard (port 3000)...
start "Argus Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo ===================================
echo System is starting up!
echo Access the dashboard at: http://localhost:3000
echo API docs at: http://localhost:8000/docs
echo ===================================
echo.
echo IMPORTANT: Close this window to stop both servers.
echo.
echo If running manually, always run from the PROJECT ROOT folder:
echo   cd C:\Users\hp\Downloads\argus-merge-work\Argus
echo.
echo In CMD (recommended):  venv\Scripts\activate ^&^& python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
echo In PowerShell:          .\venv\Scripts\Activate.ps1; python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
echo.
pause
