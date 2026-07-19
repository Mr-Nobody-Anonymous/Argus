@echo off
setlocal enabledelayedexpansion
REM Argus Webcam Tester Launcher
REM This script runs the backend with PC webcam for testing face recognition, emotion detection, etc.

echo ============================================
echo Argus Webcam Tester
echo ============================================
echo.

REM Go to project root
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install required packages
echo Installing dependencies...
pip install -q deepface insightface onnxruntime

REM Create necessary directories
if not exist "data\known_faces" mkdir data\known_faces
if not exist "data\snapshots" mkdir data\snapshots

echo.
echo Starting webcam tester...
echo Press 'q' in the video window to quit
echo.

python webcam_tester.py --camera 0

echo.
echo Webcam tester stopped.
