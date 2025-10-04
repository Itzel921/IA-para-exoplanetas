@echo off
REM NASA Space Apps Challenge 2025 - Exoplanet Detection System
REM Startup script for Windows

echo 🚀 NASA Space Apps Challenge 2025 - Exoplanet Detection System
echo ================================================================

REM Check Python version
echo 📋 Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 🔧 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo ⚡ Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Create logs directory
if not exist "logs" mkdir logs

REM Start backend server
echo 🚀 Starting FastAPI backend server...
echo Backend will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo Frontend: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

cd src\backend
python main.py

pause