@echo off
rem =============================================================================
rem Virtual Environment Setup Script for Exoplanet Detection Backend (Windows)
rem NASA Space Apps Challenge 2025
rem =============================================================================

setlocal enabledelayedexpansion

rem Configuration
set VENV_NAME=exoplanet_backend_env
set PYTHON_VERSION=3.8
set PROJECT_DIR=%~dp0

rem Colors for output (Windows compatible)
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "WARNING=[WARNING]"
set "ERROR=[ERROR]"

rem Change to project directory
cd /d "%PROJECT_DIR%"
echo %INFO% Working directory: %CD%

rem Function to check if command exists
where python >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Python not found in PATH
    echo Please install Python 3.8 or higher and ensure it's in your PATH
    pause
    exit /b 1
)

rem Check Python version
for /f "tokens=2" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION_OUTPUT=%%a
echo %INFO% Found Python version: %PYTHON_VERSION_OUTPUT%

rem Extract major and minor version
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION_OUTPUT%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% LSS 3 (
    echo %ERROR% Python version too old. Requires Python 3.8+
    pause
    exit /b 1
)

if %MAJOR% EQU 3 if %MINOR% LSS 8 (
    echo %ERROR% Python version too old. Requires Python 3.8+
    pause
    exit /b 1
)

echo %SUCCESS% Python version is compatible

rem Remove existing virtual environment if it exists
if exist "%VENV_NAME%" (
    echo %WARNING% Existing virtual environment found. Removing...
    rmdir /s /q "%VENV_NAME%"
)

rem Create virtual environment
echo %INFO% Creating virtual environment: %VENV_NAME%
python -m venv "%VENV_NAME%"
if errorlevel 1 (
    echo %ERROR% Failed to create virtual environment
    pause
    exit /b 1
)

rem Activate virtual environment
echo %INFO% Activating virtual environment...
call "%VENV_NAME%\Scripts\activate.bat"

rem Upgrade pip
echo %INFO% Upgrading pip...
python -m pip install --upgrade pip

rem Install wheel for faster package installations
echo %INFO% Installing wheel...
pip install wheel

rem Install requirements
if exist "requirements.txt" (
    echo %INFO% Installing Python dependencies from requirements.txt...
    pip install -r requirements.txt
) else if exist "..\..\requirements.txt" (
    echo %INFO% Installing Python dependencies from ..\..\requirements.txt...
    pip install -r "..\..\requirements.txt"
) else (
    echo %WARNING% No requirements.txt found. Installing basic dependencies...
    pip install fastapi uvicorn pandas numpy scikit-learn
)

if errorlevel 1 (
    echo %ERROR% Failed to install dependencies
    pause
    exit /b 1
)

rem Create .env file if it doesn't exist
if not exist ".env" (
    if exist ".env.template" (
        echo %INFO% Creating .env file from template...
        copy ".env.template" ".env" >nul
        echo %INFO% Please review and customize .env file as needed
    )
)

rem Create necessary directories
echo %INFO% Creating project directories...
if not exist "logs" mkdir "logs"
if not exist "models" mkdir "models" 
if not exist "data" mkdir "data"
if not exist "temp" mkdir "temp"
if not exist "tests" mkdir "tests"

rem Generate activation script
echo %INFO% Creating activation script...
(
echo @echo off
echo rem Activation script for Exoplanet Detection Backend
echo.
echo cd /d "%%~dp0"
echo.
echo if exist "exoplanet_backend_env" ^(
echo     echo Activating virtual environment...
echo     call exoplanet_backend_env\Scripts\activate.bat
echo     echo Virtual environment activated. Project directory: %%CD%%
echo     echo To deactivate, run: deactivate
echo ^) else ^(
echo     echo Virtual environment not found. Run setup_env.bat first.
echo     pause
echo     exit /b 1
echo ^)
) > activate_env.bat

rem Generate Linux activation script  
echo %INFO% Creating Linux activation script...
(
echo #!/bin/bash
echo # Activation script for Exoplanet Detection Backend
echo.
echo PROJECT_DIR=$^(dirname "$^(readlink -f "$0"^)"^)
echo cd "$PROJECT_DIR"
echo.
echo if [[ -d "exoplanet_backend_env" ]]; then
echo     echo "Activating virtual environment..."
echo     source exoplanet_backend_env/bin/activate
echo     echo "Virtual environment activated. Project directory: $^(pwd^)"
echo     echo "To deactivate, run: deactivate"
echo else
echo     echo "Virtual environment not found. Run setup_env.sh first."
echo     exit 1
echo fi
) > activate_env.sh

echo.
echo %SUCCESS% === Environment Setup Complete ===
echo.
echo %INFO% Virtual environment created: %VENV_NAME%
echo %INFO% Python version: %PYTHON_VERSION_OUTPUT%
for /f "tokens=*" %%a in ('pip --version') do echo %INFO% Pip version: %%a
echo.
echo %INFO% To activate the environment:
echo %INFO%   Windows:   activate_env.bat
echo %INFO%   Linux/Mac: source activate_env.sh  
echo %INFO%   Manual:    %VENV_NAME%\Scripts\activate.bat
echo.
echo %INFO% To run the backend:
echo %INFO%   python main.py
echo %INFO%   or: uvicorn main:app --reload
echo.
echo %INFO% To deactivate: deactivate
echo.

rem Check installation
echo %INFO% Verifying installation...
set missing_packages=
for %%p in (fastapi uvicorn pandas numpy scikit-learn) do (
    pip show %%p >nul 2>&1
    if errorlevel 1 (
        set missing_packages=!missing_packages! %%p
    )
)

if "!missing_packages!"=="" (
    echo %SUCCESS% All required packages are installed
) else (
    echo %ERROR% Missing packages:!missing_packages!
)

echo.
echo Press any key to exit...
pause >nul