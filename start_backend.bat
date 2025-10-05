@echo off
echo 🚀 Iniciando Backend de Detección de Exoplanetas...
echo.

REM Cambiar al directorio del backend
cd /d "%~dp0\src\backend"

REM Verificar si existe el entorno virtual
if not exist "..\..\venv\Scripts\python.exe" (
    echo ❌ Entorno virtual no encontrado. Por favor ejecuta:
    echo    python -m venv venv
    echo    venv\Scripts\activate
    echo    pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activar entorno virtual y ejecutar servidor
echo 📦 Activando entorno virtual...
call "..\..\venv\Scripts\activate.bat"

echo 📦 Instalando dependencias del backend...
pip install -r requirements.txt

echo 🌐 Iniciando servidor FastAPI...
echo    • URL: http://localhost:8000
echo    • API Docs: http://localhost:8000/api/docs
echo    • Presiona Ctrl+C para detener
echo.

python run_server.py