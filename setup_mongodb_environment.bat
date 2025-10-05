@echo off
echo ==================================================
echo  Backend Exoplanetas - Setup MongoDB Environment
echo  NASA Space Apps Challenge 2025
echo ==================================================

echo.
echo [1] Creando entorno virtual...
python -m venv venv_mongodb

echo.
echo [2] Activando entorno virtual...
call venv_mongodb\Scripts\activate.bat

echo.
echo [3] Actualizando pip...
python -m pip install --upgrade pip

echo.
echo [4] Instalando dependencias MongoDB...
pip install pandas>=2.1.0
pip install numpy>=1.24.0
pip install motor>=3.3.0
pip install pymongo>=4.5.0
pip install beanie>=1.23.0
pip install dnspython>=2.4.0

echo.
echo [5] Instalando dependencias adicionales...
pip install asyncio
pip install logging

echo.
echo [6] Verificando instalacion...
python -c "import motor; print('Motor version:', motor.version)"
python -c "import pymongo; print('PyMongo version:', pymongo.version)"
python -c "import pandas; print('Pandas version:', pandas.__version__)"

echo.
echo ==================================================
echo  CONFIGURACION MONGODB COMPLETADA
echo ==================================================
echo.
echo  Conexion MongoDB:
echo  Host: toiletcrafters.us.to:8081
echo  Database: ExoData
echo  Collection: datossatelite
echo  Usuario: manu
echo  Password: tele123
echo.
echo  Para usar el backend:
echo  1. Activar entorno: venv_mongodb\Scripts\activate.bat
echo  2. Ejecutar: python src\backend\main.py
echo.
echo ==================================================

pause