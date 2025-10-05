"""
Script de inicio del backend de detección de exoplanetas
"""

import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Cambiar al directorio del backend
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    # Configuración del servidor
    config = {
        "app": "main:app",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "log_level": "info",
        "access_log": True
    }
    
    print("🚀 Iniciando Exoplanet Detection API...")
    print(f"   • URL: http://localhost:8000")
    print(f"   • API Docs: http://localhost:8000/api/docs")
    print(f"   • Frontend: http://localhost:8000")
    
    uvicorn.run(**config)