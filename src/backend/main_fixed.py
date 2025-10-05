"""
Sistema Backend para Detección de Exoplanetas con IA
FastAPI Backend que integra el modelo ML con la interfaz web
NASA Space Apps Challenge 2025
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import os
import sys
from pathlib import Path
import json
import asyncio
from datetime import datetime
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
import pandas as pd
import time

# Configurar path para importar el modelo ML
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
ml_dev_path = project_root / "ML DEV"
sys.path.append(str(ml_dev_path))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos Pydantic
class ExoplanetFeatures(BaseModel):
    """Modelo para las características del exoplaneta"""
    # Parámetros básicos
    period: Optional[float] = None
    radius: Optional[float] = None
    temp: Optional[float] = None
    
    # Parámetros estelares
    star_radius: Optional[float] = None
    star_mass: Optional[float] = None
    star_temp: Optional[float] = None
    
    # Métricas de tránsito
    depth: Optional[float] = None
    duration: Optional[float] = None
    snr: Optional[float] = None
    
    class Config:
        json_encoders = {
            float: lambda v: round(v, 6) if v is not None else None
        }

class PredictionResponse(BaseModel):
    """Respuesta de predicción individual"""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time: float
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Respuesta de predicción en lote"""
    total_processed: int
    successful_predictions: int
    failed_predictions: int
    results: List[Dict[str, Any]]
    processing_time: float
    timestamp: str

class ModelInfoResponse(BaseModel):
    """Información del modelo"""
    model_name: str
    version: str
    accuracy: float
    last_updated: str
    features_count: int

# Servicios integrados
class MLService:
    """Servicio de Machine Learning integrado"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        
    async def load_model(self):
        """Cargar el modelo ML"""
        try:
            # Intentar importar y usar el modelo real
            try:
                import predict_exoplanets
                self.model = predict_exoplanets
                self.model_loaded = True
                logger.info("✅ Modelo ML cargado exitosamente")
            except ImportError as e:
                logger.warning(f"⚠️ No se pudo cargar el modelo real: {e}")
                logger.info("🤖 Usando predictor mock para desarrollo")
                self.model_loaded = False
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            self.model_loaded = False
    
    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predicción individual"""
        try:
            if self.model_loaded and hasattr(self.model, 'predict_single'):
                return self.model.predict_single(features)
            else:
                # Predicción mock para testing
                import random
                confidence = random.uniform(0.6, 0.95)
                is_planet = confidence > 0.8
                
                return {
                    'prediction': 'CONFIRMED' if is_planet else 'FALSE_POSITIVE',
                    'confidence': confidence,
                    'probabilities': {
                        'CONFIRMED': confidence if is_planet else 1 - confidence,
                        'FALSE_POSITIVE': 1 - confidence if is_planet else confidence
                    }
                }
        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")
    
    def predict_batch(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predicción en lote"""
        try:
            if self.model_loaded and hasattr(self.model, 'predict_batch'):
                return self.model.predict_batch(data)
            else:
                # Predicción mock para testing
                results = []
                for index, row in data.iterrows():
                    import random
                    confidence = random.uniform(0.6, 0.95)
                    is_planet = confidence > 0.8
                    
                    result = {
                        'index': index,
                        'prediction': 'CONFIRMED' if is_planet else 'FALSE_POSITIVE',
                        'confidence': confidence,
                        'probabilities': {
                            'CONFIRMED': confidence if is_planet else 1 - confidence,
                            'FALSE_POSITIVE': 1 - confidence if is_planet else confidence
                        }
                    }
                    results.append(result)
                return results
        except Exception as e:
            logger.error(f"❌ Error en predicción batch: {e}")
            raise HTTPException(status_code=500, detail=f"Error en predicción batch: {str(e)}")

# Instanciar servicio ML
ml_service = MLService()

# Gestor de ciclo de vida de la aplicación
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestor del ciclo de vida de la aplicación"""
    # Startup
    logger.info("🚀 Iniciando backend de detección de exoplanetas...")
    await ml_service.load_model()
    yield
    # Shutdown
    logger.info("🛑 Cerrando backend...")

# Configuración de la aplicación
app = FastAPI(
    title="Exoplanet Detection API",
    description="API para detección de exoplanetas usando ensemble ML - NASA Space Apps Challenge 2025",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Configurar CORS para el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas de archivos
PROJECT_ROOT = Path(__file__).parent.parent.parent
FRONTEND_PATH = PROJECT_ROOT / "web" / "frontend"
UPLOAD_PATH = PROJECT_ROOT / "data" / "temp_uploads"
RESULTS_PATH = PROJECT_ROOT / "exoPlanet_results"

# Crear directorios necesarios
UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Servir archivos estáticos del frontend
try:
    if FRONTEND_PATH.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_PATH)), name="static")
        logger.info(f"✅ Sirviendo frontend desde: {FRONTEND_PATH}")
    else:
        logger.warning(f"⚠️ Directorio frontend no encontrado: {FRONTEND_PATH}")
except Exception as e:
    logger.error(f"❌ Error configurando archivos estáticos: {e}")

# ==================== RUTAS DE LA API ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Página principal - sirve el frontend"""
    try:
        index_path = FRONTEND_PATH / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        else:
            return HTMLResponse("""
            <html>
                <head><title>Exoplanet Detection API</title></head>
                <body>
                    <h1>🌟 Exoplanet Detection API</h1>
                    <p>Backend funcionando correctamente</p>
                    <p><a href="/api/docs">📚 Documentación API</a></p>
                    <p><a href="/api/health">🏥 Estado del servicio</a></p>
                </body>
            </html>
            """)
    except Exception as e:
        logger.error(f"❌ Error sirviendo página principal: {e}")
        return HTMLResponse(f"<h1>Error: {str(e)}</h1>", status_code=500)

@app.get("/api/health")
async def health_check():
    """Health check del servicio"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": ml_service.model_loaded,
        "service": "Exoplanet Detection API"
    }

@app.get("/api/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Información del modelo ML"""
    return ModelInfoResponse(
        model_name="Ensemble Exoplanet Detector",
        version="1.0.0",
        accuracy=0.83,
        last_updated="2025-10-05",
        features_count=50
    )

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_single(features: ExoplanetFeatures):
    """Predicción individual de exoplaneta"""
    start_time = time.time()
    
    try:
        # Convertir a diccionario y filtrar valores None
        feature_dict = {k: v for k, v in features.dict().items() if v is not None}
        
        # Hacer predicción
        result = ml_service.predict_single(feature_dict)
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error en predicción individual: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...)):
    """Predicción en lote desde archivo CSV"""
    start_time = time.time()
    
    try:
        # Validar archivo
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")
        
        # Leer archivo CSV
        contents = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
        
        logger.info(f"📊 Procesando archivo: {file.filename} ({len(df)} filas)")
        
        # Hacer predicciones
        results = ml_service.predict_batch(df)
        
        processing_time = time.time() - start_time
        
        # Contar resultados
        successful = len([r for r in results if 'prediction' in r])
        failed = len(results) - successful
        
        return BatchPredictionResponse(
            total_processed=len(df),
            successful_predictions=successful,
            failed_predictions=failed,
            results=results,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error en predicción batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/{filename}")
async def get_result_file(filename: str):
    """Descargar archivo de resultados"""
    try:
        file_path = RESULTS_PATH / filename
        if file_path.exists():
            return FileResponse(file_path)
        else:
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
    except Exception as e:
        logger.error(f"❌ Error obteniendo archivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results")
async def list_result_files():
    """Listar archivos de resultados disponibles"""
    try:
        files = []
        if RESULTS_PATH.exists():
            for file_path in RESULTS_PATH.glob("*.csv"):
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return {"files": files}
    except Exception as e:
        logger.error(f"❌ Error listando archivos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WEBSOCKET ====================

class ConnectionManager:
    """Gestor de conexiones WebSocket"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"❌ Error enviando mensaje WebSocket: {e}")

manager = ConnectionManager()

@app.websocket("/ws/batch-progress")
async def websocket_batch_progress(websocket: WebSocket):
    """WebSocket para progreso de procesamiento en lote"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Esperar datos del cliente
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Simular procesamiento con progreso
            total_items = request_data.get('total', 100)
            
            for i in range(total_items):
                # Simular trabajo
                await asyncio.sleep(0.05)
                
                progress = {
                    'type': 'progress',
                    'progress': (i + 1) / total_items * 100,
                    'processed': i + 1,
                    'total': total_items,
                    'message': f"Procesando objeto {i + 1}/{total_items}"
                }
                
                await manager.send_message(progress, websocket)
            
            # Enviar resultado final
            final_result = {
                'type': 'completed',
                'message': 'Procesamiento completado',
                'summary': {
                    'total_processed': total_items,
                    'confirmed_planets': max(1, total_items // 10),
                    'false_positives': total_items - max(1, total_items // 10)
                }
            }
            
            await manager.send_message(final_result, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("🔌 Cliente WebSocket desconectado")
    except Exception as e:
        logger.error(f"❌ Error en WebSocket: {e}")
        await manager.send_message({
            'type': 'error',
            'message': f'Error: {str(e)}'
        }, websocket)

# ==================== RUTAS DE DEBUG ====================

@app.get("/api/debug/features")
async def debug_features():
    """Debug: mostrar características esperadas"""
    return {
        "expected_features": [
            "period", "radius", "temp", "star_radius", 
            "star_mass", "star_temp", "depth", "duration", "snr"
        ],
        "optional_features": "All features are optional",
        "model_status": {
            "loaded": ml_service.model_loaded,
            "type": "Real ML Model" if ml_service.model_loaded else "Mock Predictor"
        }
    }

@app.get("/api/debug/paths")
async def debug_paths():
    """Debug: mostrar rutas configuradas"""
    return {
        "project_root": str(PROJECT_ROOT),
        "frontend_path": str(FRONTEND_PATH),
        "frontend_exists": FRONTEND_PATH.exists(),
        "upload_path": str(UPLOAD_PATH),
        "results_path": str(RESULTS_PATH),
        "ml_dev_path": str(ml_dev_path)
    }

# ==================== SERVIDOR ====================

if __name__ == "__main__":
    print("🚀 Iniciando servidor Exoplanet Detection API...")
    print("📍 Documentación disponible en: http://localhost:8000/api/docs")
    print("🏠 Frontend disponible en: http://localhost:8000/")
    
    uvicorn.run(
        "main_fixed:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src/backend"]
    )