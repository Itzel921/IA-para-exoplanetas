"""
Sistema Backend para Detección de Exoplanetas con IA
FastAPI Backend simplificado que integra el modelo ML con la interfaz web
NASA Space Apps Challenge 2025
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import os
from pathlib import Path
import json
import asyncio
from datetime import datetime
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

# Configurar logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============= MODELOS PYDANTIC BÁSICOS =============

class ExoplanetFeatures(BaseModel):
    """Características de entrada para predicción individual"""
    period: float = Field(..., description="Período orbital (días)", gt=0)
    radius: float = Field(..., description="Radio planetario (radios terrestres)", gt=0)
    temp: Optional[float] = Field(None, description="Temperatura de equilibrio (K)", gt=0)
    starRadius: float = Field(..., description="Radio estelar (radios solares)", gt=0)
    starMass: Optional[float] = Field(None, description="Masa estelar (masas solares)", gt=0) 
    starTemp: float = Field(..., description="Temperatura estelar (K)", gt=0)
    depth: float = Field(..., description="Profundidad del tránsito (ppm)", gt=0)
    duration: float = Field(..., description="Duración del tránsito (hrs)", gt=0) 
    snr: float = Field(..., description="Signal-to-noise ratio", gt=0)

class PredictionResponse(BaseModel):
    """Respuesta de predicción individual"""
    prediction: str = Field(..., description="Clasificación: CONFIRMED o FALSE_POSITIVE")
    confidence: float = Field(..., description="Confianza del modelo (0-1)", ge=0, le=1)
    probabilities: Dict[str, float] = Field(..., description="Probabilidades por clase")
    model_version: str = Field(..., description="Versión del modelo utilizado")
    timestamp: str = Field(..., description="Timestamp de la predicción")
    interpretation: str = Field(..., description="Interpretación en lenguaje natural")
    confidence_level: str = Field(..., description="Nivel de confianza: HIGH, MEDIUM, LOW")

class BatchPredictionResponse(BaseModel):
    """Respuesta de predicción en lote"""
    total_processed: int = Field(..., description="Total de objetos procesados")
    confirmed_planets: int = Field(..., description="Número de exoplanetas confirmados")
    false_positives: int = Field(..., description="Número de falsos positivos")
    confirmed_percentage: float = Field(..., description="Porcentaje de confirmados")
    average_confidence: float = Field(..., description="Confianza promedio")
    output_file: str = Field(..., description="Archivo de resultados generado")
    processing_time: float = Field(..., description="Tiempo de procesamiento en segundos")
    timestamp: str = Field(..., description="Timestamp del procesamiento")

class ModelInfoResponse(BaseModel):
    """Información del modelo ML"""
    model_type: str = Field(..., description="Tipo de modelo")
    model_version: str = Field(..., description="Versión del modelo")
    training_accuracy: float = Field(..., description="Accuracy de entrenamiento")
    base_models: List[str] = Field(..., description="Modelos base del ensemble")
    training_datasets: List[str] = Field(..., description="Datasets utilizados")
    features_used: int = Field(..., description="Número de características")
    last_updated: str = Field(..., description="Última actualización")

# ============= SERVICIOS MOCK =============

class MockPredictionService:
    """Servicio de predicción mock para desarrollo"""
    
    def __init__(self):
        self.is_initialized = False
        self.model_loaded = False
    
    async def initialize(self) -> bool:
        """Inicializar el servicio mock"""
        try:
            logger.info("🔄 Inicializando servicio de predicción mock...")
            await asyncio.sleep(1)  # Simular carga
            self.is_initialized = True
            self.model_loaded = True
            logger.info("✅ Servicio mock inicializado")
            return True
        except Exception as e:
            logger.error(f"❌ Error inicializando servicio mock: {e}")
            return False
    
    def is_model_loaded(self) -> bool:
        return self.model_loaded
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo mock"""
        return {
            "model_type": "Stacking Ensemble (Mock)",
            "model_version": "v1.0-mock",
            "training_accuracy": 0.8308,
            "base_models": ["Random Forest", "AdaBoost", "Extra Trees", "LightGBM"],
            "training_datasets": ["KOI", "TOI", "K2"],
            "features_used": 42,
            "last_updated": datetime.now().isoformat()
        }
    
    async def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predicción individual mock"""
        # Simular predicción basada en algunos parámetros
        period = features.get('period', 1.0)
        radius = features.get('radius', 1.0)
        snr = features.get('snr', 10.0)
        
        # Lógica simple para generar predicción realista
        score = min(1.0, (snr / 100.0) + (radius / 10.0) + (1.0 / period))
        is_planet = score > 0.5
        confidence = score if is_planet else 1 - score
        
        # Determinar nivel de confianza
        if confidence >= 0.8:
            confidence_level = "HIGH"
        elif confidence >= 0.6:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        prediction = "CONFIRMED" if is_planet else "FALSE_POSITIVE"
        prob_confirmed = score
        prob_false_positive = 1 - score
        
        interpretation = (
            f"Este candidato tiene una probabilidad del {prob_confirmed:.1%} de ser un exoplaneta real." 
            if is_planet else 
            f"Este candidato probablemente es un falso positivo ({prob_false_positive:.1%} de confianza)."
        )
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                "CONFIRMED": prob_confirmed,
                "FALSE_POSITIVE": prob_false_positive
            },
            "model_version": "StackingEnsemble_v1.0_mock",
            "timestamp": datetime.now().isoformat(),
            "interpretation": interpretation,
            "confidence_level": confidence_level
        }
    
    async def predict_batch(self, file_path: Path) -> Dict[str, Any]:
        """Predicción en lote mock"""
        # Simular procesamiento
        await asyncio.sleep(2)
        
        # Valores simulados
        total = 1000
        confirmed = 85
        false_positives = total - confirmed
        
        return {
            "total_processed": total,
            "confirmed_planets": confirmed,
            "false_positives": false_positives,
            "confirmed_percentage": round((confirmed / total) * 100, 2),
            "average_confidence": 0.742,
            "output_file": f"mock_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "processing_time": 2.0,
            "timestamp": datetime.now().isoformat()
        }

# ============= CONFIGURACIÓN DE LA APLICACIÓN =============

# Rutas de archivos
PROJECT_ROOT = Path(__file__).parent.parent.parent
FRONTEND_PATH = PROJECT_ROOT / "web" / "frontend"
UPLOAD_PATH = PROJECT_ROOT / "data" / "temp_uploads"
RESULTS_PATH = PROJECT_ROOT / "exoPlanet_results"

# Crear directorios necesarios
UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Inicializar servicios
prediction_service = MockPredictionService()

# Evento de inicio usando lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejo del ciclo de vida de la aplicación"""
    # Startup
    logger.info("🚀 Iniciando Exoplanet Detection API...")
    
    # Inicializar servicio de predicción
    if await prediction_service.initialize():
        logger.info("✅ Modelo ML cargado exitosamente")
    else:
        logger.error("❌ Error cargando modelo ML - continuando con mock")
    
    yield
    
    # Shutdown
    logger.info("👋 Cerrando Exoplanet Detection API...")

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

# Servir archivos estáticos del frontend
if FRONTEND_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_PATH)), name="static")

# ============= ENDPOINTS PRINCIPALES =============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Servir la página principal del frontend"""
    index_path = FRONTEND_PATH / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return HTMLResponse("""
        <html>
        <head><title>Exoplanet Detection API</title></head>
        <body>
            <h1>🌟 Exoplanet Detection API</h1>
            <p>Backend funcionando correctamente!</p>
            <p><a href="/api/docs">📖 Ver documentación API</a></p>
        </body>
        </html>
        """)

@app.get("/api/health")
async def health_check():
    """Health check del sistema"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": prediction_service.is_model_loaded(),
        "version": "1.0.0",
        "backend_type": "mock" if isinstance(prediction_service, MockPredictionService) else "production"
    }

@app.get("/api/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Información del modelo ML"""
    try:
        model_info = await prediction_service.get_model_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        logger.error(f"Error obteniendo info del modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_single(features: ExoplanetFeatures):
    """Predicción individual de exoplaneta"""
    try:
        logger.info(f"🔮 Predicción individual solicitada")
        
        # Realizar predicción
        result = await prediction_service.predict_single(features.dict())
        
        logger.info(f"✅ Predicción completada: {result['prediction']}")
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Error en predicción individual: {e}")
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")

@app.post("/api/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...)):
    """Predicción en lote desde archivo CSV"""
    try:
        logger.info(f"📄 Predicción en lote solicitada: {file.filename}")
        
        # Validar archivo
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos CSV")
        
        # Simular guardado de archivo temporal
        temp_file_path = UPLOAD_PATH / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        # Realizar predicción en lote
        result = await prediction_service.predict_batch(temp_file_path)
        
        logger.info(f"✅ Predicción en lote completada: {result['confirmed_planets']} planetas confirmados")
        return BatchPredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Error en predicción en lote: {e}")
        raise HTTPException(status_code=400, detail=f"Error en procesamiento: {str(e)}")

@app.get("/api/results")
async def list_results():
    """Listar archivos de resultados disponibles"""
    try:
        # Buscar archivos en el directorio de resultados
        result_files = []
        if RESULTS_PATH.exists():
            for file_path in RESULTS_PATH.glob("*.csv"):
                stat = file_path.stat()
                result_files.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "type": "predictions"
                })
        
        return {"files": result_files}
        
    except Exception as e:
        logger.error(f"❌ Error listando resultados: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/{filename}")
async def download_results(filename: str):
    """Descargar archivo de resultados"""
    try:
        file_path = RESULTS_PATH / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"❌ Error descargando archivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= WEBSOCKET PARA PROGRESO =============

class ConnectionManager:
    """Gestor de conexiones WebSocket"""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"🔌 Nueva conexión WebSocket: {len(self.active_connections)} activas")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"🔌 Conexión WebSocket cerrada: {len(self.active_connections)} activas")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error enviando mensaje WebSocket: {e}")
            self.disconnect(websocket)

manager = ConnectionManager()

@app.websocket("/ws/batch-progress")
async def websocket_batch_progress(websocket: WebSocket):
    """WebSocket para progreso de procesamiento en lote"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Recibir datos del cliente
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            logger.info(f"📨 WebSocket: procesamiento solicitado")
            
            # Simular procesamiento con actualizaciones de progreso
            total_items = request_data.get('total', 100)
            
            for i in range(total_items):
                # Simular trabajo
                await asyncio.sleep(0.05)
                
                progress = {
                    'type': 'progress',
                    'progress': (i + 1) / total_items * 100,
                    'processed': i + 1,
                    'total': total_items,
                    'current_item': f"Procesando objeto {i + 1}",
                    'timestamp': datetime.now().isoformat()
                }
                
                await manager.send_personal_message(progress, websocket)
            
            # Enviar resultado final
            final_result = {
                'type': 'completed',
                'status': 'success',
                'summary': {
                    'total_processed': total_items,
                    'confirmed_planets': max(1, total_items // 10),
                    'false_positives': total_items - max(1, total_items // 10),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            await manager.send_personal_message(final_result, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("🔌 Cliente WebSocket desconectado")
    except Exception as e:
        logger.error(f"❌ Error en WebSocket: {e}")
        error_message = {
            'type': 'error',
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }
        await manager.send_personal_message(error_message, websocket)
        manager.disconnect(websocket)

# ============= CONFIGURACIÓN DEL SERVIDOR =============

if __name__ == "__main__":
    # Configuración para desarrollo
    print("🚀 Iniciando Exoplanet Detection API...")
    print(f"   • URL: http://localhost:8000")
    print(f"   • API Docs: http://localhost:8000/api/docs")
    print(f"   • Frontend: http://localhost:8000")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )