#!/usr/bin/env python3
"""
Servidor FastAPI simple para testing del frontend
NASA Space Apps Challenge 2025

Servidor mínimo que responde con datos mock para permitir testing
del frontend sin dependencias de MongoDB o modelos ML.
"""

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import random
import io
import csv

# Crear app FastAPI
app = FastAPI(
    title="Exoplanet Detection API - Testing Mode",
    version="1.0.0",
    description="API mínima para testing del frontend"
)

# Configurar CORS para permitir requests del frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic para requests
class PredictionRequest(BaseModel):
    period: float
    radius: float
    temp: float
    starRadius: float
    starMass: float
    starTemp: float
    depth: float
    duration: float
    snr: float

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    analysis_timestamp: str
    feature_importance: Optional[Dict[str, float]] = None

class ModelInfoResponse(BaseModel):
    model_type: str
    base_models: List[str]
    training_accuracy: float
    features_used: int
    last_updated: str
    target_accuracy: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# Datos mock para respuestas
MOCK_MODEL_INFO = {
    "model_type": "Ensemble (Stacking)",
    "base_models": ["Random Forest", "AdaBoost", "Extra Trees", "LightGBM"],
    "training_accuracy": 0.8308,
    "features_used": 13,
    "last_updated": "2025-01-01",
    "target_accuracy": 0.83
}

def generate_mock_prediction(features: Dict[str, float]) -> Dict[str, Any]:
    """Genera una predicción mock basada en los features de entrada"""
    
    # Lógica simple para generar predicciones realistas
    score = 0
    
    # Factores que aumentan probabilidad de confirmación
    if 7 < features.get('snr', 0) < 50:
        score += 0.2
    
    if 50 < features.get('depth', 0) < 5000:
        score += 0.2
    
    if 1 < features.get('period', 0) < 1000:
        score += 0.2
    
    if 0.5 < features.get('radius', 0) < 20:
        score += 0.2
    
    if 1000 < features.get('starTemp', 0) < 50000:
        score += 0.1
    
    # Añadir algo de randomización
    score += random.uniform(-0.1, 0.1)
    score = max(0, min(1, score))  # Clamp entre 0 y 1
    
    # Determinar predicción
    if score > 0.6:
        prediction = "CONFIRMED"
        confidence = score
        prob_confirmed = score
        prob_false_positive = 1 - score
    else:
        prediction = "FALSE_POSITIVE"
        confidence = 1 - score
        prob_confirmed = score
        prob_false_positive = 1 - score
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            "CONFIRMED": prob_confirmed,
            "FALSE_POSITIVE": prob_false_positive
        },
        "analysis_timestamp": datetime.utcnow().isoformat(),
        "feature_importance": {
            "snr": random.uniform(0.1, 0.3),
            "depth": random.uniform(0.1, 0.25),
            "period": random.uniform(0.1, 0.2),
            "radius": random.uniform(0.05, 0.15),
            "duration": random.uniform(0.05, 0.15)
        }
    }

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz"""
    return {
        "message": "Exoplanet Detection API - Testing Mode",
        "status": "Running",
        "version": "1.0.0"
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Predicción individual de exoplaneta"""
    
    try:
        # Convertir request a dict
        features = request.model_dump()
        
        # Validaciones básicas
        if any(val <= 0 for val in features.values()):
            raise HTTPException(
                status_code=400,
                detail="Todos los valores deben ser positivos"
            )
        
        # Validaciones astronómicas básicas
        if not (0.1 <= features['period'] <= 10000):
            raise HTTPException(
                status_code=400,
                detail="Período orbital debe estar entre 0.1 y 10,000 días"
            )
        
        if not (0.1 <= features['radius'] <= 100):
            raise HTTPException(
                status_code=400,
                detail="Radio planetario debe estar entre 0.1 y 100 radios terrestres"
            )
        
        # Generar predicción mock
        result = generate_mock_prediction(features)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando predicción: {str(e)}"
        )

@app.post("/api/batch-predict")
async def predict_batch(file: UploadFile = File(...)):
    """Predicción en lote desde archivo CSV"""
    
    try:
        # Validar archivo
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Solo se permiten archivos CSV"
            )
        
        # Leer contenido del archivo
        contents = await file.read()
        csv_data = contents.decode('utf-8')
        
        # Parsear CSV
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        results = []
        
        for i, row in enumerate(csv_reader):
            try:
                # Mapear nombres de columnas comunes
                features = {
                    'period': float(row.get('period', row.get('Period', 0))),
                    'radius': float(row.get('radius', row.get('Radius', 0))),
                    'temp': float(row.get('temp', row.get('Temperature', 0))),
                    'starRadius': float(row.get('starRadius', row.get('Star_Radius', 1))),
                    'starMass': float(row.get('starMass', row.get('Star_Mass', 1))),
                    'starTemp': float(row.get('starTemp', row.get('Star_Temperature', 5778))),
                    'depth': float(row.get('depth', row.get('Depth', 100))),
                    'duration': float(row.get('duration', row.get('Duration', 3))),
                    'snr': float(row.get('snr', row.get('SNR', 10)))
                }
                
                # Generar predicción
                prediction = generate_mock_prediction(features)
                
                results.append({
                    'index': i,
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'prob_confirmed': prediction['probabilities']['CONFIRMED'],
                    'prob_false_positive': prediction['probabilities']['FALSE_POSITIVE']
                })
                
            except (ValueError, KeyError) as e:
                # Continuar con siguiente fila si hay error
                results.append({
                    'index': i,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'prob_confirmed': 0.0,
                    'prob_false_positive': 0.0,
                    'error': str(e)
                })
        
        # Calcular estadísticas
        confirmed_count = sum(1 for r in results if r['prediction'] == 'CONFIRMED')
        false_positive_count = len(results) - confirmed_count
        
        return {
            'total_processed': len(results),
            'confirmed_planets': confirmed_count,
            'false_positives': false_positive_count,
            'processing_time': len(results) * 0.1,  # Mock timing
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'results': results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando archivo: {str(e)}"
        )

@app.get("/api/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Información del modelo ML"""
    return ModelInfoResponse(**MOCK_MODEL_INFO)

@app.get("/api/stats")
async def get_stats():
    """Estadísticas del sistema"""
    return {
        "total_predictions_today": random.randint(50, 200),
        "confirmed_planets_found": random.randint(10, 50),
        "false_positives_detected": random.randint(20, 100),
        "api_uptime": "2h 34m",
        "database_status": "mock_mode",
        "model_accuracy": 83.08,
        "last_updated": datetime.utcnow().isoformat()
    }

# Endpoints adicionales para testing
@app.get("/api/test/exoplanets")
async def get_test_exoplanets():
    """Datos de prueba de exoplanetas"""
    return {
        "exoplanets": [
            {
                "object_id": "TEST-EARTH-001",
                "mission": "Kepler",
                "disposition": "CONFIRMED",
                "period": 365.25,
                "radius": 1.0,
                "confidence": 0.95
            },
            {
                "object_id": "TEST-JUPITER-001", 
                "mission": "TESS",
                "disposition": "CANDIDATE",
                "period": 3.5,
                "radius": 11.2,
                "confidence": 0.78
            }
        ]
    }

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": f"Endpoint no encontrado: {request.url.path}",
            "available_endpoints": [
                "/api/health",
                "/api/predict", 
                "/api/batch-predict",
                "/api/model-info",
                "/api/stats"
            ]
        }
    )

if __name__ == "__main__":
    print("🚀 Iniciando servidor FastAPI para testing...")
    print("📍 Servidor disponible en: http://localhost:8000")
    print("📖 Documentación API: http://localhost:8000/docs")
    print("🔧 Modo: Testing (datos mock)")
    
    uvicorn.run(
        "fastapi_testing_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )