# MongoDB Integration - Exoplanet Detection System

## 🎯 Overview

Esta branch implementa la integración completa con MongoDB para el sistema de detección de exoplanetas del NASA Space Apps Challenge 2025.

## 🏗️ Arquitectura de Base de Datos

### Colecciones MongoDB

#### 1. `prediction_results`
Almacena resultados individuales de predicciones:
```javascript
{
  "_id": ObjectId,
  "features": {
    "period": 365.25,
    "radius": 1.0,
    "temp": 288,
    // ... otros parámetros
  },
  "prediction": "CONFIRMED", // o "FALSE_POSITIVE"
  "confidence": 0.85,
  "probabilities": {
    "CONFIRMED": 0.85,
    "FALSE_POSITIVE": 0.15
  },
  "model_version": "stacking_ensemble_v1.0",
  "timestamp": ISODate,
  "processing_time_ms": 150.5,
  "user_session": "uuid-string"
}
```

#### 2. `batch_analyses`
Análisis de lotes de datos:
```javascript
{
  "_id": ObjectId,
  "batch_id": "uuid-string",
  "filename": "candidate_list.csv",
  "total_objects": 1000,
  "confirmed_planets": 25,
  "false_positives": 975,
  "status": "completed",
  "processing_start": ISODate,
  "processing_end": ISODate,
  "results": [/* array de resultados */]
}
```

#### 3. `model_metrics`
Métricas de rendimiento del modelo:
```javascript
{
  "_id": ObjectId,
  "model_name": "stacking_ensemble",
  "model_version": "v1.0",
  "accuracy": 0.8308,
  "precision": 0.825,
  "recall": 0.812,
  "f1_score": 0.818,
  "roc_auc": 0.948,
  "evaluation_date": ISODate
}
```

#### 4. `user_sessions`
Seguimiento de sesiones de usuario:
```javascript
{
  "_id": ObjectId,
  "session_id": "uuid-string",
  "start_time": ISODate,
  "last_activity": ISODate,
  "predictions_count": 15,
  "batch_analyses_count": 2
}
```

## 🚀 Instalación y Configuración

### 1. Instalar Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configurar MongoDB

#### Opción A: MongoDB Local
```bash
# Descargar e instalar MongoDB Community Edition
# Windows: https://www.mongodb.com/try/download/community
# Iniciar servicio
mongod --dbpath "C:\data\db"
```

#### Opción B: Docker (Recomendado)
```bash
# Iniciar con Docker Compose
docker-compose -f docker-compose.dev.yml up -d

# Ver logs
docker-compose -f docker-compose.dev.yml logs -f

# Acceder a MongoDB Admin (opcional)
# http://localhost:8081
```

### 3. Variables de Entorno
```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar configuración
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=exoplanets_db
```

## 📊 Nuevos Endpoints API

### Analytics y Estadísticas
```http
GET /api/recent-predictions?limit=50&session_id=uuid
GET /api/prediction-stats?days=7
GET /api/batch-analysis/{batch_id}
```

### Health Check Mejorado
```http
GET /api/health
```
Ahora incluye estado de la base de datos.

## 🔧 Servicios Implementados

### `PredictionService`
- `save_prediction()`: Guardar resultado individual
- `get_recent_predictions()`: Obtener predicciones recientes
- `get_prediction_stats()`: Estadísticas de predicciones

### `BatchAnalysisService`
- `create_batch_analysis()`: Crear análisis de lote
- `update_batch_progress()`: Actualizar progreso
- `get_batch_analysis()`: Obtener resultados

### `ModelMetricsService`
- `save_model_metrics()`: Guardar métricas del modelo
- `get_latest_metrics()`: Obtener métricas más recientes
- `get_metrics_history()`: Historial de rendimiento

### `UserSessionService`
- `create_or_update_session()`: Gestión de sesiones
- `increment_prediction_count()`: Contadores de uso

## 📈 Beneficios de la Integración

### 1. Persistencia de Datos
- Todas las predicciones se almacenan permanentemente
- Historial completo de análisis
- Seguimiento de rendimiento del modelo

### 2. Analytics Avanzados
- Estadísticas en tiempo real
- Tendencias de uso
- Performance monitoring

### 3. Sesiones de Usuario
- Seguimiento de actividad individual
- Personalización futura
- Analytics de comportamiento

### 4. Escalabilidad
- Base preparada para múltiples usuarios
- Arquitectura asíncrona
- Sharding y clustering futuros

## 🧪 Testing

### Probar Conexión
```bash
# Desde el backend
python -c "
from database import database
import asyncio
asyncio.run(database.connect_to_database())
print('MongoDB connected successfully!')
"
```

### Endpoints de Desarrollo
```bash
# Health check
curl http://localhost:8000/api/health

# Predicción simple
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "period": 365.25,
    "radius": 1.0,
    "temp": 288,
    "starRadius": 1.0,
    "starMass": 1.0,
    "starTemp": 5778,
    "depth": 84,
    "duration": 13.2,
    "snr": 25.5
  }'

# Ver predicciones recientes
curl http://localhost:8000/api/recent-predictions
```

## 📁 Estructura de Archivos Actualizada

```
src/backend/
├── main.py                 # FastAPI app con MongoDB
├── database/              # Módulo MongoDB
│   ├── __init__.py
│   ├── config.py          # Configuración de conexión
│   ├── models.py          # Modelos Pydantic/Beanie
│   └── services.py        # Servicios de datos
├── requirements.txt       # Dependencies actualizadas
└── .env.example          # Configuración de ejemplo

scripts/
└── init-mongo.js         # Script de inicialización

docker-compose.dev.yml    # Stack completo con MongoDB
```

## 🎯 Objetivos Cumplidos

- ✅ **Persistencia**: Todas las predicciones se guardan
- ✅ **Analytics**: Estadísticas en tiempo real
- ✅ **Escalabilidad**: Arquitectura asíncrona con MongoDB
- ✅ **Monitoreo**: Health checks y métricas
- ✅ **Sesiones**: Seguimiento de usuarios
- ✅ **Docker**: Stack completo containerizado

## 🔄 Próximos Pasos

1. **Frontend Integration**: Conectar React con nuevos endpoints
2. **Real-time Updates**: WebSockets para actualizaciones en vivo
3. **Advanced Analytics**: Dashboards con métricas detalladas
4. **User Authentication**: Sistema de usuarios completo
5. **Model Versioning**: Versionado de modelos ML

---

**Target Accuracy**: 83.08% (Stacking Ensemble)  
**Architecture**: FastAPI + MongoDB + React + Docker  
**Challenge**: NASA Space Apps 2025