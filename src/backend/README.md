# Backend - Detección de Exoplanetas con IA

Backend FastAPI para el sistema de detección de exoplanetas desarrollado para el NASA Space Apps Challenge 2025.

## 🏗️ Arquitectura

```
src/backend/
├── main.py                 # Aplicación FastAPI principal
├── run_server.py          # Script de inicio del servidor
├── requirements.txt       # Dependencias del backend
├── models/               
│   └── schemas.py         # Modelos Pydantic para validación
├── services/             
│   ├── prediction_service.py  # Servicio de predicción ML
│   └── file_service.py        # Manejo de archivos
└── utils/               
    └── logger.py          # Configuración de logging
```

## 🚀 Instalación y Uso

### 1. Instalación

```bash
# Desde la raíz del proyecto
cd src/backend

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Iniciar el servidor

```bash
# Opción 1: Script directo
python run_server.py

# Opción 2: Uvicorn directo
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Opción 3: Desde la raíz del proyecto
start_backend.bat
```

### 3. Verificar funcionamiento

- **Frontend**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/api/health

## 📡 Endpoints de la API

### Información del Sistema

- `GET /api/health` - Health check del sistema
- `GET /api/model-info` - Información del modelo ML

### Predicciones

- `POST /api/predict` - Predicción individual
- `POST /api/batch-predict` - Predicción en lote (CSV)

### Archivos y Resultados

- `GET /api/results` - Listar archivos de resultados
- `GET /api/results/{filename}` - Descargar archivo específico

### WebSocket

- `WS /ws/batch-progress` - Progreso en tiempo real

## 📊 Ejemplos de Uso

### Predicción Individual

```javascript
const response = await fetch('http://localhost:8000/api/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        period: 3.52474859,
        radius: 2.26,
        temp: 1285.0,
        starRadius: 1.046,
        starTemp: 6091.0,
        depth: 455.8,
        duration: 2.95,
        snr: 18.4
    })
});

const result = await response.json();
console.log(result.prediction); // "CONFIRMED" o "FALSE_POSITIVE"
```

### Predicción en Lote

```javascript
const formData = new FormData();
formData.append('file', csvFile);

const response = await fetch('http://localhost:8000/api/batch-predict', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(`${result.confirmed_planets} planetas confirmados`);
```

## 🔧 Configuración

### Variables de Entorno

```bash
# Configuración del servidor (opcional)
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Configuración de archivos
MAX_UPLOAD_SIZE=100MB
TEMP_FILE_RETENTION=7  # días
```

### Logging

Los logs se guardan en:
- **Consola**: Salida estándar con formato colorizado
- **Archivo**: `logs/exoplanet_api_YYYYMMDD.log`

## 🔗 Integración con el Modelo ML

El backend integra directamente con el modelo existente en `ML DEV/predict_exoplanets.py`:

```python
# El servicio de predicción utiliza:
from predict_exoplanets import ExoplanetPredictor

# Y los modelos entrenados en:
# - ML DEV/trained_models/
# - models/ (legacy)
```

## 📈 Monitoreo y Salud

### Health Check

```bash
curl http://localhost:8000/api/health
```

Respuesta:
```json
{
    "status": "healthy",
    "timestamp": "2025-10-05T10:30:00Z",
    "model_loaded": true,
    "version": "1.0.0"
}
```

### Métricas del Modelo

```bash
curl http://localhost:8000/api/model-info
```

## 🐛 Troubleshooting

### Problemas Comunes

1. **Error "Modelo no cargado"**
   - Verificar que existan modelos en `ML DEV/trained_models/`
   - Ejecutar `train_ensemble.py` si es necesario

2. **Error de CORS**
   - El backend permite todas las origins en desarrollo
   - En producción, configurar origins específicos

3. **Archivos CSV no se procesan**
   - Verificar formato CSV válido
   - Máximo 100MB de tamaño
   - Solo extensiones .csv permitidas

4. **WebSocket no conecta**
   - Verificar que el puerto 8000 esté disponible
   - Comprobar firewall/antivirus

## 🔒 Seguridad

### Validaciones Implementadas

- ✅ Validación de tipos con Pydantic
- ✅ Sanitización de nombres de archivo
- ✅ Límites de tamaño de archivo
- ✅ Validación de contenido CSV
- ✅ Timeout en requests
- ✅ Logging de actividad

### Para Producción

- [ ] Configurar HTTPS
- [ ] Limitar CORS origins
- [ ] Implementar autenticación
- [ ] Rate limiting
- [ ] Monitoring avanzado

## 📚 Dependencias Principales

- **FastAPI**: Framework web moderno
- **Uvicorn**: Servidor ASGI
- **Pydantic**: Validación de datos
- **aiofiles**: Manejo asíncrono de archivos
- **pandas/numpy**: Procesamiento de datos
- **scikit-learn**: Compatibilidad con modelo ML

## 🤝 Contribución

El backend está diseñado para ser:
- **Modular**: Servicios separados y reutilizables
- **Escalable**: Async/await y WebSockets
- **Documentado**: OpenAPI automático
- **Testeable**: Estructura clara para testing

---

**NASA Space Apps Challenge 2025** - Detección de Exoplanetas con IA