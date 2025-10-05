# Arquitectura del Backend - Detección de Exoplanetas

## Descripción General

Backend modular desarrollado para el **NASA Space Apps Challenge 2025**, diseñado específicamente para la detección de exoplanetas usando algoritmos de ensemble learning. La arquitectura sigue principios de **Clean Code** y **Domain-Driven Design** con separación clara de responsabilidades.

## Estructura del Proyecto

```
src/backend/
├── app/                          # Código principal de la aplicación
│   ├── core/                     # Núcleo del sistema
│   │   ├── config.py            # Configuración centralizada
│   │   ├── exceptions.py        # Excepciones personalizadas
│   │   └── __init__.py
│   ├── models/                   # Modelos de datos (Pydantic)
│   │   ├── schemas.py           # Esquemas de validación
│   │   └── __init__.py
│   ├── services/                # Capa de servicios de negocio
│   │   ├── data_processing.py   # Procesamiento de datos astronómicos
│   │   ├── file_processing.py   # Manejo de archivos CSV
│   │   ├── logging_service.py   # Logging estructurado
│   │   └── __init__.py
│   ├── utils/                   # Utilidades y helpers
│   │   ├── helpers.py           # Funciones auxiliares
│   │   ├── config_utils.py      # Utilidades de configuración
│   │   └── __init__.py
│   └── __init__.py              # Inicialización del módulo app
├── main.py                      # Punto de entrada de la aplicación
├── .env.template               # Plantilla de variables de entorno
├── setup_env.sh               # Script de configuración (Linux/Mac)
├── setup_env.bat              # Script de configuración (Windows)
└── logs/                       # Archivos de log (creado automáticamente)
```

## Principios Arquitectónicos

### 1. Separación de Responsabilidades

- **Core**: Configuración, excepciones, constantes astronómicas
- **Models**: Validación de datos de entrada/salida con Pydantic
- **Services**: Lógica de negocio (procesamiento de datos, feature engineering)
- **Utils**: Funciones auxiliares reutilizables

### 2. Inyección de Dependencias

```python
# Ejemplo de uso de servicios
from app.services import DataProcessor, FileProcessor
from app.utils import get_config

# Los servicios se inicializan con configuración
data_processor = DataProcessor()
file_processor = FileProcessor(max_file_size=get_config().get('file_processing.max_file_size'))
```

### 3. Manejo Centralizado de Errores

```python
from app.core.exceptions import ValidationError, DataProcessingError

try:
    result = data_processor.validate_single_input(data)
except ValidationError as e:
    # Error específico con contexto astronómico
    return {"error": e.to_dict()}
```

## Componentes Principales

### Core Module

#### `config.py`
- **Settings**: Configuración centralizada con soporte para variables de entorno
- **ExoplanetConstants**: Constantes astronómicas (radios, masas, zonas habitables)
- **Validación automática**: Rangos válidos para parámetros astronómicos

```python
from app.core.config import settings, ExoplanetConstants

# Acceso a configuración
max_file_size = settings.max_file_size
solar_temp = ExoplanetConstants.SOLAR_TEMPERATURE

# Clasificación automática
planet_type = ExoplanetConstants.get_planet_type(radius=1.2)  # "Earth-size"
```

#### `exceptions.py`
- **Jerarquía de excepciones**: Específicas para diferentes tipos de errores
- **Contexto detallado**: Información adicional para debugging
- **Status codes HTTP**: Mapeo automático para APIs

### Services Layer

#### `DataProcessor`
Servicio principal para procesamiento de datos astronómicos:

```python
processor = DataProcessor()

# Validación individual
features = processor.validate_single_input(data_dict)

# Procesamiento por lotes
df_clean = processor.validate_batch_data(df)

# Feature engineering astronómico
enhanced_features = processor.apply_feature_engineering(data)
```

**Características principales:**
- Validación de rangos astronómicos realistas
- Feature engineering automático (ratios planetarios, zonas habitables)
- Detección de inconsistencias físicas
- Estadísticas de procesamiento

#### `FileProcessor`
Manejo especializado de archivos CSV con datos de NASA:

```python
file_processor = FileProcessor()

# Validación de archivos
validation_result = file_processor.validate_file(file_content, filename)

# Procesamiento con mapeo automático de columnas
result = file_processor.process_csv_file(file_content, filename)
```

**Características:**
- Mapeo automático de columnas de diferentes misiones (Kepler, TESS, K2)
- Validación de formato y codificación
- Estimación de recursos y tiempo de procesamiento

#### `LoggingService`
Sistema de logging estructurado para operaciones astronómicas:

```python
logging_service = LoggingService()
logger = logging_service.get_logger('exoplanet_data')

# Logging específico para ML
logging_service.log_model_prediction(
    logger, 'single', features, 'CONFIRMED', 0.85, 125.3
)

# Logging de procesamiento de datos
logging_service.log_data_processing(
    logger, 'feature_engineering', 1000, 987, 2345.6
)
```

### Models Layer

#### Esquemas Pydantic
Validación automática de datos astronómicos:

```python
class ExoplanetFeatures(BaseModel):
    period: float = Field(..., gt=0.1, lt=5000.0)  # días
    radius: float = Field(..., gt=0.1, lt=50.0)    # radios terrestres
    temp: float = Field(..., gt=100.0, lt=10000.0) # Kelvin
    # ... más campos con validación astronómica
```

#### Tipos de Respuesta
```python
class PredictionResult(BaseModel):
    prediction: DispositionType
    confidence: float
    probabilities: Dict[str, float]
    enhanced_features: Optional[EnhancedFeatures]
    processing_time_ms: float
```

### Utils Layer

#### `helpers.py`
Funciones auxiliares especializadas:

```python
# Conversiones seguras para datos astronómicos
radius = safe_float_conversion(raw_value, default=1.0)

# Validación de rangos astronómicos
is_valid, error = validate_astronomical_range(
    temperature, 'stellar_temp', 2000.0, 50000.0
)

# Estadísticas para análisis de datos
stats = calculate_statistics(planet_radii_list)
```

#### `config_utils.py`
Gestión avanzada de configuración:

```python
config = ConfigurationManager('config.json')

# Acceso con notación de puntos
debug_mode = config.get('app.debug', False)

# Validación de configuración
is_valid, errors = config.validate()
```

## Flujo de Procesamiento de Datos

### 1. Validación de Entrada
```python
# Validación automática con Pydantic
features = ExoplanetFeatures(**input_data)

# Validación astronómica adicional
processor = DataProcessor()
validated_features = processor.validate_single_input(input_data)
```

### 2. Feature Engineering
```python
# Derivación automática de características físicas
enhanced = processor.apply_feature_engineering(validated_data)

# Características derivadas incluyen:
# - Ratios planeta/estrella
# - Distancia a zona habitable
# - Métricas de calidad de señal
# - Clasificaciones automáticas
```

### 3. Procesamiento por Lotes
```python
# Validación y limpieza de datasets completos
df_clean = processor.validate_batch_data(dataframe)

# Estadísticas de procesamiento
stats = processor.get_processing_stats(df_clean)
```

## Configuración y Deployment

### Variables de Entorno
El sistema usa un enfoque de configuración por capas:

1. **Valores por defecto** (en `config.py`)
2. **Archivo de configuración** (JSON opcional)
3. **Variables de entorno** (máxima prioridad)

### Scripts de Configuración

#### Linux/Mac:
```bash
./setup_env.sh        # Configuración completa
./setup_env.sh check   # Verificar instalación
./setup_env.sh clean   # Limpiar entorno
```

#### Windows:
```cmd
setup_env.bat          # Configuración completa
```

### Estructura de Directorios Automática
```
logs/                   # Logs con rotación automática
models/                 # Modelos ML entrenados
data/                   # Datasets de NASA
temp/                   # Archivos temporales
```

## Manejo de Errores

### Jerarquía de Excepciones
```python
ExoplanetBackendError
├── ValidationError              # Datos inválidos
├── DataProcessingError         # Errores de procesamiento
├── FeatureEngineeringError     # Errores en derivación de características
├── FileProcessingError         # Errores de archivos
├── AstronomicalDataError       # Inconsistencias físicas
├── ModelError                  # Errores de ML
├── ConfigurationError          # Configuración inválida
└── ServiceUnavailableError     # Servicios no disponibles
```

### Respuestas de Error Estandarizadas
```python
{
    "error": "ValidationError",
    "message": "Field 'period' value 50000 outside valid range [0.1, 5000.0]",
    "error_code": "VALIDATION_ERROR",
    "details": {
        "field": "period",
        "provided_value": 50000,
        "expected_range": [0.1, 5000.0]
    },
    "timestamp": "2025-01-01T12:00:00Z"
}
```

## Logging y Monitoreo

### Logging Estructurado
```python
# Logs automáticos para operaciones ML
{
    "timestamp": "2025-01-01T12:00:00Z",
    "level": "INFO",
    "logger": "exoplanet_backend.model",
    "message": "Model prediction: CONFIRMED (confidence: 0.851, time: 125.3ms)",
    "extra": {
        "model_prediction": true,
        "prediction_type": "single",
        "prediction_result": "CONFIRMED", 
        "confidence": 0.851,
        "processing_time_ms": 125.3
    }
}
```

### Métricas de Performance
- Tiempo de procesamiento por operación
- Estadísticas de validación de datos
- Uso de memoria y recursos
- Tasas de éxito/error

## Extensibilidad

### Agregar Nuevos Servicios
```python
# 1. Crear servicio en app/services/
class NewAstronomicalService:
    def __init__(self):
        self.config = get_config()
    
    def process(self, data):
        # Lógica específica
        pass

# 2. Registrar en app/services/__init__.py
from .new_service import NewAstronomicalService

# 3. Usar en la aplicación
service = NewAstronomicalService()
```

### Agregar Nuevas Validaciones
```python
# En DataProcessor
def _validate_new_constraint(self, data: Dict) -> None:
    # Nueva validación astronómica
    if custom_check_fails(data):
        raise AstronomicalDataError(
            "Custom constraint failed",
            parameter="custom_param",
            astronomical_constraint="New physical law"
        )
```

## Seguridad y Performance

### Validación de Datos
- Rangos astronómicos realistas
- Sanitización de strings
- Validación de tipos de archivos
- Límites de tamaño de archivos

### Optimizaciones
- Procesamiento por chunks para datasets grandes
- Caché de configuración
- Logging asíncrono
- Validación lazy para operaciones costosas

## Testing

### Estructura de Tests
```
tests/
├── unit/
│   ├── test_data_processing.py
│   ├── test_file_processing.py
│   └── test_utils.py
├── integration/
│   └── test_services_integration.py
└── fixtures/
    ├── sample_data.csv
    └── test_config.json
```

### Comandos de Testing
```bash
# Tests unitarios
pytest tests/unit/

# Tests de integración
pytest tests/integration/

# Coverage
pytest --cov=app tests/
```

## Notas de Implementación

### Dependencias Opcionales
El sistema está diseñado para funcionar con dependencias opcionales:
- **Pydantic**: Validación avanzada (fallback a clases simples)
- **Pandas/NumPy**: Procesamiento de datos (fallback a Python puro)
- **PSUtil**: Monitoreo de sistema (fallback sin métricas)

### Compatibilidad
- **Python**: >= 3.8
- **Sistemas**: Linux, macOS, Windows
- **Arquitecturas**: x86_64, ARM64

### Performance Targets
- **Validación individual**: < 10ms
- **Procesamiento por lotes**: < 1ms por registro
- **Feature engineering**: < 5ms por objeto
- **Memoria**: < 100MB para datasets de 10K registros