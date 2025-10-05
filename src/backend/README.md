# Backend Exoplanetas - NASA Space Apps Challenge 2025

## Descripción

Backend modular desarrollado exclusivamente para la detección de exoplanetas usando algoritmos de ensemble learning. Esta implementación se enfoca únicamente en la **lógica interna del backend**, **estructura del proyecto**, **modularización del código**, **servicios**, **utilidades** y **manejo de datos**.

> **Nota Importante**: Este backend NO incluye APIs REST, machine learning o frontend. Esos componentes son desarrollados por otros equipos.

## Características Principales

- ✅ **Arquitectura Modular**: Separación clara de responsabilidades (core, services, models, utils)
- ✅ **Procesamiento de Datos Astronómicos**: Validación y feature engineering especializado
- ✅ **Manejo de Archivos NASA**: Soporte para datasets KOI, TOI, K2 con mapeo automático
- ✅ **Logging Estructurado**: Sistema de logging avanzado para debugging y monitoreo
- ✅ **Configuración Centralizada**: Gestión de configuración por capas con variables de entorno
- ✅ **Manejo de Errores**: Jerarquía de excepciones específicas para astronomía
- ✅ **Validación Física**: Verificación de consistencia astronómica en los datos
- ✅ **Utilidades Especializadas**: Helpers para conversiones, estadísticas y validaciones

## Arquitectura del Sistema

```
src/backend/
├── app/                          # Aplicación principal
│   ├── core/                     # Núcleo del sistema
│   │   ├── config.py            # Configuración y constantes astronómicas
│   │   ├── exceptions.py        # Excepciones especializadas
│   │   └── __init__.py
│   ├── models/                   # Modelos de datos (Pydantic)
│   │   ├── schemas.py           # Esquemas de validación
│   │   └── __init__.py
│   ├── services/                # Servicios de negocio
│   │   ├── data_processing.py   # Procesamiento de datos astronómicos
│   │   ├── file_processing.py   # Manejo de archivos CSV
│   │   ├── logging_service.py   # Logging estructurado
│   │   └── __init__.py
│   ├── utils/                   # Utilidades y helpers
│   │   ├── helpers.py           # Funciones auxiliares
│   │   ├── config_utils.py      # Utilidades de configuración
│   │   └── __init__.py
│   └── __init__.py
├── ARCHITECTURE.md              # Documentación de arquitectura
├── DEVELOPMENT.md               # Guía de desarrollo
├── .env.template               # Plantilla de configuración
├── setup_env.sh               # Configuración Linux/Mac
├── setup_env.bat              # Configuración Windows
└── logs/                       # Logs del sistema
```

## Inicio Rápido

### 1. Configuración del Entorno

#### Linux/Mac:
```bash
cd src/backend
chmod +x setup_env.sh
./setup_env.sh
source activate_env.sh
```

#### Windows:
```cmd
cd src\backend
setup_env.bat
activate_env.bat
```

### 2. Configuración Personalizada

```bash
# Copiar plantilla de configuración
cp .env.template .env

# Editar variables según necesidades
# Principales configuraciones:
DEBUG=True
LOG_LEVEL=INFO
ENVIRONMENT=development
MAX_FILE_SIZE=52428800  # 50MB
```

### 3. Uso Básico

```python
# Ejemplo de uso del sistema
from app.services import DataProcessor, FileProcessor
from app.utils import get_config, Timer

# Configuración
config = get_config()

# Procesamiento de datos individuales
processor = DataProcessor()
validated_data = processor.validate_single_input({
    'period': 365.25,
    'radius': 1.0,
    'temp': 288.0,
    'starRadius': 1.0,
    'starMass': 1.0,
    'starTemp': 5778.0,
    'depth': 84.0,
    'duration': 6.5,
    'snr': 15.0
})

# Feature engineering astronómico
enhanced_features = processor.apply_feature_engineering(validated_data)

# Procesamiento de archivos CSV
file_processor = FileProcessor()
with open('exoplanet_data.csv', 'rb') as f:
    result = file_processor.process_csv_file(f.read(), 'exoplanet_data.csv')

print(f"Procesados {result['processed_rows']} objetos")
```

## Componentes Principales

### 🔧 Core Module

#### Configuration (`config.py`)
- **Settings**: Configuración centralizada con variables de entorno
- **ExoplanetConstants**: Constantes astronómicas (radios, masas, zonas habitables)
- **Validación automática**: Rangos físicos para parámetros astronómicos

#### Exceptions (`exceptions.py`)
- **Jerarquía especializada**: 8 tipos de excepciones específicas
- **Contexto detallado**: Información adicional para debugging
- **Mapeo HTTP**: Status codes automáticos para APIs futuras

### 🛠️ Services Layer

#### DataProcessor
Procesamiento y validación de datos astronómicos:
- ✅ Validación de rangos físicos realistas
- ✅ Feature engineering automático (ratios planetarios, zonas habitables)
- ✅ Detección de inconsistencias astronómicas
- ✅ Estadísticas de procesamiento por lotes

#### FileProcessor  
Manejo especializado de archivos NASA:
- ✅ Soporte para datasets KOI (Kepler), TOI (TESS), K2
- ✅ Mapeo automático de columnas entre misiones
- ✅ Validación de formato y codificación
- ✅ Estimación de recursos necesarios

#### LoggingService
Sistema de logging estructurado:
- ✅ Logs contextuales para operaciones astronómicas
- ✅ Rotación automática de archivos
- ✅ Métricas de performance integradas
- ✅ Formato JSON estructurado (opcional)

### 🔨 Utils Layer

#### Helpers (`helpers.py`)
Utilidades especializadas:
- ✅ Conversiones seguras para datos astronómicos
- ✅ Validación de rangos físicos
- ✅ Estadísticas para análisis de datos
- ✅ Formateo de archivos y duraciones
- ✅ Context manager para timing

#### Config Utils (`config_utils.py`)
Gestión avanzada de configuración:
- ✅ Configuración por capas (defaults → archivo → env vars)
- ✅ Validación automática de configuración
- ✅ Acceso con notación de puntos
- ✅ Recarga dinámica

## Validaciones Astronómicas

### Rangos Físicos Validados
```python
VALIDATION_RANGES = {
    'period': (0.1, 5000.0),        # días
    'radius': (0.1, 50.0),          # radios terrestres  
    'temp': (100.0, 10000.0),       # Kelvin
    'starRadius': (0.1, 10.0),      # radios solares
    'starMass': (0.1, 10.0),        # masas solares
    'starTemp': (2000.0, 50000.0),  # Kelvin
    'depth': (0.0, 1000000.0),      # ppm
    'duration': (0.1, 100.0),       # horas
    'snr': (0.0, 1000.0)            # ratio
}
```

### Consistencia Física
- **Profundidad de tránsito**: Verificación geométrica planeta/estrella
- **Temperatura de equilibrio**: Validación vs parámetros orbitales  
- **Estabilidad orbital**: Criterios de Hill (opcional)
- **Zona habitable**: Cálculo automático basado en temperatura estelar

## Feature Engineering Automático

### Características Derivadas
```python
# Ratios físicos
planet_star_radius_ratio = radius / starRadius
equilibrium_temp_ratio = temp / starTemp

# Características orbitales
transit_depth_expected = (radius / starRadius)² × 10⁶  # ppm
orbital_velocity = 2π / period

# Métricas de habitabilidad
hz_distance = (calculated_distance - hz_center) / hz_width
is_habitable_zone = hz_inner ≤ distance ≤ hz_outer

# Calidad de señal
depth_snr_ratio = depth / snr
duration_period_ratio = duration / (period × 24)

# Clasificaciones automáticas
planet_type = classify_by_radius(radius)  # "Earth-size", "Super-Earth", etc.
stellar_type = classify_by_temperature(starTemp)  # "G dwarf", "M dwarf", etc.
```

## Manejo de Datasets NASA

### Mapeo Automático de Columnas

#### Kepler (KOI)
```python
'koi_period' → 'period'
'koi_prad' → 'radius' 
'koi_teq' → 'temp'
'koi_srad' → 'starRadius'
'koi_steff' → 'starTemp'
# ... más mapeos
```

#### TESS (TOI)  
```python
'pl_orbper' → 'period'
'pl_rade' → 'radius'
'st_teff' → 'starTemp'
# ... más mapeos
```

#### Nombres Alternativos
```python
'orbital_period' → 'period'
'planet_radius' → 'radius'
'stellar_temperature' → 'starTemp'
# ... más variaciones
```

## Logging y Monitoreo

### Logs Estructurados
```json
{
    "timestamp": "2025-01-01T12:00:00Z",
    "level": "INFO",
    "logger": "exoplanet_backend.data_processing",
    "message": "Data processing: feature_engineering - 987/1000 records (2345.6ms)",
    "extra": {
        "data_processing": true,
        "operation": "feature_engineering",
        "input_count": 1000,
        "output_count": 987,
        "success_rate": 0.987,
        "processing_time_ms": 2345.6
    }
}
```

### Métricas de Performance
- ⏱️ **Tiempo de procesamiento** por operación
- 📊 **Tasas de éxito** en validación
- 💾 **Uso de memoria** en tiempo real
- 📈 **Estadísticas de datos** procesados

## Configuración Avanzada

### Variables de Entorno Principales
```bash
# Aplicación
ENVIRONMENT=development|testing|production
DEBUG=True|False

# Procesamiento  
MAX_FILE_SIZE=52428800  # 50MB
MAX_BATCH_SIZE=10000
ENABLE_FEATURE_ENGINEERING=True

# Logging
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
LOG_MAX_SIZE=10485760  # 10MB
LOG_BACKUP_COUNT=5

# Paths
MODELS_DIR=models
DATA_DIR=data  
LOGS_DIR=logs
```

### Configuración por Archivo (JSON)
```json
{
    "app": {
        "name": "Exoplanet Detection Backend",
        "debug": false
    },
    "validation": {
        "ranges": {
            "period": [0.1, 5000.0],
            "radius": [0.1, 50.0]
        }
    },
    "file_processing": {
        "max_file_size": 52428800,
        "allowed_extensions": [".csv", ".txt"]
    }
}
```

## Testing

### Ejecutar Tests
```bash
# Tests unitarios
pytest tests/unit/ -v

# Tests de integración  
pytest tests/integration/ -v

# Coverage
pytest --cov=app tests/ --cov-report=html
```

### Estructura de Tests
```
tests/
├── unit/
│   ├── test_data_processing.py    # Tests de validación y feature engineering
│   ├── test_file_processing.py    # Tests de manejo de archivos
│   ├── test_utils.py             # Tests de utilidades
│   └── test_config.py            # Tests de configuración
├── integration/
│   └── test_pipeline.py          # Tests de pipeline completo
└── fixtures/
    ├── sample_kepler.csv         # Datos de prueba Kepler
    ├── sample_tess.csv           # Datos de prueba TESS
    └── test_config.json          # Configuración de prueba
```

## Dependencias y Compatibilidad

### Dependencias Core
```txt
# Procesamiento de datos
pandas>=2.0.0
numpy>=1.24.0

# Configuración y validación  
pydantic>=2.0.0
python-dotenv>=1.0.0

# Logging y monitoreo
structlog>=23.1.0
psutil>=5.9.0  # opcional

# Cálculos astronómicos
astropy>=5.3.0  # opcional
```

### Compatibilidad
- **Python**: 3.8+
- **Sistemas**: Linux, macOS, Windows
- **Arquitecturas**: x86_64, ARM64

### Dependencias Opcionales
El sistema funciona con fallbacks si ciertas dependencias no están disponibles:
- **Pydantic**: Fallback a clases simples para validación
- **Pandas/NumPy**: Fallback a Python puro (con performance reducido)
- **Astropy**: Fallback a cálculos simplificados

## Troubleshooting

### Problemas Comunes

#### Error de importación
```bash
# Error: ModuleNotFoundError
# Solución: Activar entorno virtual
source activate_env.sh  # Linux/Mac
# o
activate_env.bat  # Windows
```

#### Error de configuración
```python
# Debug de configuración
from app.utils import get_config
config = get_config()
is_valid, errors = config.validate()
print(f"Config valid: {is_valid}")
print(f"Errors: {errors}")
```

#### Error en datos astronómicos
```python
# Verificar rangos válidos
from app.services import DataProcessor
processor = DataProcessor()
print("Valid ranges:")
for param, (min_val, max_val) in processor.VALIDATION_RANGES.items():
    print(f"  {param}: [{min_val}, {max_val}]")
```

## Contribución

### Estándares de Código
```bash
# Formateo
black app/ tests/
isort app/ tests/

# Linting  
flake8 app/ tests/
mypy app/
```

### Conventional Commits
```bash
feat(data_processor): implementar validación de estabilidad orbital
fix(file_processor): corregir mapeo de columnas TESS  
docs(architecture): actualizar documentación de servicios
test(utils): agregar tests para helpers astronómicos
```

## Documentación Adicional

- 📖 **[ARCHITECTURE.md](ARCHITECTURE.md)**: Arquitectura detallada del sistema
- 🛠️ **[DEVELOPMENT.md](DEVELOPMENT.md)**: Guía completa de desarrollo  
- 🔧 **[.env.template](.env.template)**: Configuración de variables de entorno

## Licencia

NASA Space Apps Challenge 2025 - Proyecto educativo y de investigación.

---

**Desarrollado exclusivamente para el backend modular del sistema de detección de exoplanetas. No incluye APIs, ML o frontend por diseño.**