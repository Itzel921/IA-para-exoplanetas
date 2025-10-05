# Guía de Desarrollo - Backend Exoplanetas

## Configuración del Entorno de Desarrollo

### 1. Requisitos Previos

**Software requerido:**
- Python 3.8 o superior
- Git
- Editor de código (VS Code recomendado)

**Dependencias del sistema (opcionales):**
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev python3-venv build-essential

# CentOS/RHEL
sudo yum install python3-devel python3-venv gcc

# macOS (con Homebrew)
brew install python@3.9
```

### 2. Configuración Inicial

#### Paso 1: Clonar y configurar entorno
```bash
# Navegar al directorio del backend
cd src/backend

# Ejecutar script de configuración
# Linux/Mac:
chmod +x setup_env.sh
./setup_env.sh

# Windows:
setup_env.bat
```

#### Paso 2: Activar entorno virtual
```bash
# Linux/Mac:
source activate_env.sh

# Windows:
activate_env.bat

# Manual:
# Linux/Mac: source exoplanet_backend_env/bin/activate
# Windows: exoplanet_backend_env\Scripts\activate.bat
```

#### Paso 3: Configuración de variables de entorno
```bash
# Copiar plantilla de configuración
cp .env.template .env

# Editar configuración según necesidades
# Las variables más importantes:
DEBUG=True
LOG_LEVEL=DEBUG
ENVIRONMENT=development
```

### 3. Estructura de Desarrollo

#### Organización de código
```python
# Importaciones estándar primero
import os
import logging
from typing import Dict, List, Optional

# Importaciones de terceros
import pandas as pd
from fastapi import FastAPI

# Importaciones locales
from app.core.exceptions import ValidationError
from app.services import DataProcessor
```

#### Convenciones de naming
```python
# Variables y funciones: snake_case
stellar_temperature = 5778.0
def validate_astronomical_data():
    pass

# Clases: PascalCase
class DataProcessor:
    pass

# Constantes: UPPER_SNAKE_CASE
SOLAR_RADIUS_KM = 695700

# Variables privadas: prefijo _
class MyService:
    def __init__(self):
        self._internal_data = {}
```

## Desarrollo de Nuevas Funcionalidades

### 1. Agregar Nueva Validación Astronómica

#### Paso 1: Definir la validación
```python
# En app/services/data_processing.py
def _validate_orbital_stability(self, data: Dict) -> None:
    """
    Validar estabilidad orbital usando criterios de Hill
    """
    period = data['period']
    star_mass = data['starMass']
    
    # Calcular radio de Hill simplificado
    hill_radius = period * (star_mass / 3) ** (1/3)
    
    if hill_radius < 0.1:  # Criterio astronómico
        raise AstronomicalDataError(
            f"Orbital configuration unstable (Hill radius: {hill_radius})",
            parameter="orbital_stability",
            astronomical_constraint="Hill radius > 0.1"
        )
```

#### Paso 2: Integrar en el flujo
```python
def _validate_astronomical_consistency(self, data: Dict) -> None:
    # Validaciones existentes...
    
    # Nueva validación
    self._validate_orbital_stability(data)
```

#### Paso 3: Agregar tests
```python
# En tests/unit/test_data_processing.py
def test_orbital_stability_validation():
    processor = DataProcessor()
    
    # Caso inválido
    unstable_data = {
        'period': 0.5,  # Muy corto
        'starMass': 0.1,  # Estrella pequeña
        # ... otros campos
    }
    
    with pytest.raises(AstronomicalDataError) as exc_info:
        processor.validate_single_input(unstable_data)
    
    assert "orbital_stability" in str(exc_info.value)
```

### 2. Crear Nuevo Servicio

#### Plantilla base
```python
# app/services/new_service.py
"""
Nuevo servicio para [descripción]

Funcionalidades:
- Feature 1
- Feature 2
"""

import logging
from typing import Dict, Any, Optional

from ..core.exceptions import ServiceError
from ..utils import get_config, Timer

logger = logging.getLogger(__name__)


class NewAstronomicalService:
    """
    Servicio para [propósito específico]
    """
    
    def __init__(self):
        """Inicializar servicio con configuración"""
        self.config = get_config()
        self.parameter = self.config.get('new_service.parameter', 'default_value')
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar datos astronómicos
        
        Args:
            input_data: Datos de entrada
            
        Returns:
            Datos procesados
            
        Raises:
            ServiceError: Si el procesamiento falla
        """
        try:
            with Timer() as timer:
                # Validar entrada
                self._validate_input(input_data)
                
                # Procesar
                result = self._perform_processing(input_data)
                
                # Log de performance
                logger.info(
                    f"Processing completed in {timer.elapsed_ms:.2f}ms",
                    extra={
                        'service': self.__class__.__name__,
                        'operation': 'process_data',
                        'duration_ms': timer.elapsed_ms,
                        'input_size': len(input_data)
                    }
                )
                
                return result
                
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise ServiceError(f"Service processing failed: {str(e)}")
    
    def _validate_input(self, data: Dict[str, Any]) -> None:
        """Validar datos de entrada"""
        required_fields = ['field1', 'field2']
        missing = [f for f in required_fields if f not in data]
        
        if missing:
            raise ServiceError(f"Missing required fields: {missing}")
    
    def _perform_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Lógica principal de procesamiento"""
        # Implementar lógica específica
        processed_data = {
            'original': data,
            'processed_field': self._custom_calculation(data),
            'metadata': {
                'service': self.__class__.__name__,
                'version': '1.0.0'
            }
        }
        
        return processed_data
    
    def _custom_calculation(self, data: Dict[str, Any]) -> float:
        """Cálculo astronómico específico"""
        # Implementar cálculo
        return 42.0
```

#### Registrar el servicio
```python
# app/services/__init__.py
from .new_service import NewAstronomicalService

__all__ = [
    "DataProcessor",
    "FileProcessor", 
    "LoggingService",
    "NewAstronomicalService"  # Agregar aquí
]
```

### 3. Implementar Nueva Feature Engineering

#### Característica astronómica personalizada
```python
# En app/services/data_processing.py
def _calculate_tidal_heating(self, data: Dict) -> float:
    """
    Calcular calentamiento por fuerzas de marea
    Importante para habitabilidad de exolunas
    """
    try:
        period = data['period']
        star_mass = data['starMass']
        planet_radius = data['radius']
        
        # Simplified tidal heating calculation
        # Real implementation would be more complex
        distance = (period ** 2 * star_mass) ** (1/3)
        tidal_force = star_mass / (distance ** 3)
        heating = tidal_force * planet_radius ** 5
        
        return float(heating)
        
    except Exception as e:
        logger.warning(f"Tidal heating calculation failed: {e}")
        return 0.0

# Integrar en feature engineering
def _engineer_single_features(self, data: Dict) -> EnhancedFeatures:
    # ... características existentes ...
    
    # Nueva característica
    tidal_heating = self._calculate_tidal_heating(data)
    
    return EnhancedFeatures(
        # ... campos existentes ...
        tidal_heating=tidal_heating,
        # ...
    )
```

## Debugging y Testing

### 1. Logging para Debug

#### Configuración de debug
```python
# .env
DEBUG=True
LOG_LEVEL=DEBUG
SHOW_ERROR_DETAILS=True
```

#### Logging contextual
```python
import logging
from app.utils import generate_request_id

logger = logging.getLogger(__name__)

def debug_processing_flow(data):
    request_id = generate_request_id()
    
    logger.debug(
        f"Starting processing for request {request_id}",
        extra={
            'request_id': request_id,
            'data_shape': len(data) if data else 0,
            'operation': 'debug_flow'
        }
    )
    
    # Procesamiento con logs intermedios
    try:
        step1_result = step1(data)
        logger.debug(f"Step 1 completed", extra={'request_id': request_id})
        
        step2_result = step2(step1_result)
        logger.debug(f"Step 2 completed", extra={'request_id': request_id})
        
        return step2_result
        
    except Exception as e:
        logger.error(
            f"Processing failed at step: {e}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise
```

### 2. Unit Testing

#### Test de servicio completo
```python
# tests/unit/test_data_processing.py
import pytest
import pandas as pd
from app.services import DataProcessor
from app.core.exceptions import ValidationError, AstronomicalDataError

class TestDataProcessor:
    
    @pytest.fixture
    def processor(self):
        return DataProcessor()
    
    @pytest.fixture
    def valid_data(self):
        return {
            'period': 365.25,
            'radius': 1.0,
            'temp': 288.0,
            'starRadius': 1.0,
            'starMass': 1.0,
            'starTemp': 5778.0,
            'depth': 84.0,
            'duration': 6.5,
            'snr': 15.0
        }
    
    def test_valid_data_passes(self, processor, valid_data):
        """Test que datos válidos pasan la validación"""
        result = processor.validate_single_input(valid_data)
        assert result is not None
        assert result.period == 365.25
    
    def test_invalid_period_raises_error(self, processor, valid_data):
        """Test que período inválido genera error"""
        valid_data['period'] = -10  # Inválido
        
        with pytest.raises(ValidationError) as exc_info:
            processor.validate_single_input(valid_data)
        
        assert 'period' in str(exc_info.value)
    
    def test_feature_engineering(self, processor, valid_data):
        """Test de feature engineering"""
        enhanced = processor.apply_feature_engineering(valid_data)
        
        # Verificar que se calcularon características derivadas
        assert hasattr(enhanced, 'planet_star_radius_ratio')
        assert enhanced.planet_star_radius_ratio == 1.0  # 1.0 / 1.0
        
        assert hasattr(enhanced, 'planet_type')
        assert enhanced.planet_type == "Earth-size"
    
    def test_batch_processing(self, processor):
        """Test de procesamiento por lotes"""
        df = pd.DataFrame([
            {'period': 365, 'radius': 1.0, 'temp': 288, 'starRadius': 1.0, 
             'starMass': 1.0, 'starTemp': 5778, 'depth': 84, 'duration': 6.5, 'snr': 15},
            {'period': 88, 'radius': 0.9, 'temp': 700, 'starRadius': 1.1, 
             'starMass': 1.05, 'starTemp': 5950, 'depth': 79, 'duration': 3.2, 'snr': 22}
        ])
        
        result = processor.validate_batch_data(df)
        assert len(result) == 2
        assert list(result.columns) == list(df.columns)
```

#### Test de utilidades
```python
# tests/unit/test_utils.py
from app.utils.helpers import (
    safe_float_conversion, validate_astronomical_range, 
    calculate_statistics, Timer
)

def test_safe_float_conversion():
    assert safe_float_conversion("123.45") == 123.45
    assert safe_float_conversion("invalid", 0.0) == 0.0
    assert safe_float_conversion(None, 1.0) == 1.0

def test_astronomical_range_validation():
    # Caso válido
    valid, error = validate_astronomical_range(5778.0, "stellar_temp", 2000.0, 50000.0)
    assert valid is True
    assert error is None
    
    # Caso inválido
    valid, error = validate_astronomical_range(100.0, "stellar_temp", 2000.0, 50000.0)
    assert valid is False
    assert "below minimum" in error

def test_timer_context_manager():
    with Timer() as timer:
        import time
        time.sleep(0.01)  # 10ms
    
    assert timer.elapsed_ms >= 10
    assert timer.elapsed_seconds >= 0.01
```

### 3. Integration Testing

```python
# tests/integration/test_full_pipeline.py
import pytest
from app.services import DataProcessor, FileProcessor

class TestIntegrationPipeline:
    
    def test_csv_to_prediction_pipeline(self):
        """Test del pipeline completo: CSV -> Validación -> Feature Engineering"""
        
        # Crear CSV de prueba
        csv_content = """period,radius,temp,starRadius,starMass,starTemp,depth,duration,snr
365.25,1.0,288.0,1.0,1.0,5778.0,84.0,6.5,15.0
88.0,0.95,700.0,1.1,1.05,5950.0,79.0,3.2,22.0"""
        
        # Procesar archivo
        file_processor = FileProcessor()
        result = file_processor.process_csv_file(csv_content.encode(), "test.csv")
        
        assert result['success'] is True
        assert result['processed_rows'] == 2
        
        # Aplicar feature engineering
        data_processor = DataProcessor()
        df_enhanced = data_processor.apply_feature_engineering(result['data'])
        
        # Verificar características derivadas
        assert 'planet_star_radius_ratio' in df_enhanced.columns
        assert 'planet_type' in df_enhanced.columns
        assert len(df_enhanced) == 2
```

## Performance y Optimización

### 1. Profiling de Performance

```python
# Usando el Timer interno
from app.utils import Timer

def profile_operation():
    with Timer() as timer:
        # Operación a medir
        result = expensive_calculation()
    
    print(f"Operation took {timer.elapsed_ms:.2f}ms")
    return result

# Profiling más detallado con cProfile
import cProfile
import pstats

def profile_detailed():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Código a perfilar
    perform_complex_operation()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 funciones más lentas
```

### 2. Optimización de Memoria

```python
# Procesamiento en chunks para datasets grandes
def process_large_dataset(df, chunk_size=1000):
    results = []
    
    for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
        processed_chunk = data_processor.validate_batch_data(chunk)
        results.append(processed_chunk)
    
    return pd.concat(results, ignore_index=True)

# Monitoring de memoria
from app.utils import get_memory_usage

def monitor_memory_intensive_operation():
    initial_memory = get_memory_usage()
    
    # Operación que usa mucha memoria
    result = memory_intensive_calculation()
    
    final_memory = get_memory_usage()
    memory_diff = final_memory['rss_mb'] - initial_memory['rss_mb']
    
    logger.info(f"Memory usage increased by {memory_diff:.2f} MB")
    return result
```

## Deployment y Producción

### 1. Configuración de Producción

```bash
# .env para producción
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Paths absolutos
MODELS_DIR=/app/models
DATA_DIR=/app/data
LOGS_DIR=/app/logs

# Seguridad
CORS_ORIGINS=https://yourdomain.com
API_RATE_LIMIT=1000/hour
```

### 2. Health Checks

```python
# app/utils/health.py
def check_system_health():
    """Verificar salud del sistema para deployment"""
    health_status = {
        'status': 'healthy',
        'timestamp': get_timestamp(),
        'checks': {}
    }
    
    # Check memoria
    memory = get_memory_usage()
    health_status['checks']['memory'] = {
        'status': 'ok' if memory['rss_mb'] < 1000 else 'warning',
        'usage_mb': memory['rss_mb']
    }
    
    # Check configuración
    config = get_config()
    is_valid, errors = config.validate()
    health_status['checks']['config'] = {
        'status': 'ok' if is_valid else 'error',
        'errors': errors
    }
    
    # Check modelos (si existen)
    models_dir = Path(config.get('paths.models'))
    health_status['checks']['models'] = {
        'status': 'ok' if models_dir.exists() else 'warning',
        'path': str(models_dir)
    }
    
    return health_status
```

## Resolución de Problemas Comunes

### 1. Problemas de Importación

```python
# Error: ModuleNotFoundError
# Solución: Verificar PYTHONPATH
import sys
sys.path.append('/path/to/project/src')

# O usar imports relativos correctos
from ..core.exceptions import ValidationError  # Correcto
from app.core.exceptions import ValidationError  # También correcto
```

### 2. Problemas de Configuración

```python
# Error: ConfigurationError
# Debug de configuración
from app.utils import get_config

config = get_config()
is_valid, errors = config.validate()

if not is_valid:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")

# Ver configuración actual
print("Current config:")
print(json.dumps(config.to_dict(), indent=2))
```

### 3. Problemas de Datos

```python
# Error: ValidationError en datos astronómicos
# Debug paso a paso
try:
    processor = DataProcessor()
    result = processor.validate_single_input(data)
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Error details: {e.details}")
    
    # Revisar rangos válidos
    print("Valid ranges:")
    for param, (min_val, max_val) in processor.VALIDATION_RANGES.items():
        if param in data:
            value = data[param]
            status = "✓" if min_val <= value <= max_val else "✗"
            print(f"  {param}: {value} [{min_val}, {max_val}] {status}")
```

## Contribución al Código

### 1. Estándares de Código

```bash
# Formateo automático
black app/ tests/
isort app/ tests/

# Linting
flake8 app/ tests/
mypy app/

# Tests antes de commit
pytest tests/ --cov=app
```

### 2. Commits Siguiendo Conventional Commits

```bash
# Ejemplos de commits válidos
git commit -m "feat(data_processor): implementar validación de estabilidad orbital"
git commit -m "fix(file_processor): corregir mapeo de columnas TESS"
git commit -m "docs(architecture): actualizar documentación de servicios"
git commit -m "test(utils): agregar tests para helpers astronómicos"
git commit -m "refactor(core): mejorar manejo de excepciones"
```

### 3. Pull Request Checklist

- [ ] Código formateado con Black
- [ ] Tests unitarios agregados/actualizados
- [ ] Documentación actualizada
- [ ] Logs apropiados agregados
- [ ] Manejo de errores implementado
- [ ] Performance considerado
- [ ] Backward compatibility verificada