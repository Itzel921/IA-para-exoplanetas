# Backend Exoplanetas - MongoDB Integration

## NASA Space Apps Challenge 2025

Backend simplificado para procesamiento de datos de exoplanetas con integración MongoDB.

---

## 🚀 Configuración Rápida

### 1. Configurar Entorno Virtual

```bash
# Windows
setup_mongodb_environment.bat

# O manualmente:
python -m venv venv_mongodb
venv_mongodb\Scripts\activate.bat
pip install -r requirements.txt
```

### 2. Ejecutar Backend

```bash
# Activar entorno
venv_mongodb\Scripts\activate.bat

# Ejecutar backend
python src\backend\main.py
```

---

## 🗄️ Configuración MongoDB

### Conexión a Base de Datos

```
Host: toiletcrafters.us.to:8081
Database: ExoData
Collection: datossatelite
Usuario: manu
Password: tele123
```

### Cadena de Conexión

```
mongodb://manu:tele123@toiletcrafters.us.to:8081/ExoData
```

---

## 📁 Estructura del Backend

```
src/backend/
├── main.py                    # Punto de entrada principal
├── app/
│   ├── __init__.py           # Inicialización del módulo
│   └── database/             # Módulo de base de datos
│       ├── __init__.py       # Exports del módulo database
│       ├── config.py         # Configuración MongoDB
│       ├── models.py         # Modelos de datos
│       └── services.py       # Servicios CRUD
```

---

## 🧩 Componentes Principales

### ExoplanetBackend (main.py)

Clase principal que coordina todas las operaciones:

```python
from src.backend.main import ExoplanetBackend

backend = ExoplanetBackend()
await backend.initialize()

# Procesar datos individuales
result = await backend.process_exoplanet_data(exoplanet_data)

# Procesamiento masivo desde archivo
await backend.bulk_process_from_file("data/datasets/cumulative_2025.10.04_11.46.06.csv")
```

### Modelos de Datos

#### ExoplanetData
```python
@dataclass
class ExoplanetData:
    object_name: str
    period: float                    # Período orbital en días
    radius: float                    # Radio planetario en radios terrestres
    temperature: float               # Temperatura de equilibrio en Kelvin
    star_radius: float              # Radio estelar en radios solares
    star_mass: float                # Masa estelar en masas solares
    star_temperature: float         # Temperatura estelar en Kelvin
    transit_depth: float            # Profundidad de tránsito en ppm
    transit_duration: float         # Duración de tránsito en horas
    signal_noise_ratio: float       # Relación señal-ruido
    mission_source: str             # Misión origen (Kepler, TESS, K2)
```

#### SatelliteData
```python
@dataclass
class SatelliteData:
    satellite_id: str
    name: str
    mission: str                    # Kepler, TESS, K2, etc.
    launch_date: str
    status: str                     # Active, Inactive, etc.
    observations_count: int = 0
```

### Servicios MongoDB

#### ExoplanetService
- `insert_exoplanet()`: Insertar exoplaneta individual
- `bulk_insert_exoplanets()`: Inserción masiva
- `find_exoplanet_by_name()`: Buscar por nombre
- `find_exoplanets_by_mission()`: Buscar por misión
- `count_exoplanets()`: Contar total de registros

#### DataManagerService
- `initialize_all_services()`: Inicializar todos los servicios
- `get_database_status()`: Estado de la base de datos

---

## 🔧 Uso del Backend

### Ejemplo Básico

```python
import asyncio
from src.backend.main import ExoplanetBackend
from src.backend.app.database.models import ExoplanetData

async def main():
    # Inicializar backend
    backend = ExoplanetBackend()
    await backend.initialize()
    
    # Crear datos de exoplaneta
    exoplanet = ExoplanetData(
        object_name="Kepler-452b",
        period=384.84,
        radius=1.63,
        temperature=265.0,
        star_radius=1.11,
        star_mass=1.04,
        star_temperature=5757.0,
        transit_depth=50.0,
        transit_duration=12.5,
        signal_noise_ratio=15.2,
        mission_source="Kepler"
    )
    
    # Procesar y guardar en MongoDB
    result = await backend.process_exoplanet_data(exoplanet)
    print(f"Procesado: {result}")
    
    # Obtener estado de la base de datos
    status = await backend.get_database_status()
    print(f"Estado BD: {status}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Procesamiento Masivo

```python
# Procesar archivo CSV completo
await backend.bulk_process_from_file("data/datasets/cumulative_2025.10.04_11.46.06.csv")

# Cargar datos de satélites
await backend.load_satellite_data("data/datasets/k2pandc_2025.10.04_11.46.18.csv")
```

---

## 📊 Datasets Soportados

### Archivos CSV en `data/datasets/`

1. **cumulative_2025.10.04_11.46.06.csv** - Datos de Kepler
2. **k2pandc_2025.10.04_11.46.18.csv** - Datos de K2
3. **TOI_2025.10.04_11.44.53.csv** - Datos de TESS

### Campos Mapeados

Los datasets tienen diferentes nombres de columnas que se mapean automáticamente:

- **KOI (Kepler)**: `koi_disposition` → Clasificación
- **TOI (TESS)**: `tfopwg_disp` → Clasificación  
- **K2**: `archive_disp` → Clasificación

---

## 🔍 Logging y Monitoreo

### Configuración de Logs

```python
import logging

# El backend configura logging automáticamente
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Logs Principales

- Conexiones a MongoDB
- Operaciones CRUD exitosas
- Errores de procesamiento
- Estadísticas de inserción masiva

---

## ⚡ Características Clave

### ✅ Implementado

- ✅ Conexión async a MongoDB con Motor
- ✅ Modelos de datos con dataclasses
- ✅ Servicios CRUD completos
- ✅ Procesamiento individual y masivo
- ✅ Logging básico
- ✅ Validación de datos simplificada
- ✅ Manejo de errores MongoDB

### 🚫 Removido Intencionalmente

- ❌ APIs REST (delegado al módulo ML)
- ❌ Validaciones astronómicas complejas
- ❌ Rotación de logs avanzada
- ❌ FastAPI y endpoints HTTP
- ❌ Autenticación y autorización

---

## 🚀 Próximos Pasos

1. **Ejecutar setup**: `setup_mongodb_environment.bat`
2. **Probar conexión**: Ejecutar `python src\backend\main.py`
3. **Cargar datos**: Usar `bulk_process_from_file()` con CSV
4. **Integrar con ML**: Conectar con módulo de Machine Learning

---

## 🤝 Contribución

Este backend está diseñado para ser:
- **Simple**: Sin validaciones complejas
- **Modular**: Fácil de extender
- **Async**: Compatible con cargas masivas
- **MongoDB-first**: Optimizado para NoSQL

---

## 📞 Soporte

Para problemas de conexión MongoDB:
1. Verificar credenciales en `app/database/config.py`
2. Comprobar conectividad a `toiletcrafters.us.to:8081`
3. Revisar logs para errores específicos