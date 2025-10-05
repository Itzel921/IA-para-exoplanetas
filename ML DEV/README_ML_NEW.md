# ML DEV - Módulo de Machine Learning para Detección de Exoplanetas

**NAS### Estructura de Directorios del Proyecto
```
IA-para-exoplanetas/
├── data/
│   ├── datasets/               # Datasets NASA originales
│   │   ├── cumulative_2025.10.04_11.46.06.csv  # KOI (Kepler)
│   │   ├── TOI_2025.10.04_11.44.53.csv         # TOI (TESS)
│   │   └── k2pandc_20## 5. Para Desarrolladores (Arquitectura)

### Estructura Real del Módulo ML DEV

```
ML DEV/
├── Process.py                      # ⭐ PUNTO DE ENTRADA - Menú interactivo
├── Clasification.py               # 📊 DataLoader - Carga datasets NASA
├── train_ensemble.py              # 🧠 ExoplanetMLSystem + StackingEnsemble
├── train_ensemble_FAST.py         # ⚡ FastStackingEnsemble optimizado
├── train_ensemble_CORRECTED.py    # 🔧 FeatureMapper + ImprovedDataPreprocessor
├── simple_predictor_fixed.py      # 🔮 SimplePredictorFixed producción
├── simple_retrain.py             # 🔄 Reentrenamiento rápido
├── advanced_visualization.py      # 📈 ExoplanetVisualizer
├── model_imports.py               # 📦 Imports para joblib
├── README_ML.md                   # 📄 Documentación original
├── README_ML_NEW.md               # 📄 Esta documentación
├── __pycache__/                   # Python bytecode cache
└── trained_models/                # Modelos locales del módulo
    ├── exoplanet_ensemble_20251005_002211.pkl
    ├── exoplanet_ensemble_FAST_20251004_195930.pkl
    └── exoplanet_simple_20251005_011244.pkl
```

### Flujo de Datos Real (Basado en Process.py)

```
1. Process.main() → show_menu()
    ↓
2. Usuario selecciona opción
    ↓
3a. option_1_load_datasets()
    → DataLoader.load_all_datasets()
    → Análisis automático de KOI, TOI, K2
    
3b. option_2_train_model()
    → ExoplanetMLSystem.train_system()
    → FeatureEngineer + DataPreprocessor
    → StackingEnsemble (RF + GB + LGB + LogReg)
    → Modelo guardado en trained_models/
    
3c. option_4_predict_single() / option_5_predict_all()
    → SimplePredictorFixed.load_model()
    → SimplePredictorFixed.process_file()
    → Resultados en exoPlanet_results/
    
3d. option_6_full_analysis()
    → ExoplanetVisualizer.generate_complete_analysis()
    → Gráficas en exoPlanet_results/charts/
```

### Componentes Técnicos Clave

#### DataLoader (Clasification.py)
- **Mapeo de etiquetas unificado**: Convierte KOI, TOI, K2 a formato binario consistente
- **Manejo de metadatos NASA**: Procesa archivos CSV con comentarios '#'
- **Validación automática**: Verifica integridad de datasets al cargar

#### StackingEnsemble (train_ensemble.py)
- **Base learners**: RandomForest (100 est.) + GradientBoosting (50 est.) + LightGBM (50 est.)
- **Meta-learner**: LogisticRegression para combinación final
- **Cross-validation**: StratifiedKFold (5 splits) para generación de meta-features
- **Target**: Clasificación binaria (1=planeta confirmado, 0=candidato/falso positivo)

#### SimplePredictorFixed (simple_predictor_fixed.py)
- **Features limitadas**: Solo usa ['ra', 'dec'] para compatibilidad
- **Carga automática**: Selecciona modelo más reciente por timestamp
- **Procesamiento batch**: Maneja múltiples archivos CSV automáticamente
- **Output format**: CSV original + columnas ML_Probability, ML_Prediction, ML_Classification

#### ExoplanetVisualizer (advanced_visualization.py)
- **4 tipos de gráficas principales**: overview, class distributions, correlations, scatter plots
- **Comparaciones automáticas**: NASA datasets vs nuevos datasets
- **Timestamps automáticos**: Todas las visualizaciones incluyen fecha/hora de generación
- **Exportación PNG**: Guarda automáticamente en exoPlanet_results/charts/

### Consideraciones Técnicas

#### Performance
- **Modelos ensemble**: ~50-200MB en disco, carga <5 segundos
- **Datasets NASA**: ~500MB combinados, requiere 2GB+ RAM para entrenamiento
- **Predicciones**: ~1000 objetos/segundo en hardware estándar

#### Compatibilidad
- **Serialización**: joblib para modelos, garantiza compatibilidad entre versiones
- **Features**: SimplePredictorFixed maneja diferencias de características automáticamente
- **Paths**: pathlib para compatibilidad Windows/Linux

#### Escalabilidad
- **Modular**: Cada componente funciona independientemente
- **Extensible**: Fácil agregar nuevos algoritmos de ensemble
- **Batch processing**: Procesamiento paralelo de múltiples archivosv     # K2
│   └── new_datasets/          # Nuevos CSV para predicción
│       └── cumulative_2025.10.04_18.21.10Prueba1.csv
├── ML DEV/                    # 🎯 MÓDULO PRINCIPAL
│   ├── Process.py             # ⭐ PUNTO DE ENTRADA PRINCIPAL
│   ├── Clasification.py      # Carga y análisis de datos
│   ├── train_ensemble*.py     # Entrenamiento de modelos
│   ├── simple_predictor_fixed.py  # Predictor de producción
│   ├── advanced_visualization.py  # Visualizaciones
│   └── trained_models/        # Modelos locales del módulo
├── models/                    # Modelos globales del proyecto
└── exoPlanet_results/         # Resultados y gráficas
    ├── charts/                # Gráficos individuales
    └── comparative_charts/    # Comparaciones
```allenge 2025 - Sistema de Ensemble Learning para Clasificación de Objetos Astronómicos**

## 1. Visión General

El módulo **ML DEV** es un sistema completo de Machine Learning especializado en la **detección y clasificación de exoplanetas** mediante el análisis de datos de las misiones espaciales Kepler, TESS y K2. El punto de entrada principal es **Process.py**, que proporciona un menú interactivo para todas las funcionalidades del sistema.

### Propósito Principal
- Cargar y analizar datasets de NASA (KOI, TOI, K2) con objetos astronómicos
- Entrenar modelos de ensemble learning para clasificación binaria (planeta/no-planeta)
- Generar predicciones automatizadas para nuevos datasets con métricas de confianza
- Crear visualizaciones científicas para análisis exploratorio

### Caso de Uso Principal
```python
# Punto de entrada principal del módulo
python Process.py  # Ejecuta menú interactivo completo

# O uso directo de componentes específicos
from Clasification import DataLoader
from train_ensemble import ExoplanetMLSystem
from simple_predictor_fixed import SimplePredictorFixed
```

### Dependencias Críticas
```python
# Machine Learning y Preprocesamiento
pandas>=2.0.0           # Manipulación de datasets NASA
numpy>=1.24.0           # Operaciones numéricas
scikit-learn>=1.3.0     # Algoritmos de ensemble
lightgbm>=3.3.5         # Gradient boosting
joblib>=1.3.0           # Serialización de modelos

# Visualización
matplotlib>=3.7.0       # Gráficos base
seaborn>=0.12.0         # Visualizaciones estadísticas

# Utilidades Python estándar
pathlib, warnings, datetime
```

## 2. Instalación y Configuración

### Pasos de Instalación

```bash
# 1. Navegar al módulo
cd "IA-para-exoplanetas/ML DEV"

# 2. Instalar dependencias
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm joblib

# 3. Ejecutar sistema principal
python Process.py
```

### Configuración Automática

El módulo configura automáticamente las rutas necesarias:

```python
# DataLoader configura automáticamente:
project_root = Path(__file__).parent.parent
datasets_path = project_root / "data" / "datasets"      # Datasets NASA
new_datasets_path = project_root / "data" / "new_datasets"  # Nuevos CSV
models_path = project_root / "models"                   # Modelos entrenados
results_path = project_root / "exoPlanet_results"       # Resultados
```

### Estructura de Directorios Requerida
```
IA-para-exoplanetas/
├── data/
│   ├── datasets/               # ⚠️ Colocar aquí CSVs de NASA
│   │   ├── cumulative_*.csv    # Dataset KOI (Kepler Objects of Interest)
│   │   ├── TOI_*.csv          # Dataset TOI (TESS Objects of Interest)
│   │   └── k2pandc_*.csv      # Dataset K2 Planets and Candidates
│   └── new_datasets/          # 📁 Nuevos archivos para predicción
├── ML DEV/                    # 🧠 Código fuente del módulo
│   ├── models/                # 🤖 Modelos entrenados guardados
└── exoPlanet_results/         # 📊 Resultados y visualizaciones
    ├── charts/                # Gráficos individuales
    └── comparative_charts/    # Comparaciones entre datasets
```

## 3. Referencia de Uso

### Funciones Principales de Process.py (Punto de Entrada)

#### 3.1 `main()` - Función Principal del Sistema

```python
def main():
    """Función principal del sistema con menú interactivo"""
```

**Descripción**: Ejecuta el menú interactivo principal que conecta todas las funcionalidades del módulo.

**Menú de Opciones**:
1. `option_1_load_datasets()` - Cargar y analizar datasets NASA
2. `option_2_train_model()` - Entrenar modelo ensemble completo
3. `option_3_simple_train()` - Entrenamiento simplificado (rápido)
4. `option_4_predict_single()` - Predecir dataset específico
5. `option_5_predict_all()` - Procesar todos los archivos en new_datasets
6. `option_6_full_analysis()` - Análisis exploratorio completo con visualizaciones
7. `option_7_help()` - Ayuda y documentación

### Clases Principales

#### 3.2 `DataLoader` (Clasification.py) - Carga de Datasets NASA

```python
class DataLoader:
    def __init__(self, project_root)
    def load_dataset(self, filename, dataset_type='KOI')
    def load_all_datasets()
    def analyze_dataset(self, df, dataset_name)
```

**Descripción**: Carga y unifica datasets de NASA (KOI, TOI, K2) con mapeo consistente de etiquetas.

**Atributos**:
- `label_mapping`: Dict para unificar etiquetas entre datasets
- `datasets_path`: Ruta a data/datasets/
- `new_datasets_path`: Ruta a data/new_datasets/

**Métodos**:
- `load_dataset()`: Retorna (DataFrame, tipo_dataset)
- `load_all_datasets()`: Retorna dict {nombre: DataFrame}
- `analyze_dataset()`: Retorna dict con estadísticas

#### 3.3 `ExoplanetMLSystem` (train_ensemble.py) - Sistema de Ensemble

```python
class ExoplanetMLSystem:
    def __init__(self, project_root)
    def train_system(self, datasets)
```

**Descripción**: Sistema completo de ML que combina preprocesamiento, feature engineering y stacking ensemble.

**Componentes Internos**:
- `DataPreprocessor`: Imputación, escalado, encoding
- `FeatureEngineer`: Características derivadas astronómicas
- `StackingEnsemble`: RandomForest + GradientBoosting + LightGBM + LogisticRegression

**Métodos**:
- `train_system()`: Retorna dict con accuracy y ruta del modelo guardado

#### 3.4 `SimplePredictorFixed` (simple_predictor_fixed.py) - Predictor de Producción

```python
class SimplePredictorFixed:
    def __init__(self, project_root)
    def load_model()
    def load_dataset(self, filename)
    def prepare_features(self, df)
    def predict(self, X, original_df)
    def save_results(self, results, original_filename)
    def process_file(self, filename)
    def process_all_new_datasets()
```

**Descripción**: Sistema optimizado para predicciones en nuevos datasets con compatibilidad garantizada.

**Características**:
- Carga automática del modelo más reciente
- Procesamiento por lotes de archivos CSV
- Generación automática de resultados con timestamp

#### 3.5 `ExoplanetVisualizer` (advanced_visualization.py) - Visualizaciones

```python
class ExoplanetVisualizer:
    def __init__(self, project_root)
    def create_dataset_overview(self, datasets)
    def create_class_distributions(self, datasets)
    def create_correlation_matrices(self, datasets)
    def create_scatter_plots(self, datasets)
    def generate_complete_analysis(self, nasa_datasets, new_datasets=None)
```

**Descripción**: Genera visualizaciones científicas automáticas para análisis exploratorio.

**Visualizaciones Generadas**:
- Dataset overviews, distribuciones de clases, matrices de correlación
- Scatter plots de características astronómicas clave
- Comparaciones entre datasets NASA y nuevos datos

### Ejemplo de Código Funcional Mínimo

```python
"""
Flujo completo usando las funciones reales del módulo - Ejemplo funcional
"""
from pathlib import Path

# Método 1: Usar el menú interactivo (recomendado)
def usar_menu_interactivo():
    """Forma más simple de usar el módulo"""
    from Process import main
    main()  # Ejecuta menú con todas las opciones

# Método 2: Uso directo de componentes
def flujo_programatico():
    """Uso directo de las clases principales"""
    
    # 1. Cargar datasets NASA
    from Clasification import DataLoader
    project_root = Path(__file__).parent.parent
    loader = DataLoader(project_root)
    datasets = loader.load_all_datasets()
    
    if datasets:
        print(f"✅ {len(datasets)} datasets cargados")
        
        # 2. Entrenar modelo (usando train_ensemble.py)
        from train_ensemble import ExoplanetMLSystem
        ml_system = ExoplanetMLSystem(project_root)
        model_info = ml_system.train_system(datasets)
        print(f"Accuracy: {model_info['accuracy']:.4f}")
        
        # 3. Hacer predicciones
        from simple_predictor_fixed import SimplePredictorFixed
        predictor = SimplePredictorFixed(project_root)
        if predictor.load_model():
            results = predictor.process_all_new_datasets()
            print(f"Procesados: {len(results)} archivos")
    
    return datasets

# Método 3: Solo visualizaciones
def generar_visualizaciones():
    """Solo generar gráficas de análisis"""
    from Clasification import DataLoader
    from advanced_visualization import ExoplanetVisualizer
    
    project_root = Path(__file__).parent.parent
    loader = DataLoader(project_root)
    datasets = loader.load_all_datasets()
    
    if datasets:
        visualizer = ExoplanetVisualizer(project_root)
        generated_files = visualizer.generate_complete_analysis(datasets)
        print(f"Gráficas generadas: {len(generated_files)}")

# Ejecutar ejemplo
if __name__ == "__main__":
    # Opción recomendada: menú interactivo
    usar_menu_interactivo()
```

## 4. Prueba de Generación de Salida (Gráficas)

### Verificación Visual del Sistema

El módulo genera automáticamente visualizaciones de análisis exploratorio usando `ExoplanetVisualizer` y guarda resultados en `exoPlanet_results/`.

#### Minimal Test - Generación de Gráficas

```python
"""
Test funcional para generar gráficas usando ExoplanetVisualizer
"""
def test_generar_graficas():
    """Ejecuta option_6_full_analysis() de Process.py"""
    
    # Método 1: Usar función del menú (recomendado)
    from Process import option_6_full_analysis
    option_6_full_analysis()  # Genera todas las visualizaciones
    
    # Método 2: Uso directo de ExoplanetVisualizer
    from pathlib import Path
    from Clasification import DataLoader
    from advanced_visualization import ExoplanetVisualizer
    
    project_root = Path(__file__).parent.parent
    loader = DataLoader(project_root)
    datasets = loader.load_all_datasets()
    
    if datasets:
        visualizer = ExoplanetVisualizer(project_root)
        files = visualizer.generate_complete_analysis(datasets)
        print(f"✅ {len(files)} visualizaciones generadas")
        return True
    return False

# Ejecutar desde Process.py (opción 6) o usar el test directo
if __name__ == "__main__":
    test_generar_graficas()
```

#### Archivos de Salida Reales Generados

**Gráficas en `exoPlanet_results/charts/`:**
- `01_dataset_overview_20251005_012151.png` - Visión general de datasets NASA
- `02_class_distributions_20251005_012151.png` - Distribuciones de clases por dataset
- `03_correlation_matrices_20251005_012151.png` - Matrices de correlación de características
- `04_scatter_plots_20251005_012151.png` - Scatter plots de variables clave

**Gráficas Comparativas en `exoPlanet_results/comparative_charts/`:**
- `nasa_vs_new_comparison_20251005_012151.png` - Comparación NASA vs nuevos datasets
- `prediction_results_analysis_20251005_012151.png` - Análisis de resultados de predicción

**Archivos de Resultados CSV:**
- `cumulative_2025.10.04_18.21.10Prueba1_predictions_20251005_012139.csv` - Predicciones con ML_Probability, ML_Prediction, ML_Classification
- `accuracy_metrics_20251004_202348.csv` - Métricas de accuracy del modelo
- `confidence_distribution_20251004_202348.csv` - Distribución de confianza de predicciones

#### Verificación Manual de Salidas

```bash
# Verificar que las gráficas se generaron correctamente
ls "exoPlanet_results/charts/" | grep .png
ls "exoPlanet_results/comparative_charts/" | grep .png

# Verificar archivos CSV de resultados
ls "exoPlanet_results/" | grep .csv
```

## 5. Para Desarrolladores (Arquitectura)

### Estructura de Directorios del Módulo

```
ML DEV/
├── Process.py                    # 🎯 PUNTO DE ENTRADA PRINCIPAL
├── Clasification.py             # 📊 DataLoader y análisis exploratorio
├── train_ensemble.py            # 🧠 Algoritmos de ensemble learning
├── train_ensemble_FAST.py       # ⚡ Entrenamiento rápido optimizado
├── simple_predictor_fixed.py    # 🔮 Predictor de producción
├── simple_retrain.py           # 🔄 Reentrenamiento simplificado
├── advanced_visualization.py    # 📈 Visualizaciones científicas
├── model_imports.py             # 📦 Imports para deserialización
├── __pycache__/                 # 🗂️ Bytecode compilado de Python
└── trained_models/              # 🤖 Modelos persistidos localmente
    ├── exoplanet_ensemble_*.pkl     # Modelos ensemble completos
    ├── exoplanet_simple_*.pkl       # Modelos simplificados
    └── exoplanet_ensemble_FAST_*.pkl # Modelos optimizados
```

---

**🌟 NASA Space Apps Challenge 2025**  
*Documentación técnica del módulo ML DEV - Sistema de ensemble learning para detección automatizada de exoplanetas basado en el código real implementado*

**Archivos Principales:**
- **Process.py**: Punto de entrada con menú interactivo
- **Clasification.py**: DataLoader para datasets NASA
- **train_ensemble.py**: ExoplanetMLSystem con StackingEnsemble
- **simple_predictor_fixed.py**: SimplePredictorFixed para producción
- **advanced_visualization.py**: ExoplanetVisualizer para análisis visual

---

**🌟 NASA Space Apps Challenge 2025**  
*Documentación técnica completa del módulo ML DEV - Sistema de ensemble learning para detección automatizada de exoplanetas*