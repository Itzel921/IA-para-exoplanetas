# 🌟 Sistema de Detección de Exoplanetas - Machine Learning

**NASA Space Apps Challenge 2025**

Sistema de Machine Learning basado en ensemble learning para detectar exoplanetas usando datos de las misiones Kepler, TESS y K2.

## 🎯 Objetivos

- **Accuracy objetivo**: 83.08% (basado en research: Electronics 2024)
- **Algoritmo principal**: Stacking Ensemble
- **Datasets**: KOI (Kepler), TOI (TESS), K2 Planets and Candidates
- **Output**: Predicciones automáticas en archivos CSV

## 📊 Datasets Soportados

### 1. KOI (Kepler Objects of Interest)
- **Archivo**: `cumulative_2025.10.04_11.46.06.csv`
- **Objetos**: ~9,600
- **Columnas**: 49 características astronómicas
- **Target**: `koi_disposition` → ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']

### 2. TOI (TESS Objects of Interest)  
- **Archivo**: `TOI_2025.10.04_11.44.53.csv`
- **Objetos**: ~6,000
- **Target**: `tfopwg_disp` → ['KP', 'PC', 'FP', 'APC']

### 3. K2 Planets and Candidates
- **Archivo**: `k2pandc_2025.10.04_11.46.18.csv`
- **Target**: `archive_disp` → ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']

## 🧠 Algoritmos de Machine Learning

### Ensemble Learning (Stacking)
```python
Base Learners (Nivel 1):
├── Random Forest (1600 estimators)
├── Gradient Boosting  
├── AdaBoost
└── LightGBM

Meta-Learner (Nivel 2):
└── Logistic Regression
```

### Resultados Esperados
- **Stacking**: 83.08% accuracy
- **Random Forest**: 82.64% accuracy  
- **AdaBoost**: 82.52% accuracy

## 🚀 Uso del Sistema

### 1. Instalación de Dependencias

```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm joblib
```

### 2. Estructura de Carpetas

```
IA-para-exoplanetas/
├── data/
│   ├── datasets/           # Datasets originales (KOI, TOI, K2)
│   └── new_datasets/       # Nuevos CSV para predicción
├── ML DEV/
│   ├── Clasification.py    # Análisis de datos
│   ├── train_ensemble.py   # Entrenamiento
│   ├── predict_exoplanets.py # Predicciones
│   └── Process.py          # Menú principal
├── models/                 # Modelos entrenados (.pkl)
└── exoPlanet_results/      # Resultados CSV
```

### 3. Ejecutar el Sistema

```bash
cd "ML DEV"
python Process.py
```

### 4. Menú Interactivo

```
1. 📊 Cargar y analizar datasets (KOI, TOI, K2)
2. 🎯 Entrenar modelo ensemble  
3. 🔮 Predecir exoplanetas en nuevo dataset
4. 📁 Procesar todos los archivos en new_datasets
5. 📈 Análisis exploratorio completo
6. ❓ Ayuda y documentación
7. 🚪 Salir
```

## 📋 Flujo de Trabajo

### Paso 1: Análisis de Datos
```bash
python Clasification.py
```
- Carga los 3 datasets de NASA
- Muestra TODAS las columnas disponibles
- Análisis exploratorio detallado
- Identificación de características clave

### Paso 2: Entrenamiento del Modelo
```bash
python train_ensemble.py
```
- Feature engineering astronómico
- Preprocesamiento unificado
- Entrenamiento de Stacking Ensemble
- Validación cruzada estratificada
- Guardado del modelo en `models/`

### Paso 3: Predicción en Nuevos Datos
```bash
# Dataset específico
python predict_exoplanets.py nuevo_dataset.csv

# Todos los archivos en new_datasets/
python predict_exoplanets.py
```

## 📊 Feature Engineering Astronómico

### Características Derivadas
```python
# Ratio radio planeta/estrella
planet_star_radius_ratio = koi_prad / koi_srad

# Ratio temperatura de equilibrio
equilibrium_temp_ratio = koi_teq / koi_steff

# Profundidad de tránsito esperada (ppm)
transit_depth_expected = (koi_prad / koi_srad) ** 2 * 1e6

# Distancia a zona habitable
habitable_zone_distance = abs(koi_teq - 288) / 288
```

### Preprocesamiento
- **Imputación**: Mediana para parámetros estelares
- **Outliers**: Winsorizing (1%-99% percentiles)
- **Escalado**: RobustScaler (robusto para datos astronómicos)
- **Encoding**: LabelEncoder para variables categóricas

## 📈 Formato de Resultados

### Archivo CSV de Output
```csv
# Columnas originales del dataset +
ML_Probability,      # Probabilidad de ser exoplaneta (0.0-1.0)
ML_Prediction,       # Predicción binaria (1=exoplaneta, 0=no)
ML_Confidence,       # Confianza de la predicción (0.0-1.0)
ML_Classification    # 'CONFIRMED' o 'NOT_CONFIRMED'
```

### Archivo JSON de Resumen
```json
{
  "input_file": "nuevo_dataset.csv",
  "total_objects": 1500,
  "confirmed_exoplanets": 380,
  "confirmed_percentage": 25.33,
  "average_confidence": 0.847,
  "model_accuracy": 0.8308,
  "high_confidence_predictions": 920
}
```

## 🔬 Base Científica

### Papers de Referencia
- **Electronics 2024**: "Ensemble Learning for Exoplanet Detection"
- **MNRAS 2022**: "Machine Learning Applications in Transit Photometry"

### Métricas Astronómicas
- **Completeness** = Recall (fracción de planetas reales detectados)
- **Reliability** = Precision (fracción de detecciones que son planetas reales)
- **AUC-ROC > 0.9** = Excelente discriminación astronómica

### Validación
- **Cross-Validation**: StratifiedKFold (n_splits=5)
- **Métricas**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Imbalance**: Manejo de clases desbalanceadas (~25% confirmados)

## 🛠️ Comandos Rápidos

```bash
# Análisis completo
python Clasification.py

# Entrenamiento
python train_ensemble.py

# Predicción individual
python predict_exoplanets.py mi_dataset.csv

# Menú interactivo
python Process.py
```

## 📁 Agregar Nuevos Datasets

1. **Colocar archivo CSV** en `data/new_datasets/`
2. **Ejecutar predicción**:
   - Opción 3: Dataset específico
   - Opción 4: Todos los archivos
3. **Revisar resultados** en `exoPlanet_results/`

## 🎯 Objetivos de Performance

- ✅ **Accuracy**: ≥83.08%
- ✅ **Procesamiento**: Datasets de 10K+ objetos
- ✅ **Tiempo**: <5 minutos entrenamiento, <30s predicción
- ✅ **Robustez**: Manejo de valores faltantes y outliers
- ✅ **Escalabilidad**: Nuevos datasets automáticamente

## 🚀 Próximas Mejoras

- [ ] Integración con APIs de NASA TAP
- [ ] Visualizaciones interactivas con Plotly
- [ ] API REST con FastAPI
- [ ] Interfaz web con React
- [ ] Containerización con Docker

---

**🌟 NASA Space Apps Challenge 2025**  
*Sistema desarrollado para automatizar la detección de exoplanetas usando ensemble learning*