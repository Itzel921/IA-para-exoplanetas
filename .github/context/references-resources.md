# Referencias y Recursos - Detección de Exoplanetas con IA

## 📚 Bibliografía Científica Principal

### Artículos de Investigación Clave

#### 1. Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification
**Autores**: Thiago S. F. Luz, Rodrigo A. S. Braga, Enio R. Ribeiro  
**Publicación**: Electronics 2024, 13, 3950  
**DOI**: https://doi.org/10.3390/electronics13193950  

**Resumen**: Evaluación comprehensiva de cinco algoritmos ensemble (AdaBoost, Random Forest, Stacking, Random Subspace Method, y Extremely Randomized Trees) para clasificación de exoplanetas. El estudio demuestra que el algoritmo Stacking alcanza el mejor rendimiento (83.08% accuracy) y que la optimización de hiperparámetros mejora significativamente el rendimiento.

**Contribuciones Clave**:
- Primera evaluación exclusiva de algoritmos ensemble para exoplanetas
- Metodología de optimización de hiperparámetros con 100 iteraciones
- Análisis comparativo detallado con métricas astronómicas específicas
- Demostración de accuracy >80% en todos los algoritmos evaluados

#### 2. Exoplanet Detection Using Machine Learning
**Autores**: Abhishek Malik, Benjamin P. Moster, Christian Obermeier  
**Publicación**: MNRAS 513, 5505–5516 (2022)  
**DOI**: https://doi.org/10.1093/mnras/stab3692  

**Resumen**: Introducción de técnica de ML basada en gradient boosting para detección de exoplanetas usando análisis de series temporales. Utiliza TSFRESH para extraer 789 características de curvas de luz y LightGBM para clasificación.

**Logros Destacados**:
- AUC = 0.948 para datos de Kepler (94.8% de planetas reales rankeados correctamente)
- Recall = 0.96 (96% de planetas reales clasificados correctamente)
- Accuracy = 0.98 para datos TESS con recall = 0.82 y precision = 0.63
- Competitivo con métodos Box Least Squares tradicionales
- Más eficiente computacionalmente que modelos de deep learning

### Artículos de Referencia Adicionales

#### Machine Learning en Astronomía
1. **Fluke, C. and Jacobs, C.** (2020). "Surveying the reach and maturity of machine learning and artificial intelligence in astronomy." *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 10(2), e1349.

2. **Baron, D.** (2019). "Machine learning in astronomy: a practical overview." *arXiv preprint arXiv:1904.07248*.

#### Detección de Exoplanetas - Métodos Tradicionales
3. **Borucki, W. J. et al.** (2010). "Kepler planet-detection mission: introduction and first results." *Science*, 327(5968), 977-980.

4. **Ricker, G. R. et al.** (2014). "Transiting Exoplanet Survey Satellite (TESS)." *Journal of Astronomical Telescopes, Instruments, and Systems*, 1(1), 014003.

#### Ensemble Learning - Fundamentos Teóricos
5. **Rokach, L.** (2010). "Ensemble-based classifiers." *Artificial intelligence review*, 33(1-2), 1-39.

6. **Dietterich, T. G.** (2000). "Ensemble methods in machine learning." *International workshop on multiple classifier systems* (pp. 1-15). Springer.

## 🗄️ Datasets y Fuentes de Datos

### Datasets Principales

#### 1. Kepler Objects of Interest (KOI)
**Fuente**: NASA Exoplanet Archive  
**URL**: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi  
**Descripción**: Base de datos completa de objetos de interés detectados por la misión Kepler  
**Tamaño**: ~9,600 objetos con 50+ parámetros cada uno  
**Período**: 2009-2017  

**Columnas Clave**:
- `koi_disposition`: Clasificación final (CONFIRMED, CANDIDATE, FALSE POSITIVE)  
- `koi_period`: Período orbital en días  
- `koi_prad`: Radio planetario en radios terrestres  
- `koi_teq`: Temperatura de equilibrio en Kelvin  
- `koi_srad`, `koi_smass`, `koi_steff`: Parámetros estelares  

**Acceso**: 
```python
# Ejemplo de descarga
import pandas as pd
koi_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+koi&format=csv"
koi_data = pd.read_csv(koi_url)
```

#### 2. TESS Objects of Interest (TOI)
**Fuente**: NASA Exoplanet Archive  
**URL**: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=toi  
**Descripción**: Candidatos a exoplanetas de la misión TESS  
**Tamaño**: ~6,000+ objetos (actualizándose continuamente)  
**Período**: 2018-presente  

**Acceso Programático**:
```python
# API TAP para TESS
toi_query = """
SELECT toi, tic_id, pl_orbper, pl_rade, pl_eqt, tfopwg_disp
FROM toi 
WHERE tfopwg_disp IS NOT NULL
"""
toi_url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={toi_query}&format=csv"
```

#### 3. K2 Planets and Candidates
**Fuente**: NASA Exoplanet Archive  
**URL**: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2candidates  
**Descripción**: Datos de la misión K2 (extensión de Kepler)  
**Campañas**: C0-C19 (diferentes campos del cielo)  

### Datasets de Curvas de Luz

#### MAST (Mikulski Archive for Space Telescopes)
**URL**: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html  
**Contenido**: Curvas de luz completas de Kepler, K2, y TESS  
**Formato**: FITS files con series temporales  
**Acceso**: API programática disponible  

```python
# Ejemplo con astroquery
from astroquery.mast import Observations
obs_table = Observations.query_criteria(
    target_name="Kepler-7",
    obs_collection="Kepler"
)
```

### Datasets de Entrenamiento Preparados

#### Kaggle - Kepler Exoplanet Search Results
**URL**: https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results  
**Descripción**: Dataset limpio y preprocesado para ML  
**Ventajas**: Listo para uso educativo y prototipado  

#### Kaggle - Exoplanet Hunting in Deep Space
**URL**: https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data  
**Descripción**: Curvas de luz etiquetadas para deep learning  
**Formato**: Series temporales normalizadas  

## 🛠️ Herramientas y Librerías

### Librerías de Machine Learning

#### Python - Ecosystem Principal
```python
# Core ML libraries
scikit-learn==1.3.0      # Algoritmos ML fundamentales
lightgbm==4.0.0          # Gradient boosting eficiente  
xgboost==1.7.6           # Gradient boosting alternativo
catboost==1.2.0          # Gradient boosting para categóricas

# Deep Learning
tensorflow==2.13.0       # Framework DL principal
torch==2.0.1             # PyTorch para investigación
keras==2.13.1            # API high-level para TF

# Ensemble específicos
imbalanced-learn==0.11.0 # Handling class imbalance
mlxtend==0.22.0          # Extended ML algorithms
```

#### Feature Engineering para Series Temporales
```python
# TSFRESH - Feature extraction automática
tsfresh==0.20.0          # 789+ características automáticas
cesium==0.11.0           # Features astronómicas específicas
pyts==0.12.0             # Time series classification

# Análisis de periodicidad
astropy==5.3.1           # Herramientas astronómicas
scipy==1.11.1            # Análisis de señales
```

### Herramientas de Visualización

#### Plotting y Dashboard
```python
# Visualización estática
matplotlib==3.7.2        # Plotting fundamental
seaborn==0.12.2          # Statistical plotting
plotly==5.15.0           # Interactive plots

# Dashboard web
streamlit==1.25.0        # Rapid prototyping dashboard
dash==2.12.1             # Production-ready dashboards
bokeh==3.2.1             # Interactive visualizations
```

#### Visualizaciones Específicas para Astronomía
```python
# Astro-specific plotting
astroML==1.0.2           # Machine learning for astronomy
astroplan==0.8           # Observation planning
lightkurve==2.4.0        # Kepler/TESS light curve analysis
```

### APIs y Acceso a Datos

#### NASA APIs
```python
# Acceso programático a datos NASA
astroquery==0.4.6        # Query astronomical databases
requests==2.31.0         # HTTP client for APIs

# Ejemplo de uso
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
koi_table = NasaExoplanetArchive.query_criteria(
    table="koi", 
    where="koi_disposition like 'CONFIRMED'"
)
```

### Frameworks Web para Deployment

#### Backend APIs
```python
# API frameworks
fastapi==0.103.0         # Modern, fast API framework
uvicorn==0.23.2          # ASGI server for FastAPI
flask==2.3.2             # Traditional web framework
django==4.2.4            # Full-featured web framework

# Database and storage
sqlalchemy==2.0.19       # SQL toolkit and ORM
redis==4.6.0             # In-memory data store
postgresql              # Production database
```

#### Frontend Frameworks
```javascript
// React ecosystem
react: "^18.2.0"         // Core React library
typescript: "^5.0.0"     // Type safety
@mui/material: "^5.14.0" // Material-UI components
chart.js: "^4.3.0"       // Charting library
axios: "^1.4.0"          // HTTP client

// Alternative frameworks
vue: "^3.3.0"           // Vue.js alternative
angular: "^16.0.0"      // Angular alternative
svelte: "^4.0.0"        // Svelte alternative
```

## 🌐 Recursos Online y Comunidades

### Repositorios de Código

#### Implementaciones de Referencia
1. **AstroNet (Google)**: https://github.com/google-research/exoplanet-ml  
   - Implementación CNN para detección de exoplanetas  
   - Datos pre-procesados de Kepler  
   - Paper: "Identifying Exoplanets with Deep Learning"  

2. **ExoNet**: https://github.com/vincentwang25/ExoNet  
   - Red neuronal para clasificación de tránsitos  
   - Comparación con métodos tradicionales  

3. **Kepler ML**: https://github.com/wintermydream/kepler-ml  
   - Implementaciones múltiples algoritmos ML  
   - Notebooks educativos incluidos  

#### Notebooks y Tutoriales
4. **NASA Exoplanet ML Tutorial**: https://github.com/nasa/kepler-robovetter  
   - Tutorial oficial NASA para ML en exoplanetas  
   - Implementación del Robovetter de Kepler  

5. **Kaggle Kernels**: https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results/kernels  
   - Múltiples implementaciones de la comunidad  
   - Diferentes enfoques y técnicas  

### Cursos y Material Educativo

#### Cursos Online
1. **Coursera - "Introduction to Astronomy" (Caltech)**  
   Fundamentos de astronomía incluyendo detección de exoplanetas  

2. **edX - "Astrobiology and the Search for Extraterrestrial Life" (Edinburgh)**  
   Contexto científico para la búsqueda de exoplanetas  

3. **Udacity - "Machine Learning Engineer Nanodegree"**  
   Fundamentos de ML aplicables al problema  

#### Documentación Técnica
4. **NASA Exoplanet Archive Documentation**  
   URL: https://exoplanetarchive.ipac.caltech.edu/docs/  
   Guías completas para uso de datos  

5. **Kepler/K2 Data Processing Handbook**  
   URL: https://archive.stsci.edu/kepler/documents.html  
   Detalles técnicos del procesamiento de datos  

### Conferencias y Workshops

#### Eventos Científicos Relevantes
1. **American Astronomical Society (AAS) Meetings**  
   Conferencias semestrales con sesiones sobre exoplanetas  

2. **Exoplanets Conference Series**  
   Conferencia específica cada 2-3 años  

3. **Machine Learning in Astronomy Workshops**  
   Workshops especializados en la intersección ML-Astronomía  

#### Competencias y Challenges
4. **NASA Space Apps Challenge**  
   Evento anual con challenges relacionados a exoplanetas  

5. **Kaggle Competitions**  
   Competencias periódicas de ML en astronomía  

## 📖 Recursos Adicionales

### Libros Recomendados

#### Exoplanetas y Astronomía
1. **"Exoplanets: Detection and characterization"** - Sara Seager  
2. **"The Planet Hunters"** - Dennis Overbye  
3. **"Five Billion Years of Solitude"** - Lee Billings  

#### Machine Learning
4. **"Hands-On Machine Learning"** - Aurélien Géron  
5. **"Pattern Recognition and Machine Learning"** - Christopher Bishop  
6. **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman  

### Herramientas de Desarrollo

#### IDEs y Notebooks
```bash
# Jupyter ecosystem
jupyter-lab==4.0.5       # Modern notebook interface
jupyter-notebook==7.0.0  # Classic notebook
voila==0.4.1            # Dashboard from notebooks

# IDEs
vscode                  # Visual Studio Code
pycharm                 # JetBrains PyCharm
spyder                  # Scientific Python IDE
```

#### Version Control y Collaboration
```bash
# Git and collaboration
git                     # Version control
github-cli              # GitHub command line
dvc                     # Data version control
mlflow                  # ML experiment tracking
```

### Datos de Validación y Testing

#### Test Datasets
1. **Kepler DR25 Certified Catalog**  
   Dataset oficial con etiquetas verificadas manualmente  

2. **TESS Input Catalog (TIC)**  
   Catálogo de estrellas objetivo para TESS  

3. **Gaia Data Release 3**  
   Datos complementarios de paralajes y movimientos propios  

#### Métricas de Benchmark
- **Kepler Robovetter Performance**: 95% precision, 85% recall  
- **TESS SPOC Pipeline**: Baseline para comparación  
- **Box Least Squares**: Algoritmo estándar de referencia  

## 🔗 Enlaces Útiles

### Sitios Web Oficiales
- **NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/  
- **TESS Mission**: https://tess.mit.edu/  
- **Kepler/K2 Mission**: https://www.nasa.gov/kepler  
- **MAST Portal**: https://mast.stsci.edu/  

### Herramientas Online
- **Lightkurve Online**: https://docs.lightkurve.org/  
- **TESS Data Portal**: https://heasarc.gsfc.nasa.gov/docs/tess/  
- **Exoplanet Follow-up**: https://tfop.ipac.caltech.edu/  

### APIs y Servicios
- **NASA Open Data API**: https://api.nasa.gov/  
- **Astroquery Documentation**: https://astroquery.readthedocs.io/  
- **SIMBAD Astronomical Database**: http://simbad.u-strasbg.fr/simbad/  

---

**Anterior**: [Interfaz Web y Deployment](./web-interface-deployment.md) | **Inicio**: [Instructions](../instructions.md)