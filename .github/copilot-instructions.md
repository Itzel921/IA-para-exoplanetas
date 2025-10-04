# Copilot Instructions - IA para Exoplanetas

## Project Overview
This is a **NASA Space Apps Challenge 2025** project that builds an AI/ML system for **exoplanet detection** using ensemble learning algorithms. The goal is to classify astronomical objects from Kepler, TESS, and K2 missions as confirmed planets, candidates, or false positives.

## Architecture & Key Components

### Documentation-First Approach
- **Context-driven**: All technical docs are in `.github/context/` (8 specialized files)
- **Critical docs**: Always reference `.github/context/ensemble-algorithms.md` and `.github/context/implementation-methodology.md`
- **Scientific foundation**: Based on research achieving 83.08% accuracy with Stacking ensemble
- **Access pattern**: Use read_file tool to access context documents when implementing features

### System Architecture
**Always read `.github/context/implementation-methodology.md` for complete architecture details**
```
Raw Data (KOI/TOI/K2) → Preprocessing → Feature Engineering → Ensemble Models → FastAPI → React Frontend
```

### Data Pipeline Patterns
- **3 NASA datasets**: KOI (Kepler), TOI (TESS), K2 with different target columns:
  - KOI: `koi_disposition` → ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']  
  - TOI: `tfopwg_disp` → ['KP', 'PC', 'FP', 'APC']
  - K2: `archive_disp` → ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
- **Unified labels**: All mapped to binary (1=CONFIRMED, 0=FALSE_POSITIVE/CANDIDATE)
- **Label mapping class**: `DataLoader` with `label_mapping` dict for cross-dataset consistency

## ML-Specific Conventions

### Ensemble Algorithm Priority
**Always read `.github/context/ensemble-algorithms.md` for complete algorithm details**
1. **Stacking** (best: 83.08% accuracy) - meta-learning with LGBM + Gradient Boosting
2. **Random Forest** (82.64%) - 1600 estimators after hyperparameter tuning  
3. **AdaBoost** (82.52%) - biggest improvement +1.15% with tuning
4. **Extra Trees** (82.36%) - minimal tuning gains due to randomness
5. **Random Subspace** (81.91%) - lowest performer

### Feature Engineering Patterns
**Read `.github/context/implementation-methodology.md` for complete feature engineering details**
```python
# Key derived features (from .github/context/implementation-methodology.md)
planet_star_radius_ratio = koi_prad / koi_srad
equilibrium_temp_ratio = koi_teq / koi_steff  
transit_depth_expected = (koi_prad / koi_srad) ** 2 * 1e6  # ppm
```

### Model Class Structure
- **Base class**: `ExoplanetEnsemble(BaseEstimator, ClassifierMixin)`
- **Stacking implementation**: `StackingEnsemble` with 2-phase training (base models → meta-model)
- **Cross-validation**: Always use `StratifiedKFold(n_splits=5)` for imbalanced classes
- **Serialization**: Use `joblib` for model persistence

### Preprocessing Patterns
- **Missing values**: Median imputation for stellar params, conditional median by disposition for planetary params
- **Outlier handling**: Winsorizing (1%-99% percentiles) instead of removal to preserve astronomical data
- **Feature scaling**: RobustScaler preferred over StandardScaler due to astronomical outliers
- **Time series**: TSFRESH for automated extraction of 789 features from light curves

## Development Workflow

### Tech Stack (From .github/context/web-interface-deployment.md)
**Read `.github/context/web-interface-deployment.md` for complete deployment details**
- **Backend**: FastAPI + Pydantic + uvicorn
- **Frontend**: React 18+ + TypeScript + Material-UI
- **ML**: scikit-learn + LightGBM + TSFRESH (789 time-series features)
- **Data**: pandas + numpy + NASA Exoplanet Archive APIs
- **Deploy**: Docker + nginx + PostgreSQL/Redis

### API Endpoints Pattern
```python
@app.post("/api/predict", response_model=PredictionResponse)  # Single prediction
@app.post("/api/batch-predict")  # CSV file upload
@app.get("/api/model-info")  # Model metadata
@app.websocket("/ws/batch-progress")  # Real-time batch processing
```

### Feature Engineering Classes
- **`DataLoader`**: Unified dataset loading with label mapping
- **`DataPreprocessor`**: Missing value imputation using median/conditional strategies  
- **`FeatureEngineer`**: Astronomical feature derivation (orbital parameters, habitability zones)
- **`PreprocessingPipeline`**: sklearn ColumnTransformer with RobustScaler

### Dataset Specifications 
- **KOI**: 9,654 objects × 50+ features (2009-2017 Kepler mission)
- **TOI**: 6,000+ objects (2018-present TESS mission, continuously updated)
- **K2**: Multiple campaigns (C0-C19) with diverse stellar fields
- **Key features**: Period, radius, temperature, stellar parameters, SNR, transit depth/duration

## Astronomical Domain Knowledge

### Critical Metrics (From .github/context/metrics-evaluation.md)
**Read `.github/context/metrics-evaluation.md` for complete metrics details**
- **Completeness** = Recall (fraction of real planets detected) 
- **Reliability** = Precision (fraction of detections that are real planets)
- **AUC-ROC > 0.9** = Excellent discrimination for astronomy
- **Confusion Matrix**: TP=real planets found, FP=wasted telescope time, FN=lost discoveries

### Physical Parameters 
- **Transit depth** ∝ (planet radius/star radius)²
- **Habitable zone distance**: Scaled by stellar temperature (5778K = Sun)
- **Signal-to-noise ratio**: Critical for detection confidence
- **Orbital period**: Determines observation frequency requirements

## File Organization
```
├── .github/
│   ├── copilot-instructions.md # This file - AI agent guide
│   └── context/                # Comprehensive technical docs (8 files)
│       ├── ensemble-algorithms.md
│       ├── implementation-methodology.md
│       ├── challenge-description.md
│       ├── theoretical-foundations.md
│       ├── datasets-preprocessing.md
│       ├── metrics-evaluation.md
│       ├── web-interface-deployment.md
│       └── references-resources.md
├── docs/                       # Research papers (PDFs)
├── src/                       # Implementation code (to be created)  
├── data/                      # NASA datasets (to be downloaded)
├── models/                    # Trained model artifacts
└── web/                       # React frontend
```

## Getting Started Commands

**CRITICAL: Use read_file tool to access these context documents when implementing features:**

### 🏗️ System Architecture & Design
- **[.github/context/implementation-methodology.md](./context/implementation-methodology.md)** - Complete system architecture, tech stack, and data pipeline
- **[.github/context/ensemble-algorithms.md](./context/ensemble-algorithms.md)** - ML algorithms: Stacking (83.08%), Random Forest, AdaBoost

### 🛠️ Implementation-Specific Details  
- **[.github/context/datasets-preprocessing.md](./context/datasets-preprocessing.md)** - NASA data access, KOI/TOI/K2 datasets, preprocessing pipeline
- **[.github/context/metrics-evaluation.md](./context/metrics-evaluation.md)** - Astronomical metrics: completeness, reliability, AUC-ROC
- **[.github/context/web-interface-deployment.md](./context/web-interface-deployment.md)** - FastAPI patterns, React frontend, Docker deployment

### 🔬 Scientific Foundation
- **[.github/context/theoretical-foundations.md](./context/theoretical-foundations.md)** - Physics principles, transit photometry, domain knowledge
- **[.github/context/challenge-description.md](./context/challenge-description.md)** - NASA Space Apps requirements and success criteria
- **[.github/context/references-resources.md](./context/references-resources.md)** - NASA TAP APIs, research papers, external resources

## Context Document Access Pattern

### Implementation Workflow
```
1. User requests feature/implementation
2. Use read_file tool to access relevant context documents
3. Extract specific technical requirements from documentation
4. Implement based on documented specifications (Stacking ensemble, FastAPI patterns, etc.)
5. Reference specific document sections in responses
```

### Implementation Workflow
```
1. User requests feature/implementation
2. Copilot automatically accesses relevant context documents via links
3. Extract specific technical requirements from documentation
4. Implement based on documented specifications (Stacking ensemble, FastAPI patterns, etc.)
5. Reference specific document sections in responses
```

### Priority Order for Implementation
1. **Start with**: System architecture and ensemble algorithms (83.08% accuracy target)
2. **Data pipeline**: NASA datasets (KOI/TOI/K2) with unified label mapping  
3. **Model implementation**: StackingEnsemble class with LGBM + Gradient Boosting
4. **API development**: FastAPI endpoints with batch processing and WebSocket support
5. **Frontend**: React with astronomical visualizations and real-time predictions

## Critical Context
- **Success criteria**: >90% accuracy target, 83.08% already achieved in research
- **Class imbalance**: ~25% confirmed planets, 75% false positives/candidates  
- **Deployment ready**: Full Docker + CI/CD configs provided in docs
- **Scientific rigor**: All methods based on peer-reviewed research (Electronics 2024, MNRAS 2022)

## 📝 Git Commit Standards (MANDATORY)

**ALWAYS follow Conventional Commits in Spanish:**

```bash
# Pattern: <type>[scope]: <description>
feat(sella_cruce): implementar matching por referencia bancaria
fix(file_toolbox): corregir extracción de saldos BBVA  
docs: actualizar documentación de arquitectura
test(parsers): agregar coverage para BBVA parser
```

### Required Types
- **feat**: Nueva funcionalidad
- **fix**: Corrección de bug  
- **docs**: Cambios solo en documentación
- **refactor**: Refactorización sin cambios funcionales
- **test**: Agregar o modificar tests