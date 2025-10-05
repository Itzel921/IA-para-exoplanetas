# ML DEV - Machine Learning Module for Exoplanet Detection

**NASA Space Apps Challenge 2025 - Ensemble Learning System for Astronomical Object Classification**

## 1. Overview

The **ML DEV** module is a complete Machine Learning system specialized in **exoplanet detection and classification** through analysis of data from Kepler, TESS, and K2 space missions. The main entry point is **Process.py**, which provides an interactive menu for all system functionalities.

### Main Purpose
- Load and analyze NASA datasets (KOI, TOI, K2) with astronomical objects
- Train ensemble learning models for binary classification (planet/non-planet)
- Generate automated predictions for new datasets with confidence metrics
- Create scientific visualizations for exploratory analysis

### Main Use Case
```python
# Main module entry point
python Process.py  # Runs complete interactive menu

# Or direct use of specific components
from Clasification import DataLoader
from train_ensemble import ExoplanetMLSystem
from simple_predictor_fixed import SimplePredictorFixed
```

### Critical Dependencies
```python
# Machine Learning and Preprocessing
pandas>=2.0.0           # NASA datasets manipulation
numpy>=1.24.0           # Numerical operations
scikit-learn>=1.3.0     # Ensemble algorithms
lightgbm>=3.3.5         # Gradient boosting
joblib>=1.3.0           # Model serialization

# Visualization
matplotlib>=3.7.0       # Base plotting
seaborn>=0.12.0         # Statistical visualizations

# Standard Python utilities
pathlib, warnings, datetime
```

## 2. Installation and Configuration

### Installation Steps

```bash
# 1. Navigate to module
cd "IA-para-exoplanetas/ML DEV"

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm joblib

# 3. Run main system
python Process.py
```

### Automatic Configuration

The module automatically configures necessary paths:

```python
# DataLoader automatically configures:
project_root = Path(__file__).parent.parent
datasets_path = project_root / "data" / "datasets"      # NASA datasets
new_datasets_path = project_root / "data" / "new_datasets"  # New CSVs
models_path = project_root / "models"                   # Trained models
results_path = project_root / "exoPlanet_results"       # Results
```

### Project Directory Structure
```
IA-para-exoplanetas/
├── data/
│   ├── datasets/               # Original NASA datasets
│   │   ├── cumulative_2025.10.04_11.46.06.csv  # KOI (Kepler)
│   │   ├── TOI_2025.10.04_11.44.53.csv         # TOI (TESS)
│   │   └── k2pandc_2025.10.04_11.46.18.csv     # K2
│   └── new_datasets/          # New CSVs for prediction
│       └── cumulative_2025.10.04_18.21.10Prueba1.csv
├── ML DEV/                    # 🎯 MAIN MODULE
│   ├── Process.py             # ⭐ MAIN ENTRY POINT
│   ├── Clasification.py      # Data loading and analysis
│   ├── train_ensemble*.py     # Model training
│   ├── simple_predictor_fixed.py  # Production predictor
│   ├── advanced_visualization.py  # Visualizations
│   └── trained_models/        # Local module models
├── models/                    # Global project models
└── exoPlanet_results/         # Results and charts
    ├── charts/                # Individual plots
    └── comparative_charts/    # Comparisons
```

## 3. Usage Reference

### Main Process.py Functions (Entry Point)

#### 3.1 `main()` - Main System Function

```python
def main():
    """Main system function with interactive menu"""
```

**Description**: Runs the main interactive menu that connects all module functionalities.

**Menu Options**:
1. `option_1_load_datasets()` - Load and analyze NASA datasets
2. `option_2_train_model()` - Train complete ensemble model
3. `option_3_simple_train()` - Simplified training (fast)
4. `option_4_predict_single()` - Predict specific dataset
5. `option_5_predict_all()` - Process all files in new_datasets
6. `option_6_full_analysis()` - Complete exploratory analysis with visualizations
7. `option_7_help()` - Help and documentation

### Main Classes

#### 3.2 `DataLoader` (Clasification.py) - NASA Datasets Loader

```python
class DataLoader:
    def __init__(self, project_root)
    def load_dataset(self, filename, dataset_type='KOI')
    def load_all_datasets()
    def analyze_dataset(self, df, dataset_name)
```

**Description**: Loads and unifies NASA datasets (KOI, TOI, K2) with consistent label mapping.

**Attributes**:
- `label_mapping`: Dict to unify labels between datasets
- `datasets_path`: Path to data/datasets/
- `new_datasets_path`: Path to data/new_datasets/

**Methods**:
- `load_dataset()`: Returns (DataFrame, dataset_type)
- `load_all_datasets()`: Returns dict {name: DataFrame}
- `analyze_dataset()`: Returns dict with statistics

#### 3.3 `ExoplanetMLSystem` (train_ensemble.py) - Ensemble System

```python
class ExoplanetMLSystem:
    def __init__(self, project_root)
    def train_system(self, datasets)
```

**Description**: Complete ML system combining preprocessing, feature engineering and stacking ensemble.

**Internal Components**:
- `DataPreprocessor`: Imputation, scaling, encoding
- `FeatureEngineer`: Derived astronomical features
- `StackingEnsemble`: RandomForest + GradientBoosting + LightGBM + LogisticRegression

**Methods**:
- `train_system()`: Returns dict with accuracy and saved model path

#### 3.4 `SimplePredictorFixed` (simple_predictor_fixed.py) - Production Predictor

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

**Description**: Optimized system for predictions on new datasets with guaranteed compatibility.

**Features**:
- Automatic loading of most recent model
- Batch processing of CSV files
- Automatic result generation with timestamps

#### 3.5 `ExoplanetVisualizer` (advanced_visualization.py) - Visualizations

```python
class ExoplanetVisualizer:
    def __init__(self, project_root)
    def create_dataset_overview(self, datasets)
    def create_class_distributions(self, datasets)
    def create_correlation_matrices(self, datasets)
    def create_scatter_plots(self, datasets)
    def generate_complete_analysis(self, nasa_datasets, new_datasets=None)
```

**Description**: Generates automatic scientific visualizations for exploratory analysis.

**Generated Visualizations**:
- Dataset overviews, class distributions, correlation matrices
- Scatter plots of key astronomical features
- Comparisons between NASA datasets and new data

### Minimal Functional Code Example

```python
"""
Complete workflow using real module functions - Functional example
"""
from pathlib import Path

# Method 1: Use interactive menu (recommended)
def use_interactive_menu():
    """Simplest way to use the module"""
    from Process import main
    main()  # Runs menu with all options

# Method 2: Direct component usage
def programmatic_workflow():
    """Direct use of main classes"""
    
    # 1. Load NASA datasets
    from Clasification import DataLoader
    project_root = Path(__file__).parent.parent
    loader = DataLoader(project_root)
    datasets = loader.load_all_datasets()
    
    if datasets:
        print(f"✅ {len(datasets)} datasets loaded")
        
        # 2. Train model (using train_ensemble.py)
        from train_ensemble import ExoplanetMLSystem
        ml_system = ExoplanetMLSystem(project_root)
        model_info = ml_system.train_system(datasets)
        print(f"Accuracy: {model_info['accuracy']:.4f}")
        
        # 3. Make predictions
        from simple_predictor_fixed import SimplePredictorFixed
        predictor = SimplePredictorFixed(project_root)
        if predictor.load_model():
            results = predictor.process_all_new_datasets()
            print(f"Processed: {len(results)} files")
    
    return datasets

# Method 3: Visualizations only
def generate_visualizations():
    """Generate analysis charts only"""
    from Clasification import DataLoader
    from advanced_visualization import ExoplanetVisualizer
    
    project_root = Path(__file__).parent.parent
    loader = DataLoader(project_root)
    datasets = loader.load_all_datasets()
    
    if datasets:
        visualizer = ExoplanetVisualizer(project_root)
        generated_files = visualizer.generate_complete_analysis(datasets)
        print(f"Charts generated: {len(generated_files)}")

# Run example
if __name__ == "__main__":
    # Recommended option: interactive menu
    use_interactive_menu()
```

## 4. Output Generation Test (Charts)

### Visual System Verification

The module automatically generates exploratory analysis visualizations using `ExoplanetVisualizer` and saves results in `exoPlanet_results/`.

#### Minimal Test - Chart Generation

```python
"""
Functional test to generate charts using ExoplanetVisualizer
"""
def test_generate_charts():
    """Runs option_6_full_analysis() from Process.py"""
    
    # Method 1: Use menu function (recommended)
    from Process import option_6_full_analysis
    option_6_full_analysis()  # Generates all visualizations
    
    # Method 2: Direct ExoplanetVisualizer usage
    from pathlib import Path
    from Clasification import DataLoader
    from advanced_visualization import ExoplanetVisualizer
    
    project_root = Path(__file__).parent.parent
    loader = DataLoader(project_root)
    datasets = loader.load_all_datasets()
    
    if datasets:
        visualizer = ExoplanetVisualizer(project_root)
        files = visualizer.generate_complete_analysis(datasets)
        print(f"✅ {len(files)} visualizations generated")
        return True
    return False

# Run from Process.py (option 6) or use direct test
if __name__ == "__main__":
    test_generate_charts()
```

#### Real Generated Output Files

**Charts in `exoPlanet_results/charts/`:**
- `01_dataset_overview_20251005_012151.png` - NASA datasets overview
- `02_class_distributions_20251005_012151.png` - Class distributions per dataset
- `03_correlation_matrices_20251005_012151.png` - Feature correlation matrices
- `04_scatter_plots_20251005_012151.png` - Key variable scatter plots

**Comparative Charts in `exoPlanet_results/comparative_charts/`:**
- `nasa_vs_new_comparison_20251005_012151.png` - NASA vs new datasets comparison
- `prediction_results_analysis_20251005_012151.png` - Prediction results analysis

**CSV Result Files:**
- `cumulative_2025.10.04_18.21.10Prueba1_predictions_20251005_012139.csv` - Predictions with ML_Probability, ML_Prediction, ML_Classification
- `accuracy_metrics_20251004_202348.csv` - Model accuracy metrics
- `confidence_distribution_20251004_202348.csv` - Prediction confidence distribution

#### Manual Output Verification

```bash
# Verify charts were generated correctly
ls "exoPlanet_results/charts/" | grep .png
ls "exoPlanet_results/comparative_charts/" | grep .png

# Verify CSV result files
ls "exoPlanet_results/" | grep .csv
```

## 5. For Developers (Architecture)

### Real ML DEV Module Structure

```
ML DEV/
├── Process.py                      # ⭐ ENTRY POINT - Interactive menu
├── Clasification.py               # 📊 DataLoader - NASA datasets loading
├── train_ensemble.py              # 🧠 ExoplanetMLSystem + StackingEnsemble
├── train_ensemble_FAST.py         # ⚡ Optimized FastStackingEnsemble
├── train_ensemble_CORRECTED.py    # 🔧 FeatureMapper + ImprovedDataPreprocessor
├── simple_predictor_fixed.py      # 🔮 Production SimplePredictorFixed
├── simple_retrain.py             # 🔄 Fast retraining
├── advanced_visualization.py      # 📈 ExoplanetVisualizer
├── model_imports.py               # 📦 Joblib imports
├── README_ML.md                   # 📄 Original documentation
├── README_ML_EN.md               # 📄 This documentation
├── __pycache__/                   # Python bytecode cache
└── trained_models/                # Local module models
    ├── exoplanet_ensemble_20251005_002211.pkl
    ├── exoplanet_ensemble_FAST_20251004_195930.pkl
    └── exoplanet_simple_20251005_011244.pkl
```

### Real Data Flow (Based on Process.py)

```
1. Process.main() → show_menu()
    ↓
2. User selects option
    ↓
3a. option_1_load_datasets()
    → DataLoader.load_all_datasets()
    → Automatic analysis of KOI, TOI, K2
    
3b. option_2_train_model()
    → ExoplanetMLSystem.train_system()
    → FeatureEngineer + DataPreprocessor
    → StackingEnsemble (RF + GB + LGB + LogReg)
    → Model saved in trained_models/
    
3c. option_4_predict_single() / option_5_predict_all()
    → SimplePredictorFixed.load_model()
    → SimplePredictorFixed.process_file()
    → Results in exoPlanet_results/
    
3d. option_6_full_analysis()
    → ExoplanetVisualizer.generate_complete_analysis()
    → Charts in exoPlanet_results/charts/
```

### Key Technical Components

#### DataLoader (Clasification.py)
- **Unified label mapping**: Converts KOI, TOI, K2 to consistent binary format
- **NASA metadata handling**: Processes CSV files with '#' comments
- **Automatic validation**: Verifies dataset integrity on loading

#### StackingEnsemble (train_ensemble.py)
- **Base learners**: RandomForest (100 est.) + GradientBoosting (50 est.) + LightGBM (50 est.)
- **Meta-learner**: LogisticRegression for final combination
- **Cross-validation**: StratifiedKFold (5 splits) for meta-feature generation
- **Target**: Binary classification (1=confirmed planet, 0=candidate/false positive)

#### SimplePredictorFixed (simple_predictor_fixed.py)
- **Limited features**: Only uses ['ra', 'dec'] for compatibility
- **Automatic loading**: Selects most recent model by timestamp
- **Batch processing**: Handles multiple CSV files automatically
- **Output format**: Original CSV + ML_Probability, ML_Prediction, ML_Classification columns

#### ExoplanetVisualizer (advanced_visualization.py)
- **4 main chart types**: overview, class distributions, correlations, scatter plots
- **Automatic comparisons**: NASA datasets vs new datasets
- **Automatic timestamps**: All visualizations include generation date/time
- **PNG export**: Automatically saves in exoPlanet_results/charts/

### Technical Considerations

#### Performance
- **Ensemble models**: ~50-200MB on disk, load <5 seconds
- **NASA datasets**: ~500MB combined, requires 2GB+ RAM for training
- **Predictions**: ~1000 objects/second on standard hardware

#### Compatibility
- **Serialization**: joblib for models, ensures cross-version compatibility
- **Features**: SimplePredictorFixed handles feature differences automatically
- **Paths**: pathlib for Windows/Linux compatibility

#### Scalability
- **Modular**: Each component works independently
- **Extensible**: Easy to add new ensemble algorithms
- **Batch processing**: Parallel processing of multiple files

---

**🌟 NASA Space Apps Challenge 2025**  
*Technical documentation of ML DEV module - Ensemble learning system for automated exoplanet detection based on real implemented code*

**Main Files:**
- **Process.py**: Entry point with interactive menu
- **Clasification.py**: DataLoader for NASA datasets
- **train_ensemble.py**: ExoplanetMLSystem with StackingEnsemble
- **simple_predictor_fixed.py**: SimplePredictorFixed for production
- **advanced_visualization.py**: ExoplanetVisualizer for visual analysis