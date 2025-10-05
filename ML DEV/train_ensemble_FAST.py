"""
Entrenamiento RÁPIDO - Versión optimizada para monitoreo
Parámetros reducidos para entrenamiento eficiente y progreso visible
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb

# Importar componentes del sistema
from Clasification import DataLoader
from train_ensemble_CORRECTED import FeatureMapper, ImprovedDataPreprocessor

class FastStackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble rápido para pruebas y monitoreo
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # Base learners optimizados para velocidad
        self.base_learners = {
            'rf': RandomForestClassifier(
                n_estimators=50,  # Reducido para velocidad
                max_depth=8,
                min_samples_split=10,
                random_state=random_state,
                n_jobs=1,  # Un solo core para evitar problemas
                verbose=0
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=20,  # Muy reducido
                learning_rate=0.2,  # Más alto para convergencia rápida
                max_depth=4,
                random_state=random_state,
                verbose=0
            )
        }
        
        # Meta-learner
        self.meta_learner = LogisticRegression(
            random_state=random_state,
            max_iter=100
        )
        
        self.is_fitted = False
    
    def fit(self, X, y):
        """Entrenamiento rápido con progreso visible"""
        print(f"🎯 Entrenando ensemble rápido...")
        print(f"   📊 Datos: {X.shape[0]} muestras, {X.shape[1]} características")
        
        # Entrenar base learners
        print(f"🔄 Fase 1: Entrenando base learners...")
        base_predictions = np.zeros((X.shape[0], len(self.base_learners)))
        
        for i, (name, learner) in enumerate(self.base_learners.items()):
            print(f"   🌳 Entrenando {name}...")
            
            # Cross-validation predictions para meta-learner
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)  # Reducido a 3 folds
            cv_preds = np.zeros(X.shape[0])
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                print(f"      📂 Fold {fold+1}/3")
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                
                learner_fold = type(learner)(**learner.get_params())
                learner_fold.fit(X_train_fold, y_train_fold)
                cv_preds[val_idx] = learner_fold.predict_proba(X_val_fold)[:, 1]
            
            base_predictions[:, i] = cv_preds
            
            # Entrenar en todo el dataset
            learner.fit(X, y)
            print(f"      ✅ {name} completado")
        
        # Entrenar meta-learner
        print(f"🔄 Fase 2: Entrenando meta-learner...")
        self.meta_learner.fit(base_predictions, y)
        print(f"   ✅ Meta-learner completado")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predicción del ensemble"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        base_predictions = np.zeros((X.shape[0], len(self.base_learners)))
        
        for i, (name, learner) in enumerate(self.base_learners.items()):
            base_predictions[:, i] = learner.predict_proba(X)[:, 1]
        
        return self.meta_learner.predict(base_predictions)
    
    def predict_proba(self, X):
        """Probabilidades del ensemble"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        base_predictions = np.zeros((X.shape[0], len(self.base_learners)))
        
        for i, (name, learner) in enumerate(self.base_learners.items()):
            base_predictions[:, i] = learner.predict_proba(X)[:, 1]
        
        return self.meta_learner.predict_proba(base_predictions)

def main():
    """
    Entrenamiento rápido con monitoreo de progreso
    """
    print("⚡ ENTRENAMIENTO RÁPIDO - Exoplanetas ML")
    print("🚀 Versión optimizada para velocidad y monitoreo")
    print("="*60)
    
    # Cargar datos
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    loader = DataLoader(project_root)
    datasets = loader.load_all_datasets()
    
    # Preparar datasets con mapeo de columnas objetivo
    target_mapping = {
        'KOI': 'koi_disposition',
        'TOI': 'tfopwg_disp', 
        'K2': 'disposition'
    }
    
    datasets_dict = {}
    for name, df in datasets.items():
        target_col = target_mapping[name]
        if target_col in df.columns:
            # Usar solo una muestra para entrenamiento rápido
            sample_size = min(1000, len(df))  # Máximo 1000 por dataset
            df_sample = df.sample(n=sample_size, random_state=42)
            datasets_dict[name] = {'df': df_sample, 'target_col': target_col}
            print(f"📊 {name}: {len(df_sample):,} muestras (de {len(df):,} totales)")
    
    # Mapear etiquetas a binario (1=CONFIRMED, 0=resto)
    label_mapping = {
        'CONFIRMED': 1,
        'KP': 1,  # TOI confirmed planet
        'CANDIDATE': 0,
        'PC': 0,  # TOI planet candidate  
        'APC': 0, # TOI ambiguous planet candidate
        'FALSE POSITIVE': 0,
        'FP': 0,  # TOI false positive
        'REFUTED': 0
    }
    
    # Aplicar mapeo de etiquetas
    for data in datasets_dict.values():
        df = data['df']
        target_col = data['target_col']
        df[target_col] = df[target_col].map(label_mapping).fillna(0)
    
    # Preprocesamiento con mapeo
    print(f"\n🔄 PREPROCESAMIENTO RÁPIDO")
    print("="*40)
    
    preprocessor = ImprovedDataPreprocessor()
    X, y = preprocessor.fit_transform(datasets_dict)
    
    print(f"\n🎯 ENTRENAMIENTO DEL ENSEMBLE RÁPIDO")
    print("="*40)
    
    # Entrenar modelo ensemble rápido
    model = FastStackingEnsemble(random_state=42)
    model.fit(X, y)
    
    # Evaluación rápida
    print(f"\n📊 EVALUACIÓN RÁPIDA")
    print("="*30)
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    train_accuracy = accuracy_score(y, y_pred)
    print(f"🎯 Accuracy de entrenamiento: {train_accuracy:.4f}")
    
    # Distribución de predicciones
    pred_dist = pd.Series(y_pred).value_counts().sort_index()
    print(f"📊 Distribución de predicciones:")
    print(f"   No-Planeta (0): {pred_dist.get(0, 0):,}")
    print(f"   Planeta (1): {pred_dist.get(1, 0):,}")
    
    # Confianza promedio
    confidences = np.max(y_proba, axis=1)
    avg_confidence = np.mean(confidences)
    print(f"🔮 Confianza promedio: {avg_confidence:.4f}")
    
    # Guardar modelo rápido
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"exoplanet_ensemble_FAST_{timestamp}.pkl"
    model_path = project_root / "models" / model_filename
    
    model_data = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': preprocessor.feature_names,
        'label_mapping': label_mapping,
        'training_datasets': list(datasets_dict.keys()),
        'timestamp': timestamp,
        'accuracy': train_accuracy,
        'sample_sizes': {name: len(data['df']) for name, data in datasets_dict.items()}
    }
    
    joblib.dump(model_data, model_path)
    print(f"\n💾 MODELO GUARDADO")
    print("="*25)
    print(f"📁 Archivo: {model_filename}")
    print(f"🎯 Accuracy: {train_accuracy:.4f}")
    print(f"🔢 Características: {len(preprocessor.feature_names)}")
    print(f"📊 Muestras totales: {len(X):,}")
    
    # Mostrar características principales
    print(f"\n🧬 CARACTERÍSTICAS PRINCIPALES")
    print("="*35)
    for i, feature in enumerate(preprocessor.feature_names[:10], 1):
        print(f"   {i:2d}. {feature}")
    if len(preprocessor.feature_names) > 10:
        print(f"   ... y {len(preprocessor.feature_names) - 10} más")
    
    print(f"\n✅ ENTRENAMIENTO RÁPIDO COMPLETADO")
    print(f"🚀 El modelo está listo para pruebas de predicción")
    
    return model_data

if __name__ == "__main__":
    main()