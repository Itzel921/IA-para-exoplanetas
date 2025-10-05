"""
Sistema de entrenamiento CORREGIDO - Mapeo de características entre datasets
Soluciona el problema de intersección que resultaba en solo 2 características (ra, dec)
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

class FeatureMapper:
    """
    Mapea características entre diferentes datasets de exoplanetas
    """
    
    def __init__(self):
        # Mapeo de características principales entre datasets
        self.feature_mapping = {
            # Parámetros orbitales
            'period': {
                'KOI': 'koi_period',
                'TOI': 'pl_orbper', 
                'K2': 'pl_orbper'
            },
            # Radio planetario
            'planet_radius': {
                'KOI': 'koi_prad',
                'TOI': 'pl_rade',
                'K2': 'pl_rade'
            },
            # Radio estelar
            'stellar_radius': {
                'KOI': 'koi_srad',
                'TOI': 'st_rad',
                'K2': 'st_rad'
            },
            # Temperatura estelar
            'stellar_temp': {
                'KOI': 'koi_steff',
                'TOI': 'st_teff',
                'K2': 'st_teff'
            },
            # Temperatura de equilibrio planetaria
            'planet_temp': {
                'KOI': 'koi_teq',
                'TOI': 'pl_eqt',
                'K2': 'pl_eqt'
            },
            # Profundidad del tránsito
            'transit_depth': {
                'KOI': 'koi_depth',
                'TOI': 'pl_trandep',
                'K2': None  # No disponible directamente
            },
            # Duración del tránsito
            'transit_duration': {
                'KOI': 'koi_duration',
                'TOI': 'pl_trandurh',
                'K2': None  # No disponible directamente
            },
            # SNR del modelo
            'model_snr': {
                'KOI': 'koi_model_snr',
                'TOI': None,
                'K2': None
            },
            # Coordenadas (común en todos)
            'ra': {
                'KOI': 'ra',
                'TOI': 'ra', 
                'K2': 'ra'
            },
            'dec': {
                'KOI': 'dec',
                'TOI': 'dec',
                'K2': 'dec'  
            }
        }
    
    def create_unified_features(self, df, dataset_type):
        """
        Crea un conjunto unificado de características para el dataset
        """
        df_unified = pd.DataFrame()
        
        print(f"🔄 Mapeando características para {dataset_type}...")
        
        # Mapear características principales
        for unified_name, mapping in self.feature_mapping.items():
            original_col = mapping.get(dataset_type)
            
            if original_col and original_col in df.columns:
                df_unified[unified_name] = df[original_col]
                print(f"   ✅ {unified_name} <- {original_col}")
            else:
                # Llenar con NaN si no está disponible
                df_unified[unified_name] = np.nan
                print(f"   ❌ {unified_name} (no disponible)")
        
        # Crear características derivadas
        df_unified = self._create_derived_features(df_unified)
        
        return df_unified
    
    def _create_derived_features(self, df):
        """
        Crea características derivadas a partir de las características base
        """
        df_derived = df.copy()
        
        # Planet-Star Radius Ratio
        if 'planet_radius' in df.columns and 'stellar_radius' in df.columns:
            df_derived['planet_star_radius_ratio'] = df['planet_radius'] / df['stellar_radius']
        
        # Temperature Ratio
        if 'planet_temp' in df.columns and 'stellar_temp' in df.columns:
            df_derived['temp_ratio'] = df['planet_temp'] / df['stellar_temp']
        
        # Habitability Index (distance from habitable temperature ~288K)
        if 'planet_temp' in df.columns:
            df_derived['habitability_index'] = np.abs(df['planet_temp'] - 288) / 288
        
        # Transit Depth Expected (theoretical)
        if 'planet_radius' in df.columns and 'stellar_radius' in df.columns:
            df_derived['transit_depth_theoretical'] = (df['planet_radius'] / df['stellar_radius']) ** 2 * 1e6
        
        # Period-Radius relationship
        if 'period' in df.columns and 'planet_radius' in df.columns:
            df_derived['period_radius_product'] = df['period'] * df['planet_radius']
        
        print(f"   🧬 Características derivadas creadas: {len(df_derived.columns) - len(df.columns)}")
        
        return df_derived

class ImprovedDataPreprocessor:
    """
    Preprocessor mejorado que usa mapeo de características
    """
    
    def __init__(self):
        self.feature_mapper = FeatureMapper()
        self.scaler = None
        self.feature_names = None
        
    def fit_transform(self, datasets_dict):
        """
        Procesa múltiples datasets y los unifica
        """
        processed_data = []
        
        for name, data in datasets_dict.items():
            df, target_col = data['df'], data['target_col']
            
            print(f"\n🔄 Procesando {name}...")
            print(f"   📊 Shape original: {df.shape}")
            
            # 1. Crear características unificadas
            df_unified = self.feature_mapper.create_unified_features(df, name)
            
            # 2. Extraer variable objetivo
            y = df[target_col]
            
            # 3. Preparar datos para el conjunto unificado
            processed_data.append({
                'X': df_unified,
                'y': y,
                'dataset': name
            })
        
        # Combinar todos los datasets
        X_combined = pd.concat([data['X'] for data in processed_data], ignore_index=True)
        y_combined = pd.concat([data['y'] for data in processed_data], ignore_index=True)
        
        print(f"\n🔗 Dataset combinado:")
        print(f"   📊 Shape: {X_combined.shape}")
        print(f"   🎯 Distribución de clases: {dict(y_combined.value_counts())}")
        
        # 4. Imputación y preprocesamiento
        X_processed = self._preprocess_features(X_combined)
        
        # 5. Guardar nombres de características
        self.feature_names = list(X_processed.columns)
        
        return X_processed, y_combined
    
    def _preprocess_features(self, X):
        """
        Preprocesamiento de características unificadas
        """
        X_processed = X.copy()
        
        # 1. Imputación con mediana
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        self.imputer = SimpleImputer(strategy='median')
        X_processed[numeric_cols] = self.imputer.fit_transform(X_processed[numeric_cols])
        
        # 2. Manejo de outliers (winsorizing)
        for col in numeric_cols:
            q01 = X_processed[col].quantile(0.01)
            q99 = X_processed[col].quantile(0.99)
            X_processed[col] = X_processed[col].clip(lower=q01, upper=q99)
        
        # 3. Escalado robusto
        self.scaler = RobustScaler()
        X_processed[numeric_cols] = self.scaler.fit_transform(X_processed[numeric_cols])
        
        print(f"   ✅ Preprocesamiento completado")
        print(f"   🔢 Características finales: {len(X_processed.columns)}")
        
        return X_processed
    
    def transform(self, df, dataset_type):
        """
        Transforma un nuevo dataset usando los parámetros ajustados
        """
        # 1. Mapear características
        df_unified = self.feature_mapper.create_unified_features(df, dataset_type)
        
        # 2. Asegurar mismo orden de columnas
        for col in self.feature_names:
            if col not in df_unified.columns:
                df_unified[col] = 0.0  # Valor por defecto
        
        df_unified = df_unified[self.feature_names]
        
        # 3. Imputar valores faltantes con la misma estrategia de entrenamiento
        numeric_cols = df_unified.select_dtypes(include=[np.number]).columns
        
        # Imputar valores faltantes usando SimpleImputer
        if hasattr(self, 'imputer'):
            df_unified[numeric_cols] = self.imputer.transform(df_unified[numeric_cols])
        else:
            # Fallback: imputación manual
            for col in numeric_cols:
                df_unified[col] = df_unified[col].fillna(df_unified[col].median())
            df_unified = df_unified.fillna(0)  # Backup final
        
        # 4. Aplicar escalado
        df_unified[numeric_cols] = self.scaler.transform(df_unified[numeric_cols])
        
        return df_unified

# Usar las mismas clases de ensemble del archivo original
from train_ensemble import StackingEnsemble

def main():
    """
    Entrenamiento principal con mapeo de características corregido
    """
    print("🚀 Sistema de Entrenamiento CORREGIDO - Exoplanetas ML")
    print("🔧 Usando mapeo inteligente de características")
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
            datasets_dict[name] = {'df': df, 'target_col': target_col}
    
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
    preprocessor = ImprovedDataPreprocessor()
    X, y = preprocessor.fit_transform(datasets_dict)
    
    print(f"\n🎯 ENTRENAMIENTO DEL ENSEMBLE")
    print("="*40)
    
    # Entrenar modelo ensemble
    model = StackingEnsemble(random_state=42)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print(f"📊 Cross-validation scores: {cv_scores}")
    print(f"🎯 Accuracy promedio: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Entrenar modelo final
    model.fit(X, y)
    
    # Predicciones finales
    y_pred = model.predict(X)
    final_accuracy = accuracy_score(y, y_pred)
    print(f"✅ Accuracy final: {final_accuracy:.4f}")
    
    # Guardar modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"exoplanet_ensemble_CORRECTED_{timestamp}.pkl"
    model_path = project_root / "models" / model_filename
    
    model_data = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': preprocessor.feature_names,
        'label_mapping': label_mapping,
        'cv_scores': cv_scores,
        'training_datasets': list(datasets_dict.keys()),
        'timestamp': timestamp,
        'accuracy': final_accuracy
    }
    
    joblib.dump(model_data, model_path)
    print(f"💾 Modelo guardado: {model_filename}")
    
    # Reporte detallado
    print(f"\n📋 REPORTE FINAL")
    print("="*30)
    print(f"✅ Modelo entrenado exitosamente")
    print(f"🎯 Accuracy: {final_accuracy:.4f}")
    print(f"🔢 Características: {len(preprocessor.feature_names)}")
    print(f"📊 Muestras totales: {len(X)}")
    print(f"🎲 Datasets: {list(datasets_dict.keys())}")
    print(f"💾 Archivo: {model_filename}")

if __name__ == "__main__":
    main()