"""
Reentrenamiento de Modelo con Características Astronómicas Específicas
Soluciona el problema de que solo se usan 'ra' y 'dec'
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys
from datetime import datetime
import warnings
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Importar clases necesarias
import model_imports
from Clasification import DataLoader

class FixedTrainer:
    """
    Reentrenador que usa características astronómicas específicas
    """
    
    def __init__(self):
        self.current_dir = Path(__file__).parent
        self.project_root = self.current_dir.parent
        
        # Paths
        self.datasets_path = self.project_root / "data" / "datasets"
        self.trained_models_path = self.current_dir / "trained_models"
        self.trained_models_path.mkdir(exist_ok=True)
        
        # Datos
        self.datasets = {}
        self.combined_data = None
        self.scaler = StandardScaler()
        
        # Características astronómicas específicas que usaremos
        self.target_features = [
            # Coordenadas (por compatibilidad)
            'ra', 'dec',
            
            # Parámetros planetarios clave
            'period', 'radius', 'depth', 'duration', 'temp',
            
            # Parámetros estelares  
            'star_radius', 'star_mass', 'star_temp',
            
            # Métricas de detección
            'snr', 'impact'
        ]
    
    def load_datasets(self):
        """Cargar y unificar datasets con mapeo de características"""
        print("🚀 Iniciando reentrenamiento con características astronómicas específicas")
        print("="*80)
        
        # Cargar datasets
        data_loader = DataLoader(str(self.project_root))
        self.datasets = data_loader.load_all_datasets()
        
        print(f"✅ Datasets cargados:")
        for name, df in self.datasets.items():
            print(f"   • {name}: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    
    def create_unified_features(self):
        """Crear características unificadas específicas para astronomía"""
        print(f"\\n🔧 Creando características astronómicas unificadas...")
        
        unified_datasets = []
        
        for dataset_name, df in self.datasets.items():
            print(f"\\n📊 Procesando {dataset_name}...")
            
            # Crear DataFrame unificado
            unified_df = pd.DataFrame()
            
            # Mapear características según el dataset
            if dataset_name == 'KOI':
                # Coordenadas
                unified_df['ra'] = df.get('ra', 0.0)
                unified_df['dec'] = df.get('dec', 0.0)
                
                # Parámetros planetarios
                unified_df['period'] = df.get('koi_period', 0.0)
                unified_df['radius'] = df.get('koi_prad', 0.0)
                unified_df['depth'] = df.get('koi_depth', 0.0) 
                unified_df['duration'] = df.get('koi_duration', 0.0)
                unified_df['temp'] = df.get('koi_teq', 0.0)
                
                # Parámetros estelares
                unified_df['star_radius'] = df.get('koi_srad', 0.0)
                unified_df['star_mass'] = df.get('koi_smass', 1.0)  # Default solar mass
                unified_df['star_temp'] = df.get('koi_steff', 5778.0)  # Default solar temp
                
                # Métricas
                unified_df['snr'] = df.get('koi_model_snr', 0.0)
                unified_df['impact'] = df.get('koi_impact', 0.0)
                
                # Target
                unified_df['target'] = df['koi_disposition'].map({
                    'CONFIRMED': 1,
                    'CANDIDATE': 0, 
                    'FALSE POSITIVE': 0
                })
                
            elif dataset_name == 'TOI':
                # Coordenadas
                unified_df['ra'] = df.get('ra', 0.0)
                unified_df['dec'] = df.get('dec', 0.0)
                
                # Parámetros planetarios
                unified_df['period'] = df.get('pl_orbper', 0.0)
                unified_df['radius'] = df.get('pl_rade', 0.0)
                unified_df['depth'] = df.get('pl_trandep', 0.0)
                unified_df['duration'] = df.get('pl_trandurh', 0.0)
                unified_df['temp'] = df.get('pl_eqt', 0.0)
                
                # Parámetros estelares
                unified_df['star_radius'] = df.get('st_rad', 0.0)
                unified_df['star_mass'] = 1.0  # No disponible en TOI
                unified_df['star_temp'] = df.get('st_teff', 5778.0)
                
                # Métricas
                unified_df['snr'] = 0.0  # No disponible directo
                unified_df['impact'] = 0.0  # No disponible directo
                
                # Target
                unified_df['target'] = df['tfopwg_disp'].map({
                    'KP': 1,
                    'PC': 0,
                    'FP': 0,
                    'APC': 0
                })
                
            elif dataset_name == 'K2':
                # Coordenadas
                unified_df['ra'] = df.get('ra', 0.0)
                unified_df['dec'] = df.get('dec', 0.0)
                
                # Parámetros planetarios
                unified_df['period'] = df.get('pl_orbper', 0.0)
                unified_df['radius'] = df.get('pl_rade', 0.0)
                unified_df['depth'] = 0.0  # Calcular aproximado
                unified_df['duration'] = 0.0  # No disponible directo
                unified_df['temp'] = df.get('pl_eqt', 0.0)
                
                # Parámetros estelares
                unified_df['star_radius'] = df.get('st_rad', 0.0)
                unified_df['star_mass'] = df.get('st_mass', 1.0)
                unified_df['star_temp'] = df.get('st_teff', 5778.0)
                
                # Métricas
                unified_df['snr'] = 0.0  # No disponible
                unified_df['impact'] = 0.0  # No disponible
                
                # Target
                unified_df['target'] = df['disposition'].map({
                    'CONFIRMED': 1,
                    'CANDIDATE': 0,
                    'FALSE POSITIVE': 0,
                    'REFUTED': 0
                })
            
            # Agregar identificador de dataset
            unified_df['source'] = dataset_name
            
            # Limpiar datos
            unified_df = unified_df.fillna(0.0)
            unified_df = unified_df[unified_df['target'].notna()]
            
            print(f"   • Características mapeadas: {len(self.target_features)}")
            print(f"   • Objetos válidos: {len(unified_df):,}")
            print(f"   • Confirmados: {(unified_df['target'] == 1).sum():,}")
            
            unified_datasets.append(unified_df)
        
        # Combinar todos los datasets
        self.combined_data = pd.concat(unified_datasets, ignore_index=True)
        
        print(f"\\n✅ Dataset unificado creado:")
        print(f"   • Total objetos: {len(self.combined_data):,}")
        print(f"   • Total confirmados: {(self.combined_data['target'] == 1).sum():,}")
        print(f"   • Características: {len(self.target_features)}")
        
        return self.combined_data
    
    def create_derived_features(self, df):
        """Crear características derivadas astronómicamente significativas"""
        print(f"\\n🔬 Creando características derivadas...")
        
        df_enhanced = df.copy()
        
        # Ratios físicos
        df_enhanced['planet_star_radius_ratio'] = np.where(
            df['star_radius'] > 0,
            df['radius'] / df['star_radius'],
            0
        )
        
        df_enhanced['temp_ratio'] = np.where(
            df['star_temp'] > 0,
            df['temp'] / df['star_temp'],
            0
        )
        
        # Características orbitales
        df_enhanced['period_log'] = np.log10(df['period'] + 1)
        df_enhanced['radius_log'] = np.log10(df['radius'] + 1)
        
        # Zona habitable aproximada
        df_enhanced['habitable_zone_distance'] = np.where(
            df['star_temp'] > 0,
            np.abs(df['temp'] - 300) / 100,  # Distancia de temp habitable
            0
        )
        
        # Métricas de tránsito
        df_enhanced['depth_expected'] = np.where(
            df['star_radius'] > 0,
            (df['radius'] / df['star_radius']) ** 2 * 1e6,
            0
        )
        
        print(f"   • Nuevas características creadas: 6")
        print(f"   • Total características: {df_enhanced.shape[1] - 2}")  # -2 por target y source
        
        return df_enhanced
    
    def train_ensemble_model(self):
        """Entrenar modelo ensemble con características astronómicas"""
        print(f"\\n🎯 Entrenando modelo ensemble...")
        
        # Preparar datos
        feature_cols = [col for col in self.combined_data.columns 
                       if col not in ['target', 'source']]
        
        X = self.combined_data[feature_cols]
        y = self.combined_data['target']
        
        print(f"   • Características de entrada: {len(feature_cols)}")
        print(f"   • Muestras de entrenamiento: {len(X):,}")
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   • Training set: {len(X_train):,}")
        print(f"   • Test set: {len(X_test):,}")
        
        # Definir modelos base
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
            ('extra', ExtraTreesClassifier(n_estimators=100, random_state=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42))
        ]
        
        # Meta-modelo
        meta_model = LogisticRegression(random_state=42)
        
        # Crear ensemble
        ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        
        print(f"\\n📚 Entrenando Stacking Ensemble...")
        print(f"   • Modelos base: {len(base_models)}")
        print(f"   • Meta-modelo: Logistic Regression")
        print(f"   • Cross-validation: 5-fold")
        
        # Entrenar
        ensemble.fit(X_train, y_train)
        
        # Evaluar
        train_score = ensemble.score(X_train, y_train)
        test_score = ensemble.score(X_test, y_test)
        
        print(f"\\n📊 Resultados:")
        print(f"   • Accuracy (train): {train_score:.4f}")
        print(f"   • Accuracy (test): {test_score:.4f}")
        
        # Predicciones detalladas
        y_pred = ensemble.predict(X_test)
        
        print(f"\\n🎯 Reporte de clasificación:")
        print(classification_report(y_test, y_pred, target_names=['No-Planeta', 'Planeta']))
        
        # Guardar modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_info = {
            'model': ensemble,
            'scaler': self.scaler,
            'feature_names': feature_cols,
            'common_features': feature_cols,  # Todas las características
            'accuracy': test_score,
            'trained_on': timestamp,
            'n_features': len(feature_cols),
            'n_samples': len(X),
            'datasets_used': ['KOI', 'TOI', 'K2']
        }
        
        model_filename = f"exoplanet_ensemble_FIXED_{timestamp}.pkl"
        model_path = self.trained_models_path / model_filename
        
        joblib.dump(model_info, model_path)
        
        print(f"\\n💾 Modelo guardado:")
        print(f"   • Archivo: {model_filename}")
        print(f"   • Características: {len(feature_cols)}")
        print(f"   • Accuracy: {test_score:.4f}")
        print(f"   • Tamaño: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        return model_info

def main():
    """Función principal de reentrenamiento"""
    trainer = FixedTrainer()
    
    # Cargar datasets
    trainer.load_datasets()
    
    # Crear características unificadas
    unified_data = trainer.create_unified_features()
    
    # Crear características derivadas
    trainer.combined_data = trainer.create_derived_features(unified_data)
    
    # Entrenar modelo
    model_info = trainer.train_ensemble_model()
    
    print(f"\\n✅ ¡Reentrenamiento completado exitosamente!")
    print(f"El modelo ahora usa {model_info['n_features']} características astronómicas")

if __name__ == "__main__":
    main()