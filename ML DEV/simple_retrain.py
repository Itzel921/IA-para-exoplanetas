"""
Reentrenamiento Simplificado - Versión Rápida
Soluciona el problema de características con un enfoque más simple y eficiente
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# Importar clases necesarias
import model_imports
from Clasification import DataLoader

class SimpleTrainer:
    """
    Reentrenador simplificado que usa solo Random Forest
    Más rápido y menos propenso a errores
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
    
    def load_datasets(self):
        """Cargar datasets usando el DataLoader existente"""
        print("🚀 Iniciando reentrenamiento simplificado")
        print("="*60)
        
        # Cargar datasets
        data_loader = DataLoader(str(self.project_root))
        self.datasets = data_loader.load_all_datasets()
        
        print(f"✅ Datasets cargados:")
        for name, df in self.datasets.items():
            print(f"   • {name}: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    
    def create_unified_features(self):
        """Crear características unificadas de manera simple"""
        print(f"\\n🔧 Creando características astronómicas unificadas...")
        
        unified_datasets = []
        
        for dataset_name, df in self.datasets.items():
            print(f"\\n📊 Procesando {dataset_name}...")
            
            # Crear DataFrame unificado con características básicas
            unified_df = pd.DataFrame()
            
            # Mapear características según el dataset
            if dataset_name == 'KOI':
                # Coordenadas (siempre disponibles)
                unified_df['ra'] = df.get('ra', 0.0)
                unified_df['dec'] = df.get('dec', 0.0)
                
                # Parámetros astronómicos clave
                unified_df['period'] = df.get('koi_period', 0.0)
                unified_df['radius'] = df.get('koi_prad', 0.0)
                unified_df['depth'] = df.get('koi_depth', 0.0)
                unified_df['duration'] = df.get('koi_duration', 0.0)
                unified_df['temp'] = df.get('koi_teq', 0.0)
                unified_df['star_radius'] = df.get('koi_srad', 1.0)
                unified_df['star_temp'] = df.get('koi_steff', 5778.0)
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
                
                # Parámetros astronómicos
                unified_df['period'] = df.get('pl_orbper', 0.0)
                unified_df['radius'] = df.get('pl_rade', 0.0)
                unified_df['depth'] = df.get('pl_trandep', 0.0)
                unified_df['duration'] = df.get('pl_trandurh', 0.0)
                unified_df['temp'] = df.get('pl_eqt', 0.0)
                unified_df['star_radius'] = df.get('st_rad', 1.0)
                unified_df['star_temp'] = df.get('st_teff', 5778.0)
                unified_df['snr'] = 10.0  # Valor por defecto
                unified_df['impact'] = 0.5  # Valor por defecto
                
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
                
                # Parámetros astronómicos
                unified_df['period'] = df.get('pl_orbper', 0.0)
                unified_df['radius'] = df.get('pl_rade', 0.0)
                unified_df['depth'] = 100.0  # Valor aproximado por defecto
                unified_df['duration'] = 3.0   # Valor aproximado por defecto
                unified_df['temp'] = df.get('pl_eqt', 0.0)
                unified_df['star_radius'] = df.get('st_rad', 1.0)
                unified_df['star_temp'] = df.get('st_teff', 5778.0)
                unified_df['snr'] = 15.0  # Valor por defecto
                unified_df['impact'] = 0.3  # Valor por defecto
                
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
            
            # Filtrar valores extremos
            for col in ['period', 'radius', 'temp', 'star_radius', 'star_temp']:
                if col in unified_df.columns:
                    # Winsorizing básico (1%-99%)
                    q1 = unified_df[col].quantile(0.01)
                    q99 = unified_df[col].quantile(0.99)
                    unified_df[col] = unified_df[col].clip(q1, q99)
            
            print(f"   • Objetos procesados: {len(unified_df):,}")
            print(f"   • Confirmados: {(unified_df['target'] == 1).sum():,}")
            print(f"   • No confirmados: {(unified_df['target'] == 0).sum():,}")
            
            unified_datasets.append(unified_df)
        
        # Combinar todos los datasets
        self.combined_data = pd.concat(unified_datasets, ignore_index=True)
        
        print(f"\\n✅ Dataset unificado creado:")
        print(f"   • Total objetos: {len(self.combined_data):,}")
        print(f"   • Total confirmados: {(self.combined_data['target'] == 1).sum():,} ({(self.combined_data['target'] == 1).mean()*100:.1f}%)")
        print(f"   • Características base: 11")
        
        return self.combined_data
    
    def add_derived_features(self, df):
        """Agregar características derivadas simples"""
        print(f"\\n🔬 Creando características derivadas...")
        
        df_enhanced = df.copy()
        
        # Ratios físicos seguros
        df_enhanced['planet_star_ratio'] = np.where(
            df['star_radius'] > 0,
            df['radius'] / df['star_radius'],
            0.01  # Valor por defecto pequeño
        )
        
        df_enhanced['temp_ratio'] = np.where(
            df['star_temp'] > 0,
            df['temp'] / df['star_temp'],
            0.5  # Valor por defecto
        )
        
        # Logaritmos para normalizar distribuciones
        df_enhanced['log_period'] = np.log10(df['period'] + 1)
        df_enhanced['log_radius'] = np.log10(df['radius'] + 0.1)
        
        # Índice de habitabilidad simple
        df_enhanced['habitability_index'] = np.where(
            (df['temp'] >= 200) & (df['temp'] <= 400),
            1.0,  # Zona habitable
            0.0   # Fuera de zona habitable
        )
        
        print(f"   • Características derivadas agregadas: 5")
        print(f"   • Total características: {df_enhanced.shape[1] - 2}")  # -2 por target y source
        
        return df_enhanced
    
    def train_simple_model(self):
        """Entrenar modelo simple usando solo Random Forest"""
        print(f"\\n🎯 Entrenando modelo Random Forest...")
        
        # Preparar datos
        feature_cols = [col for col in self.combined_data.columns 
                       if col not in ['target', 'source']]
        
        X = self.combined_data[feature_cols]
        y = self.combined_data['target']
        
        print(f"   • Características de entrada: {len(feature_cols)}")
        print(f"   • Muestras de entrenamiento: {len(X):,}")
        
        # Verificar distribución de clases
        class_dist = y.value_counts(normalize=True)
        print(f"   • Distribución de clases:")
        print(f"     - No exoplanetas: {class_dist[0]*100:.1f}%")
        print(f"     - Exoplanetas: {class_dist[1]*100:.1f}%")
        
        # Escalar características
        print(f"\\n📐 Escalando características...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Split train/test estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   • Training set: {len(X_train):,}")
        print(f"   • Test set: {len(X_test):,}")
        
        # Definir modelo simple pero efectivo
        print(f"\\n🌳 Entrenando Random Forest...")
        model = RandomForestClassifier(
            n_estimators=200,      # Menos árboles para velocidad
            max_depth=15,          # Limitar profundidad
            min_samples_split=5,   # Evitar overfitting
            min_samples_leaf=2,    # Evitar overfitting
            random_state=42,
            n_jobs=-1              # Usar todos los cores
        )
        
        # Entrenar
        model.fit(X_train, y_train)
        
        # Evaluar
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"\\n📊 Resultados del entrenamiento:")
        print(f"   • Accuracy (train): {train_score:.4f} ({train_score*100:.1f}%)")
        print(f"   • Accuracy (test): {test_score:.4f} ({test_score*100:.1f}%)")
        
        # Predicciones detalladas
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\\n🎯 Métricas detalladas:")
        print(f"   • True Positives (planetas detectados): {tp}")
        print(f"   • True Negatives (no-planetas correctos): {tn}")
        print(f"   • False Positives (falsos descubrimientos): {fp}")
        print(f"   • False Negatives (planetas perdidos): {fn}")
        
        # Calcular métricas astronómicas
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\\n🔬 Métricas astronómicas:")
        print(f"   • Precisión (reliability): {precision:.3f} ({precision*100:.1f}%)")
        print(f"   • Recall (completeness): {recall:.3f} ({recall*100:.1f}%)")
        print(f"   • F1-Score: {f1:.3f} ({f1*100:.1f}%)")
        
        # Importancia de características
        feature_importance = model.feature_importances_
        important_features = sorted(zip(feature_cols, feature_importance), 
                                   key=lambda x: x[1], reverse=True)
        
        print(f"\\n🔍 Características más importantes (Top 5):")
        for i, (feature, importance) in enumerate(important_features[:5]):
            print(f"   {i+1}. {feature}: {importance:.3f} ({importance*100:.1f}%)")
        
        # Guardar modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_info = {
            'model': model,
            'scaler': self.scaler,
            'feature_names': feature_cols,
            'common_features': feature_cols,  # Todas las características
            'accuracy': test_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'trained_on': timestamp,
            'n_features': len(feature_cols),
            'n_samples': len(X),
            'datasets_used': list(self.datasets.keys()),
            'model_type': 'Random Forest (Simplified)',
            'feature_importance': dict(important_features)
        }
        
        model_filename = f"exoplanet_simple_{timestamp}.pkl"
        model_path = self.trained_models_path / model_filename
        
        joblib.dump(model_info, model_path)
        
        print(f"\\n💾 Modelo guardado exitosamente:")
        print(f"   • Archivo: {model_filename}")
        print(f"   • Ubicación: trained_models/")
        print(f"   • Características: {len(feature_cols)}")
        print(f"   • Accuracy: {test_score:.4f} ({test_score*100:.1f}%)")
        print(f"   • Tamaño: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        return model_info

def main():
    """Función principal de reentrenamiento simplificado"""
    trainer = SimpleTrainer()
    
    try:
        # Cargar datasets
        trainer.load_datasets()
        
        # Crear características unificadas
        unified_data = trainer.create_unified_features()
        
        # Agregar características derivadas
        trainer.combined_data = trainer.add_derived_features(unified_data)
        
        # Entrenar modelo
        model_info = trainer.train_simple_model()
        
        print(f"\\n✅ ¡Reentrenamiento simplificado completado exitosamente!")
        print(f"\\n🎉 RESUMEN FINAL:")
        print(f"   • Modelo: Random Forest")
        print(f"   • Características: {model_info['n_features']}")
        print(f"   • Accuracy: {model_info['accuracy']:.4f} ({model_info['accuracy']*100:.1f}%)")
        print(f"   • Precision: {model_info['precision']:.3f}")
        print(f"   • Recall: {model_info['recall']:.3f}")
        print(f"   • F1-Score: {model_info['f1_score']:.3f}")
        print(f"\\nEl nuevo modelo está listo para hacer predicciones astronómicamente útiles! 🚀")
        
        return model_info
        
    except Exception as e:
        print(f"❌ Error durante el reentrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model_info = main()
    if model_info:
        print("\\n🔮 Para usar el nuevo modelo, ejecuta:")
        print("   python fixed_predictor.py <archivo.csv>")