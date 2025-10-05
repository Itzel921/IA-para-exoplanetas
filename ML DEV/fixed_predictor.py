"""
Predictor Corregido para Exoplanetas
Soluciona el problema de compatibilidad de características
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar clases necesarias
import model_imports
try:
    from train_ensemble import StackingEnsemble, DataPreprocessor, FeatureEngineer
except ImportError as e:
    print(f"⚠️ Advertencia de importación: {e}")

from Clasification import DataLoader

class FixedExoplanetPredictor:
    """
    Predictor corregido que maneja la incompatibilidad de características
    """
    
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.ml_dev_path = Path(__file__).parent
        self.models_path_new = self.ml_dev_path / "trained_models"
        self.models_path_legacy = self.project_root / "models"
        self.new_datasets_path = self.project_root / "data" / "new_datasets"
        self.results_path = self.project_root / "exoPlanet_results"
        
        self.results_path.mkdir(exist_ok=True)
        
        self.model_info = None
        self.loaded_model_path = None
        
    def load_latest_model(self):
        """Carga el modelo más reciente con manejo de errores mejorado"""
        try:
            # Buscar modelos en ambas ubicaciones
            model_files_new = list(self.models_path_new.glob("exoplanet_ensemble_*.pkl")) if self.models_path_new.exists() else []
            model_files_legacy = list(self.models_path_legacy.glob("exoplanet_ensemble_*.pkl")) if self.models_path_legacy.exists() else []
            
            all_model_files = model_files_new + model_files_legacy
            
            if not all_model_files:
                print("❌ No se encontraron modelos entrenados")
                return False
            
            # Obtener el modelo más reciente
            latest_model = max(all_model_files, key=lambda x: x.stat().st_mtime)
            
            print(f"📦 Cargando modelo: {latest_model.name}")
            self.model_info = joblib.load(latest_model)
            self.loaded_model_path = latest_model
            
            print(f"✅ Modelo cargado exitosamente!")
            print(f"   • Accuracy: {self.model_info['accuracy']:.4f}")
            
            # Mostrar características del modelo
            model_features = self.model_info.get('common_features', [])
            print(f"   • Características del modelo: {len(model_features)}")
            if len(model_features) <= 10:
                print(f"   • Features: {model_features}")
            else:
                print(f"   • Features: {model_features[:5]} ... (+{len(model_features)-5} más)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_new_dataset(self, filename):
        """Carga nuevo dataset para predicción"""
        file_path = self.new_datasets_path / filename
        
        if not file_path.exists():
            print(f"❌ Archivo no encontrado: {file_path}")
            return None
        
        try:
            df = pd.read_csv(file_path, comment='#')
            print(f"✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
            return df
            
        except Exception as e:
            print(f"❌ Error cargando {filename}: {e}")
            return None
    
    def create_minimal_features(self, df):
        """
        Crea un conjunto mínimo de características compatibles con el modelo
        Basado en las características que realmente tiene el modelo
        """
        print(f"🔧 Creando características compatibles...")
        
        try:
            model_features = self.model_info.get('common_features', [])
            
            # Crear DataFrame con las características esperadas por el modelo
            X_pred = pd.DataFrame(index=df.index)
            
            # Mapear características disponibles a las esperadas por el modelo
            feature_mapping = {
                # Coordenadas astronómicas (siempre disponibles)
                'ra': 'ra',
                'dec': 'dec',
            }
            
            # Agregar características disponibles
            available_count = 0
            for model_feature in model_features:
                if model_feature in df.columns:
                    X_pred[model_feature] = df[model_feature]
                    available_count += 1
                elif model_feature in feature_mapping and feature_mapping[model_feature] in df.columns:
                    X_pred[model_feature] = df[feature_mapping[model_feature]]
                    available_count += 1
                else:
                    # Imputar con valor por defecto
                    if model_feature == 'ra':
                        X_pred[model_feature] = df.get('ra', 0.0)
                    elif model_feature == 'dec':
                        X_pred[model_feature] = df.get('dec', 0.0)
                    else:
                        X_pred[model_feature] = 0.0  # Valor por defecto
            
            # Llenar valores faltantes
            X_pred = X_pred.fillna(0.0)
            
            print(f"   • Características del modelo: {len(model_features)}")
            print(f"   • Características mapeadas: {available_count}")
            print(f"   • Características imputadas: {len(model_features) - available_count}")
            print(f"   • Shape final: {X_pred.shape}")
            
            return X_pred
            
        except Exception as e:
            print(f"❌ Error en feature mapping: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_dataset(self, filename, confidence_threshold=0.5):
        """Realiza predicciones usando el enfoque corregido"""
        print(f"\\n🔮 PREDICCIÓN CORREGIDA - {filename}")
        print("="*60)
        
        # Cargar dataset
        df = self.load_new_dataset(filename)
        if df is None:
            return None
        
        # Crear características compatibles
        X_pred = self.create_minimal_features(df)
        if X_pred is None:
            return None
        
        # Realizar predicciones
        print(f"🎯 Realizando predicciones...")
        
        try:
            model = self.model_info['model']
            
            # Predicciones
            y_proba = model.predict_proba(X_pred)[:, 1]
            y_pred = (y_proba >= confidence_threshold).astype(int)
            
            # Crear resultados
            results_df = df.copy()
            results_df['ML_Probability'] = y_proba
            results_df['ML_Prediction'] = y_pred
            results_df['ML_Confidence'] = np.where(y_pred == 1, y_proba, 1 - y_proba)
            results_df['ML_Classification'] = np.where(y_pred == 1, 'EXOPLANET', 'NOT_EXOPLANET')
            
            # Estadísticas
            n_confirmed = (y_pred == 1).sum()
            n_total = len(y_pred)
            confirmed_pct = (n_confirmed / n_total) * 100
            avg_confidence = results_df['ML_Confidence'].mean()
            
            print(f"\\n📊 RESULTADOS:")
            print(f"   • Total objetos: {n_total:,}")
            print(f"   • Exoplanetas detectados: {n_confirmed:,} ({confirmed_pct:.1f}%)")
            print(f"   • No exoplanetas: {n_total - n_confirmed:,} ({100-confirmed_pct:.1f}%)")
            print(f"   • Confianza promedio: {avg_confidence:.3f}")
            
            # Distribución de confianza
            high_conf = (results_df['ML_Confidence'] >= 0.8).sum()
            medium_conf = ((results_df['ML_Confidence'] >= 0.6) & (results_df['ML_Confidence'] < 0.8)).sum()
            low_conf = (results_df['ML_Confidence'] < 0.6).sum()
            
            print(f"\\n📈 Confianza:")
            print(f"   • Alta (≥0.8): {high_conf:,} ({high_conf/n_total*100:.1f}%)")
            print(f"   • Media (0.6-0.8): {medium_conf:,} ({medium_conf/n_total*100:.1f}%)")
            print(f"   • Baja (<0.6): {low_conf:,} ({low_conf/n_total*100:.1f}%)")
            
            # Guardar resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = Path(filename).stem
            output_filename = f"{base_filename}_FIXED_predictions_{timestamp}.csv"
            output_path = self.results_path / output_filename
            
            results_df.to_csv(output_path, index=False)
            print(f"\\n💾 Resultados guardados: {output_filename}")
            
            return results_df
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Función principal para probar el predictor corregido"""
    if len(sys.argv) != 2:
        print("Uso: python fixed_predictor.py <archivo.csv>")
        
        # Mostrar archivos disponibles
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        new_datasets_path = project_root / "data" / "new_datasets"
        
        if new_datasets_path.exists():
            csv_files = list(new_datasets_path.glob("*.csv"))
            if csv_files:
                print(f"\\nArchivos disponibles:")
                for i, file in enumerate(csv_files, 1):
                    print(f"   {i}. {file.name}")
        return
    
    filename = sys.argv[1]
    
    # Crear predictor
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    predictor = FixedExoplanetPredictor(project_root)
    
    # Cargar modelo
    if not predictor.load_latest_model():
        print("❌ No se pudo cargar el modelo")
        return
    
    # Realizar predicción
    results = predictor.predict_dataset(filename)
    
    if results is not None:
        print("\\n✅ ¡Predicción completada exitosamente!")
    else:
        print("\\n❌ Error en la predicción")

if __name__ == "__main__":
    main()