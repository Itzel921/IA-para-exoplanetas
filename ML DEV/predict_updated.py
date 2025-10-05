"""
Sistema de Predicción ACTUALIZADO - Usa el nuevo modelo corregido
Compatible con el mapeo inteligente de características y el modelo FAST
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar clases necesarias del entrenamiento
import model_imports  # Esto asegura compatibilidad con todos los modelos
from train_ensemble_FAST import FastStackingEnsemble
from train_ensemble_CORRECTED import FeatureMapper, ImprovedDataPreprocessor

class UpdatedExoplanetPredictor:
    """
    Predictor actualizado que usa el nuevo modelo corregido
    """
    
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.models_path = self.project_root / "models"
        self.new_datasets_path = self.project_root / "data" / "new_datasets"
        self.results_path = self.project_root / "exoPlanet_results"
        
        # Crear carpeta de resultados si no existe
        self.results_path.mkdir(exist_ok=True)
        
        self.model_info = None
        self.loaded_model_path = None
    
    def load_latest_model(self):
        """Carga el modelo más reciente (preferentemente el FAST)"""
        try:
            # Buscar modelos FAST primero
            fast_models = list(self.models_path.glob("exoplanet_ensemble_FAST_*.pkl"))
            corrected_models = list(self.models_path.glob("exoplanet_ensemble_CORRECTED_*.pkl"))
            all_models = list(self.models_path.glob("exoplanet_ensemble_*.pkl"))
            
            # Priorizar modelos FAST, luego CORRECTED, luego cualquier otro
            if fast_models:
                model_files = fast_models
                print("🚀 Usando modelo FAST (optimizado)")
            elif corrected_models:
                model_files = corrected_models
                print("🔧 Usando modelo CORRECTED")
            elif all_models:
                model_files = all_models
                print("⚠️ Usando modelo legacy")
            else:
                raise FileNotFoundError("No se encontraron modelos entrenados")
            
            # Obtener el modelo más reciente
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            print(f"📦 Cargando modelo: {latest_model.name}")
            self.model_info = joblib.load(latest_model)
            self.loaded_model_path = latest_model
            
            print(f"✅ Modelo cargado exitosamente!")
            print(f"   • Accuracy: {self.model_info['accuracy']:.4f}")
            print(f"   • Datasets: {self.model_info['training_datasets']}")
            print(f"   • Características: {len(self.model_info['feature_names'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def predict_dataset(self, filename, confidence_threshold=0.5):
        """Predice exoplanetas en un nuevo dataset"""
        if self.model_info is None:
            print("❌ Primero debes cargar un modelo")
            return None, None
        
        print(f"\n🔮 PREDICIENDO EXOPLANETAS EN: {filename}")
        print("="*60)
        
        try:
            # 1. Cargar dataset
            file_path = self.new_datasets_path / filename
            print(f"📁 Cargando dataset: {filename}")
            
            # Intentar cargar con comment='#' para archivos de NASA
            try:
                df = pd.read_csv(file_path, comment='#', low_memory=False)
            except:
                df = pd.read_csv(file_path, low_memory=False)
                
            print(f"✅ Dataset cargado: {len(df):,} filas × {len(df.columns)} columnas")
            
            # 2. Preprocesar con el nuevo sistema
            print(f"🔄 Preprocesando dataset...")
            
            # Usar el preprocessor del modelo cargado
            preprocessor = self.model_info['preprocessor']
            
            # Determinar tipo de dataset (asumimos KOI por el nombre del archivo)
            dataset_type = 'KOI'  # Puede refinarse basado en el nombre del archivo
            
            # Aplicar transformación
            X_pred = preprocessor.transform(df, dataset_type)
            print(f"✅ Preprocesamiento completado")
            print(f"   • Características del modelo: {len(self.model_info['feature_names'])}")
            print(f"   • Características procesadas: {X_pred.shape[1]}")
            
            # 3. Realizar predicciones
            print(f"🧠 Ejecutando predicciones...")
            model = self.model_info['model']
            
            y_proba = model.predict_proba(X_pred)[:, 1]  # Probabilidad de ser exoplaneta
            y_pred = (y_proba >= confidence_threshold).astype(int)
            
            print(f"✅ Predicciones completadas!")
            
            # 4. Preparar resultados
            results_df = df.copy()
            
            # Agregar columnas de ML
            results_df['ML_Probability'] = y_proba
            results_df['ML_Prediction'] = y_pred
            results_df['ML_Confidence'] = np.maximum(y_proba, 1 - y_proba)
            results_df['ML_Classification'] = ['CONFIRMED' if pred == 1 else 'NOT_CONFIRMED' for pred in y_pred]
            
            # 5. Guardar resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = Path(filename).stem
            output_filename = f"{base_filename}_predictions_{timestamp}.csv"
            output_path = self.results_path / output_filename
            
            results_df.to_csv(output_path, index=False)
            
            # 6. Generar estadísticas
            total_objects = len(results_df)
            predicted_planets = int(np.sum(y_pred))
            predicted_non_planets = total_objects - predicted_planets
            high_conf = int(np.sum(results_df['ML_Confidence'] >= 0.8))
            avg_confidence = np.mean(results_df['ML_Confidence'])
            
            print(f"\n📊 RESULTADOS DE PREDICCIÓN")
            print("="*40)
            print(f"🌟 Exoplanetas predichos: {predicted_planets:,}")
            print(f"🚫 No-planetas predichos: {predicted_non_planets:,}")
            print(f"🎯 Confianza promedio: {avg_confidence:.1%}")
            print(f"🔥 Predicciones alta confianza (>80%): {high_conf:,}")
            print(f"💾 Resultados guardados: {output_filename}")
            
            # Estadísticas adicionales
            if predicted_planets > 0:
                planet_confidences = results_df[results_df['ML_Prediction'] == 1]['ML_Confidence']
                print(f"🌟 Confianza promedio planetas: {np.mean(planet_confidences):.1%}")
                top_candidates = results_df.nlargest(5, 'ML_Probability')[['kepoi_name', 'ML_Probability', 'ML_Confidence']] if 'kepoi_name' in results_df.columns else None
                if top_candidates is not None:
                    print(f"\n🏆 TOP 5 CANDIDATOS:")
                    for idx, row in top_candidates.iterrows():
                        print(f"   • {row['kepoi_name']}: {row['ML_Probability']:.1%} (conf: {row['ML_Confidence']:.1%})")
            
            summary = {
                'total_objects': total_objects,
                'predicted_planets': predicted_planets,
                'predicted_non_planets': predicted_non_planets,
                'avg_confidence': round(avg_confidence, 4),
                'high_confidence_predictions': high_conf,
                'model_accuracy': round(self.model_info['accuracy'], 4),
                'output_file': str(output_filename)
            }
            
            return results_df, summary
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """
    Función principal para probar predicciones
    """
    print("🔮 SISTEMA DE PREDICCIÓN ACTUALIZADO")
    print("🚀 Usando nuevo modelo corregido con mapeo inteligente")
    print("="*60)
    
    # Inicializar predictor
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    predictor = UpdatedExoplanetPredictor(project_root)
    
    # Cargar modelo
    if not predictor.load_latest_model():
        print("❌ No se pudo cargar el modelo. Ejecuta primero train_ensemble_FAST.py")
        return
    
    # Listar archivos disponibles
    csv_files = list(predictor.new_datasets_path.glob("*.csv"))
    if not csv_files:
        print("❌ No hay archivos CSV en new_datasets")
        return
    
    print(f"\n📁 Archivos disponibles:")
    for i, file in enumerate(csv_files, 1):
        print(f"   {i}. {file.name}")
    
    # Procesar todos los archivos
    print(f"\n🔄 Procesando todos los archivos...")
    for csv_file in csv_files:
        result_df, summary = predictor.predict_dataset(csv_file.name)
        if result_df is not None:
            print(f"✅ {csv_file.name} procesado exitosamente")
        else:
            print(f"❌ Error procesando {csv_file.name}")

if __name__ == "__main__":
    main()