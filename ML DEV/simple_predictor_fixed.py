"""
Predictor corregido para el modelo simplificado
Compatible con las características limitadas del modelo actual
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimplePredictorFixed:
    """
    Predictor específico para el modelo simplificado que solo usa 2 características (ra, dec)
    """
    
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.models_path = self.project_root / "models"
        self.new_datasets_path = self.project_root / "data" / "new_datasets"
        self.results_path = self.project_root / "exoPlanet_results"
        
        # Crear directorios si no existen
        self.results_path.mkdir(exist_ok=True)
        
        self.model = None
        self.model_features = ['ra', 'dec']  # Solo las 2 características que usa el modelo actual
    
    def load_model(self):
        """Carga el modelo más reciente"""
        try:
            model_files = list(self.models_path.glob("exoplanet_ensemble_*.pkl"))
            
            if not model_files:
                print("❌ No se encontraron modelos entrenados")
                return False
            
            # Obtener el modelo más reciente
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            print(f"📦 Cargando modelo: {latest_model.name}")
            
            # Cargar modelo
            loaded_data = joblib.load(latest_model)
            
            # El modelo puede estar guardado como dict o como objeto directo
            if isinstance(loaded_data, dict):
                # Si es un diccionario, extraer el modelo
                if 'model' in loaded_data:
                    self.model = loaded_data['model']
                elif 'ensemble' in loaded_data:
                    self.model = loaded_data['ensemble']
                else:
                    print(f"❌ Estructura de modelo no reconocida. Claves disponibles: {loaded_data.keys()}")
                    return False
            else:
                # Si es el modelo directamente
                self.model = loaded_data
            
            print(f"✅ Modelo cargado exitosamente!")
            print(f"   • Tipo: {type(self.model).__name__}")
            
            # Verificar que tiene los métodos necesarios
            if not hasattr(self.model, 'predict') or not hasattr(self.model, 'predict_proba'):
                print(f"❌ El modelo no tiene los métodos predict/predict_proba necesarios")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_dataset(self, filename):
        """Carga un nuevo dataset para predicción"""
        file_path = self.new_datasets_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        print(f"📁 Cargando dataset: {filename}")
        
        try:
            df = pd.read_csv(file_path, comment='#', sep=',', engine='python')
            print(f"✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
            return df
            
        except Exception as e:
            print(f"❌ Error cargando {filename}: {e}")
            return None
    
    def prepare_features(self, df):
        """
        Prepara solo las características que necesita el modelo (ra, dec)
        """
        print(f"🔄 Preparando características para predicción...")
        
        # Verificar que existan las columnas necesarias
        missing_cols = [col for col in self.model_features if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Faltan columnas necesarias: {missing_cols}")
            print(f"   Columnas disponibles: {list(df.columns)}")
            return None
        
        # Extraer solo las características que usa el modelo
        X = df[self.model_features].copy()
        
        # Manejar valores faltantes
        X = X.fillna(X.median())
        
        print(f"✅ Características preparadas: {X.shape}")
        print(f"   • Columnas: {list(X.columns)}")
        
        return X
    
    def predict(self, X, original_df):
        """Hace predicciones usando el modelo cargado"""
        try:
            print(f"🔮 Generando predicciones...")
            
            # Hacer predicciones
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # Crear DataFrame de resultados
            results = original_df.copy()
            results['prediction'] = predictions
            results['prediction_label'] = ['CONFIRMED' if p == 1 else 'FALSE_POSITIVE' for p in predictions]
            results['confidence'] = probabilities.max(axis=1)
            results['prob_confirmed'] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
            results['prob_false_positive'] = probabilities[:, 0] if probabilities.shape[1] > 1 else 1 - probabilities[:, 0]
            
            print(f"✅ Predicciones completadas!")
            
            # Estadísticas
            confirmed = sum(predictions == 1)
            false_positive = sum(predictions == 0)
            
            print(f"📊 Resultados:")
            print(f"   • Planetas confirmados: {confirmed} ({confirmed/len(predictions)*100:.1f}%)")
            print(f"   • Falsos positivos: {false_positive} ({false_positive/len(predictions)*100:.1f}%)")
            
            return results
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return None
    
    def save_results(self, results, original_filename):
        """Guarda los resultados en CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(original_filename).stem
        output_filename = f"{base_name}_predictions_{timestamp}.csv"
        output_path = self.results_path / output_filename
        
        try:
            results.to_csv(output_path, index=False)
            print(f"💾 Resultados guardados en: {output_filename}")
            return output_path
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")
            return None
    
    def process_file(self, filename):
        """Procesa un archivo completo"""
        print(f"\n{'='*60}")
        print(f"🚀 Procesando: {filename}")
        print(f"{'='*60}")
        
        # Cargar dataset
        df = self.load_dataset(filename)
        if df is None:
            return False
        
        # Preparar características
        X = self.prepare_features(df)
        if X is None:
            return False
        
        # Hacer predicciones
        results = self.predict(X, df)
        if results is None:
            return False
        
        # Guardar resultados
        output_path = self.save_results(results, filename)
        if output_path is None:
            return False
        
        print(f"✅ Archivo procesado exitosamente!")
        return True
    
    def process_all_new_datasets(self):
        """Procesa todos los archivos en new_datasets"""
        csv_files = list(self.new_datasets_path.glob("*.csv"))
        
        if not csv_files:
            print("❌ No se encontraron archivos CSV en new_datasets")
            return
        
        print(f"📂 Encontrados {len(csv_files)} archivos CSV")
        
        successful = 0
        for file_path in csv_files:
            try:
                success = self.process_file(file_path.name)
                if success:
                    successful += 1
            except Exception as e:
                print(f"❌ Error procesando {file_path.name}: {e}")
        
        print(f"\n🏁 Procesamiento completado:")
        print(f"   • Archivos procesados exitosamente: {successful}/{len(csv_files)}")

def main():
    """Función principal para ejecutar el predictor"""
    import sys
    
    # Determinar directorio del proyecto
    current_dir = Path(__file__).parent
    project_root = current_dir.parent  # Subir un nivel desde ML DEV
    
    # Crear predictor
    predictor = SimplePredictorFixed(project_root)
    
    # Cargar modelo
    if not predictor.load_model():
        return
    
    # Procesar archivos
    if len(sys.argv) > 1:
        # Archivo específico
        filename = sys.argv[1]
        predictor.process_file(filename)
    else:
        # Todos los archivos
        predictor.process_all_new_datasets()

if __name__ == "__main__":
    main()