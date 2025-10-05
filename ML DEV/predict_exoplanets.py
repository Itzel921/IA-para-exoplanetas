"""
Sistema de Predicción de Exoplanetas para Nuevos Datasets
Usa el modelo entrenado para predecir en archivos CSV de la carpeta new_datasets
Exporta resultados a exoPlanet_results
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
from train_ensemble import StackingEnsemble, DataPreprocessor, FeatureEngineer
from Clasification import DataLoader

class ExoplanetPredictor:
    """
    Predictor de exoplanetas para nuevos datasets
    """
    
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.models_path = self.project_root / "models"
        self.new_datasets_path = self.project_root / "data" / "new_datasets"
        self.results_path = self.project_root / "exoPlanet_results"
        
        # Crear directorios si no existen
        self.results_path.mkdir(exist_ok=True)
        
        self.model_info = None
        self.loaded_model_path = None
    
    def load_latest_model(self):
        """Carga el modelo más reciente"""
        try:
            model_files = list(self.models_path.glob("exoplanet_ensemble_*.pkl"))
            if not model_files:
                raise FileNotFoundError("No se encontraron modelos entrenados")
            
            # Obtener el modelo más reciente
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            print(f"📦 Cargando modelo: {latest_model.name}")
            self.model_info = joblib.load(latest_model)
            self.loaded_model_path = latest_model
            
            print(f"✅ Modelo cargado exitosamente!")
            print(f"   • Accuracy entrenamiento: {self.model_info['accuracy']:.4f}")
            print(f"   • Datasets de entrenamiento: {self.model_info['training_datasets']}")
            print(f"   • Características comunes: {len(self.model_info['common_features'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def load_new_dataset(self, filename):
        """Carga un nuevo dataset para predicción"""
        file_path = self.new_datasets_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        print(f"📁 Cargando nuevo dataset: {filename}")
        
        try:
            # Intentar cargar con diferentes métodos (similar al DataLoader)
            df = pd.read_csv(file_path, 
                           comment='#',          # Ignorar comentarios NASA
                           sep=',',              
                           engine='python')
            
            print(f"✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
            return df
            
        except Exception as e:
            print(f"❌ Error cargando {filename}: {e}")
            return None
    
    def preprocess_new_dataset(self, df, dataset_filename):
        """
        Preprocesa el nuevo dataset usando los mismos pasos del entrenamiento
        """
        print(f"🔄 Preprocesando dataset para predicción...")
        
        try:
            # Usar el preprocessor entrenado
            preprocessor = self.model_info['preprocessor']
            
            # Determinar tipo de dataset basado en columnas
            if 'koi_disposition' in df.columns or 'kepoi_name' in df.columns:
                dataset_type = 'KOI'
            elif 'tfopwg_disp' in df.columns or 'toi' in dataset_filename.lower():
                dataset_type = 'TOI'  
            elif 'archive_disp' in df.columns or 'k2' in dataset_filename.lower():
                dataset_type = 'K2'
            else:
                dataset_type = 'UNKNOWN'
                print(f"⚠️ Tipo de dataset no identificado, usando procesamiento genérico")
            
            # Aplicar feature engineering
            df_engineered = preprocessor.feature_engineer.create_astronomical_features(df, dataset_type)
            
            # Identificar columnas disponibles vs características del modelo
            model_features = self.model_info['common_features']
            available_features = [col for col in model_features if col in df_engineered.columns]
            missing_features = [col for col in model_features if col not in df_engineered.columns]
            
            print(f"   • Características del modelo: {len(model_features)}")
            print(f"   • Características disponibles: {len(available_features)}")
            print(f"   • Características faltantes: {len(missing_features)}")
            
            if len(missing_features) > 0:
                print(f"⚠️ Características faltantes: {missing_features[:5]}...")
            
            # Crear matriz de características con todas las columnas necesarias
            X_pred = pd.DataFrame(index=df_engineered.index, columns=model_features)
            
            # Llenar características disponibles
            for col in available_features:
                X_pred[col] = df_engineered[col]
            
            # Imputar características faltantes con mediana (estrategia conservadora)
            X_pred = X_pred.fillna(X_pred.median())
            X_pred = X_pred.fillna(0)  # Para casos extremos
            
            # Aplicar mismo escalado que en entrenamiento
            numeric_cols = X_pred.select_dtypes(include=[np.number]).columns.tolist()
            
            # Intentar aplicar scaler si está disponible
            if dataset_type in preprocessor.scalers:
                scaler = preprocessor.scalers[dataset_type]
                X_pred[numeric_cols] = scaler.transform(X_pred[numeric_cols])
            else:
                print(f"⚠️ Scaler para {dataset_type} no disponible, usando normalización básica")
                # Normalización básica
                for col in numeric_cols:
                    if X_pred[col].std() > 0:
                        X_pred[col] = (X_pred[col] - X_pred[col].mean()) / X_pred[col].std()
            
            print(f"✅ Preprocesamiento completado: {X_pred.shape}")
            
            return X_pred, available_features, missing_features
            
        except Exception as e:
            print(f"❌ Error en preprocesamiento: {e}")
            return None, [], []
    
    def predict_dataset(self, filename, confidence_threshold=0.5):
        """
        Realiza predicciones en un nuevo dataset
        """
        print(f"\n🔮 PREDICIENDO EXOPLANETAS EN: {filename}")
        print("="*60)
        
        # Cargar dataset
        df = self.load_new_dataset(filename)
        if df is None:
            return None
        
        # Preprocesamiento  
        X_pred, available_features, missing_features = self.preprocess_new_dataset(df, filename)
        if X_pred is None:
            return None
        
        # Realizar predicciones
        print(f"🎯 Realizando predicciones...")
        
        model = self.model_info['model']
        
        try:
            # Predicciones de probabilidad
            y_proba = model.predict_proba(X_pred)[:, 1]  # Probabilidad de ser exoplaneta
            
            # Predicciones binarias
            y_pred = (y_proba >= confidence_threshold).astype(int)
            
            # Crear DataFrame de resultados
            results_df = df.copy()
            results_df['ML_Probability'] = y_proba
            results_df['ML_Prediction'] = y_pred
            results_df['ML_Confidence'] = np.where(y_pred == 1, y_proba, 1 - y_proba)
            results_df['ML_Classification'] = np.where(y_pred == 1, 'CONFIRMED', 'NOT_CONFIRMED')
            
            # Estadísticas de predicción
            n_confirmed = (y_pred == 1).sum()
            n_total = len(y_pred)
            confirmed_pct = (n_confirmed / n_total) * 100
            avg_confidence = results_df['ML_Confidence'].mean()
            
            print(f"\n📊 RESULTADOS DE PREDICCIÓN:")
            print(f"   • Total de objetos analizados: {n_total:,}")
            print(f"   • Exoplanetas confirmados: {n_confirmed:,} ({confirmed_pct:.1f}%)")
            print(f"   • Candidatos/False Positives: {n_total - n_confirmed:,} ({100-confirmed_pct:.1f}%)")
            print(f"   • Confianza promedio: {avg_confidence:.3f}")
            
            # Distribución de confianza
            high_conf = (results_df['ML_Confidence'] >= 0.8).sum()
            medium_conf = ((results_df['ML_Confidence'] >= 0.6) & (results_df['ML_Confidence'] < 0.8)).sum()
            low_conf = (results_df['ML_Confidence'] < 0.6).sum()
            
            print(f"\n📈 Distribución de confianza:")
            print(f"   • Alta confianza (≥0.8): {high_conf:,} ({high_conf/n_total*100:.1f}%)")
            print(f"   • Confianza media (0.6-0.8): {medium_conf:,} ({medium_conf/n_total*100:.1f}%)")
            print(f"   • Baja confianza (<0.6): {low_conf:,} ({low_conf/n_total*100:.1f}%)")
            
            # Exportar resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = Path(filename).stem
            output_filename = f"{base_filename}_predictions_{timestamp}.csv"
            output_path = self.results_path / output_filename
            
            results_df.to_csv(output_path, index=False)
            
            print(f"\n💾 Resultados exportados a: {output_filename}")
            
            # Crear resumen ejecutivo
            summary = {
                'input_file': filename,
                'output_file': output_filename,
                'timestamp': timestamp,
                'total_objects': n_total,
                'confirmed_exoplanets': int(n_confirmed),
                'confirmed_percentage': round(confirmed_pct, 2),
                'average_confidence': round(avg_confidence, 3),
                'model_accuracy': round(self.model_info['accuracy'], 4),
                'features_available': len(available_features),
                'features_missing': len(missing_features),
                'high_confidence_predictions': int(high_conf)
            }
            
            # Guardar resumen
            summary_path = self.results_path / f"{base_filename}_summary_{timestamp}.json"
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"📋 Resumen guardado en: {summary_path.name}")
            print(f"\n✅ PREDICCIÓN COMPLETADA EXITOSAMENTE!")
            
            return results_df, summary
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return None, None
    
    def process_all_new_datasets(self):
        """Procesa todos los datasets en la carpeta new_datasets"""
        print(f"\n🔍 Buscando nuevos datasets en: {self.new_datasets_path}")
        
        csv_files = list(self.new_datasets_path.glob("*.csv"))
        
        if not csv_files:
            print(f"📂 No se encontraron archivos CSV en new_datasets")
            print(f"   Coloca archivos CSV en: {self.new_datasets_path}")
            return
        
        print(f"📁 Encontrados {len(csv_files)} archivos CSV:")
        for file in csv_files:
            print(f"   • {file.name}")
        
        # Procesar cada archivo
        results = {}
        for csv_file in csv_files:
            try:
                result_df, summary = self.predict_dataset(csv_file.name)
                if result_df is not None:
                    results[csv_file.name] = summary
                    
            except Exception as e:
                print(f"❌ Error procesando {csv_file.name}: {e}")
        
        # Resumen final
        if results:
            print(f"\n🎉 PROCESAMIENTO COMPLETADO!")
            print(f"   • Archivos procesados: {len(results)}")
            print(f"   • Resultados en: {self.results_path}")
        
        return results

def main():
    """Función principal del script"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Inicializar predictor
    predictor = ExoplanetPredictor(project_root)
    
    # Cargar modelo
    if not predictor.load_latest_model():
        print("❌ No se pudo cargar el modelo. Primero ejecuta train_ensemble.py")
        return
    
    # Si se proporciona un archivo específico como argumento
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print(f"🎯 Prediciendo archivo específico: {filename}")
        predictor.predict_dataset(filename)
    else:
        # Procesar todos los archivos en new_datasets
        predictor.process_all_new_datasets()

if __name__ == "__main__":
    main()