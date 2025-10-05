"""
Servicio de Predicción de Exoplanetas
Integra el modelo ML existente con la API FastAPI
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Importar el predictor existente
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "ML DEV"))

try:
    from predict_exoplanets import ExoplanetPredictor
    import model_imports  # Para asegurar compatibilidad con modelos
except ImportError as e:
    logging.error(f"Error importando predictor: {e}")
    ExoplanetPredictor = None

logger = logging.getLogger(__name__)

class PredictionService:
    """
    Servicio de predicción que encapsula el modelo ML
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.predictor = None
        self.model_info = None
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Inicializar el servicio cargando el modelo"""
        try:
            if ExoplanetPredictor is None:
                logger.error("ExoplanetPredictor no disponible")
                return False
                
            logger.info("🔄 Inicializando servicio de predicción...")
            
            # Crear instancia del predictor
            self.predictor = ExoplanetPredictor(self.project_root)
            
            # Cargar modelo en un hilo separado para no bloquear
            success = await asyncio.get_event_loop().run_in_executor(
                None, self.predictor.load_latest_model
            )
            
            if success:
                self.model_info = self.predictor.model_info
                self.is_initialized = True
                logger.info("✅ Servicio de predicción inicializado")
                return True
            else:
                logger.error("❌ Error cargando modelo ML")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error inicializando servicio: {e}")
            return False
    
    def is_model_loaded(self) -> bool:
        """Verificar si el modelo está cargado"""
        return self.is_initialized and self.model_info is not None
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        if not self.is_model_loaded():
            raise RuntimeError("Modelo no cargado")
        
        try:
            info = {
                "model_type": "Stacking Ensemble",
                "model_version": "v1.0",
                "training_accuracy": float(self.model_info.get('accuracy', 0.0)),
                "base_models": ["Random Forest", "AdaBoost", "Extra Trees", "LightGBM"],
                "training_datasets": self.model_info.get('training_datasets', []),
                "features_used": len(self.model_info.get('common_features', [])),
                "training_samples": self.model_info.get('training_samples', 0),
                "metrics": {
                    "accuracy": float(self.model_info.get('accuracy', 0.0)),
                    "precision": float(self.model_info.get('precision', 0.0)),
                    "recall": float(self.model_info.get('recall', 0.0)),
                    "f1_score": float(self.model_info.get('f1_score', 0.0)),
                    "roc_auc": float(self.model_info.get('roc_auc', 0.0))
                },
                "last_trained": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error obteniendo info del modelo: {e}")
            raise
    
    async def get_model_features(self) -> List[str]:
        """Obtener lista de características del modelo"""
        if not self.is_model_loaded():
            raise RuntimeError("Modelo no cargado")
        
        return self.model_info.get('common_features', [])
    
    async def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicción individual
        """
        if not self.is_model_loaded():
            raise RuntimeError("Modelo no cargado")
        
        try:
            logger.info("🔮 Realizando predicción individual...")
            
            # Convertir características a DataFrame
            df = pd.DataFrame([features])
            
            # Mapear nombres de columnas si es necesario
            column_mapping = {
                'star_radius': 'koi_srad',
                'star_mass': 'koi_smass', 
                'star_temp': 'koi_steff',
                'period': 'koi_period',
                'radius': 'koi_prad',
                'temp': 'koi_teq',
                'depth': 'koi_depth',
                'duration': 'koi_duration',
                'snr': 'koi_model_snr',
                'impact': 'koi_impact',
                'semi_major_axis': 'koi_sma'
            }
            
            # Aplicar mapeo de columnas
            for api_name, model_name in column_mapping.items():
                if api_name in df.columns:
                    df[model_name] = df[api_name]
            
            # Simular el preprocesamiento del predictor
            try:
                # Usar el preprocesador del modelo cargado
                X_pred, available_features, missing_features = self.predictor.preprocess_new_dataset(
                    df, "single_prediction.csv"
                )
                
                if X_pred is None:
                    raise ValueError("Error en preprocesamiento")
                
                # Realizar predicción
                model = self.model_info['model']
                y_proba = model.predict_proba(X_pred)[:, 1][0]  # Probabilidad de ser exoplaneta
                y_pred = int(y_proba >= 0.5)
                
                # Calcular confianza
                confidence = float(y_proba if y_pred == 1 else 1 - y_proba)
                
                # Determinar nivel de confianza
                if confidence >= 0.8:
                    confidence_level = "HIGH"
                elif confidence >= 0.6:
                    confidence_level = "MEDIUM"
                else:
                    confidence_level = "LOW"
                
                # Generar interpretación
                if y_pred == 1:
                    interpretation = f"Este candidato tiene una alta probabilidad ({y_proba:.1%}) de ser un exoplaneta real. Se recomienda seguimiento observacional."
                else:
                    interpretation = f"Este candidato probablemente es un falso positivo ({1-y_proba:.1%} de confianza). La señal puede deberse a eclipses binarios o ruido instrumental."
                
                # Resultado
                result = {
                    "prediction": "CONFIRMED" if y_pred == 1 else "FALSE_POSITIVE",
                    "confidence": confidence,
                    "probabilities": {
                        "CONFIRMED": float(y_proba),
                        "FALSE_POSITIVE": float(1 - y_proba)
                    },
                    "model_version": "StackingEnsemble_v1.0",
                    "timestamp": datetime.now().isoformat(),
                    "interpretation": interpretation,
                    "confidence_level": confidence_level
                }
                
                # Agregar feature importance si está disponible
                try:
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        features_list = self.model_info.get('common_features', [])
                        if len(importance) == len(features_list):
                            # Top 5 características más importantes
                            feature_importance = dict(zip(features_list, importance))
                            top_features = dict(sorted(feature_importance.items(), 
                                                     key=lambda x: x[1], reverse=True)[:5])
                            result["feature_importance"] = top_features
                except Exception as e:
                    logger.warning(f"No se pudo calcular feature importance: {e}")
                
                logger.info(f"✅ Predicción completada: {result['prediction']} ({confidence:.3f})")
                return result
                
            except Exception as e:
                logger.error(f"Error en predicción: {e}")
                raise ValueError(f"Error procesando características: {str(e)}")
                
        except Exception as e:
            logger.error(f"❌ Error en predicción individual: {e}")
            raise
    
    async def predict_batch(self, file_path: Path) -> Dict[str, Any]:
        """
        Predicción en lote desde archivo CSV
        """
        if not self.is_model_loaded():
            raise RuntimeError("Modelo no cargado")
        
        try:
            logger.info(f"📄 Iniciando predicción en lote: {file_path.name}")
            
            start_time = datetime.now()
            
            # Usar el predictor existente en un hilo separado
            result_df, summary = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.predictor.predict_dataset, 
                file_path.name,
                0.5  # confidence_threshold
            )
            
            if result_df is None or summary is None:
                raise ValueError("Error en predicción en lote")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calcular estadísticas adicionales
            n_total = len(result_df)
            n_confirmed = int((result_df['ML_Prediction'] == 1).sum())
            n_false_positives = n_total - n_confirmed
            
            # Distribución de confianza
            high_conf = int((result_df['ML_Confidence'] >= 0.8).sum())
            medium_conf = int(((result_df['ML_Confidence'] >= 0.6) & 
                              (result_df['ML_Confidence'] < 0.8)).sum())
            low_conf = int((result_df['ML_Confidence'] < 0.6).sum())
            
            # Top candidatos
            top_candidates_df = result_df.nlargest(5, 'ML_Confidence')
            top_candidates = []
            
            for _, row in top_candidates_df.iterrows():
                candidate = {
                    "confidence": float(row['ML_Confidence']),
                    "prediction": row['ML_Classification'],
                    "probability": float(row['ML_Probability'])
                }
                
                # Agregar algunos parámetros clave si están disponibles
                for param in ['koi_period', 'koi_prad', 'koi_teq', 'period', 'radius', 'temp']:
                    if param in row and pd.notna(row[param]):
                        candidate[param] = float(row[param])
                        break
                
                top_candidates.append(candidate)
            
            # Resultado de la predicción en lote
            batch_result = {
                "total_processed": n_total,
                "confirmed_planets": n_confirmed,
                "candidates": 0,  # Por simplicidad, no diferenciamos candidatos
                "false_positives": n_false_positives,
                "confirmed_percentage": round((n_confirmed / n_total) * 100, 2),
                "average_confidence": round(float(result_df['ML_Confidence'].mean()), 3),
                "confidence_distribution": {
                    "HIGH": high_conf,
                    "MEDIUM": medium_conf,
                    "LOW": low_conf
                },
                "output_file": summary['output_file'],
                "summary_file": f"{Path(summary['output_file']).stem}_summary.json",
                "processing_time": round(processing_time, 2),
                "timestamp": datetime.now().isoformat(),
                "top_candidates": top_candidates
            }
            
            logger.info(f"✅ Predicción en lote completada: {n_confirmed}/{n_total} confirmados")
            return batch_result
            
        except Exception as e:
            logger.error(f"❌ Error en predicción en lote: {e}")
            raise
    
    async def validate_features(self, features: Dict[str, Any]) -> bool:
        """Validar que las características sean válidas para el modelo"""
        if not self.is_model_loaded():
            return False
        
        try:
            # Verificar características mínimas requeridas
            required_features = ['period', 'radius', 'star_radius', 'star_temp', 'depth', 'duration', 'snr']
            
            for feature in required_features:
                if feature not in features or features[feature] is None:
                    logger.warning(f"Característica requerida faltante: {feature}")
                    return False
                    
                # Validaciones básicas
                if features[feature] <= 0:
                    logger.warning(f"Característica inválida {feature}: debe ser > 0")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando características: {e}")
            return False