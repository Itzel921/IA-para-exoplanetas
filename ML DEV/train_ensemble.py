"""
Entrenamiento de Ensemble para Detección de Exoplanetas
Implementa Stacking (83.08% accuracy target) + Random Forest + AdaBoost
Basado en research: Electronics 2024, MNRAS 2022
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importaciones de sklearn para ensemble learning
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb

class FeatureEngineer:
    """
    Feature Engineering especializado para datos astronómicos
    Implementa derivaciones físicas según .github/context/implementation-methodology.md
    """
    
    def __init__(self):
        pass
    
    def create_astronomical_features(self, df, dataset_type='KOI'):
        """
        Crea características derivadas específicas para exoplanetas
        """
        df_features = df.copy()
        
        try:
            # Mapeo de columnas según dataset
            if dataset_type == 'KOI':
                period_col = 'koi_period'
                prad_col = 'koi_prad'  
                srad_col = 'koi_srad'
                teq_col = 'koi_teq'
                steff_col = 'koi_steff'
                depth_col = 'koi_depth'
                duration_col = 'koi_duration'
                
            elif dataset_type == 'TOI':
                # Mapear columnas TOI (adaptar según estructura real)
                period_col = 'pl_orbper' if 'pl_orbper' in df.columns else None
                prad_col = 'pl_rade' if 'pl_rade' in df.columns else None
                
            elif dataset_type == 'K2':
                # Mapear columnas K2 (adaptar según estructura real)
                period_col = 'koi_period' if 'koi_period' in df.columns else None
                prad_col = 'koi_prad' if 'koi_prad' in df.columns else None
            
            # Feature 1: Planet-Star Radius Ratio
            if prad_col and srad_col and prad_col in df.columns and srad_col in df.columns:
                df_features['planet_star_radius_ratio'] = df[prad_col] / df[srad_col]
            
            # Feature 2: Equilibrium Temperature Ratio
            if teq_col and steff_col and teq_col in df.columns and steff_col in df.columns:
                df_features['equilibrium_temp_ratio'] = df[teq_col] / df[steff_col]
            
            # Feature 3: Transit Depth Expected (ppm)
            if prad_col and srad_col and prad_col in df.columns and srad_col in df.columns:
                df_features['transit_depth_expected'] = (df[prad_col] / df[srad_col]) ** 2 * 1e6
            
            # Feature 4: Habitability Zone Distance (scaled by stellar temperature)
            if teq_col in df.columns:
                df_features['habitable_zone_distance'] = abs(df[teq_col] - 288) / 288  # 288K ≈ Earth temp
            
            # Feature 5: Period-Radius Relationship
            if period_col and prad_col and period_col in df.columns and prad_col in df.columns:
                df_features['period_radius_product'] = df[period_col] * df[prad_col]
            
            print(f"✅ Feature engineering completado: +{len(df_features.columns) - len(df.columns)} características")
            
        except Exception as e:
            print(f"⚠️ Error en feature engineering: {e}")
        
        return df_features

class DataPreprocessor:
    """
    Preprocesamiento unificado para datasets de NASA
    Implementa estrategias de imputación específicas para astronomía
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.scalers = {}
        self.label_encoders = {}
        self.imputers = {}
    
    def preprocess_dataset(self, df, dataset_type='KOI', target_col=None):
        """
        Preprocesamiento completo para un dataset
        """
        print(f"\n🔄 Preprocesando dataset {dataset_type}...")
        
        df_processed = df.copy()
        
        # 1. Feature Engineering
        df_processed = self.feature_engineer.create_astronomical_features(df_processed, dataset_type)
        
        # 2. Identificar columnas numéricas y categóricas
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Remover target de features si existe
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        # 3. Imputación de valores faltantes
        # Para parámetros estelares: mediana
        stellar_params = [col for col in numeric_cols if any(x in col.lower() for x in ['steff', 'srad', 'smass', 'slogg']) and col in df_processed.columns]
        if stellar_params:
            imputer_stellar = SimpleImputer(strategy='median')
            # Verificar que las columnas existen antes de procesar
            stellar_data = df_processed[stellar_params].copy()
            imputed_data = imputer_stellar.fit_transform(stellar_data)
            # Asignar de vuelta columna por columna verificando índices
            for i, col in enumerate(stellar_params):
                if i < imputed_data.shape[1]:  # Verificar que el índice esté dentro de los límites
                    df_processed[col] = imputed_data[:, i]
            self.imputers[f'{dataset_type}_stellar'] = imputer_stellar
        
        # Para parámetros planetarios: mediana condicional por disposición
        planetary_params = [col for col in numeric_cols if col not in stellar_params and col in df_processed.columns]
        if planetary_params:
            imputer_planetary = SimpleImputer(strategy='median')
            # Verificar que las columnas existen antes de procesar
            planetary_data = df_processed[planetary_params].copy()
            imputed_data = imputer_planetary.fit_transform(planetary_data)
            # Asignar de vuelta columna por columna verificando índices
            for i, col in enumerate(planetary_params):
                if i < imputed_data.shape[1]:  # Verificar que el índice esté dentro de los límites
                    df_processed[col] = imputed_data[:, i]
            self.imputers[f'{dataset_type}_planetary'] = imputer_planetary
        
        # 4. Manejo de outliers astronómicos (Winsorizing 1%-99%)
        for col in numeric_cols:
            if col in df_processed.columns:
                q1 = df_processed[col].quantile(0.01)
                q99 = df_processed[col].quantile(0.99)
                df_processed[col] = df_processed[col].clip(lower=q1, upper=q99)
        
        # 5. Escalado robusto (preferido para datos astronómicos)
        if numeric_cols:
            scaler = RobustScaler()
            df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
            self.scalers[dataset_type] = scaler
        
        # 6. Encoding de variables categóricas
        for col in categorical_cols:
            if col != target_col and col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[f'{dataset_type}_{col}'] = le
        
        print(f"✅ Preprocesamiento {dataset_type} completado")
        print(f"   • Características numéricas: {len(numeric_cols)}")
        print(f"   • Características categóricas: {len(categorical_cols)}")
        print(f"   • Shape final: {df_processed.shape}")
        
        return df_processed, numeric_cols + categorical_cols

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Implementación de Stacking Ensemble para exoplanetas
    Basado en research achieving 83.08% accuracy
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # Base learners (Nivel 1) - Versión simplificada para pruebas
        self.base_learners = {
            'rf': RandomForestClassifier(
                n_estimators=100,  # Reducido para pruebas
                max_depth=10,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=2  # Limitado para evitar problemas
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=50,  # Reducido para pruebas
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=50,  # Reducido para pruebas
                random_state=random_state,
                verbose=-1
            )
        }
        
        # Meta-learner (Nivel 2)
        self.meta_learner = LogisticRegression(random_state=random_state)
        
        self.cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    def fit(self, X, y):
        """
        Entrenamiento en dos fases:
        1. Base learners con cross-validation
        2. Meta-learner con predicciones de base learners
        """
        print("🎯 Entrenando Stacking Ensemble...")
        
        # Fase 1: Entrenar base learners y generar meta-features
        meta_features = np.zeros((X.shape[0], len(self.base_learners)))
        
        for fold, (train_idx, val_idx) in enumerate(self.cv_folds.split(X, y)):
            print(f"  Fold {fold + 1}/5...")
            
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            for i, (name, learner) in enumerate(self.base_learners.items()):
                learner_fold = learner.__class__(**learner.get_params())
                learner_fold.fit(X_train_fold, y_train_fold)
                meta_features[val_idx, i] = learner_fold.predict_proba(X_val_fold)[:, 1]
        
        # Entrenar base learners en todos los datos
        print("  Entrenando base learners finales...")
        for name, learner in self.base_learners.items():
            learner.fit(X, y)
        
        # Fase 2: Entrenar meta-learner
        print("  Entrenando meta-learner...")
        self.meta_learner.fit(meta_features, y)
        
        print("✅ Stacking Ensemble entrenado exitosamente!")
        return self
    
    def predict_proba(self, X):
        """Predicción de probabilidades"""
        # Obtener predicciones de base learners
        meta_features = np.zeros((X.shape[0], len(self.base_learners)))
        
        for i, (name, learner) in enumerate(self.base_learners.items()):
            meta_features[:, i] = learner.predict_proba(X)[:, 1]
        
        # Predicción final del meta-learner
        return self.meta_learner.predict_proba(meta_features)
    
    def predict(self, X):
        """Predicción de clases"""
        return self.predict_proba(X)[:, 1] > 0.5

class ExoplanetMLSystem:
    """
    Sistema principal de Machine Learning para detección de exoplanetas
    """
    
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.models_path = self.project_root / "models"
        self.results_path = self.project_root / "exoPlanet_results"
        
        # Crear directorios si no existen
        self.models_path.mkdir(exist_ok=True)
        self.results_path.mkdir(exist_ok=True)
        
        # Componentes del sistema
        self.preprocessor = DataPreprocessor()
        self.ensemble_model = StackingEnsemble()
        
        # Mapeo de etiquetas unificado
        self.label_mapping = {
            'CONFIRMED': 1,
            'CANDIDATE': 0, 
            'FALSE POSITIVE': 0,
            'KP': 1,
            'PC': 0,
            'FP': 0,
            'APC': 0
        }
    
    def train_system(self, datasets):
        """
        Entrenamiento del sistema completo con los 3 datasets
        """
        print("\n🚀 INICIANDO ENTRENAMIENTO DEL SISTEMA")
        print("="*60)
        
        # Procesar cada dataset
        processed_datasets = {}
        all_features = []
        
        for name, df in datasets.items():
            # Identificar columna target
            target_cols = [col for col in df.columns if 'disposition' in col.lower()]
            target_col = target_cols[0] if target_cols else None
            
            if target_col is None:
                print(f"⚠️ No se encontró columna target en {name}")
                continue
            
            print(f"\n📊 Procesando dataset {name} (target: {target_col})...")
            
            # Preprocesamiento
            df_processed, feature_cols = self.preprocessor.preprocess_dataset(df, name, target_col)
            
            # Mapeo de etiquetas a binario
            y_mapped = df_processed[target_col].map(self.label_mapping)
            y_mapped = y_mapped.fillna(0)  # Valores no mapeados = 0
            
            processed_datasets[name] = {
                'X': df_processed[feature_cols],
                'y': y_mapped,
                'target_col': target_col,
                'feature_cols': feature_cols
            }
            
            all_features.extend(feature_cols)
            
            print(f"   • Shape X: {processed_datasets[name]['X'].shape}")
            print(f"   • Distribución y: {dict(y_mapped.value_counts())}")
        
        # Encontrar características comunes entre datasets
        common_features = set(processed_datasets[list(processed_datasets.keys())[0]]['feature_cols'])
        for dataset_data in processed_datasets.values():
            common_features &= set(dataset_data['feature_cols'])
        
        common_features = list(common_features)
        print(f"\n🔗 Características comunes entre datasets: {len(common_features)}")
        
        # Combinar datasets usando solo características comunes
        X_combined = []
        y_combined = []
        dataset_labels = []
        
        for name, dataset_data in processed_datasets.items():
            X_common = dataset_data['X'][common_features]
            y_common = dataset_data['y']
            
            X_combined.append(X_common)
            y_combined.extend(y_common)
            dataset_labels.extend([name] * len(y_common))
        
        # Concatenar todos los datasets
        X_final = pd.concat(X_combined, ignore_index=True)
        y_final = pd.Series(y_combined)
        
        print(f"\n📈 Dataset combinado final:")
        print(f"   • Shape: {X_final.shape}")
        print(f"   • Distribución clases: {dict(y_final.value_counts())}")
        print(f"   • Porcentaje planetas confirmados: {(y_final.sum() / len(y_final)) * 100:.1f}%")
        
        # Entrenamiento del ensemble
        print(f"\n🎯 Entrenando Stacking Ensemble...")
        self.ensemble_model.fit(X_final, y_final)
        
        # Evaluación con cross-validation
        cv_scores = cross_val_score(
            self.ensemble_model, X_final, y_final,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1
        )
        
        print(f"\n📊 RESULTADOS DEL ENTRENAMIENTO:")
        print(f"   • Accuracy promedio CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"   • Objetivo (research): 0.8308")
        print(f"   • Estado: {'✅ OBJETIVO ALCANZADO' if cv_scores.mean() >= 0.83 else '⚠️ Necesita optimización'}")
        
        # Guardar modelo y metadatos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_info = {
            'model': self.ensemble_model,
            'preprocessor': self.preprocessor,
            'common_features': common_features,
            'label_mapping': self.label_mapping,
            'cv_scores': cv_scores,
            'training_datasets': list(datasets.keys()),
            'timestamp': timestamp,
            'accuracy': cv_scores.mean()
        }
        
        model_path = self.models_path / f"exoplanet_ensemble_{timestamp}.pkl"
        joblib.dump(model_info, model_path)
        
        print(f"\n💾 Modelo guardado en: {model_path}")
        print(f"🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        
        return model_info

if __name__ == "__main__":
    # Cargar datasets para entrenamiento
    from Clasification import DataLoader
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Cargar datasets
    loader = DataLoader(project_root)
    datasets = loader.load_all_datasets()
    
    if len(datasets) > 0:
        # Inicializar y entrenar sistema
        ml_system = ExoplanetMLSystem(project_root)
        model_info = ml_system.train_system(datasets)
        
        print(f"\n🚀 Sistema listo para predicciones!")
        print(f"   • Usar: python predict_exoplanets.py <nuevo_dataset.csv>")
    else:
        print("❌ No se pudieron cargar los datasets para entrenamiento")