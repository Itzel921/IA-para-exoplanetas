"""
Script para inspeccionar las características usadas en el modelo entrenado
"""
import sys
from pathlib import Path
import joblib
import pandas as pd

# Agregar el directorio actual al path para las importaciones
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(current_dir))

# Importar las clases necesarias ANTES de cargar el modelo
try:
    from train_ensemble import StackingEnsemble, DataPreprocessor, FeatureEngineer
    print("✅ Clases importadas correctamente")
except ImportError as e:
    print(f"❌ Error importando clases: {e}")
    sys.exit(1)

# Cargar el modelo
models_path = current_dir.parent / "models"
model_files = list(models_path.glob("*.pkl"))

if model_files:
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"🔄 Inspeccionando modelo: {latest_model.name}")
    
    try:
        model_data = joblib.load(latest_model)
        print(f"✅ Modelo cargado exitosamente!")
        
        # Inspeccionar las características
        print(f"\n📊 INFORMACIÓN DEL MODELO:")
        print("="*50)
        
        # Ver las características comunes
        if 'common_features' in model_data:
            features = model_data['common_features']
            print(f"🔢 Total de características entrenadas: {len(features)}")
            print(f"\n📋 Lista completa de características:")
            for i, feature in enumerate(features, 1):
                print(f"   {i:3d}. {feature}")
        
        # Ver información del preprocessor
        if 'preprocessor' in model_data:
            preprocessor = model_data['preprocessor']
            print(f"\n🔧 Información del preprocessor:")
            print(f"   Tipo: {type(preprocessor)}")
            
            # Ver escaladores disponibles
            if hasattr(preprocessor, 'scalers'):
                print(f"   Escaladores: {list(preprocessor.scalers.keys())}")
            
            # Ver imputadores disponibles  
            if hasattr(preprocessor, 'imputers'):
                print(f"   Imputadores: {list(preprocessor.imputers.keys())}")
                
        # Ver información del modelo real
        if 'model' in model_data:
            actual_model = model_data['model']
            print(f"\n🎯 Información del modelo ensemble:")
            print(f"   Tipo: {type(actual_model)}")
            if hasattr(actual_model, 'base_learners'):
                print(f"   Base learners: {list(actual_model.base_learners.keys())}")
            if hasattr(actual_model, 'meta_learner'):
                print(f"   Meta learner: {type(actual_model.meta_learner)}")
        
        # Ver datasets de entrenamiento
        if 'training_datasets' in model_data:
            print(f"\n📊 Datasets de entrenamiento: {model_data['training_datasets']}")
        
        # Ver accuracy
        if 'accuracy' in model_data:
            print(f"🎯 Accuracy alcanzado: {model_data['accuracy']:.4f}")
            
    except Exception as e:
        print(f"❌ Error inspeccionando modelo: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ No se encontraron archivos de modelo")