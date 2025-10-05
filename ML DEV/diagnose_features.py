"""
Script para inspeccionar qué características fueron guardadas en el modelo
y diagnosticar el problema de compatibilidad
"""

from pathlib import Path
import joblib

def inspect_model_features():
    """Inspecciona las características guardadas en el modelo más reciente"""
    
    current_dir = Path(__file__).parent
    
    # Buscar modelos
    models_path_new = current_dir / "trained_models"
    models_path_legacy = current_dir.parent / "models"
    
    # Encontrar el modelo más reciente
    all_models = []
    for path in [models_path_new, models_path_legacy]:
        if path.exists():
            all_models.extend(list(path.glob("exoplanet_ensemble_*.pkl")))
    
    if not all_models:
        print("❌ No se encontraron modelos")
        return
    
    latest_model = max(all_models, key=lambda x: x.stat().st_mtime)
    
    print(f"🔍 INSPECCIONANDO MODELO: {latest_model.name}")
    print("="*60)
    
    # Cargar modelo
    model_info = joblib.load(latest_model)
    
    print(f"📊 Información del modelo:")
    print(f"   • Accuracy: {model_info['accuracy']:.4f}")
    print(f"   • Datasets entrenamiento: {model_info['training_datasets']}")
    print(f"   • Timestamp: {model_info['timestamp']}")
    
    # Características comunes
    common_features = model_info['common_features']
    print(f"\n🎯 CARACTERÍSTICAS COMUNES ({len(common_features)}):")
    for i, feature in enumerate(common_features, 1):
        print(f"   {i:2d}. {feature}")
    
    # Mapeo de etiquetas
    label_mapping = model_info['label_mapping']
    print(f"\n🏷️ MAPEO DE ETIQUETAS:")
    for dataset, mapping in label_mapping.items():
        print(f"   {dataset}: {mapping}")
    
    # Información del preprocessor
    if 'preprocessor' in model_info:
        preprocessor = model_info['preprocessor']
        print(f"\n🔧 INFORMACIÓN DEL PREPROCESSOR:")
        print(f"   • Tipo: {type(preprocessor).__name__}")
        
        # Ver qué componentes tiene el preprocessor
        if hasattr(preprocessor, 'feature_engineer'):
            print(f"   • Feature Engineer: {type(preprocessor.feature_engineer).__name__}")
        
        if hasattr(preprocessor, 'scalers'):
            print(f"   • Scalers disponibles: {list(preprocessor.scalers.keys())}")
        
        if hasattr(preprocessor, 'imputers'):
            print(f"   • Imputers disponibles: {list(preprocessor.imputers.keys())}")
    
    return model_info

def test_feature_engineering():
    """Prueba la creación de características en un dataset de ejemplo"""
    print(f"\n🧪 PROBANDO FEATURE ENGINEERING")
    print("="*40)
    
    try:
        from Clasification import DataLoader
        from train_ensemble import FeatureEngineer
        
        # Cargar un dataset pequeño para probar
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        
        loader = DataLoader(project_root)
        datasets = loader.load_all_datasets()
        
        if 'KOI' in datasets:
            df_koi = datasets['KOI'].head(10)  # Solo 10 filas para prueba
            
            print(f"📊 Dataset original KOI: {df_koi.shape}")
            print(f"   Columnas: {list(df_koi.columns[:10])}")
            
            # Probar feature engineering
            fe = FeatureEngineer()
            df_engineered = fe.create_astronomical_features(df_koi, 'KOI')
            
            print(f"\n🔧 Después del feature engineering: {df_engineered.shape}")
            
            # Nuevas características creadas
            original_cols = set(df_koi.columns)
            new_cols = set(df_engineered.columns) - original_cols
            
            print(f"\n✨ NUEVAS CARACTERÍSTICAS CREADAS ({len(new_cols)}):")
            for col in sorted(new_cols):
                print(f"   • {col}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 DIAGNÓSTICO DE COMPATIBILIDAD DE CARACTERÍSTICAS")
    print("="*70)
    
    # Paso 1: Inspeccionar modelo
    model_info = inspect_model_features()
    
    # Paso 2: Probar feature engineering
    if model_info:
        success = test_feature_engineering()
        
        if success:
            print(f"\n📋 DIAGNÓSTICO:")
            print("="*20)
            print("✅ El feature engineering funciona correctamente")
            print("⚠️ Problema probablemente en la compatibilidad de características")
            print("🔧 Solución: Ajustar el proceso de predicción")
        else:
            print(f"\n❌ PROBLEMA ENCONTRADO:")
            print("="*25)
            print("El feature engineering tiene errors")
    else:
        print("❌ No se pudo cargar el modelo para diagnóstico")