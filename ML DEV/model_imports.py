"""
Importaciones compartidas para carga de modelos ML
Asegura que todas las clases necesarias estén disponibles para joblib
"""

# Importaciones para modelos legacy
try:
    from train_ensemble import StackingEnsemble, DataPreprocessor, FeatureEngineer
except ImportError:
    print("⚠️ No se pudieron importar clases legacy de train_ensemble")
    StackingEnsemble = None
    DataPreprocessor = None  
    FeatureEngineer = None

# Importaciones para modelos corregidos
try:
    from train_ensemble_CORRECTED import FeatureMapper, ImprovedDataPreprocessor
except ImportError:
    print("⚠️ No se pudieron importar clases de train_ensemble_CORRECTED")
    FeatureMapper = None
    ImprovedDataPreprocessor = None

# Importaciones para modelos rápidos
try:
    from train_ensemble_FAST import FastStackingEnsemble
except ImportError:
    print("⚠️ No se pudieron importar clases de train_ensemble_FAST")
    FastStackingEnsemble = None

# Lista de todas las clases disponibles para verificación
AVAILABLE_CLASSES = {
    'StackingEnsemble': StackingEnsemble,
    'DataPreprocessor': DataPreprocessor,
    'FeatureEngineer': FeatureEngineer,
    'FeatureMapper': FeatureMapper,
    'ImprovedDataPreprocessor': ImprovedDataPreprocessor,
    'FastStackingEnsemble': FastStackingEnsemble,
}

def check_model_compatibility():
    """
    Verifica qué clases están disponibles para la carga de modelos
    """
    print("🔍 Verificando compatibilidad de modelos:")
    
    for class_name, class_obj in AVAILABLE_CLASSES.items():
        status = "✅" if class_obj is not None else "❌"
        print(f"   {status} {class_name}")
    
    available_count = sum(1 for cls in AVAILABLE_CLASSES.values() if cls is not None)
    print(f"\n📊 Total disponibles: {available_count}/{len(AVAILABLE_CLASSES)}")
    
    return available_count > 0

if __name__ == "__main__":
    check_model_compatibility()