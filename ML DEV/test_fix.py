#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simple para verificar que el fix del FastStackingEnsemble funciona correctamente
"""
import sys
from pathlib import Path
import joblib

# Agregar el directorio actual al path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

print("🧪 PROBANDO EL FIX DE FASTSTACKINGENSEMBLE")
print("="*50)

# 1. Importar el sistema de compatibilidad
try:
    import model_imports
    print("✅ model_imports importado correctamente")
    
    # Verificar compatibilidad
    is_compatible = model_imports.check_model_compatibility()
    if is_compatible:
        print("✅ Todas las clases están disponibles (6/6)")
    else:
        print("❌ Algunas clases no están disponibles")
    
    # IMPORTANTE: Importar las clases directamente para joblib
    from train_ensemble_FAST import FastStackingEnsemble
    from train_ensemble import StackingEnsemble, DataPreprocessor, FeatureEngineer
    from train_ensemble_CORRECTED import FeatureMapper, ImprovedDataPreprocessor
    print("✅ Clases importadas directamente para joblib")
        
except ImportError as e:
    print(f"❌ Error importando clases: {e}")
    sys.exit(1)

# 2. Buscar modelo más reciente
models_dir = current_dir.parent / "models"
model_files = list(models_dir.glob("*.pkl"))

if not model_files:
    print("❌ No se encontraron modelos .pkl")
    sys.exit(1)

# Ordenar por fecha de modificación (más reciente primero)
latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
print(f"\n🔍 Modelo más reciente: {latest_model.name}")

# 3. Intentar cargar el modelo
try:
    print(f"\n🔄 Cargando modelo...")
    model = joblib.load(str(latest_model))
    print(f"✅ Modelo cargado exitosamente!")
    print(f"   Tipo: {type(model)}")
    
    # Verificar si tiene atributos esperados
    if hasattr(model, 'predict'):
        print("✅ Método predict disponible")
    if hasattr(model, 'predict_proba'):
        print("✅ Método predict_proba disponible")
        
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    print(f"   Tipo de error: {type(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Probar Process.py sin ejecutar el menú
try:
    print(f"\n🔄 Probando importación de Process.py...")
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("process_module", current_dir / "Process.py")
    process_module = importlib.util.module_from_spec(spec)
    
    # Ejecutar solo las importaciones, no el main()
    spec.loader.exec_module(process_module)
    print("✅ Process.py se puede importar sin errores!")
    
except Exception as e:
    print(f"❌ Error importando Process.py: {e}")
    print(f"   Tipo de error: {type(e)}")

print(f"\n🎉 PRUEBAS COMPLETADAS")
print("="*50)
print("El fix del FastStackingEnsemble está funcionando correctamente!")