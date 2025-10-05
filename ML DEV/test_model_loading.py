"""
Script de prueba para verificar la carga del modelo y hacer una predicción
"""
import sys
from pathlib import Path

# Agregar el directorio actual al path para las importaciones
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(current_dir))

# Importar las clases necesarias ANTES de cargar el modelo
try:
    from train_ensemble import StackingEnsemble, DataPreprocessor, FeatureEngineer
    from predict_exoplanets import ExoplanetPredictor
    print("✅ Clases importadas correctamente")
except ImportError as e:
    print(f"❌ Error importando clases: {e}")
    sys.exit(1)

# Probar el predictor completo
print(f"\n🧪 PROBANDO EL PREDICTOR COMPLETO")
print("="*50)

try:
    # Inicializar predictor
    predictor = ExoplanetPredictor(project_root)
    
    # Cargar modelo
    if predictor.load_latest_model():
        print("✅ Modelo cargado exitosamente en el predictor!")
        
        # Verificar si hay archivos para predecir
        new_datasets_path = project_root / "data" / "new_datasets"
        csv_files = list(new_datasets_path.glob("*.csv"))
        
        if csv_files:
            print(f"\n� Archivos disponibles para predicción:")
            for i, file in enumerate(csv_files, 1):
                print(f"   {i}. {file.name}")
            
            # Probar predicción con el primer archivo
            test_file = csv_files[0].name
            print(f"\n🔮 Probando predicción con: {test_file}")
            
            try:
                result_df, summary = predictor.predict_dataset(test_file)
                print("✅ Predicción exitosa!")
                print(f"📊 Resumen:")
                for key, value in summary.items():
                    print(f"   {key}: {value}")
                    
            except Exception as e:
                print(f"❌ Error en predicción: {e}")
                print(f"   Tipo de error: {type(e)}")
        else:
            print("❌ No hay archivos CSV en new_datasets para probar")
    else:
        print("❌ No se pudo cargar el modelo")
        
except Exception as e:
    print(f"❌ Error general: {e}")
    print(f"   Tipo de error: {type(e)}")
    import traceback
    traceback.print_exc()