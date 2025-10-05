"""
Diagnóstico del Sistema de Entrenamiento
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def test_basic_imports():
    """Prueba las importaciones básicas"""
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import lightgbm as lgb
        import joblib
        print("✅ Todas las librerías básicas importadas correctamente")
        return True
    except Exception as e:
        print(f"❌ Error importando librerías: {e}")
        return False

def test_data_loading():
    """Prueba la carga de datos"""
    try:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        
        # Importar DataLoader
        sys.path.append(str(current_dir))
        from Clasification import DataLoader
        
        loader = DataLoader(project_root)
        datasets = loader.load_all_datasets()
        
        print(f"✅ Datasets cargados: {len(datasets)}")
        for name, df in datasets.items():
            print(f"   • {name}: {df.shape}")
        
        return datasets
        
    except Exception as e:
        print(f"❌ Error cargando datasets: {e}")
        import traceback
        traceback.print_exc()
        return {}

def test_model_training_simple():
    """Prueba un entrenamiento simplificado"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        import joblib
        
        # Datos de prueba
        X_test = pd.DataFrame(np.random.rand(100, 5))
        y_test = np.random.choice([0, 1], 100)
        
        # Modelo simple
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_test, y_test)
        
        # Guardar modelo de prueba
        current_dir = Path(__file__).parent
        models_path = current_dir.parent / "models"
        models_path.mkdir(exist_ok=True)
        
        test_model_path = models_path / "test_model.pkl"
        joblib.dump(model, test_model_path)
        
        print(f"✅ Modelo de prueba guardado en: {test_model_path}")
        print(f"✅ El archivo existe: {test_model_path.exists()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en entrenamiento de prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 DIAGNÓSTICO DEL SISTEMA DE ENTRENAMIENTO")
    print("="*50)
    
    # Test 1: Importaciones
    print("\n1. Probando importaciones...")
    if not test_basic_imports():
        return
    
    # Test 2: Carga de datos
    print("\n2. Probando carga de datos...")
    datasets = test_data_loading()
    if not datasets:
        return
    
    # Test 3: Entrenamiento simple
    print("\n3. Probando entrenamiento simple...")
    if not test_model_training_simple():
        return
    
    # Test 4: Verificar carpetas
    print("\n4. Verificando estructura de carpetas...")
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    folders_to_check = [
        project_root / "models",
        project_root / "exoPlanet_results",
        project_root / "data" / "datasets",
        project_root / "data" / "new_datasets"
    ]
    
    for folder in folders_to_check:
        print(f"   • {folder.name}: {'✅' if folder.exists() else '❌ NO EXISTE'}")
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            print(f"     └─ Carpeta creada: {folder}")
    
    print("\n🎯 Ahora intentemos el entrenamiento real...")
    try:
        from train_ensemble import ExoplanetMLSystem
        
        ml_system = ExoplanetMLSystem(project_root)
        model_info = ml_system.train_system(datasets)
        
        if model_info:
            print("🎉 ¡ENTRENAMIENTO EXITOSO!")
        else:
            print("❌ Error en entrenamiento")
            
    except Exception as e:
        print(f"❌ Error en entrenamiento completo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()