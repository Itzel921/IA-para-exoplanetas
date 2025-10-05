#!/usr/bin/env python3
"""
Script de prueba para el predictor corregido
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("🚀 Probando predictor corregido...")
    
    from simple_predictor_fixed import SimplePredictorFixed
    from pathlib import Path
    
    # Crear predictor
    project_root = Path(__file__).parent.parent
    predictor = SimplePredictorFixed(project_root)
    
    print("✅ SimplePredictorFixed importado correctamente")
    
    # Cargar modelo
    if predictor.load_model():
        print("✅ Modelo cargado exitosamente")
        
        # Listar archivos disponibles
        print("\n📁 Archivos disponibles para predicción:")
        csv_files = list(predictor.new_datasets_path.glob("*.csv"))
        for i, file in enumerate(csv_files, 1):
            print(f"   {i}. {file.name}")
        
        if csv_files:
            # Probar con el primer archivo
            test_file = csv_files[0].name
            print(f"\n🧪 Probando predicción con: {test_file}")
            
            result = predictor.process_file(test_file)
            if result:
                print("✅ Predicción completada exitosamente!")
            else:
                print("❌ Error en la predicción")
        else:
            print("❌ No hay archivos CSV para probar")
    else:
        print("❌ No se pudo cargar el modelo")
        
except ImportError as e:
    print(f"❌ Error de importación: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()