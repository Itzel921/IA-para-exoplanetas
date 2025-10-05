"""
Script para migrar modelos existentes a la nueva ubicación
y probar la nueva configuración
"""

from pathlib import Path
import shutil
import os

def migrate_models_to_new_location():
    """Migra modelos existentes a ML DEV/trained_models"""
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Rutas
    old_models_path = project_root / "models"
    new_models_path = current_dir / "trained_models"
    
    print("🔄 MIGRACIÓN DE MODELOS A NUEVA UBICACIÓN")
    print("="*50)
    print(f"📁 Ubicación antigua: {old_models_path}")
    print(f"📁 Ubicación nueva: {new_models_path}")
    
    # Crear nueva carpeta
    new_models_path.mkdir(exist_ok=True)
    print(f"✅ Carpeta creada: {new_models_path}")
    
    # Buscar modelos existentes
    if old_models_path.exists():
        model_files = list(old_models_path.glob("exoplanet_ensemble_*.pkl"))
        
        if model_files:
            print(f"\n📦 Modelos encontrados: {len(model_files)}")
            
            for model_file in model_files:
                new_location = new_models_path / model_file.name
                
                # Copiar (no mover) para mantener compatibilidad
                shutil.copy2(model_file, new_location)
                print(f"   ✅ Copiado: {model_file.name}")
                
                # Mostrar información del archivo
                file_size = model_file.stat().st_size / (1024*1024)  # MB
                print(f"      📏 Tamaño: {file_size:.1f} MB")
            
            print(f"\n🎉 ¡Migración completada!")
            print(f"📍 Los modelos ahora están disponibles en:")
            print(f"   {new_models_path}")
            
        else:
            print("❌ No se encontraron modelos en la ubicación antigua")
    else:
        print("❌ La carpeta de modelos antigua no existe")
    
    # Listar contenido de la nueva carpeta
    if new_models_path.exists():
        new_models = list(new_models_path.glob("*.pkl"))
        print(f"\n📂 Contenido de la nueva carpeta:")
        if new_models:
            for model in new_models:
                file_size = model.stat().st_size / (1024*1024)
                mod_time = model.stat().st_mtime
                from datetime import datetime
                mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
                print(f"   📄 {model.name}")
                print(f"      📏 {file_size:.1f} MB | 🕐 {mod_time_str}")
        else:
            print("   (Vacía)")

def test_new_model_loading():
    """Prueba la carga de modelos desde la nueva ubicación"""
    print(f"\n🧪 PROBANDO CARGA DE MODELOS")
    print("="*30)
    
    try:
        from predict_exoplanets import ExoplanetPredictor
        
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        
        predictor = ExoplanetPredictor(project_root)
        
        print("📂 Rutas configuradas:")
        print(f"   • Nueva: {predictor.models_path_new}")
        print(f"   • Legacy: {predictor.models_path_legacy}")
        
        success = predictor.load_latest_model()
        
        if success:
            print("✅ ¡Carga de modelo exitosa!")
            return True
        else:
            print("❌ Error en la carga del modelo")
            return False
            
    except Exception as e:
        print(f"❌ Error en el test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 CONFIGURACIÓN DE NUEVA UBICACIÓN DE MODELOS")
    print("="*60)
    
    # Paso 1: Migrar modelos existentes
    migrate_models_to_new_location()
    
    # Paso 2: Probar la nueva configuración
    test_success = test_new_model_loading()
    
    # Resumen
    print(f"\n📋 RESUMEN")
    print("="*20)
    if test_success:
        print("✅ Configuración completada exitosamente")
        print("🎯 Los futuros modelos se guardarán en ML DEV/trained_models/")
        print("🔍 El sistema puede cargar modelos tanto de la nueva como de la ubicación legacy")
    else:
        print("⚠️ Hay issues que resolver")
    
    print(f"\n📁 Estructura recomendada:")
    print("   IA-para-exoplanetas/")
    print("   ├── ML DEV/")
    print("   │   ├── trained_models/  ← NUEVA UBICACIÓN")
    print("   │   │   ├── exoplanet_ensemble_*.pkl")
    print("   │   ├── train_ensemble.py")
    print("   │   └── predict_exoplanets.py")
    print("   └── models/  ← LEGACY (mantenido por compatibilidad)")