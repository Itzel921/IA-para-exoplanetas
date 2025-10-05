"""
Script de prueba para verificar que el entrenamiento funciona correctamente
"""

from pathlib import Path
import sys

def test_training():
    """Prueba la funcionalidad de entrenamiento"""
    print("🧪 PROBANDO CORRECCIÓN DEL ENTRENAMIENTO")
    print("="*50)
    
    try:
        # Importar las clases necesarias
        from train_ensemble import ExoplanetMLSystem
        from Clasification import DataLoader
        
        # Configurar paths
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        
        print("📂 Cargando datasets de NASA (KOI, TOI, K2)...")
        
        # Cargar datasets
        loader = DataLoader(project_root)
        datasets = loader.load_all_datasets()
        
        if len(datasets) > 0:
            print(f"✅ {len(datasets)} datasets cargados exitosamente!")
            
            # Mostrar resumen de datasets
            total_objects = sum(len(df) for df in datasets.values())
            print(f"📊 Total de objetos astronómicos: {total_objects:,}")
            
            for name, df in datasets.items():
                print(f"   • {name}: {len(df):,} objetos")
                
                # Identificar columna objetivo
                target_cols = {
                    'KOI': 'koi_disposition',
                    'TOI': 'tfopwg_disp', 
                    'K2': 'disposition'
                }
                
                target_col = target_cols.get(name)
                if target_col and target_col in df.columns:
                    print(f"     🎯 Target: {target_col}")
                    class_dist = df[target_col].value_counts()
                    for label, count in class_dist.head(3).items():
                        pct = (count / len(df)) * 100
                        print(f"       - {label}: {count:,} ({pct:.1f}%)")
            
            print("\n🎯 Iniciando entrenamiento de prueba...")
            print("⏳ (Este proceso puede tomar 2-3 minutos...)")
            
            # Inicializar sistema ML
            ml_system = ExoplanetMLSystem(project_root)
            
            # Entrenar sistema
            model_info = ml_system.train_system(datasets)
            
            # Mostrar resultados
            print("\n🎉 ENTRENAMIENTO COMPLETADO!")
            print("="*50)
            print(f"🎯 Accuracy: {model_info['accuracy']:.4f}")
            print(f"📊 Cross-validation scores: {model_info['cv_scores']}")
            print(f"📁 Modelo guardado: {model_info['timestamp']}")
            print(f"🔗 Datasets usados: {model_info['training_datasets']}")
            
            return True
            
        else:
            print("❌ No se pudieron cargar los datasets")
            return False
        
    except Exception as e:
        print(f"❌ Error en el test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training()
    if success:
        print("\n✅ TEST EXITOSO - El problema de entrenamiento está resuelto!")
    else:
        print("\n❌ TEST FALLIDO - Necesita más debugging")