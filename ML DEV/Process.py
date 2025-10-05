"""
Sistema Principal de Procesamiento - Exoplanetas ML
Integra carga de datos, entrenamiento y predicción
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Importaciones necesarias para que joblib pueda cargar los modelos correctamente
import model_imports  # Esto asegura que todas las clases estén disponibles

def show_menu():
    """Muestra el menú principal del sistema"""
    print("\n" + "="*60)
    print("🌟 SISTEMA DE DETECCIÓN DE EXOPLANETAS")
    print("    NASA Space Apps Challenge 2025")
    print("="*60)
    print("\n📋 OPCIONES DISPONIBLES:")
    print("1. 📊 Cargar y analizar datasets (KOI, TOI, K2)")
    print("2. 🎯 Entrenar modelo ensemble (Stacking + RF + AdaBoost)")
    print("3. ⚡ Entrenar modelo simplificado (RÁPIDO - Random Forest)")
    print("4. 🔮 Predecir exoplanetas en nuevo dataset")
    print("5. 📁 Procesar todos los archivos en new_datasets")
    print("6. 📈 Análisis exploratorio completo")
    print("7. ❓ Ayuda y documentación")
    print("8. 🚪 Salir")
    print("\n" + "-"*60)

def option_1_load_datasets():
    """Opción 1: Cargar y analizar datasets"""
    print("\n🔄 Ejecutando análisis de datasets...")
    
    try:
        from Clasification import DataLoader
        
        # Configurar paths
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        
        # Cargar y mostrar información de datasets
        loader = DataLoader(project_root)
        datasets = loader.load_all_datasets()
        
        if len(datasets) > 0:
            print(f"\n📊 RESUMEN DE DATASETS CARGADOS:")
            print("="*50)
            
            total_rows = 0
            for name, df in datasets.items():
                print(f"\n📁 Dataset: {name}")
                print(f"   📏 Dimensiones: {df.shape[0]:,} × {df.shape[1]}")
                print(f"   📋 Columnas principales: {list(df.columns[:5])}")
                total_rows += df.shape[0]
                
                # Mostrar distribución de clases si existe columna objetivo
                target_cols = ['koi_disposition', 'tfopwg_disp', 'archive_disp']
                for col in target_cols:
                    if col in df.columns:
                        print(f"   🎯 Distribución {col}:")
                        value_counts = df[col].value_counts()
                        for val, count in value_counts.items():
                            percentage = (count / len(df)) * 100
                            print(f"      • {val}: {count:,} ({percentage:.1f}%)")
                        break
            
            print(f"\n📊 TOTAL COMBINADO: {total_rows:,} objetos astronómicos")
            print("✅ Análisis de datasets completado!")
            
        else:
            print("❌ No se pudieron cargar los datasets")
            print("   Verifica que los archivos CSV estén en data/datasets/")
        
    except Exception as e:
        print(f"❌ Error en análisis: {e}")
        import traceback
        traceback.print_exc()

def option_2_train_model():
    """Opción 2: Entrenar modelo ensemble"""
    print("\n🎯 Iniciando entrenamiento del modelo...")
    print("⏳ Este proceso puede tomar varios minutos...")
    
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
            
            # Inicializar y entrenar sistema
            ml_system = ExoplanetMLSystem(project_root)
            model_info = ml_system.train_system(datasets)
            
            print("✅ Entrenamiento completado!")
            print(f"🎯 Accuracy alcanzado: {model_info['accuracy']:.4f}")
            print("🚀 Sistema listo para predicciones!")
            
        else:
            print("❌ No se pudieron cargar los datasets para entrenamiento")
            print("   Verifica que los archivos CSV estén en data/datasets/")
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()

def option_3_simple_train():
    """Opción 3: Entrenar modelo simplificado (RÁPIDO)"""
    print("\n⚡ Iniciando entrenamiento simplificado...")
    print("🚀 Este proceso es más rápido y usa menos recursos")
    print("🎯 Usando Random Forest con características astronómicas optimizadas")
    
    try:
        # Importar el entrenador simplificado
        import simple_retrain
        
        print("📂 Ejecutando reentrenamiento simplificado...")
        model_info = simple_retrain.main()
        
        if model_info:
            print("✅ ¡Entrenamiento simplificado completado exitosamente!")
            print(f"🎯 Accuracy: {model_info['accuracy']:.4f} ({model_info['accuracy']*100:.1f}%)")
            print(f"🔬 Características: {model_info['n_features']}")
            print(f"📊 Precision: {model_info['precision']:.3f}")
            print(f"📊 Recall: {model_info['recall']:.3f}")
            print("🚀 El modelo está listo para predicciones!")
        else:
            print("❌ Error en el entrenamiento simplificado")
        
    except Exception as e:
        print(f"❌ Error en entrenamiento simplificado: {e}")
        import traceback
        traceback.print_exc()

def option_4_predict_single():
    """Opción 4: Predecir en dataset específico usando predictor corregido"""
    new_datasets_path = Path(__file__).parent.parent / "data" / "new_datasets"
    
    print(f"\n📁 Archivos disponibles en new_datasets:")
    csv_files = list(new_datasets_path.glob("*.csv"))
    
    if not csv_files:
        print("❌ No hay archivos CSV en la carpeta new_datasets")
        print(f"   Coloca archivos en: {new_datasets_path}")
        return
    
    for i, file in enumerate(csv_files, 1):
        print(f"   {i}. {file.name}")
    
    try:
        choice = int(input("\n🔢 Selecciona el número del archivo: "))
        if 1 <= choice <= len(csv_files):
            filename = csv_files[choice - 1].name
            
            print(f"\n🔮 Usando predictor corregido compatible con modelo actual...")
            
            # Usar el predictor corregido
            from simple_predictor_fixed import SimplePredictorFixed
            predictor = SimplePredictorFixed(Path(__file__).parent.parent)
            
            if predictor.load_model():
                predictor.process_file(filename)
            else:
                print("❌ Primero debes entrenar un modelo")
        else:
            print("❌ Selección inválida")
            
    except ValueError:
        print("❌ Por favor ingresa un número válido")
    except Exception as e:
        print(f"❌ Error: {e}")

def option_5_predict_all():
    """Opción 5: Procesar todos los archivos usando predictor corregido"""
    print("\n🔄 Procesando todos los archivos en new_datasets...")
    
    try:
        print("🔮 Usando predictor corregido compatible con modelo actual...")
        
        from simple_predictor_fixed import SimplePredictorFixed
        predictor = SimplePredictorFixed(Path(__file__).parent.parent)
        
        if predictor.load_model():
            predictor.process_all_new_datasets()
        else:
            print("❌ Primero debes entrenar un modelo")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def option_6_full_analysis():
    """Opción 6: Análisis exploratorio completo con visualizaciones avanzadas"""
    print("\n📊 Ejecutando análisis exploratorio completo...")
    
    try:
        from Clasification import DataLoader
        from advanced_visualization import ExoplanetVisualizer
        
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        
        # Cargar datasets NASA
        print("📁 Cargando datasets NASA...")
        loader = DataLoader(project_root)
        nasa_datasets = loader.load_all_datasets()
        
        # Cargar new_datasets si existen
        print("� Cargando new_datasets...")
        new_datasets = {}
        new_datasets_path = project_root / "data" / "new_datasets"
        
        if new_datasets_path.exists():
            csv_files = list(new_datasets_path.glob("*.csv"))
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, comment='#', sep=',', engine='python')
                    file_name = csv_file.stem
                    new_datasets[file_name] = df
                    print(f"   ✅ {file_name}: {df.shape[0]:,} filas × {df.shape[1]} columnas")
                except Exception as e:
                    print(f"   ⚠️ Error cargando {csv_file.name}: {e}")
        
        # Crear visualizador
        visualizer = ExoplanetVisualizer(project_root)
        
        # Generar todas las visualizaciones
        generated_files = visualizer.generate_complete_analysis(nasa_datasets, new_datasets)
        
        # Mostrar resumen estadístico
        print(f"\n📈 RESUMEN ESTADÍSTICO")
        print("="*50)
        
        print("📊 Datasets NASA:")
        for name, df in nasa_datasets.items():
            analysis = loader.analyze_dataset(df, name)
            print(f"   • {name}: {df.shape[0]:,} objetos, {df.shape[1]} características")
            if analysis.get('target_column'):
                target_col = analysis['target_column']
                if target_col in df.columns:
                    class_dist = df[target_col].value_counts()
                    print(f"     - Clases: {dict(class_dist)}")
        
        if new_datasets:
            print("\n📁 New Datasets:")
            for name, df in new_datasets.items():
                print(f"   • {name}: {df.shape[0]:,} objetos, {df.shape[1]} características")
        
        print(f"\n🎨 Visualizaciones generadas: {len(generated_files)}")
        print("   📂 Ubicaciones:")
        print("      • Charts individuales: exoPlanet_results/charts/")
        print("      • Charts comparativos: exoPlanet_results/comparative_charts/")
        
        print("\n✅ Análisis exploratorio completo finalizado!")
        
    except Exception as e:
        print(f"❌ Error en análisis: {e}")
        import traceback
        traceback.print_exc()

def option_7_help():
    """Opción 7: Ayuda y documentación"""
    print("\n📚 DOCUMENTACIÓN DEL SISTEMA")
    print("="*50)
    
    print("""
🎯 OBJETIVO:
   Detectar exoplanetas usando ensemble learning con 83.08% de accuracy

📊 DATASETS SOPORTADOS:
   • KOI (Kepler Objects of Interest): ~9,600 objetos
   • TOI (TESS Objects of Interest): ~6,000 objetos  
   • K2 Planets and Candidates: Múltiples campañas

🧠 ALGORITMOS DE ML:
   • Stacking Ensemble (mejor: 83.08% accuracy)
   • Random Forest (82.64%)
   • AdaBoost (82.52%)
   • Gradient Boosting + LightGBM

📁 ESTRUCTURA DE CARPETAS:
   • data/datasets/: Datasets originales (KOI, TOI, K2)
   • data/new_datasets/: Nuevos archivos para predicción
   • exoPlanet_results/: Resultados de predicciones
   • models/: Modelos entrenados guardados

🚀 FLUJO DE TRABAJO:
   1. Analizar datasets existentes (opción 1)
   2. Entrenar modelo ensemble (opción 2) 
   3. Colocar nuevos CSV en data/new_datasets/
   4. Predecir exoplanetas (opción 3 o 4)
   5. Revisar resultados en exoPlanet_results/

📋 FORMATO DE RESULTADOS:
   • CSV con columnas originales + predicciones ML
   • ML_Probability: Probabilidad de ser exoplaneta (0-1)
   • ML_Prediction: Predicción binaria (1=exoplaneta, 0=no)
   • ML_Confidence: Nivel de confianza de la predicción
   • ML_Classification: 'CONFIRMED' o 'NOT_CONFIRMED'

🔬 BASE CIENTÍFICA:
   • Research papers: Electronics 2024, MNRAS 2022
   • Feature engineering astronómico especializado
   • Validación cruzada estratificada para clases desbalanceadas
   """)

def main():
    """Función principal del sistema"""
    
    while True:
        show_menu()
        
        try:
            choice = input("🔢 Selecciona una opción (1-8): ").strip()
            
            if choice == '1':
                option_1_load_datasets()
            elif choice == '2':
                option_2_train_model()
            elif choice == '3':
                option_3_simple_train()
            elif choice == '4':
                option_4_predict_single()
            elif choice == '5':
                option_5_predict_all()
            elif choice == '6':
                option_6_full_analysis()
            elif choice == '7':
                option_7_help()
            elif choice == '8':
                print("\n👋 ¡Gracias por usar el Sistema de Detección de Exoplanetas!")
                print("🌟 NASA Space Apps Challenge 2025")
                break
            else:
                print("❌ Opción inválida. Por favor selecciona 1-8.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Saliendo del sistema...")
            break
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
        
        input("\n⏸️  Presiona Enter para continuar...")

if __name__ == "__main__":
    main()
