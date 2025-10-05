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
try:
    from train_ensemble import StackingEnsemble, DataPreprocessor, FeatureEngineer
except ImportError:
    # Si las importaciones fallan, las clases se definirán cuando se importen los módulos
    pass

def show_menu():
    """Muestra el menú principal del sistema"""
    print("\n" + "="*60)
    print("🌟 SISTEMA DE DETECCIÓN DE EXOPLANETAS")
    print("    NASA Space Apps Challenge 2025")
    print("="*60)
    print("\n📋 OPCIONES DISPONIBLES:")
    print("1. 📊 Cargar y analizar datasets (KOI, TOI, K2)")
    print("2. 🎯 Entrenar modelo ensemble (Stacking + RF + AdaBoost)")
    print("3. 🔮 Predecir exoplanetas en nuevo dataset")
    print("4. 📁 Procesar todos los archivos en new_datasets")
    print("5. 📈 Análisis exploratorio completo")
    print("6. ❓ Ayuda y documentación")
    print("7. 🚪 Salir")
    print("\n" + "-"*60)

def option_1_load_datasets():
    """Opción 1: Cargar y analizar datasets"""
    print("\n🔄 Ejecutando análisis de datasets...")
    
    try:
        # Importar y ejecutar Clasification.py
        import Clasification
        print("✅ Análisis de datasets completado!")
        
    except Exception as e:
        print(f"❌ Error en análisis: {e}")

def option_2_train_model():
    """Opción 2: Entrenar modelo ensemble"""
    print("\n🎯 Iniciando entrenamiento del modelo...")
    print("⏳ Este proceso puede tomar varios minutos...")
    
    try:
        import train_ensemble
        print("✅ Entrenamiento completado!")
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")

def option_3_predict_single():
    """Opción 3: Predecir en dataset específico"""
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
            
            # Asegurar que las clases están disponibles para joblib
            try:
                from train_ensemble import StackingEnsemble, DataPreprocessor, FeatureEngineer
            except ImportError:
                print("❌ Error: No se pueden importar las clases del modelo")
                return
            
            # Ejecutar predicción
            import predict_exoplanets
            predictor = predict_exoplanets.ExoplanetPredictor(Path(__file__).parent.parent)
            
            if predictor.load_latest_model():
                predictor.predict_dataset(filename)
            else:
                print("❌ Primero debes entrenar el modelo (opción 2)")
        else:
            print("❌ Selección inválida")
            
    except ValueError:
        print("❌ Por favor ingresa un número válido")
    except Exception as e:
        print(f"❌ Error: {e}")

def option_4_predict_all():
    """Opción 4: Procesar todos los archivos"""
    print("\n🔄 Procesando todos los archivos en new_datasets...")
    
    try:
        # Asegurar que las clases están disponibles para joblib
        try:
            from train_ensemble import StackingEnsemble, DataPreprocessor, FeatureEngineer
        except ImportError:
            print("❌ Error: No se pueden importar las clases del modelo")
            return
        
        import predict_exoplanets
        predictor = predict_exoplanets.ExoplanetPredictor(Path(__file__).parent.parent)
        
        if predictor.load_latest_model():
            predictor.process_all_new_datasets()
        else:
            print("❌ Primero debes entrenar el modelo (opción 2)")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def option_5_full_analysis():
    """Opción 5: Análisis exploratorio completo"""
    print("\n📊 Ejecutando análisis exploratorio completo...")
    
    try:
        from Clasification import DataLoader
        
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        
        loader = DataLoader(project_root)
        datasets = loader.load_all_datasets()
        
        # Análisis comparativo entre datasets
        print(f"\n🔍 ANÁLISIS COMPARATIVO ENTRE DATASETS")
        print("="*50)
        
        for name, df in datasets.items():
            analysis = loader.analyze_dataset(df, name)
            
            # Crear visualizaciones básicas
            if 'disposition' in str(analysis.get('target_column', '')).lower():
                target_col = analysis['target_column']
                
                plt.figure(figsize=(12, 4))
                
                # Distribución de clases
                plt.subplot(1, 2, 1)
                df[target_col].value_counts().plot(kind='bar')
                plt.title(f'{name} - Distribución de Clases')
                plt.xticks(rotation=45)
                
                # Correlación de características numéricas clave
                if len(analysis['key_features']) > 1:
                    plt.subplot(1, 2, 2)
                    numeric_df = df[analysis['key_features'][:5]].select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        correlation = numeric_df.corr()
                        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
                        plt.title(f'{name} - Correlación Características')
                
                plt.tight_layout()
                plt.show()
        
        print("✅ Análisis exploratorio completado!")
        
    except Exception as e:
        print(f"❌ Error en análisis: {e}")

def option_6_help():
    """Opción 6: Ayuda y documentación"""
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
            choice = input("🔢 Selecciona una opción (1-7): ").strip()
            
            if choice == '1':
                option_1_load_datasets()
            elif choice == '2':
                option_2_train_model()
            elif choice == '3':
                option_3_predict_single()
            elif choice == '4':
                option_4_predict_all()
            elif choice == '5':
                option_5_full_analysis()
            elif choice == '6':
                option_6_help()
            elif choice == '7':
                print("\n👋 ¡Gracias por usar el Sistema de Detección de Exoplanetas!")
                print("🌟 NASA Space Apps Challenge 2025")
                break
            else:
                print("❌ Opción inválida. Por favor selecciona 1-7.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Saliendo del sistema...")
            break
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
        
        input("\n⏸️  Presiona Enter para continuar...")

if __name__ == "__main__":
    main()
