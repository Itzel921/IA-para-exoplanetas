"""
Sistema de Detección de Exoplanetas - NASA Space Apps Challenge 2025
Basado en ensemble learning con datasets KOI, TOI, K2
Objetivo: 83.08% accuracy con Stacking ensemble
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib
from datetime import datetime

# Configuración de visualización
pd.set_option('display.max_columns', None)  # Mostrar TODAS las columnas
pd.set_option('display.max_rows', 100)     # Mostrar más filas
pd.set_option('display.width', None)       # Sin límite de ancho
pd.set_option('display.max_colwidth', 50)  # Ancho de columnas
warnings.filterwarnings('ignore')

class DataLoader:
    """
    Cargador unificado para datasets de NASA (KOI, TOI, K2)
    Implementa mapeo de etiquetas consistente entre datasets
    """
    
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.datasets_path = self.project_root / "data" / "datasets"
        self.new_datasets_path = self.project_root / "data" / "new_datasets"
        
        # Mapeo de etiquetas para consistencia entre datasets
        self.label_mapping = {
            # KOI (Kepler Objects of Interest)
            'koi_disposition': {
                'CONFIRMED': 1,
                'CANDIDATE': 0,
                'FALSE POSITIVE': 0
            },
            # TOI (TESS Objects of Interest) 
            'tfopwg_disp': {
                'KP': 1,      # Known Planet
                'PC': 0,      # Planet Candidate
                'FP': 0,      # False Positive
                'APC': 0      # Ambiguous Planet Candidate
            },
            # K2 Planets and Candidates
            'archive_disp': {
                'CONFIRMED': 1,
                'CANDIDATE': 0,
                'FALSE POSITIVE': 0
            }
        }
    
    def load_dataset(self, filename, dataset_type='KOI'):
        """Carga un dataset específico con manejo de metadatos NASA"""
        file_path = self.datasets_path / filename
        
        print(f"📁 Cargando {dataset_type}: {filename}")
        
        try:
            # Los archivos NASA tienen comentarios con '#'
            df = pd.read_csv(file_path, 
                           comment='#',          # Ignorar metadatos
                           sep=',',              
                           engine='python')      # Motor Python para manejar comentarios
            
            print(f"✅ Dataset {dataset_type} cargado exitosamente!")
            print(f"📊 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
            print(f"💾 Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            return df, dataset_type
            
        except Exception as e:
            print(f"❌ Error cargando {filename}: {e}")
            return None, dataset_type
    
    def load_all_datasets(self):
        """Carga los 3 datasets principales de NASA"""
        datasets = {}
        
        # Dataset 1: KOI (Kepler Objects of Interest)
        koi_df, _ = self.load_dataset("cumulative_2025.10.04_11.46.06.csv", "KOI")
        if koi_df is not None:
            datasets['KOI'] = koi_df
        
        # Dataset 2: TOI (TESS Objects of Interest)
        toi_df, _ = self.load_dataset("TOI_2025.10.04_11.44.53.csv", "TOI") 
        if toi_df is not None:
            datasets['TOI'] = toi_df
            
        # Dataset 3: K2 Planets and Candidates
        k2_df, _ = self.load_dataset("k2pandc_2025.10.04_11.46.18.csv", "K2")
        if k2_df is not None:
            datasets['K2'] = k2_df
        
        return datasets
    
    def analyze_dataset(self, df, dataset_name):
        """Análisis exploratorio detallado de cada dataset"""
        print(f"\n" + "="*60)
        print(f"🔬 ANÁLISIS DETALLADO - {dataset_name}")
        print(f"=" * 60)
        
        # Información básica
        print(f"\n� Información General:")
        print(f"• Filas: {df.shape[0]:,}")
        print(f"• Columnas: {df.shape[1]}")
        print(f"• Memoria total: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Mostrar TODAS las columnas disponibles
        print(f"\n🔤 TODAS las columnas disponibles ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        # Tipos de datos
        print(f"\n📋 Distribución de tipos de datos:")
        type_counts = df.dtypes.value_counts()
        for dtype, count in type_counts.items():
            print(f"  • {dtype}: {count} columnas")
        
        # Valores faltantes
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_info = pd.DataFrame({
            'Columna': missing_data.index,
            'Valores_Faltantes': missing_data.values,
            'Porcentaje': missing_percent.values
        })
        missing_info = missing_info[missing_info['Valores_Faltantes'] > 0].sort_values('Porcentaje', ascending=False)
        
        print(f"\n🔍 Valores faltantes ({missing_info.shape[0]} columnas con datos faltantes):")
        if len(missing_info) > 0:
            print(missing_info.head(10).to_string(index=False))
        else:
            print("  ✅ No hay valores faltantes")
        
        # Identificar columna de disposición/clasificación
        disposition_cols = [col for col in df.columns if 'disposition' in col.lower()]
        if disposition_cols:
            target_col = disposition_cols[0]
            print(f"\n🎯 Variable objetivo identificada: '{target_col}'")
            print(f"Distribución de clases:")
            class_counts = df[target_col].value_counts()
            for class_name, count in class_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  • {class_name}: {count:,} ({percentage:.1f}%)")
        
        # Columnas numéricas clave para exoplanetas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exoplanet_features = ['period', 'prad', 'teq', 'srad', 'smass', 'steff', 'depth', 'duration']
        key_numeric = [col for col in numeric_cols if any(feature in col.lower() for feature in exoplanet_features)]
        
        print(f"\n🌟 Características astronómicas clave identificadas ({len(key_numeric)}):")
        for col in key_numeric[:10]:  # Mostrar primeras 10
            print(f"  • {col}")
        
        # Muestra de datos (primeras 3 filas, todas las columnas)
        print(f"\n📊 Muestra de datos (primeras 3 filas):")
        print(df.head(3).to_string())
        
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'target_column': disposition_cols[0] if disposition_cols else None,
            'numeric_features': numeric_cols,
            'key_features': key_numeric,
            'missing_data': missing_info
        }

# Inicializar el sistema
print("🚀 Inicializando Sistema de Detección de Exoplanetas")
print("� NASA Space Apps Challenge 2025")
print("-" * 60)

# Obtener directorio del proyecto
current_dir = Path(__file__).parent
project_root = current_dir.parent

# Crear instancia del cargador
loader = DataLoader(project_root)

# Cargar todos los datasets
print("\n📂 Cargando datasets de NASA...")
datasets = loader.load_all_datasets()

# Análisis detallado de cada dataset
analysis_results = {}
for name, df in datasets.items():
    analysis_results[name] = loader.analyze_dataset(df, name)

# Resumen general
print(f"\n" + "="*60)
print(f"📈 RESUMEN GENERAL DEL SISTEMA")
print(f"=" * 60)
print(f"• Datasets cargados: {len(datasets)}")
for name, df in datasets.items():
    print(f"  - {name}: {df.shape[0]:,} objetos × {df.shape[1]} características")

print(f"\n🎯 Próximos pasos:")
print(f"1. Implementar preprocesamiento unificado")
print(f"2. Feature engineering astronómico") 
print(f"3. Ensemble learning (Stacking + Random Forest + AdaBoost)")
print(f"4. Validación cruzada estratificada")
print(f"5. Predicción en nuevos datasets (carpeta: data/new_datasets)")
print(f"6. Exportar resultados (carpeta: exoPlanet_results)")

# Verificar carpetas del sistema
print(f"\n📁 Estructura del sistema:")
print(f"  • Datasets originales: {loader.datasets_path}")
print(f"  • Nuevos datasets: {loader.new_datasets_path}")
print(f"  • Resultados: {project_root / 'exoPlanet_results'}")
print(f"  • Modelos: {project_root / 'models'}")

# Mostrar comandos disponibles
print(f"\n💻 Comandos disponibles:")
print(f"  • Entrenar modelo: python train_ensemble.py")
print(f"  • Predecir nuevo dataset: python predict_exoplanets.py <archivo.csv>")
print(f"  • Análisis completo: python full_analysis.py")