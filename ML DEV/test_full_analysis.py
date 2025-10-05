#!/usr/bin/env python3
"""
Script de prueba para el análisis exploratorio completo
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("🚀 Probando análisis exploratorio completo...")
    
    from advanced_visualization import ExoplanetVisualizer
    from Clasification import DataLoader
    from pathlib import Path
    import pandas as pd
    
    # Configurar paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    print("✅ Módulos importados correctamente")
    
    # Cargar datasets NASA
    print("\n📁 Cargando datasets NASA...")
    loader = DataLoader(project_root)
    nasa_datasets = loader.load_all_datasets()
    
    print(f"✅ Datasets NASA cargados: {list(nasa_datasets.keys())}")
    
    # Cargar new_datasets
    print("\n📁 Cargando new_datasets...")
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
    
    # Crear visualizador y generar análisis
    print("\n🎨 Generando visualizaciones...")
    visualizer = ExoplanetVisualizer(project_root)
    generated_files = visualizer.generate_complete_analysis(nasa_datasets, new_datasets)
    
    print(f"\n✅ Análisis completado!")
    print(f"   📊 Archivos generados: {len(generated_files)}")
    for file_path in generated_files:
        print(f"      • {file_path.name}")
        
except ImportError as e:
    print(f"❌ Error de importación: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()