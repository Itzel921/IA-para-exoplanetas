"""
Módulo de visualización avanzada para análisis exploratorio de exoplanetas
Genera gráficas comparativas entre datasets de NASA y new_datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('default')
sns.set_palette("husl")

class ExoplanetVisualizer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.charts_path = self.project_root / "exoPlanet_results" / "charts"
        self.comparative_path = self.project_root / "exoPlanet_results" / "comparative_charts"
        
        # Crear carpetas si no existen
        self.charts_path.mkdir(parents=True, exist_ok=True)
        self.comparative_path.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def create_dataset_overview(self, datasets):
        """Crear gráfica de resumen general de datasets"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📊 Resumen General de Datasets NASA', fontsize=16, fontweight='bold')
        
        # 1. Tamaños de datasets
        sizes = [len(df) for df in datasets.values()]
        names = list(datasets.keys())
        
        axes[0, 0].bar(names, sizes, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('Número de Objetos por Dataset')
        axes[0, 0].set_ylabel('Cantidad de Objetos')
        for i, v in enumerate(sizes):
            axes[0, 0].text(i, v + max(sizes)*0.01, f'{v:,}', ha='center', fontweight='bold')
        
        # 2. Número de características
        feature_counts = [df.shape[1] for df in datasets.values()]
        axes[0, 1].bar(names, feature_counts, color=['#96CEB4', '#FECA57', '#FF9FF3'])
        axes[0, 1].set_title('Número de Características por Dataset')
        axes[0, 1].set_ylabel('Cantidad de Características')
        for i, v in enumerate(feature_counts):
            axes[0, 1].text(i, v + max(feature_counts)*0.01, f'{v}', ha='center', fontweight='bold')
        
        # 3. Memoria utilizada
        memory_mb = [df.memory_usage(deep=True).sum() / 1024**2 for df in datasets.values()]
        axes[1, 0].bar(names, memory_mb, color=['#F38BA8', '#A8DADC', '#457B9D'])
        axes[1, 0].set_title('Uso de Memoria por Dataset')
        axes[1, 0].set_ylabel('Memoria (MB)')
        for i, v in enumerate(memory_mb):
            axes[1, 0].text(i, v + max(memory_mb)*0.01, f'{v:.1f} MB', ha='center', fontweight='bold')
        
        # 4. Valores faltantes promedio
        missing_pct = [(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100) for df in datasets.values()]
        axes[1, 1].bar(names, missing_pct, color=['#E63946', '#F77F00', '#FCBF49'])
        axes[1, 1].set_title('Porcentaje de Valores Faltantes')
        axes[1, 1].set_ylabel('% Valores Faltantes')
        for i, v in enumerate(missing_pct):
            axes[1, 1].text(i, v + max(missing_pct)*0.01, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.charts_path / f"01_dataset_overview_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gráfica guardada: {output_path.name}")
        return output_path
    
    def create_class_distributions(self, datasets):
        """Crear distribuciones de clases por dataset"""
        # Identificar columnas target
        target_columns = {
            'KOI': 'koi_disposition',
            'TOI': 'tfopwg_disp', 
            'K2': 'disposition'
        }
        
        fig, axes = plt.subplots(1, len(datasets), figsize=(15, 5))
        if len(datasets) == 1:
            axes = [axes]
        
        fig.suptitle('🎯 Distribución de Clases por Dataset', fontsize=16, fontweight='bold')
        
        for i, (name, df) in enumerate(datasets.items()):
            target_col = target_columns.get(name)
            
            if target_col and target_col in df.columns:
                class_counts = df[target_col].value_counts()
                
                # Crear gráfica de pie
                colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
                wedges, texts, autotexts = axes[i].pie(
                    class_counts.values, 
                    labels=class_counts.index,
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90
                )
                axes[i].set_title(f'{name}\n({len(df):,} objetos)')
                
                # Mejorar texto
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            else:
                axes[i].text(0.5, 0.5, f'Sin columna\ntarget identificada', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{name}')
        
        plt.tight_layout()
        output_path = self.charts_path / f"02_class_distributions_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gráfica guardada: {output_path.name}")
        return output_path
    
    def create_correlation_matrices(self, datasets):
        """Crear matrices de correlación para características clave"""
        fig, axes = plt.subplots(1, len(datasets), figsize=(18, 5))
        if len(datasets) == 1:
            axes = [axes]
        
        fig.suptitle('🔗 Matrices de Correlación - Características Astronómicas', fontsize=16, fontweight='bold')
        
        # Características astronómicas clave comunes
        key_features = {
            'KOI': ['koi_period', 'koi_prad', 'koi_teq', 'koi_steff', 'koi_srad'],
            'TOI': ['pl_orbper', 'pl_rade', 'pl_eqt', 'st_teff', 'st_rad'],
            'K2': ['pl_orbper', 'pl_rade', 'pl_eqt', 'st_teff', 'st_rad']
        }
        
        for i, (name, df) in enumerate(datasets.items()):
            features = key_features.get(name, [])
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) >= 2:
                # Seleccionar solo datos numéricos
                numeric_df = df[available_features].select_dtypes(include=[np.number])
                
                if not numeric_df.empty:
                    correlation = numeric_df.corr()
                    
                    # Crear heatmap
                    sns.heatmap(correlation, annot=True, cmap='RdBu_r', center=0,
                              square=True, ax=axes[i], cbar_kws={'shrink': 0.8})
                    axes[i].set_title(f'{name}\nCorrelación de Características')
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].tick_params(axis='y', rotation=0)
            else:
                axes[i].text(0.5, 0.5, f'Insuficientes\ncaracterísticas numéricas', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{name}')
        
        plt.tight_layout()
        output_path = self.charts_path / f"03_correlation_matrices_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gráfica guardada: {output_path.name}")
        return output_path
    
    def create_scatter_plots(self, datasets):
        """Crear gráficas de dispersión para características clave"""
        fig, axes = plt.subplots(2, len(datasets), figsize=(18, 10))
        if len(datasets) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('📈 Gráficas de Dispersión - Relaciones Astronómicas', fontsize=16, fontweight='bold')
        
        # Pares de características para scatter plots
        scatter_pairs = {
            'KOI': [('koi_period', 'koi_prad'), ('koi_teq', 'koi_steff')],
            'TOI': [('pl_orbper', 'pl_rade'), ('pl_eqt', 'st_teff')], 
            'K2': [('pl_orbper', 'pl_rade'), ('pl_eqt', 'st_teff')]
        }
        
        target_columns = {
            'KOI': 'koi_disposition',
            'TOI': 'tfopwg_disp',
            'K2': 'disposition'
        }
        
        for i, (name, df) in enumerate(datasets.items()):
            pairs = scatter_pairs.get(name, [])
            target_col = target_columns.get(name)
            
            for j, (x_col, y_col) in enumerate(pairs[:2]):  # Solo los primeros 2 pares
                if x_col in df.columns and y_col in df.columns:
                    # Crear scatter plot
                    if target_col and target_col in df.columns:
                        # Colorear por clase
                        for class_name in df[target_col].unique():
                            class_data = df[df[target_col] == class_name]
                            axes[j, i].scatter(class_data[x_col], class_data[y_col], 
                                             label=class_name, alpha=0.6, s=20)
                        axes[j, i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    else:
                        axes[j, i].scatter(df[x_col], df[y_col], alpha=0.6, s=20)
                    
                    axes[j, i].set_xlabel(x_col)
                    axes[j, i].set_ylabel(y_col)
                    axes[j, i].set_title(f'{name}: {x_col} vs {y_col}')
                    axes[j, i].grid(True, alpha=0.3)
                else:
                    axes[j, i].text(0.5, 0.5, f'Características\nno disponibles', 
                                   ha='center', va='center', transform=axes[j, i].transAxes)
                    axes[j, i].set_title(f'{name}')
        
        plt.tight_layout()
        output_path = self.charts_path / f"04_scatter_plots_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gráfica guardada: {output_path.name}")
        return output_path
    
    def create_nasa_vs_new_comparison(self, nasa_datasets, new_datasets):
        """Crear comparaciones entre datasets NASA y new_datasets"""
        if not new_datasets:
            print("⚠️ No hay new_datasets para comparar")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🆚 Comparación: Datasets NASA vs New Datasets', fontsize=16, fontweight='bold')
        
        # 1. Comparación de tamaños
        nasa_sizes = {name: len(df) for name, df in nasa_datasets.items()}
        new_sizes = {name: len(df) for name, df in new_datasets.items()}
        
        all_names = list(nasa_sizes.keys()) + [f"NEW_{name}" for name in new_sizes.keys()]
        all_sizes = list(nasa_sizes.values()) + list(new_sizes.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'] + ['#FFA07A', '#98FB98', '#87CEEB'][:len(new_sizes)]
        
        axes[0, 0].bar(all_names, all_sizes, color=colors[:len(all_sizes)])
        axes[0, 0].set_title('Comparación de Tamaños')
        axes[0, 0].set_ylabel('Número de Objetos')
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(all_sizes):
            axes[0, 0].text(i, v + max(all_sizes)*0.01, f'{v:,}', ha='center', fontweight='bold')
        
        # 2. Distribución de características (coordenadas)
        nasa_coords = []
        new_coords = []
        
        for name, df in nasa_datasets.items():
            if 'ra' in df.columns and 'dec' in df.columns:
                nasa_coords.extend(list(zip(df['ra'].dropna(), df['dec'].dropna())))
        
        for name, df in new_datasets.items():
            if 'ra' in df.columns and 'dec' in df.columns:
                new_coords.extend(list(zip(df['ra'].dropna(), df['dec'].dropna())))
        
        if nasa_coords and new_coords:
            nasa_ra, nasa_dec = zip(*nasa_coords) if nasa_coords else ([], [])
            new_ra, new_dec = zip(*new_coords) if new_coords else ([], [])
            
            axes[0, 1].scatter(nasa_ra, nasa_dec, alpha=0.5, s=1, label='NASA Datasets', color='blue')
            axes[0, 1].scatter(new_ra, new_dec, alpha=0.7, s=3, label='New Datasets', color='red')
            axes[0, 1].set_xlabel('Ascensión Recta (RA)')
            axes[0, 1].set_ylabel('Declinación (DEC)')
            axes[0, 1].set_title('Distribución de Coordenadas Celestes')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Comparación de características disponibles
        nasa_features = set()
        for df in nasa_datasets.values():
            nasa_features.update(df.columns)
        
        new_features = set()
        for df in new_datasets.values():
            new_features.update(df.columns)
        
        common_features = nasa_features & new_features
        nasa_only = nasa_features - new_features
        new_only = new_features - nasa_features
        
        feature_comparison = {
            'Comunes': len(common_features),
            'Solo NASA': len(nasa_only),
            'Solo NEW': len(new_only)
        }
        
        axes[1, 0].pie(feature_comparison.values(), labels=feature_comparison.keys(),
                      autopct='%1.1f%%', startangle=90, colors=['#90EE90', '#FFB6C1', '#87CEFA'])
        axes[1, 0].set_title('Comparación de Características')
        
        # 4. Estadísticas de calidad de datos
        nasa_missing_pct = np.mean([df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100 
                                   for df in nasa_datasets.values()])
        new_missing_pct = np.mean([df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100 
                                  for df in new_datasets.values()]) if new_datasets else 0
        
        quality_metrics = ['NASA Datasets', 'New Datasets']
        missing_values = [nasa_missing_pct, new_missing_pct]
        
        axes[1, 1].bar(quality_metrics, missing_values, color=['#4ECDC4', '#FFA07A'])
        axes[1, 1].set_title('Calidad de Datos (% Valores Faltantes)')
        axes[1, 1].set_ylabel('% Valores Faltantes')
        for i, v in enumerate(missing_values):
            axes[1, 1].text(i, v + max(missing_values)*0.01, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.comparative_path / f"nasa_vs_new_comparison_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparación guardada: {output_path.name}")
        return output_path
    
    def create_prediction_results_analysis(self, predictions_files):
        """Analizar resultados de predicciones"""
        if not predictions_files:
            print("⚠️ No hay archivos de predicciones para analizar")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('🔮 Análisis de Resultados de Predicciones', fontsize=16, fontweight='bold')
        
        all_predictions = []
        file_stats = {}
        
        # Cargar todos los archivos de predicciones
        for pred_file in predictions_files:
            try:
                df = pd.read_csv(pred_file)
                file_name = pred_file.stem
                
                if 'prediction_label' in df.columns:
                    confirmed = sum(df['prediction_label'] == 'CONFIRMED')
                    false_pos = sum(df['prediction_label'] == 'FALSE_POSITIVE')
                    
                    file_stats[file_name] = {
                        'confirmed': confirmed,
                        'false_positive': false_pos,
                        'total': len(df)
                    }
                    
                    all_predictions.extend(df['prediction_label'].tolist())
                    
            except Exception as e:
                print(f"⚠️ Error cargando {pred_file}: {e}")
        
        if file_stats:
            # 1. Distribución general de predicciones
            from collections import Counter
            pred_counts = Counter(all_predictions)
            
            axes[0, 0].pie(pred_counts.values(), labels=pred_counts.keys(), 
                          autopct='%1.1f%%', startangle=90, 
                          colors=['#FF6B6B', '#4ECDC4'])
            axes[0, 0].set_title('Distribución General de Predicciones')
            
            # 2. Predicciones por archivo
            files = list(file_stats.keys())
            confirmed_counts = [file_stats[f]['confirmed'] for f in files]
            false_pos_counts = [file_stats[f]['false_positive'] for f in files]
            
            x = np.arange(len(files))
            width = 0.35
            
            axes[0, 1].bar(x - width/2, confirmed_counts, width, label='CONFIRMED', color='#4ECDC4')
            axes[0, 1].bar(x + width/2, false_pos_counts, width, label='FALSE_POSITIVE', color='#FF6B6B')
            axes[0, 1].set_title('Predicciones por Archivo')
            axes[0, 1].set_xlabel('Archivos')
            axes[0, 1].set_ylabel('Cantidad')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels([f[:15] + '...' if len(f) > 15 else f for f in files], 
                                      rotation=45)
            axes[0, 1].legend()
            
            # 3. Porcentajes de confirmación por archivo
            confirmation_rates = [file_stats[f]['confirmed'] / file_stats[f]['total'] * 100 
                                 for f in files]
            
            axes[1, 0].bar(files, confirmation_rates, color='#45B7D1')
            axes[1, 0].set_title('Tasa de Confirmación por Archivo')
            axes[1, 0].set_ylabel('% Confirmados')
            axes[1, 0].tick_params(axis='x', rotation=45)
            for i, v in enumerate(confirmation_rates):
                axes[1, 0].text(i, v + max(confirmation_rates)*0.01, f'{v:.1f}%', 
                               ha='center', fontweight='bold')
            
            # 4. Confianza de predicciones (si disponible)
            confidence_data = []
            try:
                for pred_file in predictions_files:
                    df = pd.read_csv(pred_file)
                    if 'confidence' in df.columns:
                        confidence_data.extend(df['confidence'].tolist())
                
                if confidence_data:
                    axes[1, 1].hist(confidence_data, bins=20, alpha=0.7, color='#96CEB4', edgecolor='black')
                    axes[1, 1].set_title('Distribución de Confianza de Predicciones')
                    axes[1, 1].set_xlabel('Confianza')
                    axes[1, 1].set_ylabel('Frecuencia')
                    axes[1, 1].axvline(np.mean(confidence_data), color='red', linestyle='--', 
                                      label=f'Media: {np.mean(confidence_data):.3f}')
                    axes[1, 1].legend()
                else:
                    axes[1, 1].text(0.5, 0.5, 'Sin datos de\nconfianza disponibles',
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
            except:
                axes[1, 1].text(0.5, 0.5, 'Error cargando\ndatos de confianza',
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        output_path = self.comparative_path / f"prediction_results_analysis_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Análisis de predicciones guardado: {output_path.name}")
        return output_path
    
    def generate_complete_analysis(self, nasa_datasets, new_datasets=None):
        """Generar análisis completo con todas las visualizaciones"""
        print(f"\n🎨 GENERANDO VISUALIZACIONES COMPLETAS")
        print("="*60)
        
        generated_files = []
        
        # Gráficas individuales de datasets NASA
        print("\n📊 Generando gráficas de datasets NASA...")
        generated_files.append(self.create_dataset_overview(nasa_datasets))
        generated_files.append(self.create_class_distributions(nasa_datasets))
        generated_files.append(self.create_correlation_matrices(nasa_datasets))
        generated_files.append(self.create_scatter_plots(nasa_datasets))
        
        # Comparaciones NASA vs new_datasets
        if new_datasets:
            print("\n🆚 Generando comparaciones NASA vs New Datasets...")
            generated_files.append(self.create_nasa_vs_new_comparison(nasa_datasets, new_datasets))
        
        # Análisis de resultados de predicciones
        print("\n🔮 Analizando resultados de predicciones...")
        results_path = self.project_root / "exoPlanet_results"
        prediction_files = list(results_path.glob("*predictions*.csv"))
        if prediction_files:
            generated_files.append(self.create_prediction_results_analysis(prediction_files))
        
        print(f"\n✅ Análisis completo generado!")
        print(f"   📁 Charts: {len([f for f in generated_files if 'charts' in str(f)])} archivos")
        print(f"   📁 Comparative: {len([f for f in generated_files if 'comparative' in str(f)])} archivos")
        
        return [f for f in generated_files if f is not None]