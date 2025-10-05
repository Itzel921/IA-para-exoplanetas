#!/usr/bin/env python3
"""
🔬 ANÁLISIS COMPARATIVO - PREDICCIONES VS DATASETS DE ENTRENAMIENTO
NASA Space Apps Challenge 2025
================================================================

Compara los resultados de predicción con los datasets originales (KOI, TOI, K2)
para evaluar el rendimiento del modelo en diferentes fuentes de datos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'

class DatasetComparator:
    def __init__(self):
        """
        Inicializa el comparador de datasets
        """
        self.base_path = Path(__file__).parent.parent
        self.datasets_path = self.base_path / "data" / "datasets"
        self.results_path = self.base_path / "exoPlanet_results"
        self.output_dir = self.results_path / "comparative_charts"
        self.output_dir.mkdir(exist_ok=True)
        
        # Cargar datasets originales
        self.load_original_datasets()
        
        # Cargar resultados de predicción
        self.load_prediction_results()
        
    def load_original_datasets(self):
        """
        Carga los datasets originales de entrenamiento con manejo robusto de errores
        """
        print("📂 Cargando datasets originales...")
        
        # KOI Dataset
        try:
            koi_path = self.datasets_path / "cumulative_2025.10.04_11.46.06.csv"
            # Los datos NASA tienen muchas líneas de comentario, buscar la línea de headers
            with open(koi_path, 'r') as f:
                lines = f.readlines()
                header_line = 0
                for i, line in enumerate(lines):
                    if not line.startswith('#') and 'kepid' in line:
                        header_line = i
                        break
            
            self.koi_df = pd.read_csv(koi_path, skiprows=header_line, low_memory=False)
            print(f"✅ KOI cargado: {len(self.koi_df):,} objetos")
            print(f"   Columnas clave: koi_disposition, koi_period, koi_prad, ra, dec")
            
            # Verificar columnas importantes
            key_cols = ['koi_disposition', 'koi_period', 'koi_prad', 'ra', 'dec']
            available_cols = [col for col in key_cols if col in self.koi_df.columns]
            print(f"   Disponibles: {available_cols}")
            
        except Exception as e:
            print(f"❌ Error cargando KOI: {e}")
            self.koi_df = pd.DataFrame()
        
        # TOI Dataset
        try:
            toi_path = self.datasets_path / "TOI_2025.10.04_11.44.53.csv"
            # Encontrar línea de headers para TOI
            with open(toi_path, 'r') as f:
                lines = f.readlines()
                header_line = 0
                for i, line in enumerate(lines):
                    if not line.startswith('#') and ('toi' in line.lower() or 'tic_id' in line.lower()):
                        header_line = i
                        break
            
            self.toi_df = pd.read_csv(toi_path, skiprows=header_line, low_memory=False)
            print(f"✅ TOI cargado: {len(self.toi_df):,} objetos")
            print(f"   Columnas clave: tfopwg_disp, pl_orbper, pl_rade, ra, dec")
            
            # Verificar columnas importantes
            key_cols = ['tfopwg_disp', 'pl_orbper', 'pl_rade', 'ra', 'dec', 'toi', 'tic_id']
            available_cols = [col for col in key_cols if col in self.toi_df.columns]
            print(f"   Disponibles: {available_cols}")
            
        except Exception as e:
            print(f"❌ Error cargando TOI: {e}")
            self.toi_df = pd.DataFrame()
        
        # K2 Dataset  
        try:
            k2_path = self.datasets_path / "k2pandc_2025.10.04_11.46.18.csv"
            # Encontrar línea de headers para K2
            with open(k2_path, 'r') as f:
                lines = f.readlines()
                header_line = 0
                for i, line in enumerate(lines):
                    if not line.startswith('#') and ('epic' in line.lower() or 'k2' in line.lower() or 'pl_name' in line.lower()):
                        header_line = i
                        break
            
            self.k2_df = pd.read_csv(k2_path, skiprows=header_line, low_memory=False)
            print(f"✅ K2 cargado: {len(self.k2_df):,} objetos")
            print(f"   Columnas clave: pl_discmethod, pl_orbper, pl_rade, ra, dec")
            
            # Verificar columnas importantes para K2
            key_cols = ['pl_discmethod', 'pl_orbper', 'pl_rade', 'ra', 'dec', 'pl_name', 'epic_name']
            available_cols = [col for col in key_cols if col in self.k2_df.columns]
            print(f"   Disponibles: {available_cols}")
            
        except Exception as e:
            print(f"❌ Error cargando K2: {e}")
            self.k2_df = pd.DataFrame()
            
    def load_prediction_results(self):
        """
        Carga los resultados de predicción más recientes
        """
        print("📊 Cargando resultados de predicción...")
        
        # Buscar archivo de predicciones más reciente
        prediction_files = list(self.results_path.glob("*predictions*.csv"))
        
        if not prediction_files:
            raise FileNotFoundError("No se encontraron archivos de predicción")
            
        latest_prediction = max(prediction_files, key=lambda x: x.stat().st_mtime)
        self.predictions_df = pd.read_csv(latest_prediction, low_memory=False)
        print(f"✅ Predicciones cargadas: {len(self.predictions_df):,} objetos")
        print(f"   Archivo: {latest_prediction.name}")
        
    def analyze_dataset_distributions(self):
        """
        Analiza las distribuciones de clases en cada dataset original
        """
        print("📊 Analizando distribuciones de clases originales...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('black')
        
        # === KOI Distribution ===
        if not self.koi_df.empty and 'koi_disposition' in self.koi_df.columns:
            koi_counts = self.koi_df['koi_disposition'].value_counts()
            colors_koi = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#6BCF7F']
            
            wedges1, texts1, autotexts1 = ax1.pie(koi_counts.values, labels=koi_counts.index, 
                                                  colors=colors_koi, autopct='%1.1f%%', startangle=90)
            ax1.set_title('📊 KOI Dataset - Distribución Original', fontsize=14, fontweight='bold')
            
            # Estadísticas en texto
            total_koi = len(self.koi_df)
            confirmed_koi = koi_counts.get('CONFIRMED', 0)
            candidates_koi = koi_counts.get('CANDIDATE', 0)
            fp_koi = koi_counts.get('FALSE POSITIVE', 0)
            
            koi_stats = f"""KOI DATASET
Total: {total_koi:,}
Confirmed: {confirmed_koi:,} ({confirmed_koi/total_koi*100:.1f}%)
Candidates: {candidates_koi:,} ({candidates_koi/total_koi*100:.1f}%)
False Pos: {fp_koi:,} ({fp_koi/total_koi*100:.1f}%)"""
        else:
            ax1.text(0.5, 0.5, 'KOI Dataset\nNo disponible', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=14)
            ax1.set_title('📊 KOI Dataset', fontsize=14)
            koi_stats = "KOI Dataset: No disponible"
        
        # === TOI Distribution ===
        if not self.toi_df.empty and 'tfopwg_disp' in self.toi_df.columns:
            toi_counts = self.toi_df['tfopwg_disp'].value_counts()
            colors_toi = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#A8E6CF', '#FF8B94']
            
            # Filtrar valores válidos
            valid_toi = toi_counts.head(5)  # TOP 5 categorías
            
            wedges2, texts2, autotexts2 = ax2.pie(valid_toi.values, labels=valid_toi.index,
                                                  colors=colors_toi[:len(valid_toi)], 
                                                  autopct='%1.1f%%', startangle=90)
            ax2.set_title('📊 TOI Dataset - Distribución Original', fontsize=14, fontweight='bold')
            
            total_toi = len(self.toi_df)
            toi_stats = f"""TOI DATASET
Total: {total_toi:,}
Categorías: {len(toi_counts)}
Principal: {valid_toi.index[0]} ({valid_toi.iloc[0]:,})"""
        else:
            ax2.text(0.5, 0.5, 'TOI Dataset\nNo disponible', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('📊 TOI Dataset', fontsize=14)
            toi_stats = "TOI Dataset: No disponible"
        
        # === K2 Distribution ===
        if not self.k2_df.empty and 'disposition' in self.k2_df.columns:
            k2_counts = self.k2_df['disposition'].value_counts()
            colors_k2 = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#DDA0DD']
            
            wedges3, texts3, autotexts3 = ax3.pie(k2_counts.values, labels=k2_counts.index,
                                                  colors=colors_k2[:len(k2_counts)], 
                                                  autopct='%1.1f%%', startangle=90)
            ax3.set_title('📊 K2 Dataset - Distribución Original', fontsize=14, fontweight='bold')
            
            total_k2 = len(self.k2_df)
            confirmed_k2 = k2_counts.get('CONFIRMED', 0)
            candidates_k2 = k2_counts.get('CANDIDATE', 0)
            
            k2_stats = f"""K2 DATASET
Total: {total_k2:,}
Confirmed: {confirmed_k2:,} ({confirmed_k2/total_k2*100:.1f}%)
Candidates: {candidates_k2:,} ({candidates_k2/total_k2*100:.1f}%)"""
        else:
            ax3.text(0.5, 0.5, 'K2 Dataset\nNo disponible', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('📊 K2 Dataset', fontsize=14)
            k2_stats = "K2 Dataset: No disponible"
        
        # === Predictions Distribution ===
        pred_counts = self.predictions_df['ML_Prediction'].value_counts()
        pred_labels = ['NO PLANETA', 'EXOPLANETA']
        pred_colors = ['#FF6B6B', '#4ECDC4']
        
        wedges4, texts4, autotexts4 = ax4.pie(pred_counts.values, labels=pred_labels,
                                              colors=pred_colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('🔮 Predicciones ML - Resultado', fontsize=14, fontweight='bold')
        
        total_pred = len(self.predictions_df)
        exoplanets_pred = pred_counts.get(1, 0)
        no_planets_pred = pred_counts.get(0, 0)
        avg_conf = self.predictions_df['ML_Confidence'].mean()
        
        pred_stats = f"""PREDICCIONES ML
Total: {total_pred:,}
Exoplanetas: {exoplanets_pred:,} ({exoplanets_pred/total_pred*100:.1f}%)
No-Planetas: {no_planets_pred:,} ({no_planets_pred/total_pred*100:.1f}%)
Confianza Avg: {avg_conf:.3f}"""
        
        # Agregar estadísticas como texto
        stats_text = f"""{koi_stats}

{toi_stats}

{k2_stats}

{pred_stats}"""
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=9, fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8),
                   color='white', verticalalignment='bottom')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # Hacer espacio para las estadísticas
        plt.savefig(self.output_dir / "01_dataset_distributions_comparison.png", 
                   dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print(f"✅ Gráfico guardado: 01_dataset_distributions_comparison.png")
        
    def compare_prediction_accuracy(self):
        """
        Compara la accuracy del modelo en objetos que aparecen en los datasets originales
        """
        print("🎯 Comparando accuracy con datasets originales...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('black')
        
        # === Comparación con KOI ===
        # Las predicciones ya contienen toda la información de KOI, así que podemos comparar directamente
        if 'koi_disposition' in self.predictions_df.columns and 'ML_Prediction' in self.predictions_df.columns:
            # Filtrar solo objetos con etiquetas válidas
            valid_koi = self.predictions_df.dropna(subset=['koi_disposition', 'ML_Prediction'])
            
            if len(valid_koi) > 0:
                # Convertir a labels binarios
                valid_koi = valid_koi.copy()
                valid_koi['True_Label'] = (valid_koi['koi_disposition'] == 'CONFIRMED').astype(int)
                
                # Calcular métricas
                tp_koi = sum((valid_koi['True_Label'] == 1) & (valid_koi['ML_Prediction'] == 1))
                fp_koi = sum((valid_koi['True_Label'] == 0) & (valid_koi['ML_Prediction'] == 1))
                fn_koi = sum((valid_koi['True_Label'] == 1) & (valid_koi['ML_Prediction'] == 0))
                tn_koi = sum((valid_koi['True_Label'] == 0) & (valid_koi['ML_Prediction'] == 0))
                
                accuracy_koi = (tp_koi + tn_koi) / len(valid_koi) if len(valid_koi) > 0 else 0
                precision_koi = tp_koi / (tp_koi + fp_koi) if (tp_koi + fp_koi) > 0 else 0
                recall_koi = tp_koi / (tp_koi + fn_koi) if (tp_koi + fn_koi) > 0 else 0
                
                # Matriz de confusión
                cm_koi = [[tn_koi, fp_koi], [fn_koi, tp_koi]]
                im1 = ax1.imshow(cm_koi, interpolation='nearest', cmap='Blues')
                ax1.set_title(f'🎯 KOI Dataset\nAccuracy: {accuracy_koi:.3f} | Precision: {precision_koi:.3f} | Recall: {recall_koi:.3f}\n({len(valid_koi):,} objetos validados)', 
                             fontsize=10, fontweight='bold')
                
                # Agregar números a la matriz
                for i in range(2):
                    for j in range(2):
                        ax1.text(j, i, cm_koi[i][j], ha="center", va="center", 
                               color="white", fontsize=16, fontweight='bold')
                
                ax1.set_xticks([0, 1])
                ax1.set_yticks([0, 1])
                ax1.set_xticklabels(['NO PLANETA', 'EXOPLANETA'])
                ax1.set_yticklabels(['NO PLANETA', 'EXOPLANETA'])
                ax1.set_xlabel('Predicción ML')
                ax1.set_ylabel('KOI Etiqueta Real')
            else:
                ax1.text(0.5, 0.5, 'Sin datos válidos\npara comparación KOI', ha='center', va='center',
                        transform=ax1.transAxes, fontsize=12)
                ax1.set_title('🎯 KOI Comparison')
        else:
            ax1.text(0.5, 0.5, 'Columnas KOI\nno disponibles', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('🎯 KOI Comparison')
        
        # === Análisis de Confianza por Categoría Original ===
        # Analizar confianza por disposición original en las predicciones
        dataset_names = []
        avg_confidences = []
        planet_counts = []
        
        if 'koi_disposition' in self.predictions_df.columns:
            # Análisis por categoría KOI
            koi_categories = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
            
            for category in koi_categories:
                category_data = self.predictions_df[self.predictions_df['koi_disposition'] == category]
                if len(category_data) > 0:
                    avg_conf = category_data['ML_Confidence'].mean()
                    planet_pred = sum(category_data['ML_Prediction'] == 1)
                    
                    dataset_names.append(f'{category}\n({len(category_data):,} objetos)')
                    avg_confidences.append(avg_conf)
                    planet_counts.append(planet_pred)
            
            # También agregar estadísticas de otras fuentes si están disponibles
            other_sources = 0
            for dataset_name, df in [('TOI Original', self.toi_df), ('K2 Original', self.k2_df)]:
                if not df.empty:
                    other_sources += len(df)
            
            if other_sources > 0:
                dataset_names.append(f'Otros Datasets\n({other_sources:,} objetos)')
                avg_confidences.append(0.5)  # Placeholder
                planet_counts.append(0)  # Placeholder
        
        # Gráfico de confianza promedio por dataset
        if dataset_names:
            bars2 = ax2.bar(dataset_names, avg_confidences, color=['#FF6B6B', '#4ECDC4', '#FFD93D'])
            ax2.set_ylabel('Confianza Promedio ML')
            ax2.set_title('📊 Confianza ML por Dataset Original', fontsize=12, fontweight='bold')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            
            # Agregar valores en las barras
            for bar, conf in zip(bars2, avg_confidences):
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # === Gráfico de planetas encontrados por dataset ===
        if dataset_names:
            bars3 = ax3.bar(dataset_names, planet_counts, color=['#FF6B6B', '#4ECDC4', '#FFD93D'])
            ax3.set_ylabel('Exoplanetas Predichos')
            ax3.set_title('🌟 Exoplanetas Encontrados por Dataset', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Agregar valores en las barras
            for bar, count in zip(bars3, planet_counts):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(planet_counts)*0.02,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # === Estadísticas comparativas ===
        comp_stats = f"""🔍 ANÁLISIS COMPARATIVO

📊 Datasets Originales:
• KOI: {len(self.koi_df):,} objetos
• TOI: {len(self.toi_df):,} objetos  
• K2: {len(self.k2_df):,} objetos
• Total Original: {len(self.koi_df) + len(self.toi_df) + len(self.k2_df):,}

🔮 Predicciones ML:
• Total Analizado: {len(self.predictions_df):,}
• Exoplanetas: {sum(self.predictions_df['ML_Prediction'] == 1):,}
• Confianza Avg: {self.predictions_df['ML_Confidence'].mean():.3f}

🎯 Comparación:
• Coincidencias detectadas por coords
• Análisis de accuracy por dataset
• Distribución de confianza"""
        
        ax4.text(0.05, 0.95, comp_stats, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8),
                color='white')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('📋 Estadísticas Comparativas', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "02_accuracy_comparison_by_dataset.png", 
                   dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print(f"✅ Gráfico guardado: 02_accuracy_comparison_by_dataset.png")
        
    def analyze_feature_distributions(self):
        """
        Compara las distribuciones de características entre datasets y predicciones
        """
        print("📈 Analizando distribuciones de características...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('black')
        
        # === Análisis de Períodos Orbitales ===
        period_data = []
        
        # KOI periods
        if not self.koi_df.empty and 'koi_period' in self.koi_df.columns:
            koi_periods = self.koi_df['koi_period'].dropna()
            if len(koi_periods) > 0:
                period_data.append(('KOI Original', koi_periods.values))
        
        # Predictions periods (si están disponibles)
        if 'koi_period' in self.predictions_df.columns:
            pred_periods = self.predictions_df['koi_period'].dropna()
            if len(pred_periods) > 0:
                period_data.append(('Predicciones', pred_periods.values))
        
        if period_data:
            colors = ['#FF6B6B', '#4ECDC4']
            for i, (label, data) in enumerate(period_data):
                # Usar log scale para períodos
                log_data = np.log10(data[data > 0])  # Solo períodos positivos
                ax1.hist(log_data, bins=30, alpha=0.7, label=label, color=colors[i % len(colors)])
            
            ax1.set_xlabel('Log10(Período Orbital) [días]')
            ax1.set_ylabel('Frecuencia')
            ax1.set_title('📊 Distribución de Períodos Orbitales')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # === Análisis de Radios Planetarios ===
        radius_data = []
        
        # KOI radii
        if not self.koi_df.empty and 'koi_prad' in self.koi_df.columns:
            koi_radii = self.koi_df['koi_prad'].dropna()
            if len(koi_radii) > 0:
                radius_data.append(('KOI Original', koi_radii.values))
        
        # Predictions radii
        if 'koi_prad' in self.predictions_df.columns:
            pred_radii = self.predictions_df['koi_prad'].dropna()
            if len(pred_radii) > 0:
                radius_data.append(('Predicciones', pred_radii.values))
        
        if radius_data:
            for i, (label, data) in enumerate(radius_data):
                # Filtrar outliers extremos
                filtered_data = data[(data > 0) & (data < 50)]  # Radios razonables
                ax2.hist(filtered_data, bins=30, alpha=0.7, label=label, color=colors[i % len(colors)])
            
            ax2.set_xlabel('Radio Planetario [R⊕]')
            ax2.set_ylabel('Frecuencia')
            ax2.set_title('📊 Distribución de Radios Planetarios')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # === Distribución de Confianza vs Características ===
        if not self.predictions_df.empty:
            # Scatter plot: Confianza vs Período (si disponible)
            if 'koi_period' in self.predictions_df.columns:
                pred_clean = self.predictions_df.dropna(subset=['ML_Confidence', 'koi_period'])
                if len(pred_clean) > 0:
                    scatter = ax3.scatter(pred_clean['koi_period'], pred_clean['ML_Confidence'],
                                        c=pred_clean['ML_Prediction'], cmap='RdYlBu',
                                        s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
                    
                    ax3.set_xlabel('Período Orbital [días]')
                    ax3.set_ylabel('Confianza ML')
                    ax3.set_title('🎯 Confianza vs Período Orbital')
                    ax3.set_xscale('log')
                    ax3.grid(True, alpha=0.3)
                    
                    # Colorbar
                    cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8)
                    cbar.set_label('Predicción (0=No, 1=Sí)', rotation=270, labelpad=15)
        
        # === Comparación de Distribuciones de Coordenadas ===
        coord_data = []
        
        # Coordenadas originales
        for name, df in [('KOI', self.koi_df), ('TOI', self.toi_df), ('K2', self.k2_df)]:
            if not df.empty and 'ra' in df.columns and 'dec' in df.columns:
                coords = df[['ra', 'dec']].dropna()
                if len(coords) > 0:
                    coord_data.append((name, coords))
        
        # Coordenadas de predicciones
        if 'ra' in self.predictions_df.columns and 'dec' in self.predictions_df.columns:
            pred_coords = self.predictions_df[['ra', 'dec']].dropna()
            if len(pred_coords) > 0:
                coord_data.append(('Predicciones', pred_coords))
        
        if coord_data:
            colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#A8E6CF']
            for i, (label, coords) in enumerate(coord_data):
                ax4.scatter(coords['ra'], coords['dec'], 
                           s=10, alpha=0.6, label=label, color=colors[i % len(colors)])
            
            ax4.set_xlabel('Ascensión Recta [grados]')
            ax4.set_ylabel('Declinación [grados]')
            ax4.set_title('🌌 Distribución de Coordenadas Astronómicas')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "03_feature_distributions_comparison.png", 
                   dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print(f"✅ Gráfico guardado: 03_feature_distributions_comparison.png")
        
    def generate_comprehensive_summary(self):
        """
        Genera un resumen comprehensivo de la comparación
        """
        print("📋 Generando resumen comprehensivo...")
        
        # Calcular estadísticas comparativas
        summary_data = {
            'Dataset': [],
            'Original_Count': [],
            'Confirmed_Original': [],
            'Confirmed_Percentage': [],
            'ML_Predictions_Found': [],
            'ML_Exoplanets_Found': [],
            'Average_Confidence': []
        }
        
        # Análisis por dataset
        datasets_analysis = [
            ('KOI', self.koi_df, 'koi_disposition', 'CONFIRMED'),
            ('TOI', self.toi_df, 'tfopwg_disp', None),  # TOI tiene múltiples categorías
            ('K2', self.k2_df, 'disposition', 'CONFIRMED')
        ]
        
        for name, df, disp_col, confirmed_val in datasets_analysis:
            summary_data['Dataset'].append(name)
            
            if not df.empty and disp_col in df.columns:
                total_orig = len(df)
                summary_data['Original_Count'].append(total_orig)
                
                if confirmed_val:
                    confirmed_orig = sum(df[disp_col] == confirmed_val)
                    summary_data['Confirmed_Original'].append(confirmed_orig)
                    summary_data['Confirmed_Percentage'].append(f"{confirmed_orig/total_orig*100:.1f}%")
                else:
                    # Para TOI, tomar las categorías más relevantes
                    disp_counts = df[disp_col].value_counts()
                    confirmed_orig = disp_counts.get('PC', 0) + disp_counts.get('KP', 0)
                    summary_data['Confirmed_Original'].append(confirmed_orig)
                    summary_data['Confirmed_Percentage'].append(f"{confirmed_orig/total_orig*100:.1f}%")
                
                # Intentar encontrar coincidencias en predicciones
                ml_found = 0
                ml_exoplanets = 0
                total_conf = 0
                
                if 'ra' in df.columns and 'ra' in self.predictions_df.columns:
                    # Buscar por proximidad de coordenadas (sample)
                    orig_sample = df[['ra', 'dec']].dropna().head(100)
                    pred_sample = self.predictions_df[['ra', 'dec', 'ML_Prediction', 'ML_Confidence']].dropna().head(1000)
                    
                    for _, orig_row in orig_sample.iterrows():
                        ra_diff = abs(pred_sample['ra'] - orig_row['ra'])
                        dec_diff = abs(pred_sample['dec'] - orig_row['dec'])
                        matches = pred_sample[(ra_diff < 0.01) & (dec_diff < 0.01)]
                        
                        if len(matches) > 0:
                            ml_found += 1
                            if matches.iloc[0]['ML_Prediction'] == 1:
                                ml_exoplanets += 1
                            total_conf += matches.iloc[0]['ML_Confidence']
                
                summary_data['ML_Predictions_Found'].append(ml_found)
                summary_data['ML_Exoplanets_Found'].append(ml_exoplanets)
                summary_data['Average_Confidence'].append(f"{total_conf/ml_found:.3f}" if ml_found > 0 else "N/A")
            else:
                # Dataset no disponible
                summary_data['Original_Count'].append(0)
                summary_data['Confirmed_Original'].append(0)
                summary_data['Confirmed_Percentage'].append("N/A")
                summary_data['ML_Predictions_Found'].append(0)
                summary_data['ML_Exoplanets_Found'].append(0)
                summary_data['Average_Confidence'].append("N/A")
        
        # Guardar resumen en CSV
        summary_df = pd.DataFrame(summary_data)
        summary_csv = self.results_path / f"comparative_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(summary_csv, index=False)
        
        print(f"✅ Resumen CSV guardado: {summary_csv.name}")
        
        return summary_df
        
    def run_complete_analysis(self):
        """
        Ejecuta el análisis completo comparativo
        """
        print("🔬 INICIANDO ANÁLISIS COMPARATIVO COMPLETO")
        print("="*60)
        
        print("1️⃣ Analizando distribuciones de datasets...")
        self.analyze_dataset_distributions()
        
        print("2️⃣ Comparando accuracy por dataset...")
        self.compare_prediction_accuracy()
        
        print("3️⃣ Analizando distribuciones de características...")
        self.analyze_feature_distributions()
        
        print("4️⃣ Generando resumen comprehensivo...")
        summary_df = self.generate_comprehensive_summary()
        
        print(f"\n🎉 ¡ANÁLISIS COMPARATIVO COMPLETADO!")
        print(f"📁 Gráficos guardados en: {self.output_dir}")
        print(f"📊 Resumen CSV generado")
        
        # Mostrar resumen en consola
        print("\n📋 RESUMEN COMPARATIVO:")
        print(summary_df.to_string(index=False))
        
        return summary_df

def main():
    """
    Función principal
    """
    print("🔬 ANÁLISIS COMPARATIVO - DATASETS vs PREDICCIONES")
    print("NASA Space Apps Challenge 2025")
    print("="*60)
    
    try:
        comparator = DatasetComparator()
        summary = comparator.run_complete_analysis()
        
        print(f"\n✅ Proceso completado exitosamente")
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()