#!/usr/bin/env python3
"""
🎨 GENERADOR DE GRÁFICOS PARA PREDICCIONES DE EXOPLANETAS
NASA Space Apps Challenge 2025
============================================================

Genera visualizaciones automáticas de los resultados de predicción ML
- Gráficos de distribución de confianza
- Comparaciones con etiquetas reales
- Análisis de características principales
- Mapas de posición astronómica
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('dark_background')
sns.set_palette("viridis")

class ExoplanetVisualizationGenerator:
    def __init__(self, results_file_path):
        """
        Inicializa el generador de gráficos
        """
        self.results_path = Path(results_file_path)
        self.output_dir = self.results_path.parent / "charts"
        self.output_dir.mkdir(exist_ok=True)
        
        # Cargar datos
        print(f"📊 Cargando resultados: {self.results_path.name}")
        self.df = pd.read_csv(self.results_path)
        print(f"✅ Datos cargados: {len(self.df):,} objetos")
        
        # Limpiar datos para análisis
        self.df_clean = self.df.dropna(subset=['ML_Confidence', 'ML_Prediction'])
        
    def generate_confidence_distribution(self):
        """
        Gráfico de distribución de confianza
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histograma de confianza
        ax1.hist(self.df_clean['ML_Confidence'], bins=50, alpha=0.7, 
                color='cyan', edgecolor='white', linewidth=0.5)
        ax1.axvline(self.df_clean['ML_Confidence'].mean(), color='yellow', 
                   linestyle='--', linewidth=2, label=f'Promedio: {self.df_clean["ML_Confidence"].mean():.3f}')
        ax1.set_xlabel('Confianza del Modelo')
        ax1.set_ylabel('Número de Objetos')
        ax1.set_title('📊 Distribución de Confianza ML')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot por predicción
        predictions_labels = {0: 'NO PLANETA', 1: 'EXOPLANETA'}
        self.df_clean['Pred_Label'] = self.df_clean['ML_Prediction'].map(predictions_labels)
        
        sns.boxplot(data=self.df_clean, x='Pred_Label', y='ML_Confidence', ax=ax2)
        ax2.set_title('📦 Confianza por Tipo de Predicción')
        ax2.set_ylabel('Confianza del Modelo')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = self.output_dir / "01_confidence_distribution.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Gráfico guardado: {chart_path}")
        
    def generate_prediction_summary(self):
        """
        Gráfico resumen de predicciones
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Pie chart de predicciones
        pred_counts = self.df_clean['ML_Prediction'].value_counts()
        labels = ['NO PLANETA', 'EXOPLANETA']
        colors = ['#FF6B6B', '#4ECDC4']
        
        wedges, texts, autotexts = ax1.pie(pred_counts.values, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
        ax1.set_title('🎯 Predicciones ML - Distribución General', fontsize=14, fontweight='bold')
        
        # 2. Histograma de confianza por rangos
        confidence_ranges = [
            (0.9, 1.0, 'Muy Alta (90-100%)'),
            (0.8, 0.9, 'Alta (80-90%)'),
            (0.7, 0.8, 'Media (70-80%)'),
            (0.6, 0.7, 'Baja (60-70%)'),
            (0.0, 0.6, 'Muy Baja (<60%)')
        ]
        
        range_counts = []
        range_labels = []
        for min_conf, max_conf, label in confidence_ranges:
            count = sum((self.df_clean['ML_Confidence'] >= min_conf) & 
                       (self.df_clean['ML_Confidence'] < max_conf))
            range_counts.append(count)
            range_labels.append(label)
        
        bars = ax2.bar(range(len(range_labels)), range_counts, color='skyblue', alpha=0.8)
        ax2.set_xticks(range(len(range_labels)))
        ax2.set_xticklabels(range_labels, rotation=45, ha='right')
        ax2.set_ylabel('Número de Objetos')
        ax2.set_title('📈 Distribución por Rangos de Confianza')
        ax2.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bar, count in zip(bars, range_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(range_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Top candidatos con mayor confianza
        df_planets = self.df_clean[self.df_clean['ML_Prediction'] == 1].copy()
        if len(df_planets) > 0:
            top_candidates = df_planets.nlargest(10, 'ML_Confidence')
            
            # Crear etiquetas para candidatos
            candidate_labels = []
            for idx, row in top_candidates.iterrows():
                if pd.notna(row.get('kepoi_name')):
                    label = row['kepoi_name']
                elif pd.notna(row.get('kepler_name')):
                    label = row['kepler_name'][:15] + ('...' if len(str(row['kepler_name'])) > 15 else '')
                else:
                    label = f"Obj_{idx}"
                candidate_labels.append(label)
            
            y_pos = np.arange(len(candidate_labels))
            bars3 = ax3.barh(y_pos, top_candidates['ML_Confidence'], color='gold', alpha=0.8)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(candidate_labels)
            ax3.set_xlabel('Confianza ML')
            ax3.set_title('🏆 TOP 10 Candidatos a Exoplanetas')
            ax3.grid(True, alpha=0.3)
            
            # Agregar valores de confianza
            for i, (bar, conf) in enumerate(zip(bars3, top_candidates['ML_Confidence'])):
                ax3.text(conf + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{conf:.3f}', va='center', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No se encontraron\ncandidatos a exoplanetas', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14)
            ax3.set_title('🏆 TOP Candidatos a Exoplanetas')
        
        # 4. Estadísticas generales
        stats_text = f"""
📊 ESTADÍSTICAS GENERALES

Total de objetos analizados: {len(self.df_clean):,}

🌟 Exoplanetas predichos: {sum(self.df_clean['ML_Prediction'] == 1):,}
🚫 No-planetas predichos: {sum(self.df_clean['ML_Prediction'] == 0):,}

🎯 Confianza promedio: {self.df_clean['ML_Confidence'].mean():.3f}
📈 Confianza máxima: {self.df_clean['ML_Confidence'].max():.3f}
📉 Confianza mínima: {self.df_clean['ML_Confidence'].min():.3f}

🔥 Predicciones alta confianza (>80%): {sum(self.df_clean['ML_Confidence'] > 0.8):,}
💎 Predicciones muy alta confianza (>90%): {sum(self.df_clean['ML_Confidence'] > 0.9):,}
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('📋 Resumen Estadístico')
        
        plt.tight_layout()
        chart_path = self.output_dir / "02_prediction_summary.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Gráfico guardado: {chart_path}")
        
    def generate_accuracy_comparison(self):
        """
        Comparación con etiquetas reales (si están disponibles)
        """
        # Filtrar objetos con etiquetas reales
        df_labeled = self.df_clean.dropna(subset=['koi_disposition']).copy()
        
        if len(df_labeled) == 0:
            print("⚠️ No hay etiquetas reales disponibles para comparación")
            return
            
        print(f"📊 Analizando {len(df_labeled):,} objetos con etiquetas reales")
        
        # Convertir etiquetas reales a formato binario
        df_labeled['True_Label'] = (df_labeled['koi_disposition'] == 'CONFIRMED').astype(int)
        
        # Calcular métricas
        from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
        
        y_true = df_labeled['True_Label']
        y_pred = df_labeled['ML_Prediction']
        y_prob = df_labeled['ML_Confidence']
        
        # Generar gráficos
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['NO PLANETA', 'EXOPLANETA'],
                   yticklabels=['NO PLANETA', 'EXOPLANETA'])
        ax1.set_title('🎯 Matriz de Confusión')
        ax1.set_xlabel('Predicción ML')
        ax1.set_ylabel('Etiqueta Real')
        
        # 2. Distribución de confianza por clase real
        df_labeled['Real_Label_Text'] = df_labeled['True_Label'].map({0: 'NO PLANETA', 1: 'EXOPLANETA'})
        sns.boxplot(data=df_labeled, x='Real_Label_Text', y='ML_Confidence', ax=ax2)
        ax2.set_title('📦 Confianza ML por Clase Real')
        ax2.set_ylabel('Confianza ML')
        ax2.grid(True, alpha=0.3)
        
        # 3. Curva ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax3.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('Tasa de Falsos Positivos')
        ax3.set_ylabel('Tasa de Verdaderos Positivos')
        ax3.set_title('📈 Curva ROC')
        ax3.legend(loc="lower right")
        ax3.grid(True, alpha=0.3)
        
        # 4. Reporte de clasificación
        report = classification_report(y_true, y_pred, output_dict=True)
        accuracy = report['accuracy']
        precision = report['1']['precision'] if '1' in report else 0
        recall = report['1']['recall'] if '1' in report else 0
        f1 = report['1']['f1-score'] if '1' in report else 0
        
        metrics_text = f"""
🎯 MÉTRICAS DE RENDIMIENTO

Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)
Precision: {precision:.3f} ({precision*100:.1f}%)
Recall: {recall:.3f} ({recall*100:.1f}%)
F1-Score: {f1:.3f}
AUC-ROC: {roc_auc:.3f}

📊 MATRIZ DE CONFUSIÓN:
• Verdaderos Negativos: {cm[0,0]:,}
• Falsos Positivos: {cm[0,1]:,}  
• Falsos Negativos: {cm[1,0]:,}
• Verdaderos Positivos: {cm[1,1]:,}

🔍 INTERPRETACIÓN:
• Completeness: {recall:.1%} de planetas reales detectados
• Reliability: {precision:.1%} de detecciones son correctas
• False Discovery Rate: {(1-precision):.1%}
        """
        
        ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('📋 Métricas de Accuracy')
        
        plt.tight_layout()
        chart_path = self.output_dir / "03_accuracy_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Gráfico guardado: {chart_path}")
        
        return accuracy, precision, recall, f1, roc_auc
        
    def generate_astronomical_plots(self):
        """
        Gráficos específicos de características astronómicas
        """
        # Filtrar datos con coordenadas válidas
        df_coords = self.df_clean.dropna(subset=['ra', 'dec']).copy()
        
        if len(df_coords) == 0:
            print("⚠️ No hay coordenadas astronómicas disponibles")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Mapa de posiciones astronómicas
        planets_mask = df_coords['ML_Prediction'] == 1
        
        # No-planetas
        ax1.scatter(df_coords[~planets_mask]['ra'], df_coords[~planets_mask]['dec'], 
                   alpha=0.6, s=20, c='lightblue', label='NO PLANETA', marker='.')
        
        # Exoplanetas predichos
        if sum(planets_mask) > 0:
            scatter = ax1.scatter(df_coords[planets_mask]['ra'], df_coords[planets_mask]['dec'],
                                c=df_coords[planets_mask]['ML_Confidence'], cmap='Reds',
                                s=60, alpha=0.8, label='EXOPLANETA', marker='*', edgecolors='black')
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
            cbar.set_label('Confianza ML', rotation=270, labelpad=15)
        
        ax1.set_xlabel('Ascensión Recta (RA) [grados]')
        ax1.set_ylabel('Declinación (Dec) [grados]')
        ax1.set_title('🌌 Mapa Astronómico de Predicciones')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribución por características planetarias (si disponibles)
        planetary_features = ['koi_period', 'koi_prad', 'koi_teq', 'koi_depth']
        available_features = [col for col in planetary_features if col in self.df_clean.columns]
        
        if available_features:
            feature = available_features[0]  # Usar la primera característica disponible
            df_feature = self.df_clean.dropna(subset=[feature]).copy()
            
            if len(df_feature) > 0:
                # Histograma por predicción
                planets_data = df_feature[df_feature['ML_Prediction'] == 1][feature]
                no_planets_data = df_feature[df_feature['ML_Prediction'] == 0][feature]
                
                ax2.hist(no_planets_data, bins=30, alpha=0.7, label='NO PLANETA', color='lightblue', density=True)
                ax2.hist(planets_data, bins=30, alpha=0.7, label='EXOPLANETA', color='orange', density=True)
                
                ax2.set_xlabel(feature.replace('koi_', '').replace('_', ' ').title())
                ax2.set_ylabel('Densidad')
                ax2.set_title(f'📊 Distribución de {feature.replace("koi_", "").replace("_", " ").title()}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # 3. Confianza vs características (scatter plot)
        if len(available_features) >= 2:
            feature1, feature2 = available_features[0], available_features[1]
            df_scatter = self.df_clean.dropna(subset=[feature1, feature2, 'ML_Confidence']).copy()
            
            if len(df_scatter) > 0:
                scatter2 = ax3.scatter(df_scatter[feature1], df_scatter[feature2],
                                     c=df_scatter['ML_Confidence'], cmap='viridis',
                                     s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
                
                ax3.set_xlabel(feature1.replace('koi_', '').replace('_', ' ').title())
                ax3.set_ylabel(feature2.replace('koi_', '').replace('_', ' ').title())
                ax3.set_title('🎯 Confianza vs Características Planetarias')
                ax3.grid(True, alpha=0.3)
                
                cbar2 = plt.colorbar(scatter2, ax=ax3, shrink=0.8)
                cbar2.set_label('Confianza ML', rotation=270, labelpad=15)
        
        # 4. Estadísticas astronómicas
        astro_stats = f"""
🌌 ESTADÍSTICAS ASTRONÓMICAS

📍 Objetos con coordenadas: {len(df_coords):,}
  • RA rango: {df_coords['ra'].min():.1f}° - {df_coords['ra'].max():.1f}°
  • Dec rango: {df_coords['dec'].min():.1f}° - {df_coords['dec'].max():.1f}°

🌟 Exoplanetas predichos: {sum(planets_mask):,}
  • Confianza promedio: {df_coords[planets_mask]['ML_Confidence'].mean():.3f if sum(planets_mask) > 0 else 'N/A'}

📊 Características disponibles:
{chr(10).join([f'  • {feat}' for feat in available_features[:5]])}
        """
        
        ax4.text(0.05, 0.95, astro_stats, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('📋 Resumen Astronómico')
        
        plt.tight_layout()
        chart_path = self.output_dir / "04_astronomical_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Gráfico guardado: {chart_path}")
        
    def generate_all_charts(self):
        """
        Genera todos los gráficos disponibles
        """
        print(f"\n🎨 GENERANDO GRÁFICOS PARA {len(self.df_clean):,} OBJETOS")
        print("="*60)
        
        print("1️⃣ Generando distribución de confianza...")
        self.generate_confidence_distribution()
        
        print("2️⃣ Generando resumen de predicciones...")
        self.generate_prediction_summary()
        
        print("3️⃣ Generando análisis de accuracy...")
        metrics = self.generate_accuracy_comparison()
        
        print("4️⃣ Generando gráficos astronómicos...")
        self.generate_astronomical_plots()
        
        # Generar resumen HTML
        self.generate_html_report(metrics)
        
        print(f"\n✅ TODOS LOS GRÁFICOS GENERADOS EXITOSAMENTE")
        print(f"📁 Ubicación: {self.output_dir}")
        print("="*60)
        
    def generate_html_report(self, metrics=None):
        """
        Genera un reporte HTML con todos los gráficos
        """
        chart_files = list(self.output_dir.glob("*.png"))
        chart_files.sort()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔮 Análisis de Predicciones ML - Exoplanetas</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 100%);
            color: #ffffff;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(16, 16, 20, 0.9);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 255, 255, 0.3);
        }}
        h1 {{
            text-align: center;
            color: #00ffff;
            text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
            font-size: 2.5em;
            margin-bottom: 30px;
        }}
        .chart-section {{
            margin: 40px 0;
            border: 2px solid #333;
            border-radius: 10px;
            padding: 20px;
            background: rgba(26, 26, 46, 0.7);
        }}
        .chart-section h2 {{
            color: #ffdd00;
            border-bottom: 2px solid #ffdd00;
            padding-bottom: 10px;
        }}
        img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
        }}
        .metrics {{
            background: rgba(0, 100, 0, 0.2);
            border-left: 4px solid #00ff00;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            border-top: 1px solid #333;
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 Análisis ML de Detección de Exoplanetas</h1>
        <p style="text-align: center; font-size: 1.2em; color: #aaa;">
            Resultados del modelo ensemble con {len(self.df_clean):,} objetos analizados
        </p>
        
        {"".join([f'''
        <div class="chart-section">
            <h2>{chart_file.stem.replace("_", " ").title()}</h2>
            <img src="{chart_file.name}" alt="{chart_file.stem}">
        </div>
        ''' for chart_file in chart_files])}
        
        {f'''
        <div class="metrics">
            <h3>📊 Métricas de Rendimiento</h3>
            <ul>
                <li><strong>Accuracy:</strong> {metrics[0]:.3f} ({metrics[0]*100:.1f}%)</li>
                <li><strong>Precision:</strong> {metrics[1]:.3f} ({metrics[1]*100:.1f}%)</li>
                <li><strong>Recall:</strong> {metrics[2]:.3f} ({metrics[2]*100:.1f}%)</li>
                <li><strong>F1-Score:</strong> {metrics[3]:.3f}</li>
                <li><strong>AUC-ROC:</strong> {metrics[4]:.3f}</li>
            </ul>
        </div>
        ''' if metrics else ''}
        
        <div class="footer">
            <p>🚀 Generado automáticamente por el Sistema de IA para Exoplanetas</p>
            <p>NASA Space Apps Challenge 2025</p>
        </div>
    </div>
</body>
</html>
        """
        
        html_path = self.output_dir / "exoplanet_analysis_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Reporte HTML generado: {html_path}")

def main():
    """
    Función principal para generar gráficos
    """
    print("🎨 GENERADOR DE GRÁFICOS - PREDICCIONES DE EXOPLANETAS")
    print("NASA Space Apps Challenge 2025")
    print("="*60)
    
    # Buscar el archivo de resultados más reciente
    results_dir = Path(__file__).parent.parent / "exoPlanet_results"
    csv_files = list(results_dir.glob("*predictions*.csv"))
    
    if not csv_files:
        print("❌ No se encontraron archivos de resultados")
        print(f"   Buscar en: {results_dir}")
        return
    
    # Usar el archivo más reciente
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"📂 Usando archivo: {latest_file.name}")
    
    # Generar gráficos
    generator = ExoplanetVisualizationGenerator(latest_file)
    generator.generate_all_charts()
    
    print(f"\n🎉 ¡PROCESO COMPLETADO!")
    print(f"📁 Revisa la carpeta: {generator.output_dir}")
    print(f"🌐 Abre el archivo HTML para ver todos los gráficos")

if __name__ == "__main__":
    main()