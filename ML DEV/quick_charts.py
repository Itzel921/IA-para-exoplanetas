#!/usr/bin/env python3
"""
🎨 GENERADOR RÁPIDO DE GRÁFICOS - EXOPLANETAS ML
NASA Space Apps Challenge 2025
===============================================
Versión optimizada para generar gráficos esenciales rápidamente
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

def quick_analysis(results_file):
    """
    Análisis rápido con gráficos esenciales
    """
    print(f"📊 Analizando: {Path(results_file).name}")
    
    # Cargar datos
    df = pd.read_csv(results_file)
    df_clean = df.dropna(subset=['ML_Confidence', 'ML_Prediction'])
    
    print(f"✅ Datos: {len(df_clean):,} objetos procesados")
    
    # Crear directorio de salida
    output_dir = Path(results_file).parent / "charts"
    output_dir.mkdir(exist_ok=True)
    
    # === GRÁFICO 1: DISTRIBUCIÓN DE CONFIANZA ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('black')
    
    # Histograma de confianza
    ax1.hist(df_clean['ML_Confidence'], bins=30, alpha=0.8, color='cyan', edgecolor='white')
    ax1.axvline(df_clean['ML_Confidence'].mean(), color='yellow', linestyle='--', linewidth=2,
                label=f'Promedio: {df_clean["ML_Confidence"].mean():.3f}')
    ax1.set_xlabel('Confianza del Modelo', color='white')
    ax1.set_ylabel('Número de Objetos', color='white')
    ax1.set_title('📊 Distribución de Confianza ML', color='white', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pie chart de predicciones
    pred_counts = df_clean['ML_Prediction'].value_counts()
    labels = ['NO PLANETA', 'EXOPLANETA']
    colors = ['#FF6B6B', '#4ECDC4']
    
    wedges, texts, autotexts = ax2.pie(pred_counts.values, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=90)
    ax2.set_title('🎯 Predicciones ML', color='white', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_and_predictions.png", dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='white')
    plt.close()
    
    # === GRÁFICO 2: TOP CANDIDATOS ===
    df_planets = df_clean[df_clean['ML_Prediction'] == 1].copy()
    
    if len(df_planets) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('black')
        
        # TOP 10 candidatos
        top_10 = df_planets.nlargest(10, 'ML_Confidence')
        
        # Crear etiquetas
        labels = []
        for i, row in top_10.iterrows():
            if pd.notna(row.get('kepoi_name')):
                labels.append(row['kepoi_name'])
            elif pd.notna(row.get('kepler_name')):
                name = str(row['kepler_name'])[:12]
                labels.append(name + ('...' if len(str(row['kepler_name'])) > 12 else ''))
            else:
                labels.append(f"Candidato_{i}")
        
        y_pos = np.arange(len(labels))
        bars = ax1.barh(y_pos, top_10['ML_Confidence'], color='gold', alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels)
        ax1.set_xlabel('Confianza ML', color='white')
        ax1.set_title('🏆 TOP 10 Candidatos a Exoplanetas', color='white', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Agregar valores
        for bar, conf in zip(bars, top_10['ML_Confidence']):
            ax1.text(conf + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{conf:.3f}', va='center', color='white', fontweight='bold')
        
        # Rangos de confianza
        confidence_ranges = [(0.9, 1.0, 'Muy Alta'), (0.8, 0.9, 'Alta'), 
                           (0.7, 0.8, 'Media'), (0.6, 0.7, 'Baja'), (0.0, 0.6, 'Muy Baja')]
        
        range_counts = []
        range_labels = []
        for min_conf, max_conf, label in confidence_ranges:
            count = sum((df_clean['ML_Confidence'] >= min_conf) & 
                       (df_clean['ML_Confidence'] < max_conf))
            range_counts.append(count)
            range_labels.append(label)
        
        bars2 = ax2.bar(range(len(range_labels)), range_counts, color='skyblue', alpha=0.8)
        ax2.set_xticks(range(len(range_labels)))
        ax2.set_xticklabels(range_labels, rotation=45, ha='right')
        ax2.set_ylabel('Número de Objetos', color='white')
        ax2.set_title('📈 Distribución por Confianza', color='white', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Valores en barras
        for bar, count in zip(bars2, range_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(range_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "top_candidates_and_distribution.png", dpi=150, bbox_inches='tight',
                    facecolor='black', edgecolor='white')
        plt.close()
    
    # === GRÁFICO 3: ANÁLISIS DE ACCURACY (si hay etiquetas) ===
    df_labeled = df_clean.dropna(subset=['koi_disposition']).copy()
    
    if len(df_labeled) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('black')
        
        # Preparar datos para matriz de confusión
        df_labeled['True_Label'] = (df_labeled['koi_disposition'] == 'CONFIRMED').astype(int)
        y_true = df_labeled['True_Label']
        y_pred = df_labeled['ML_Prediction']
        
        # Matriz de confusión manual (sin sklearn para velocidad)
        tp = sum((y_true == 1) & (y_pred == 1))
        fp = sum((y_true == 0) & (y_pred == 1))
        fn = sum((y_true == 1) & (y_pred == 0))
        tn = sum((y_true == 0) & (y_pred == 0))
        
        cm_data = [[tn, fp], [fn, tp]]
        
        # Plot matriz de confusión
        im = ax1.imshow(cm_data, interpolation='nearest', cmap='Blues')
        ax1.set_title('🎯 Matriz de Confusión', color='white', fontsize=14)
        
        # Añadir texto
        for i in range(2):
            for j in range(2):
                text = ax1.text(j, i, cm_data[i][j], ha="center", va="center", 
                               color="white", fontsize=16, fontweight='bold')
        
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['NO PLANETA', 'EXOPLANETA'])
        ax1.set_yticklabels(['NO PLANETA', 'EXOPLANETA'])
        ax1.set_xlabel('Predicción ML', color='white')
        ax1.set_ylabel('Etiqueta Real', color='white')
        
        # Métricas
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Texto de métricas
        metrics_text = f"""🎯 MÉTRICAS DE RENDIMIENTO

Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)
Precision: {precision:.3f} ({precision*100:.1f}%)
Recall: {recall:.3f} ({recall*100:.1f}%)
F1-Score: {f1:.3f}

📊 MATRIZ DE CONFUSIÓN:
• Verdaderos Negativos: {tn:,}
• Falsos Positivos: {fp:,}  
• Falsos Negativos: {fn:,}
• Verdaderos Positivos: {tp:,}

🔍 INTERPRETACIÓN:
• Completeness: {recall:.1%}
• Reliability: {precision:.1%}
• False Discovery Rate: {(1-precision):.1%}"""
        
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8),
                color='white')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('📋 Métricas de Accuracy', color='white', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_analysis.png", dpi=150, bbox_inches='tight',
                    facecolor='black', edgecolor='white')
        plt.close()
        
        print(f"✅ Accuracy en muestra: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # === GRÁFICO 4: MAPA ASTRONÓMICO ===
    df_coords = df_clean.dropna(subset=['ra', 'dec']).copy()
    
    if len(df_coords) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.patch.set_facecolor('black')
        
        planets_mask = df_coords['ML_Prediction'] == 1
        
        # Plot no-planetas
        ax.scatter(df_coords[~planets_mask]['ra'], df_coords[~planets_mask]['dec'], 
                  alpha=0.6, s=10, c='lightblue', label='NO PLANETA', marker='.')
        
        # Plot exoplanetas con color por confianza
        if sum(planets_mask) > 0:
            scatter = ax.scatter(df_coords[planets_mask]['ra'], df_coords[planets_mask]['dec'],
                               c=df_coords[planets_mask]['ML_Confidence'], cmap='Reds',
                               s=40, alpha=0.9, label='EXOPLANETA', marker='*', edgecolors='yellow')
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Confianza ML', rotation=270, labelpad=15, color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
        
        ax.set_xlabel('Ascensión Recta (RA) [grados]', color='white')
        ax.set_ylabel('Declinación (Dec) [grados]', color='white')
        ax.set_title('🌌 Mapa Astronómico de Predicciones ML', color='white', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "astronomical_map.png", dpi=150, bbox_inches='tight',
                    facecolor='black', edgecolor='white')
        plt.close()
    
    # === ESTADÍSTICAS FINALES ===
    exoplanets_found = sum(df_clean['ML_Prediction'] == 1)
    avg_confidence = df_clean['ML_Confidence'].mean()
    high_conf = sum(df_clean['ML_Confidence'] > 0.8)
    
    print(f"\n📊 RESUMEN FINAL:")
    print(f"🌟 Exoplanetas predichos: {exoplanets_found:,}")
    print(f"🎯 Confianza promedio: {avg_confidence:.3f}")
    print(f"🔥 Alta confianza (>80%): {high_conf:,}")
    print(f"📁 Gráficos guardados en: {output_dir}")
    
    return output_dir

def main():
    """Función principal optimizada"""
    print("🎨 GENERADOR RÁPIDO DE GRÁFICOS - EXOPLANETAS")
    print("="*50)
    
    # Buscar archivo de resultados más reciente
    results_dir = Path(__file__).parent.parent / "exoPlanet_results"
    csv_files = list(results_dir.glob("*predictions*.csv"))
    
    if not csv_files:
        print("❌ No se encontraron archivos de resultados")
        return
    
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"📂 Procesando: {latest_file.name}")
    
    # Generar gráficos
    output_dir = quick_analysis(latest_file)
    
    print(f"\n🎉 ¡GRÁFICOS GENERADOS EXITOSAMENTE!")
    print(f"📁 Revisa la carpeta: {output_dir}")

if __name__ == "__main__":
    main()