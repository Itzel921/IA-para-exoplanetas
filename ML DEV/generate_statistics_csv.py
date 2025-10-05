#!/usr/bin/env python3
"""
📊 GENERADOR DE ESTADÍSTICAS CSV - EXOPLANETAS ML
NASA Space Apps Challenge 2025
===============================================
Genera un CSV detallado con todas las estadísticas de predicción
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def generate_statistics_csv(results_file):
    """
    Genera estadísticas detalladas en formato CSV
    """
    print(f"📊 Generando estadísticas para: {Path(results_file).name}")
    
    # Cargar datos
    df = pd.read_csv(results_file)
    df_clean = df.dropna(subset=['ML_Confidence', 'ML_Prediction'])
    
    # Crear directorio de salida
    output_dir = Path(results_file).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # === ESTADÍSTICAS GENERALES ===
    general_stats = {
        'Metric': [
            'Total Objects Analyzed',
            'Exoplanets Predicted',
            'Non-Planets Predicted', 
            'Average Confidence',
            'Maximum Confidence',
            'Minimum Confidence',
            'High Confidence (>80%)',
            'Very High Confidence (>90%)',
            'Medium Confidence (60-80%)',
            'Low Confidence (<60%)'
        ],
        'Value': [
            len(df_clean),
            sum(df_clean['ML_Prediction'] == 1),
            sum(df_clean['ML_Prediction'] == 0),
            f"{df_clean['ML_Confidence'].mean():.4f}",
            f"{df_clean['ML_Confidence'].max():.4f}",
            f"{df_clean['ML_Confidence'].min():.4f}",
            sum(df_clean['ML_Confidence'] > 0.8),
            sum(df_clean['ML_Confidence'] > 0.9),
            sum((df_clean['ML_Confidence'] >= 0.6) & (df_clean['ML_Confidence'] <= 0.8)),
            sum(df_clean['ML_Confidence'] < 0.6)
        ]
    }
    
    general_df = pd.DataFrame(general_stats)
    general_csv = output_dir / f"general_statistics_{timestamp}.csv"
    general_df.to_csv(general_csv, index=False)
    print(f"✅ Estadísticas generales: {general_csv.name}")
    
    # === TOP CANDIDATOS ===
    df_planets = df_clean[df_clean['ML_Prediction'] == 1].copy()
    
    if len(df_planets) > 0:
        # TOP 20 candidatos
        top_candidates = df_planets.nlargest(20, 'ML_Confidence').copy()
        
        # Seleccionar columnas relevantes
        candidate_cols = ['kepoi_name', 'kepler_name', 'ML_Confidence', 'ML_Prediction', 'ML_Classification']
        
        # Agregar información adicional si está disponible
        if 'koi_disposition' in top_candidates.columns:
            candidate_cols.insert(-1, 'koi_disposition')
        if 'koi_period' in top_candidates.columns:
            candidate_cols.insert(-1, 'koi_period')
        if 'koi_prad' in top_candidates.columns:
            candidate_cols.insert(-1, 'koi_prad')
        if 'ra' in top_candidates.columns and 'dec' in top_candidates.columns:
            candidate_cols.extend(['ra', 'dec'])
        
        # Filtrar columnas que realmente existen
        existing_cols = [col for col in candidate_cols if col in top_candidates.columns]
        top_candidates_export = top_candidates[existing_cols].copy()
        
        # Agregar ranking
        top_candidates_export.insert(0, 'Rank', range(1, len(top_candidates_export) + 1))
        
        candidates_csv = output_dir / f"top_candidates_{timestamp}.csv"
        top_candidates_export.to_csv(candidates_csv, index=False)
        print(f"✅ TOP candidatos: {candidates_csv.name}")
    
    # === ANÁLISIS DE ACCURACY (si hay etiquetas reales) ===
    df_labeled = df_clean.dropna(subset=['koi_disposition']).copy()
    
    if len(df_labeled) > 0:
        df_labeled['True_Label'] = (df_labeled['koi_disposition'] == 'CONFIRMED').astype(int)
        y_true = df_labeled['True_Label']
        y_pred = df_labeled['ML_Prediction']
        
        # Calcular métricas
        tp = sum((y_true == 1) & (y_pred == 1))
        fp = sum((y_true == 0) & (y_pred == 1))
        fn = sum((y_true == 1) & (y_pred == 0))
        tn = sum((y_true == 0) & (y_pred == 0))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        accuracy_stats = {
            'Metric': [
                'Sample Size (with labels)',
                'True Positives (TP)',
                'False Positives (FP)',
                'True Negatives (TN)',
                'False Negatives (FN)',
                'Accuracy',
                'Precision (Reliability)',
                'Recall (Completeness)',
                'Specificity',
                'F1-Score',
                'False Discovery Rate',
                'Miss Rate'
            ],
            'Value': [
                len(df_labeled),
                tp,
                fp,
                tn,
                fn,
                f"{accuracy:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{specificity:.4f}",
                f"{f1:.4f}",
                f"{1-precision:.4f}" if precision > 0 else "N/A",
                f"{1-recall:.4f}" if recall > 0 else "N/A"
            ],
            'Percentage': [
                "N/A",
                "N/A",
                "N/A",
                "N/A", 
                "N/A",
                f"{accuracy*100:.2f}%",
                f"{precision*100:.2f}%",
                f"{recall*100:.2f}%",
                f"{specificity*100:.2f}%",
                f"{f1*100:.2f}%",
                f"{(1-precision)*100:.2f}%" if precision > 0 else "N/A",
                f"{(1-recall)*100:.2f}%" if recall > 0 else "N/A"
            ]
        }
        
        accuracy_df = pd.DataFrame(accuracy_stats)
        accuracy_csv = output_dir / f"accuracy_metrics_{timestamp}.csv"
        accuracy_df.to_csv(accuracy_csv, index=False)
        print(f"✅ Métricas de accuracy: {accuracy_csv.name}")
    
    # === DISTRIBUCIÓN POR CONFIANZA ===
    confidence_ranges = [
        (0.95, 1.00, 'Extremely High (95-100%)'),
        (0.90, 0.95, 'Very High (90-95%)'),
        (0.85, 0.90, 'High (85-90%)'),
        (0.80, 0.85, 'Good (80-85%)'),
        (0.70, 0.80, 'Medium (70-80%)'),
        (0.60, 0.70, 'Low (60-70%)'),
        (0.50, 0.60, 'Very Low (50-60%)'),
        (0.00, 0.50, 'Poor (<50%)')
    ]
    
    confidence_dist = []
    for min_conf, max_conf, label in confidence_ranges:
        count = sum((df_clean['ML_Confidence'] >= min_conf) & 
                   (df_clean['ML_Confidence'] < max_conf))
        percentage = (count / len(df_clean)) * 100
        
        confidence_dist.append({
            'Confidence_Range': label,
            'Min_Confidence': min_conf,
            'Max_Confidence': max_conf,
            'Object_Count': count,
            'Percentage': f"{percentage:.2f}%"
        })
    
    confidence_df = pd.DataFrame(confidence_dist)
    confidence_csv = output_dir / f"confidence_distribution_{timestamp}.csv"
    confidence_df.to_csv(confidence_csv, index=False)
    print(f"✅ Distribución de confianza: {confidence_csv.name}")
    
    # === COORDENADAS ASTRONÓMICAS (si están disponibles) ===
    df_coords = df_clean.dropna(subset=['ra', 'dec']).copy()
    
    if len(df_coords) > 0:
        coord_stats = {
            'Metric': [
                'Objects with Coordinates',
                'RA Range (degrees)',
                'Dec Range (degrees)',
                'RA Mean',
                'Dec Mean',
                'Exoplanets in Northern Hemisphere (Dec > 0)',
                'Exoplanets in Southern Hemisphere (Dec < 0)',
                'Coverage Area (approx deg²)'
            ],
            'Value': [
                len(df_coords),
                f"{df_coords['ra'].min():.2f} - {df_coords['ra'].max():.2f}",
                f"{df_coords['dec'].min():.2f} - {df_coords['dec'].max():.2f}",
                f"{df_coords['ra'].mean():.4f}",
                f"{df_coords['dec'].mean():.4f}",
                sum((df_coords['ML_Prediction'] == 1) & (df_coords['dec'] > 0)),
                sum((df_coords['ML_Prediction'] == 1) & (df_coords['dec'] < 0)),
                f"{(df_coords['ra'].max() - df_coords['ra'].min()) * (df_coords['dec'].max() - df_coords['dec'].min()):.1f}"
            ]
        }
        
        coord_df = pd.DataFrame(coord_stats)
        coord_csv = output_dir / f"astronomical_coordinates_{timestamp}.csv"
        coord_df.to_csv(coord_csv, index=False)
        print(f"✅ Coordenadas astronómicas: {coord_csv.name}")
    
    return {
        'general': general_csv if 'general_csv' in locals() else None,
        'candidates': candidates_csv if 'candidates_csv' in locals() else None,
        'accuracy': accuracy_csv if 'accuracy_csv' in locals() else None,
        'confidence': confidence_csv if 'confidence_csv' in locals() else None,
        'coordinates': coord_csv if 'coord_csv' in locals() else None
    }

def main():
    """Función principal"""
    print("📊 GENERADOR DE ESTADÍSTICAS CSV - EXOPLANETAS")
    print("="*50)
    
    # Buscar archivo más reciente
    results_dir = Path(__file__).parent.parent / "exoPlanet_results"
    csv_files = list(results_dir.glob("*predictions*.csv"))
    
    if not csv_files:
        print("❌ No se encontraron archivos de resultados")
        return
    
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"📂 Procesando: {latest_file.name}")
    
    # Generar estadísticas
    generated_files = generate_statistics_csv(latest_file)
    
    print(f"\n🎉 ¡ESTADÍSTICAS CSV GENERADAS EXITOSAMENTE!")
    print(f"📁 Archivos creados en: {results_dir}")
    
    # Listar archivos generados
    for file_type, file_path in generated_files.items():
        if file_path:
            print(f"  • {file_type}: {file_path.name}")

if __name__ == "__main__":
    main()