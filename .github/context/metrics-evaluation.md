# Métricas y Evaluación - Detección de Exoplanetas

## 📊 Métricas de Evaluación Fundamentales

### Matriz de Confusión para Exoplanetas

#### Definición de Casos
```python
confusion_matrix_definitions = {
    'TP (True Positive)': {
        'description': 'Exoplanetas confirmados correctamente clasificados como confirmados',
        'astronomical_meaning': 'Detecciones exitosas de planetas reales',
        'impact': 'Descubrimientos válidos que contribuyen a la ciencia'
    },
    'TN (True Negative)': {
        'description': 'Candidatos/falsos positivos correctamente clasificados como no-confirmados',
        'astronomical_meaning': 'Filtrado exitoso de señales no planetarias',
        'impact': 'Reduce ruido y enfoca recursos en candidatos prometedores'
    },
    'FP (False Positive)': {
        'description': 'Candidatos clasificados incorrectamente como confirmados',
        'astronomical_meaning': 'Falsos descubrimientos reportados',
        'impact': 'Desperdicio de recursos de seguimiento, credibilidad científica'
    },
    'FN (False Negative)': {
        'description': 'Exoplanetas confirmados clasificados incorrectamente como candidatos',
        'astronomical_meaning': 'Planetas reales perdidos por el sistema',
        'impact': 'Pérdida de descubrimientos científicos importantes'
    }
}
```

#### Implementación de Matriz de Confusión
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class ExoplanetConfusionMatrix:
    def __init__(self):
        self.labels = ['FALSE_POSITIVE', 'CONFIRMED']
        self.display_labels = ['No Planeta', 'Planeta']
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False):
        """Visualización de matriz de confusión"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Matriz de Confusión Normalizada'
        else:
            fmt = 'd'
            title = 'Matriz de Confusión'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.display_labels,
                   yticklabels=self.display_labels)
        plt.title(title)
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        
        # Agregar interpretación astronómica
        tn, fp, fn, tp = cm.ravel()
        interpretation = f"""
        Interpretación Astronómica:
        • Planetas detectados correctamente: {tp}
        • Falsos positivos filtrados: {tn} 
        • Planetas perdidos: {fn}
        • Falsos descubrimientos: {fp}
        """
        
        plt.figtext(0.02, 0.02, interpretation, fontsize=9, ha='left')
        plt.tight_layout()
        return plt.gcf()
```

## 🎯 Métricas Principales de Clasificación

### 1. Accuracy (Exactitud)
**Definición**: Porcentaje de clasificaciones correctas sobre el total.

```python
def calculate_accuracy(y_true, y_pred):
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    interpretation = {
        'value': accuracy,
        'percentage': accuracy * 100,
        'meaning': f'El modelo clasifica correctamente el {accuracy*100:.1f}% de todos los objetos',
        'astronomical_context': f'De cada 100 objetos analizados, {int(accuracy*100)} son clasificados correctamente'
    }
    return interpretation
```

### 2. Sensitivity/Recall (Sensibilidad)
**Definición**: Proporción de planetas reales que son detectados correctamente.

```python
def calculate_sensitivity(y_true, y_pred):
    """
    Sensitivity = TP / (TP + FN)
    También conocido como True Positive Rate o Recall
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    interpretation = {
        'value': sensitivity,
        'percentage': sensitivity * 100,
        'meaning': f'El modelo detecta el {sensitivity*100:.1f}% de todos los planetas reales',
        'astronomical_context': f'De cada 100 planetas confirmados, {int(sensitivity*100)} son detectados por el modelo',
        'importance': 'CRÍTICA - Alta sensibilidad significa menos descubrimientos perdidos'
    }
    return interpretation
```

### 3. Specificity (Especificidad)
**Definición**: Proporción de no-planetas que son correctamente identificados como tal.

```python
def calculate_specificity(y_true, y_pred):
    """
    Specificity = TN / (TN + FP)
    También conocido como True Negative Rate
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    interpretation = {
        'value': specificity,
        'percentage': specificity * 100,
        'meaning': f'El modelo identifica correctamente el {specificity*100:.1f}% de los no-planetas',
        'astronomical_context': f'De cada 100 falsos positivos/candidatos, {int(specificity*100)} son correctamente filtrados',
        'importance': 'ALTA - Reduce recursos gastados en seguimiento de falsos positivos'
    }
    return interpretation
```

### 4. Precision (Precisión)
**Definición**: Proporción de predicciones positivas que son correctas.

```python
def calculate_precision(y_true, y_pred):
    """
    Precision = TP / (TP + FP)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    interpretation = {
        'value': precision,
        'percentage': precision * 100,
        'meaning': f'El {precision*100:.1f}% de las predicciones "planeta" son correctas',
        'astronomical_context': f'De cada 100 objetos clasificados como planetas, {int(precision*100)} son realmente planetas',
        'importance': 'CRÍTICA - Alta precisión significa menos tiempo perdido en falsos descubrimientos'
    }
    return interpretation
```

### 5. F1-Score
**Definición**: Media armónica entre precisión y recall.

```python
def calculate_f1_score(y_true, y_pred):
    """
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    precision = calculate_precision(y_true, y_pred)['value']
    recall = calculate_sensitivity(y_true, y_pred)['value']
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    interpretation = {
        'value': f1,
        'percentage': f1 * 100,
        'meaning': f'Balance entre precisión y recall: {f1*100:.1f}%',
        'astronomical_context': 'Métrica balanceada que considera tanto descubrimientos perdidos como falsos positivos',
        'importance': 'ALTA - Especialmente útil cuando las clases están desbalanceadas'
    }
    return interpretation
```

## 📈 Métricas Avanzadas para Exoplanetas

### ROC Analysis (Receiver Operating Characteristic)

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class ROCAnalysis:
    def __init__(self):
        self.optimal_threshold = None
        self.roc_auc = None
    
    def calculate_roc_metrics(self, y_true, y_prob):
        """Cálculo completo de métricas ROC"""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Encontrar threshold óptimo (máximo Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        self.optimal_threshold = thresholds[optimal_idx]
        
        metrics = {
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'optimal_threshold': self.optimal_threshold,
            'optimal_sensitivity': tpr[optimal_idx],
            'optimal_specificity': 1 - fpr[optimal_idx]
        }
        
        return metrics
    
    def plot_roc_curve(self, y_true, y_prob, model_name='Ensemble Model'):
        """Visualización de curva ROC"""
        metrics = self.calculate_roc_metrics(y_true, y_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(metrics['fpr'], metrics['tpr'], 
                color='darkorange', lw=2, 
                label=f'{model_name} (AUC = {metrics["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Clasificador Aleatorio (AUC = 0.5)')
        
        # Marcar punto óptimo
        opt_idx = np.argmax(metrics['tpr'] - metrics['fpr'])
        plt.scatter(metrics['fpr'][opt_idx], metrics['tpr'][opt_idx], 
                   color='red', s=100, zorder=5,
                   label=f'Punto Óptimo (threshold={self.optimal_threshold:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Especificidad)')
        plt.ylabel('True Positive Rate (Sensibilidad)')
        plt.title('Curva ROC - Detección de Exoplanetas')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Interpretación astronómica
        interpretation = f"""
        Interpretación AUC = {metrics['auc']:.3f}:
        • Excelente (>0.9): Discriminación excepcional entre planetas y no-planetas
        • Buena (0.8-0.9): Discriminación buena, aceptable para uso científico  
        • Regular (0.7-0.8): Discriminación moderada, necesita mejoras
        • Pobre (<0.7): Discriminación deficiente, no recomendado para producción
        """
        
        plt.figtext(0.02, 0.02, interpretation, fontsize=9, ha='left')
        plt.tight_layout()
        return plt.gcf()
```

### Precision-Recall Analysis

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

class PrecisionRecallAnalysis:
    def __init__(self):
        self.optimal_threshold = None
        self.avg_precision = None
    
    def calculate_pr_metrics(self, y_true, y_prob):
        """Cálculo de métricas Precision-Recall"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob[:, 1])
        avg_precision = average_precision_score(y_true, y_prob[:, 1])
        
        # Encontrar threshold óptimo (máximo F1-score)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
        f1_scores = np.nan_to_num(f1_scores)
        optimal_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[optimal_idx]
        
        metrics = {
            'precision': precision,
            'recall': recall, 
            'thresholds': thresholds,
            'average_precision': avg_precision,
            'optimal_threshold': self.optimal_threshold,
            'optimal_precision': precision[optimal_idx],
            'optimal_recall': recall[optimal_idx],
            'optimal_f1': f1_scores[optimal_idx]
        }
        
        return metrics
    
    def plot_pr_curve(self, y_true, y_prob, model_name='Ensemble Model'):
        """Visualización de curva Precision-Recall"""
        metrics = self.calculate_pr_metrics(y_true, y_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(metrics['recall'], metrics['precision'], 
                color='blue', lw=2,
                label=f'{model_name} (AP = {metrics["average_precision"]:.3f})')
        
        # Línea base (proporción de positivos)
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='red', linestyle='--', 
                   label=f'Línea Base (AP = {baseline:.3f})')
        
        # Marcar punto óptimo
        opt_idx = np.argmax(2 * metrics['precision'][:-1] * metrics['recall'][:-1] / 
                           (metrics['precision'][:-1] + metrics['recall'][:-1]))
        plt.scatter(metrics['recall'][opt_idx], metrics['precision'][opt_idx],
                   color='green', s=100, zorder=5,
                   label=f'Punto Óptimo F1 = {metrics["optimal_f1"]:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensibilidad)')
        plt.ylabel('Precision')
        plt.title('Curva Precision-Recall - Detección de Exoplanetas')
        plt.legend()
        plt.grid(True)
        
        # Interpretación
        interpretation = f"""
        Interpretación AP = {metrics['average_precision']:.3f}:
        • La curva PR es especialmente importante cuando hay desbalance de clases
        • AP > 0.8: Excelente balance entre precisión y recall
        • AP 0.6-0.8: Bueno, aceptable para aplicaciones científicas
        • AP < 0.6: Necesita mejoras significativas
        """
        
        plt.figtext(0.02, 0.02, interpretation, fontsize=9, ha='left')
        plt.tight_layout()
        return plt.gcf()
```

## 🔍 Métricas Específicas para Astronomía

### Completeness y Reliability

```python
class AstronomicalMetrics:
    def __init__(self):
        self.metrics = {}
    
    def calculate_completeness_reliability(self, y_true, y_pred, y_prob):
        """Métricas específicas para astronomía"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Completeness (= Recall/Sensitivity)
        completeness = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Reliability (= Precision)  
        reliability = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # False Discovery Rate
        fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Miss Rate (complemento de completeness)
        miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics = {
            'completeness': {
                'value': completeness,
                'percentage': completeness * 100,
                'interpretation': f'{completeness*100:.1f}% de planetas reales son detectados',
                'astronomical_meaning': 'Fracción de población planetaria real recuperada por el survey'
            },
            'reliability': {
                'value': reliability, 
                'percentage': reliability * 100,
                'interpretation': f'{reliability*100:.1f}% de detecciones son planetas reales',
                'astronomical_meaning': 'Fracción de detecciones que son astronómicamente válidas'
            },
            'false_discovery_rate': {
                'value': fdr,
                'percentage': fdr * 100,
                'interpretation': f'{fdr*100:.1f}% de detecciones son falsos positivos',
                'astronomical_meaning': 'Tasa de contaminación en el catálogo final'
            },
            'miss_rate': {
                'value': miss_rate,
                'percentage': miss_rate * 100,
                'interpretation': f'{miss_rate*100:.1f}% de planetas reales son perdidos',
                'astronomical_meaning': 'Fracción de población planetaria no detectada'
            }
        }
        
        return metrics
```

### Detection Efficiency por Parámetros Planetarios

```python
class DetectionEfficiency:
    def __init__(self):
        self.efficiency_curves = {}
    
    def calculate_efficiency_vs_radius(self, df, y_true, y_pred, radius_col='koi_prad'):
        """Eficiencia de detección vs radio planetario"""
        # Crear bins de radio
        radius_bins = np.logspace(np.log10(df[radius_col].min()), 
                                 np.log10(df[radius_col].max()), 10)
        
        efficiency_data = []
        
        for i in range(len(radius_bins)-1):
            mask = (df[radius_col] >= radius_bins[i]) & (df[radius_col] < radius_bins[i+1])
            
            if np.sum(mask) > 0:
                y_true_bin = y_true[mask]
                y_pred_bin = y_pred[mask]
                
                if np.sum(y_true_bin) > 0:  # Si hay planetas reales en el bin
                    efficiency = np.sum((y_true_bin == 1) & (y_pred_bin == 1)) / np.sum(y_true_bin)
                else:
                    efficiency = 0
                
                efficiency_data.append({
                    'radius_min': radius_bins[i],
                    'radius_max': radius_bins[i+1],
                    'radius_center': np.sqrt(radius_bins[i] * radius_bins[i+1]),
                    'efficiency': efficiency,
                    'n_planets': np.sum(y_true_bin),
                    'n_detected': np.sum((y_true_bin == 1) & (y_pred_bin == 1))
                })
        
        return pd.DataFrame(efficiency_data)
    
    def plot_detection_efficiency(self, efficiency_df):
        """Visualizar eficiencia de detección"""
        plt.figure(figsize=(10, 6))
        plt.semilogx(efficiency_df['radius_center'], efficiency_df['efficiency'], 
                    'bo-', linewidth=2, markersize=8)
        
        plt.xlabel('Radio Planetario (Radios Terrestres)')
        plt.ylabel('Eficiencia de Detección')
        plt.title('Eficiencia de Detección vs Radio Planetario')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        
        # Agregar anotaciones
        for _, row in efficiency_df.iterrows():
            plt.annotate(f'{row["n_detected"]}/{row["n_planets"]}',
                        (row['radius_center'], row['efficiency']),
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        return plt.gcf()
```

## ⏱️ Análisis de Tiempo de Ejecución

```python
import time
from contextlib import contextmanager

class PerformanceMetrics:
    def __init__(self):
        self.timing_results = {}
    
    @contextmanager
    def timer(self, operation_name):
        """Context manager para medir tiempo"""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.timing_results[operation_name] = elapsed_time
            print(f"{operation_name}: {elapsed_time:.3f} segundos")
    
    def benchmark_models(self, models, X_train, X_test, y_train, y_test):
        """Benchmark de múltiples modelos"""
        results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluando {model_name}...")
            
            # Tiempo de entrenamiento
            with self.timer(f"{model_name}_training"):
                model.fit(X_train, y_train)
            
            # Tiempo de predicción
            with self.timer(f"{model_name}_prediction"):
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            
            results[model_name] = {
                'accuracy': accuracy,
                'training_time': self.timing_results[f"{model_name}_training"],
                'prediction_time': self.timing_results[f"{model_name}_prediction"],
                'predictions_per_second': len(X_test) / self.timing_results[f"{model_name}_prediction"]
            }
        
        return pd.DataFrame(results).T
```

## 📊 Reportes Integrales de Evaluación

```python
class ComprehensiveEvaluationReport:
    def __init__(self):
        self.report_data = {}
    
    def generate_full_report(self, model, X_test, y_test, model_name="Ensemble Model"):
        """Genera reporte completo de evaluación"""
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Métricas básicas
        basic_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob[:, 1])
        }
        
        # Métricas astronómicas
        astro_metrics = AstronomicalMetrics().calculate_completeness_reliability(
            y_test, y_pred, y_prob
        )
        
        # Generar visualizaciones
        figures = {
            'confusion_matrix': ExoplanetConfusionMatrix().plot_confusion_matrix(
                y_test, y_pred, normalize=True
            ),
            'roc_curve': ROCAnalysis().plot_roc_curve(y_test, y_prob, model_name),
            'pr_curve': PrecisionRecallAnalysis().plot_pr_curve(y_test, y_prob, model_name)
        }
        
        # Compilar reporte
        self.report_data = {
            'model_name': model_name,
            'basic_metrics': basic_metrics,
            'astronomical_metrics': astro_metrics,
            'figures': figures,
            'sample_size': len(y_test),
            'positive_class_proportion': np.mean(y_test)
        }
        
        return self.report_data
    
    def print_summary(self):
        """Imprime resumen del reporte"""
        print(f"\n{'='*50}")
        print(f"REPORTE DE EVALUACIÓN - {self.report_data['model_name']}")
        print(f"{'='*50}")
        
        print(f"\nTamaño de muestra: {self.report_data['sample_size']}")
        print(f"Proporción de planetas: {self.report_data['positive_class_proportion']:.1%}")
        
        print(f"\n{'MÉTRICAS BÁSICAS':-^30}")
        for metric, value in self.report_data['basic_metrics'].items():
            print(f"{metric:15}: {value:.3f} ({value*100:.1f}%)")
        
        print(f"\n{'MÉTRICAS ASTRONÓMICAS':-^30}")
        for metric_name, metric_data in self.report_data['astronomical_metrics'].items():
            print(f"{metric_name:15}: {metric_data['percentage']:.1f}% - {metric_data['interpretation']}")
```

---

**Anterior**: [Datasets y Preprocesamiento](./datasets-preprocessing.md) | **Siguiente**: [Interfaz Web y Deployment](./web-interface-deployment.md)