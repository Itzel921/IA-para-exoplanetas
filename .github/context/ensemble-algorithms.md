# Algoritmos de Ensemble para Detección de Exoplanetas

## 🧠 Fundamentos de Ensemble Learning

### Concepto Fundamental
Los **algoritmos ensemble** combinan múltiples algoritmos de machine learning para obtener una predicción final mejorada. El principio básico es que las predicciones incorrectas de un algoritmo pueden ser compensadas por predicciones correctas de otro, resultando en un rendimiento superior al de cualquier algoritmo individual.

### Arquitectura Típica de Ensemble
```
Datos de Entrada
       ↓
   Algoritmo 1 → Predicción 1
   Algoritmo 2 → Predicción 2    → Esquema de Votación → Predicción Final
   Algoritmo N → Predicción N
```

## 🔄 Estrategias para Diversidad en Ensemble

### 1. Manipulación de Entrada (Input Manipulation)
- Cada algoritmo base se entrena con diferentes subconjuntos de datos
- **Bagging**: Bootstrap Aggregating
- **Random Subspace**: Selección aleatoria de características

### 2. Manipulación del Aprendizaje (Learning Manipulation)
- Modificación del proceso de búsqueda de la solución óptima
- **Boosting**: Aprendizaje secuencial con re-pesado de errores
- Diferentes funciones de pérdida o criterios de optimización

### 3. Particionamiento (Partitioning)
- División del dataset en subconjuntos especializados
- Cada algoritmo se especializa en una región del espacio de características

### 4. Hibridación (Hybridization)
- Combinación de dos o más estrategias anteriores
- Enfoques multi-nivel y jerárquicos

## 🎯 Cinco Algoritmos Ensemble Evaluados

### 1. AdaBoost (Adaptive Boosting)

#### Principio de Funcionamiento
- **Tipo**: Boosting secuencial
- **Método de Fusión**: Votación ponderada
- **Dependencia**: Secuencial (algoritmos posteriores dependen de anteriores)
- **Entrenamiento**: Adaptativo con re-pesado de muestras

#### Características Técnicas
```python
# Hiperparámetros clave
n_estimators: Número de algoritmos base
learning_rate: Tasa de aprendizaje
algorithm: SAMME o SAMME.R
```

#### Rendimiento en Exoplanetas
- **Accuracy inicial**: 81.37%
- **Accuracy optimizada**: 82.52% (+1.15%)
- **Mejora**: Incremento significativo con más estimadores
- **Fortalezas**: Bueno para reducir bias, eficaz con weak learners

### 2. Random Forest

#### Principio de Funcionamiento
- **Tipo**: Bagging con árboles de decisión
- **Método de Fusión**: Votación por mayoría
- **Dependencia**: Independiente (entrenamiento paralelo)
- **Entrenamiento**: Bootstrap sampling + random feature selection

#### Características Técnicas
```python
# Hiperparámetros clave
n_estimators: Número de árboles (100 → 1600 optimizado)
max_depth: Profundidad máxima de árboles
min_samples_split: Mínimo de muestras para división
max_features: Número de características por división
```

#### Rendimiento en Exoplanetas
- **Accuracy inicial**: 82.25%
- **Accuracy optimizada**: 82.64% (+0.39%)
- **Fortalezas**: Robusto, fácil implementación, interpretable
- **Aplicación**: Uno de los ensembles más utilizados en astronomía

### 3. Stacking (Stacked Generalization)

#### Principio de Funcionamiento
- **Tipo**: Meta-learning con múltiples niveles
- **Método de Fusión**: Meta-modelo entrenado
- **Dependencia**: Jerárquica (meta-modelo aprende de base learners)
- **Entrenamiento**: Dos fases - base learners y meta-learner

#### Arquitectura
```
Nivel 1: Base Learners (ej. Random Forest, Gradient Boosting)
          ↓
Nivel 2: Meta-Learner (aprende a combinar predicciones del Nivel 1)
```

#### Rendimiento en Exoplanetas
- **Accuracy inicial**: 82.72% (mejor inicial)
- **Accuracy optimizada**: 83.03%
- **Mejor combinación**: LGBM + Gradient Boosting = 83.08%
- **Fortalezas**: 
  - Mejor rendimiento global
  - Capacidad de aprender combinaciones óptimas
  - Flexibilidad en la elección de algoritmos base

#### Configuraciones Evaluadas
| Base Learners | Accuracy | Observaciones |
|---------------|----------|---------------|
| Random Forest + Gradient Boosting | 83.03% | Configuración estándar |
| LGBM + Gradient Boosting | 83.08% | Mejor resultado |
| Múltiples combinaciones | Variable | Exploración sistemática |

### 4. Random Subspace Method

#### Principio de Funcionamiento
- **Tipo**: Feature bagging
- **Método de Fusión**: Votación por mayoría
- **Dependencia**: Independiente
- **Entrenamiento**: Selección aleatoria de subconjuntos de características

#### Características Técnicas
```python
# Hiperparámetros clave
n_estimators: Número de algoritmos base
max_features: Fracción de características por subespacio
base_estimator: Algoritmo base (por defecto: Decision Tree)
```

#### Rendimiento en Exoplanetas
- **Accuracy inicial**: ~80.8%
- **Accuracy optimizada**: 81.91% (+1.1%)
- **Fortalezas**: Efectivo con alta dimensionalidad, reduce overfitting
- **Limitaciones**: No alcanzó el 82% incluso optimizado

### 5. Extremely Randomized Trees (Extra Trees)

#### Principio de Funcionamiento
- **Tipo**: Bagging con randomización extrema
- **Método de Fusión**: Votación por mayoría
- **Dependencia**: Independiente
- **Entrenamiento**: Randomización en selección de splits y características

#### Diferencias con Random Forest
- **Splits aleatorios**: No busca el mejor split, usa splits aleatorios
- **Todo el dataset**: Usa todo el dataset (no bootstrap sampling)
- **Mayor randomización**: Reduce variance pero puede aumentar bias

#### Rendimiento en Exoplanetas
- **Accuracy inicial**: 82.06%
- **Accuracy optimizada**: 82.36% (+0.30%)
- **Características**: Mejora menor debido a la naturaleza aleatoria
- **Ventajas**: Más rápido que Random Forest, menos propenso a overfitting

## 📊 Análisis Comparativo de Resultados

### Resumen de Performance
| Algoritmo | Accuracy Inicial | Accuracy Optimizada | Mejora | Ranking |
|-----------|------------------|---------------------|--------|---------|
| **Stacking** | 82.72% | **83.08%** | +0.36% | 🥇 1° |
| AdaBoost | 81.37% | 82.52% | +1.15% | 🥈 2° |
| Random Forest | 82.25% | 82.64% | +0.39% | 🥉 3° |
| Extra Trees | 82.06% | 82.36% | +0.30% | 4° |
| Random Subspace | ~80.8% | 81.91% | +1.10% | 5° |

### Métricas Detalladas (Stacking - Mejor Modelo)
```
Accuracy:    83.08%
Precision:   >80%
Recall:      >80%
F1-Score:    >82%
Specificity: >80%
```

## 🔧 Optimización de Hiperparámetros

### Metodología de Tuning
1. **Grid Search**: Búsqueda exhaustiva en espacio de parámetros
2. **Iteraciones**: 100 combinaciones por algoritmo
3. **Validación**: Cross-validation para evitar overfitting
4. **Métricas**: Accuracy como criterio principal de optimización

### Hiperparámetros Críticos por Algoritmo

#### AdaBoost
- `n_estimators`: 50 → Optimizado
- `learning_rate`: 1.0 → Optimizado
- **Impacto**: +1.15% accuracy

#### Random Forest
- `n_estimators`: 100 → 1600
- `max_depth`: Auto → Optimizado
- **Impacto**: +0.39% accuracy

#### Stacking
- Base learners: Configuración específica
- Meta-learner: Optimización individual
- **Impacto**: +0.36% accuracy

## 🎯 Aplicación Específica a Exoplanetas

### Ventajas de Ensemble en Astronomía

#### Manejo de Ruido
- Múltiples algoritmos compensan diferentes tipos de ruido
- Reducción de falsos positivos por consenso
- Mayor robustez ante artefactos instrumentales

#### Generalización
- Mejor performance en diferentes tipos de estrellas
- Adaptabilidad a diferentes misiones (Kepler, TESS, K2)
- Menor dependencia de características específicas

#### Interpretabilidad Mejorada
- Análisis de consenso entre algoritmos
- Identificación de características más importantes
- Confianza en predicciones basada en acuerdo

### Desafíos Específicos

#### Computational Cost
- Mayor tiempo de entrenamiento e inferencia
- Necesidad de recursos computacionales superiores
- Balance entre performance y eficiencia

#### Complexity Management
- Múltiples hiperparámetros por algoritmo base
- Riesgo de overfitting en meta-modelos
- Dificultad en debugging y troubleshooting

## 🚀 Mejores Prácticas para Implementación

### Selección de Algoritmos Base
1. **Diversidad**: Combinar algoritmos con diferentes biases
2. **Complementariedad**: Algoritmos que cometan errores diferentes
3. **Performance Individual**: Algoritmos base con rendimiento razonable

### Estrategias de Combinación
1. **Voting**: Simple majority o weighted voting
2. **Stacking**: Meta-learner para combinación óptima
3. **Blending**: Combinación lineal con pesos optimizados

### Validación y Testing
1. **Cross-Validation**: K-fold para robustez
2. **Holdout Sets**: Validación en datos no vistos
3. **Temporal Splits**: Especialmente importante para series temporales

## 📈 Futuras Direcciones

### Ensemble Avanzados
- **Deep Ensemble**: Combinación de redes neuronales
- **Bayesian Ensemble**: Incorporación de incertidumbre
- **Dynamic Ensemble**: Selección adaptativa de modelos

### Optimización Automática
- **AutoML**: Selección automática de algoritmos y parámetros
- **Neural Architecture Search**: Para componentes deep learning
- **Multi-objective Optimization**: Balance accuracy-interpretability-efficiency

### Aplicaciones Específicas
- **Real-time Processing**: Para datos de TESS en streaming
- **Multi-mission Integration**: Combinación de datos Kepler/K2/TESS
- **Rare Event Detection**: Optimización para eventos poco frecuentes

---

**Anterior**: [Fundamentos Teóricos](./theoretical-foundations.md) | **Siguiente**: [Metodología de Implementación](./implementation-methodology.md)