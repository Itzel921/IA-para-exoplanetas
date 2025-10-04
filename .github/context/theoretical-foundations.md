# Fundamentos Teóricos - Detección de Exoplanetas con Machine Learning

## 🌌 Introducción a los Exoplanetas

Los **exoplanetas** son planetas que orbitan estrellas fuera de nuestro sistema solar. Desde el descubrimiento del primer exoplaneta en 1992 orbitando un pulsar, el campo ha evolucionado dramáticamente, especialmente con el desarrollo de técnicas de detección automatizadas.

## 🔭 Métodos de Detección de Exoplanetas

### 1. Método de Tránsito
**Principio**: Cuando un planeta pasa frente a su estrella anfitriona, bloquea una pequeña cantidad de luz, causando una disminución temporal en el brillo observado.

**Características**:
- La disminución de luz es proporcional al tamaño del planeta
- La periodicidad indica el período orbital
- Es el método más exitoso para detección masiva

**Ventajas**:
- Permite detectar múltiples planetas simultáneamente
- Proporciona información sobre el tamaño del planeta
- Datos susceptibles de análisis automatizado

### 2. Velocidad Radial
**Principio**: La gravedad del planeta causa que la estrella "oscile" ligeramente, detectado por cambios Doppler en el espectro estelar.

**Aplicaciones**:
- Confirmación de candidatos de tránsito
- Determinación de masa planetaria
- Detección de planetas grandes

### 3. Astrometría
**Principio**: Detección del movimiento de la estrella causado por la atracción gravitacional del planeta.

### 4. Imagen Directa
**Principio**: Observación directa del planeta, generalmente usando coronógrafos para bloquear la luz estelar.

## 🛰️ Misiones Espaciales Clave

### Kepler Space Telescope (2009-2017)
- **Objetivo**: Monitoreo continuo de ~200,000 estrellas
- **Método**: Fotometría de tránsito de alta precisión
- **Logros**: 
  - Descubrimiento de >2,600 exoplanetas confirmados
  - >4,000 candidatos adicionales
  - Demostración de que planetas del tamaño de la Tierra son comunes

### K2 Mission (2014-2018)
- **Descripción**: Extensión de Kepler con diferentes campos de observación
- **Contribuciones**: ~400 planetas confirmados adicionales
- **Ventaja**: Mayor diversidad de tipos estelares observados

### TESS (Transiting Exoplanet Survey Satellite) (2018-presente)
- **Alcance**: Monitoreo de todo el cielo
- **Objetivo**: Planetas alrededor de estrellas brillantes cercanas
- **Fortaleza**: Ideal para seguimiento desde tierra

## 🤖 Machine Learning en Astronomía

### Evolución del Análisis
1. **Análisis Manual**: Inspección visual de curvas de luz
2. **Métodos Algorítmicos**: Box Least Squares (BLS) fitting
3. **Machine Learning**: Clasificación automatizada con alta precisión
4. **Deep Learning**: Redes neuronales para patrones complejos

### Ventajas del ML en Detección de Exoplanetas

#### Escalabilidad
- Procesamiento de millones de curvas de luz
- Análisis en tiempo real de datos de misiones activas
- Manejo eficiente de big data astronómico

#### Precisión Mejorada
- Reducción de falsos positivos
- Detección de señales débiles
- Identificación de patrones complejos

#### Automatización
- Análisis continuo sin intervención humana
- Procesamiento estandarizado
- Reducción de sesgos humanos

## 📊 Características de los Datos de Tránsito

### Curvas de Luz
**Definición**: Serie temporal de la intensidad luminosa de una estrella

**Componentes**:
- **Señal de tránsito**: Disminución periódica característica
- **Ruido estelar**: Variabilidad intrínseca de la estrella
- **Ruido instrumental**: Limitaciones del detector
- **Ruido sistemático**: Efectos ambientales y de calibración

### Parámetros Clave de Tránsito
1. **Profundidad**: Fracción de luz bloqueada
2. **Duración**: Tiempo que dura el tránsito
3. **Período**: Tiempo entre tránsitos consecutivos
4. **Forma**: Perfil detallado de la curva de tránsito

## 🧠 Enfoques de Machine Learning

### Aprendizaje Supervisado
**Datasets Etiquetados**:
- CONFIRMED: Exoplanetas verificados
- CANDIDATE: Candidatos bajo investigación
- FALSE POSITIVE: Señales que imitan tránsitos

**Algoritmos Comunes**:
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- Support Vector Machines
- Neural Networks

### Feature Engineering para Curvas de Luz

#### Features Estadísticos
- Media, mediana, desviación estándar
- Skewness y kurtosis
- Percentiles y rangos intercuartílicos

#### Features de Serie Temporal
- Análisis de Fourier (frecuencias dominantes)
- Autocorrelación
- Tendencias y estacionalidad
- Periodicidad detectada

#### Features Específicos de Tránsito
- Detección de dips periódicos
- Simetría de la señal
- Relación señal-ruido
- Consistencia temporal

### Desafíos Únicos

#### Desbalance de Clases
- Falsos positivos abundantes vs. planetas reales escasos
- Necesidad de técnicas de balanceo
- Métricas específicas para clases minoritarias

#### Ruido y Artefactos
- Variabilidad estelar intrínseca
- Efectos instrumentales
- Contaminación por objetos de fondo

#### Generalización
- Diferencias entre misiones (Kepler vs. TESS)
- Variación en tipos estelares
- Rangos de parámetros planetarios

## 📈 Estado del Arte Actual

### Métodos Tradicionales
**Box Least Squares (BLS)**:
- Algoritmo estándar para detección de tránsitos
- Busca señales periódicas en forma de caja
- Baseline para comparación de métodos ML

### Enfoques de Deep Learning

#### Redes Neuronales Convolucionales (CNN)
- Procesamiento directo de curvas de luz
- Detección automática de patrones
- Ejemplos: AstroNet (Google), ExoNet

#### Redes Recurrentes (RNN/LSTM)
- Modelado de dependencias temporales
- Memoria de patrones de largo plazo
- Procesamiento secuencial natural

### Métodos Ensemble
- Combinación de múltiples algoritmos
- Mejora en robustez y accuracy
- Reducción de overfitting
- **Resultados destacados**: >80% accuracy promedio

## 🎯 Métricas de Evaluación Específicas

### Métricas de Clasificación
- **Accuracy**: Porcentaje de clasificaciones correctas
- **Precision**: Fracción de positivos predichos que son correctos
- **Recall (Sensitivity)**: Fracción de positivos reales detectados
- **Specificity**: Fracción de negativos reales identificados correctamente
- **F1-Score**: Media armónica de precision y recall

### Métricas Astronómicas Específicas
- **Completeness**: Fracción de planetas reales detectados
- **Reliability**: Fracción de detecciones que son planetas reales
- **False Positive Rate**: Tasa de falsos positivos
- **Detection Efficiency**: Eficiencia en función del tamaño planetario

### Análisis ROC y PR
- **ROC Curve**: True Positive Rate vs. False Positive Rate
- **PR Curve**: Precision vs. Recall
- **AUC**: Área bajo la curva como métrica resumen

---

**Anterior**: [Descripción del Desafío](./challenge-description.md) | **Siguiente**: [Algoritmos de Ensemble](./ensemble-algorithms.md)