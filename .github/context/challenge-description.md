# NASA Space Apps Challenge 2025 - Descripción del Desafío

## 🚀 Challenge 2: A World Away: Hunting for Exoplanets with AI

### 📝 Contexto del Reto

El desafío consiste en crear un modelo de inteligencia artificial/aprendizaje automático (IA/ML) entrenado con datasets abiertos de NASA para identificar exoplanetas en datos nuevos.

### 🌟 Importancia del Problema

La detección de exoplanetas es un área en crecimiento dentro de la exploración astronómica. Misiones como **Kepler**, **K2** y **TESS** han permitido descubrir miles de planetas fuera del sistema solar mediante el **método de tránsito** (la disminución de luz de una estrella cuando un planeta pasa frente a ella).

Gran parte del análisis original se hizo de manera manual, pero con las técnicas modernas de ML es posible:
- Automatizar la clasificación de tránsitos
- Mejorar la precisión de identificación de exoplanetas
- Procesar grandes volúmenes de datos de manera eficiente
- Reducir falsos positivos y negativos

### 📊 Datasets Principales (Labeled, para aprendizaje supervisado)

#### 1. Kepler Objects of Interest (KOI)
- **Descripción**: Lista de exoplanetas confirmados, candidatos y falsos positivos detectados por Kepler
- **Columna clave**: `Disposition Using Kepler Data`
- **Categorías**:
  - CONFIRMED: Exoplanetas confirmados
  - CANDIDATE: Candidatos a exoplanetas
  - FALSE POSITIVE: Falsos positivos

#### 2. TESS Objects of Interest (TOI)
- **Descripción**: Datos de la misión TESS (Transiting Exoplanet Survey Satellite)
- **Columna clave**: `TFOWPG Disposition`
- **Categorías**:
  - KP: Confirmed planets (Planetas confirmados)
  - PC: Planet candidates (Candidatos planetarios)
  - FP: False positives (Falsos positivos)
  - APC: Ambiguous planet candidates (Candidatos ambiguos)

#### 3. K2 Planets and Candidates
- **Descripción**: Continuación de la misión Kepler
- **Columna clave**: `Archive Disposition`
- **Características**: Datos de diferentes campañas de observación

### 🎯 Objetivos Específicos del Proyecto

1. **Desarrollo del Modelo ML**:
   - Crear un clasificador robusto usando los datasets KOI, TOI y K2
   - Implementar técnicas de ensemble learning
   - Optimizar hiperparámetros para máximo rendimiento
   - Validar usando cross-validation y métricas apropiadas

2. **Interfaz Web Interactiva**:
   - Permitir subir o introducir nuevos datos
   - Mostrar predicciones (planeta confirmado / candidato / falso positivo)
   - Visualizar métricas de rendimiento (accuracy, precision, recall, ROC/PR curves)
   - (Opcional) Mostrar gráficas de curvas de luz preprocesadas

### 📚 Referencias Recomendadas

#### Artículos Científicos Clave:
1. **"Exoplanet Detection Using Machine Learning" (2021)**
   - Revisión general de métodos ML aplicados a detección de exoplanetas
   - Comparación de diferentes enfoques y técnicas

2. **"Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification"**
   - Estudio comparativo de algoritmos ensemble
   - Técnicas de preprocesamiento y optimización
   - Resultados con accuracy > 80% en todos los algoritmos evaluados

### 🛰️ Recursos Adicionales de Socios

#### NEOSSat (Canadá)
- Imágenes astronómicas dedicadas a exoplanetas
- Datos de asteroides y objetos cercanos a la Tierra
- Complemento para validación de resultados

#### James Webb Space Telescope (JWST)
- Información general del telescopio
- Contribuciones canadienses al proyecto
- Datos de alta resolución para validación

### 🏆 Criterios de Evaluación

#### Técnicos:
- **Accuracy del modelo**: > 90% objetivo
- **Robustez**: Rendimiento consistente en diferentes datasets
- **Eficiencia**: Tiempo de procesamiento razonable
- **Escalabilidad**: Capacidad de manejar grandes volúmenes de datos

#### Interfaz y Usabilidad:
- **Funcionalidad completa**: Todas las características implementadas
- **User Experience**: Interfaz intuitiva y responsive
- **Visualizaciones**: Gráficos claros y informativos
- **Documentación**: Instrucciones de uso claras

#### Innovación y Impacto:
- **Originalidad**: Enfoques novedosos o mejoras significativas
- **Aplicabilidad**: Potencial de uso en misiones reales
- **Reproducibilidad**: Código bien documentado y reproducible

### 📈 Métricas de Éxito

| Métrica | Objetivo | Excelente |
|---------|----------|-----------|
| Accuracy | > 85% | > 90% |
| Precision | > 80% | > 85% |
| Recall | > 80% | > 85% |
| F1-Score | > 82% | > 87% |
| AUC-ROC | > 0.90 | > 0.95 |

### 🔄 Metodología de Trabajo

1. **Fase 1**: Análisis exploratorio de datos
2. **Fase 2**: Preprocesamiento y feature engineering
3. **Fase 3**: Implementación y evaluación de modelos
4. **Fase 4**: Desarrollo de la interfaz web
5. **Fase 5**: Testing, optimización y documentación

---

**Siguiente**: [Fundamentos Teóricos](./theoretical-foundations.md)