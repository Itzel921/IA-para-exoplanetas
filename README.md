# NASA Space Apps Challenge 2025  
## Challenge 2: *A World Away: Hunting for Exoplanets with AI*

### 🚀 Contexto
El desafío consiste en crear un modelo de **IA/ML** entrenado con datasets abiertos de NASA para **identificar exoplanetas** en datos nuevos.  

Misiones como **Kepler, K2 y TESS** han permitido descubrir miles de planetas fuera del sistema solar mediante el **método de tránsito**.  
Hoy, con ML, podemos **automatizar la clasificación** de tránsitos y mejorar la detección.

---

### 📂 Datasets principales
- **Kepler Objects of Interest (KOI)** → clave: *Disposition Using Kepler Data*.  
- **TESS Objects of Interest (TOI)** → clave: *TFOWPG Disposition*.  
- **K2 Planets and Candidates** → clave: *Archive Disposition*.  

---

### 📚 Referencias
- *Exoplanet Detection Using Machine Learning (2021)*  
- *Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification*  

**Recursos extra:**  
- **NEOSSat (Canadá)** – imágenes astronómicas.  
- **JWST** – datos complementarios y contribuciones canadienses.  

---

### 🎯 Objetivo del equipo
Desarrollar un **modelo ML** que clasifique exoplanetas y exponerlo en una **interfaz web** con:
- Subida de datos.  
- Predicciones (confirmado / candidato / falso positivo).  
- Métricas (accuracy, precision, recall, ROC/PR).  
- (Opcional) Visualización de curvas de luz.  

---

### 🛠️ Stack propuesto
- **Python + scikit-learn / TensorFlow / PyTorch** → modelado.  
- **Pandas / NumPy** → preprocesamiento.  
- **Flask o FastAPI** → API backend.  
- **React / Streamlit** → interfaz web.  
- **Docker** → despliegue.  

---

### 👥 Organización
- **Project board:** GitHub Projects (Kanban).  
- **Branches:**  
  - `main` → estable  
  - `dev` → desarrollo  
  - `feature/*` → ramas de features  

---

### 📌 Próximos pasos
1. Recolectar datasets iniciales (KOI, TOI, K2).  
2. Definir baseline ML (Random Forest / XGBoost).  
3. Implementar preprocesamiento de curvas de luz.  
4. Construir API mínima (Flask/FastAPI).  
5. Conectar interfaz web.  

---
