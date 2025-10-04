# Datasets y Preprocesamiento - Datos de Exoplanetas

## 📊 Descripción Detallada de Datasets

### 1. Kepler Objects of Interest (KOI)

#### Características Generales
- **Fuente**: NASA Exoplanet Archive
- **Período**: 2009-2017 (Misión Kepler)
- **Tamaño**: 9,654 filas × 50 columnas
- **Método de detección**: Fotometría de tránsito

#### Estructura del Dataset
```python
# Columnas principales del KOI dataset
koi_columns = {
    # Identificadores
    'kepoi_name': 'Nombre del objeto Kepler',
    'kepler_name': 'Nombre confirmado (si aplica)',
    'koi_disposition': 'Disposición usando datos Kepler',
    'koi_pdisposition': 'Disposición preliminar',
    
    # Parámetros estelares
    'koi_slogg': 'Log gravedad superficial estelar',
    'koi_srad': 'Radio estelar (radios solares)',
    'koi_smass': 'Masa estelar (masas solares)',
    'koi_steff': 'Temperatura efectiva estelar (K)',
    
    # Parámetros planetarios
    'koi_period': 'Período orbital (días)',
    'koi_prad': 'Radio planetario (radios terrestres)',
    'koi_teq': 'Temperatura de equilibrio planetaria (K)',
    'koi_impact': 'Parámetro de impacto',
    'koi_duration': 'Duración del tránsito (hrs)',
    'koi_depth': 'Profundidad del tránsito (ppm)',
    
    # Métricas de detección
    'koi_max_sngle_ev': 'Máximo single event statistic',
    'koi_max_mult_ev': 'Máximo multiple event statistic',
    'koi_model_snr': 'Signal-to-noise ratio del modelo'
}
```

#### Clasificaciones Target
```python
koi_dispositions = {
    'CONFIRMED': {
        'description': 'Exoplaneta confirmado',
        'count': '~2,400 objetos',
        'confidence': 'Alta - múltiples validaciones'
    },
    'CANDIDATE': {
        'description': 'Candidato a exoplaneta',
        'count': '~4,700 objetos', 
        'confidence': 'Media - requiere validación adicional'
    },
    'FALSE POSITIVE': {
        'description': 'Falso positivo identificado',
        'count': '~2,500 objetos',
        'confidence': 'Alta - descartado por análisis'
    }
}
```

### 2. TESS Objects of Interest (TOI)

#### Características Generales
- **Fuente**: TESS Mission (NASA)
- **Período**: 2018-presente
- **Cobertura**: Todo el cielo (sectores de 27 días)
- **Ventaja**: Estrellas más brillantes y cercanas

#### Estructura del Dataset
```python
toi_columns = {
    # Identificadores TESS
    'toi': 'TESS Object of Interest number',
    'tic_id': 'TESS Input Catalog identifier',
    'tfopwg_disp': 'TFOP Working Group disposition',
    
    # Parámetros estelares TESS
    'st_rad': 'Radio estelar',
    'st_mass': 'Masa estelar', 
    'st_teff': 'Temperatura efectiva',
    'st_logg': 'Log gravedad superficial',
    
    # Parámetros del tránsito
    'pl_orbper': 'Período orbital',
    'pl_rade': 'Radio planetario',
    'pl_eqt': 'Temperatura de equilibrio',
    'pl_tranmid': 'Tiempo de tránsito medio',
    
    # Métricas TESS específicas
    'tess_mag': 'Magnitud TESS',
    'depth_ppm': 'Profundidad en partes por millón',
    'duration_hr': 'Duración en horas'
}
```

#### Clasificaciones TESS
```python
tess_dispositions = {
    'KP': 'Known Planet (Planeta conocido)',
    'PC': 'Planet Candidate (Candidato planetario)',
    'FP': 'False Positive (Falso positivo)',
    'APC': 'Ambiguous Planet Candidate (Candidato ambiguo)',
    'FA': 'False Alarm (Falsa alarma)'
}
```

### 3. K2 Planets and Candidates

#### Características Generales
- **Fuente**: K2 Mission (extensión de Kepler)
- **Período**: 2014-2018
- **Campañas**: 19 campañas diferentes (C0-C19)
- **Ventaja**: Diversidad de campos estelares

#### Estructura del Dataset
```python
k2_columns = {
    # Identificadores K2
    'k2_name': 'Nombre del objeto K2',
    'epic_id': 'EPIC catalog identifier',
    'campaign': 'Campaña de observación K2',
    'archive_disp': 'Archive disposition',
    
    # Parámetros específicos K2
    'k2_period': 'Período orbital',
    'k2_prad': 'Radio planetario',
    'k2_teq': 'Temperatura equilibrio',
    'k2_impact': 'Parámetro impacto',
    
    # Información de la campaña
    'campaign_field': 'Campo observado',
    'observing_days': 'Días de observación',
    'data_quality': 'Calidad de datos'
}
```

## 🔧 Preprocesamiento de Datos

### Análisis Exploratorio Inicial

#### Distribución de Clases
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ExploratoryAnalysis:
    def __init__(self, datasets):
        self.datasets = datasets
    
    def analyze_class_distribution(self):
        """Análisis de distribución de clases"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (name, df) in enumerate(self.datasets.items()):
            target_col = self.get_target_column(name)
            class_counts = df[target_col].value_counts()
            
            axes[i].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
            axes[i].set_title(f'{name} - Distribución de Clases')
        
        plt.tight_layout()
        return fig
    
    def missing_values_analysis(self):
        """Análisis de valores faltantes"""
        missing_stats = {}
        
        for name, df in self.datasets.items():
            missing_pct = (df.isnull().sum() / len(df)) * 100
            missing_stats[name] = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        return missing_stats
```

#### Estadísticas Descriptivas
```python
def generate_descriptive_stats(df):
    """Estadísticas descriptivas por categoría de columnas"""
    
    # Parámetros estelares
    stellar_params = ['koi_srad', 'koi_smass', 'koi_steff', 'koi_slogg']
    stellar_stats = df[stellar_params].describe()
    
    # Parámetros planetarios
    planet_params = ['koi_period', 'koi_prad', 'koi_teq', 'koi_impact']
    planet_stats = df[planet_params].describe()
    
    # Métricas de tránsito
    transit_params = ['koi_duration', 'koi_depth', 'koi_model_snr']
    transit_stats = df[transit_params].describe()
    
    return {
        'stellar': stellar_stats,
        'planetary': planet_stats,
        'transit': transit_stats
    }
```

### Limpieza y Normalización

#### Manejo de Valores Faltantes
```python
class DataCleaner:
    def __init__(self):
        self.imputation_strategies = {
            'stellar_params': 'median',  # Parámetros estelares
            'planetary_params': 'conditional_median',  # Por disposición
            'transit_metrics': 'model_based',  # Basado en modelos físicos
            'categorical': 'mode'  # Valores más frecuentes
        }
    
    def handle_missing_values(self, df):
        """Estrategia integral para valores faltantes"""
        df_clean = df.copy()
        
        # 1. Parámetros estelares - Imputación por mediana
        stellar_cols = ['koi_srad', 'koi_smass', 'koi_steff', 'koi_slogg']
        for col in stellar_cols:
            if col in df_clean.columns:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
        
        # 2. Parámetros planetarios - Imputación condicional
        planet_cols = ['koi_period', 'koi_prad', 'koi_teq']
        for col in planet_cols:
            if col in df_clean.columns:
                # Imputar por mediana dentro de cada disposición
                df_clean[col] = df_clean.groupby('koi_disposition')[col].transform(
                    lambda x: x.fillna(x.median())
                )
        
        # 3. Métricas de tránsito - Modelo físico
        df_clean = self.impute_transit_metrics(df_clean)
        
        return df_clean
    
    def impute_transit_metrics(self, df):
        """Imputación basada en relaciones físicas"""
        # Profundidad de tránsito basada en radios
        mask = df['koi_depth'].isna()
        if mask.any():
            estimated_depth = ((df['koi_prad'] / df['koi_srad']) ** 2) * 1e6  # ppm
            df.loc[mask, 'koi_depth'] = estimated_depth[mask]
        
        # Duración basada en parámetros orbitales
        mask = df['koi_duration'].isna()
        if mask.any():
            # Fórmula aproximada para duración de tránsito
            estimated_duration = (df['koi_period'] / np.pi) * np.sqrt(
                (1 + df['koi_prad'] / df['koi_srad']) ** 2 - df['koi_impact'] ** 2
            )
            df.loc[mask, 'koi_duration'] = estimated_duration[mask]
        
        return df
```

#### Detección y Manejo de Outliers
```python
from scipy import stats

class OutlierHandler:
    def __init__(self, method='iqr'):
        self.method = method
        self.outlier_bounds = {}
    
    def detect_outliers(self, df, columns):
        """Detección de outliers usando IQR o Z-score"""
        outliers = {}
        
        for col in columns:
            if self.method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            
            elif self.method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                threshold = 3
                outlier_indices = np.where(z_scores > threshold)[0]
            
            outliers[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'count': len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
            }
        
        return outliers
    
    def handle_outliers(self, df, strategy='cap'):
        """Manejo de outliers"""
        if strategy == 'cap':
            # Winsorizing - limitar a percentiles
            for col in df.select_dtypes(include=[np.number]).columns:
                lower_percentile = df[col].quantile(0.01)
                upper_percentile = df[col].quantile(0.99)
                df[col] = df[col].clip(lower_percentile, upper_percentile)
        
        elif strategy == 'remove':
            # Remover outliers extremos (usar con cuidado)
            for col in df.select_dtypes(include=[np.number]).columns:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                df = df[z_scores < 4]  # Más conservador que 3
        
        return df
```

### Feature Engineering Avanzado

#### Características Derivadas
```python
class FeatureEngineer:
    def __init__(self):
        self.feature_catalog = {}
    
    def create_derived_features(self, df):
        """Creación de características derivadas"""
        df_enhanced = df.copy()
        
        # 1. Ratios físicamente significativos
        df_enhanced['planet_star_radius_ratio'] = df['koi_prad'] / df['koi_srad']
        df_enhanced['planet_star_mass_ratio'] = df['koi_pmass'] / df['koi_smass']  # Si disponible
        df_enhanced['equilibrium_temp_ratio'] = df['koi_teq'] / df['koi_steff']
        
        # 2. Características orbitales
        df_enhanced['orbital_velocity'] = (2 * np.pi * df['koi_sma']) / df['koi_period']
        df_enhanced['orbital_angular_momentum'] = df['koi_pmass'] * df['orbital_velocity'] * df['koi_sma']
        
        # 3. Índices de habitabilidad
        df_enhanced['habitable_zone_distance'] = self.calculate_habitable_zone_distance(df)
        df_enhanced['tidal_heating_parameter'] = self.calculate_tidal_heating(df)
        
        # 4. Métricas de detección mejoradas
        df_enhanced['snr_ratio'] = df['koi_max_mult_ev'] / df['koi_max_sngle_ev']
        df_enhanced['transit_probability'] = self.calculate_transit_probability(df)
        
        # 5. Características de calidad de datos
        df_enhanced['observation_baseline'] = df['koi_dataspan']
        df_enhanced['transit_count'] = df['koi_dataspan'] / df['koi_period']
        
        return df_enhanced
    
    def calculate_habitable_zone_distance(self, df):
        """Cálculo de distancia a zona habitable"""
        # Zona habitable conservativa (0.95-1.37 AU para el Sol)
        hz_inner = 0.95 * np.sqrt(df['koi_steff'] / 5778)  # Escalado por temperatura
        hz_outer = 1.37 * np.sqrt(df['koi_steff'] / 5778)
        
        # Distancia normalizada (0 = centro de HZ, <0 = muy caliente, >0 = muy frío)
        hz_center = (hz_inner + hz_outer) / 2
        hz_distance = (df['koi_sma'] - hz_center) / (hz_outer - hz_inner)
        
        return hz_distance
    
    def calculate_transit_probability(self, df):
        """Probabilidad geométrica de tránsito"""
        # P_transit = R_star / a (aproximación para órbitas circulares)
        transit_prob = df['koi_srad'] / df['koi_sma']
        return np.clip(transit_prob, 0, 1)  # Limitar a [0,1]
```

#### Análisis de Periodicidad y Series Temporales
```python
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

class TimeSeriesFeatures:
    def __init__(self):
        self.period_detection_methods = ['fft', 'autocorr', 'bls']
    
    def extract_periodicity_features(self, lightcurve_data):
        """Extracción de características de periodicidad"""
        features = {}
        
        # 1. Análisis FFT
        fft_vals = fft(lightcurve_data['flux'])
        fft_freq = fftfreq(len(lightcurve_data), d=lightcurve_data['time'].diff().median())
        
        # Encontrar picos dominantes
        power_spectrum = np.abs(fft_vals)
        peaks, _ = find_peaks(power_spectrum, height=np.std(power_spectrum))
        
        features['dominant_frequency'] = fft_freq[peaks[0]] if len(peaks) > 0 else 0
        features['frequency_power'] = power_spectrum[peaks[0]] if len(peaks) > 0 else 0
        
        # 2. Autocorrelación
        autocorr = np.correlate(lightcurve_data['flux'], lightcurve_data['flux'], mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Encontrar picos de autocorrelación (indicativos de periodicidad)
        autocorr_peaks, _ = find_peaks(autocorr, height=0.1 * np.max(autocorr))
        features['autocorr_period'] = autocorr_peaks[0] if len(autocorr_peaks) > 0 else 0
        
        # 3. Estadísticas de variabilidad
        features['flux_std'] = np.std(lightcurve_data['flux'])
        features['flux_mad'] = np.median(np.abs(lightcurve_data['flux'] - np.median(lightcurve_data['flux'])))
        features['flux_range'] = np.ptp(lightcurve_data['flux'])
        
        return features
```

### Normalización y Escalado

#### Pipeline de Preprocesamiento
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer

class PreprocessingPipeline:
    def __init__(self):
        self.numerical_features = []
        self.categorical_features = []
        self.pipeline = None
    
    def build_preprocessing_pipeline(self, df):
        """Construcción del pipeline de preprocesamiento"""
        
        # Identificar tipos de características
        self.numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remover target de features
        if 'koi_disposition' in self.numerical_features:
            self.numerical_features.remove('koi_disposition')
        if 'koi_disposition' in self.categorical_features:
            self.categorical_features.remove('koi_disposition')
        
        # Definir transformadores
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())  # Robusto ante outliers
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(drop='first', sparse=False))
        ])
        
        # Combinar transformadores
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        return self.pipeline
    
    def fit_transform(self, X):
        """Ajustar y transformar datos"""
        return self.pipeline.fit_transform(X)
    
    def transform(self, X):
        """Solo transformar (para datos nuevos)"""
        return self.pipeline.transform(X)
```

## 📊 Integración de Datasets

### Unificación de Esquemas
```python
class DatasetIntegrator:
    def __init__(self):
        self.column_mapping = {
            'KOI': {
                'target': 'koi_disposition',
                'period': 'koi_period',
                'radius': 'koi_prad',
                'temp': 'koi_teq'
            },
            'TOI': {
                'target': 'tfopwg_disp',
                'period': 'pl_orbper',
                'radius': 'pl_rade', 
                'temp': 'pl_eqt'
            },
            'K2': {
                'target': 'archive_disp',
                'period': 'k2_period',
                'radius': 'k2_prad',
                'temp': 'k2_teq'
            }
        }
    
    def unify_datasets(self, datasets):
        """Unificación de múltiples datasets"""
        unified_data = []
        
        for dataset_name, df in datasets.items():
            # Mapear columnas al esquema común
            df_mapped = self.map_columns(df, dataset_name)
            
            # Agregar columna de fuente
            df_mapped['source_dataset'] = dataset_name
            
            # Unificar etiquetas
            df_mapped['unified_target'] = self.unify_labels(
                df_mapped[self.column_mapping[dataset_name]['target']]
            )
            
            unified_data.append(df_mapped)
        
        # Concatenar todos los datasets
        combined_df = pd.concat(unified_data, ignore_index=True, sort=False)
        
        return combined_df
    
    def unify_labels(self, labels):
        """Unificación de etiquetas a esquema común"""
        label_mapping = {
            'CONFIRMED': 'CONFIRMED',
            'KP': 'CONFIRMED',
            'CANDIDATE': 'CANDIDATE', 
            'PC': 'CANDIDATE',
            'APC': 'CANDIDATE',
            'FALSE POSITIVE': 'FALSE_POSITIVE',
            'FP': 'FALSE_POSITIVE'
        }
        
        return labels.map(label_mapping)
```

---

**Anterior**: [Metodología de Implementación](./implementation-methodology.md) | **Siguiente**: [Métricas y Evaluación](./metrics-evaluation.md)