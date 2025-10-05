"""
Modelos de datos para MongoDB - Backend Exoplanetas
NASA Space Apps Challenge 2025

Modelos simplificados para trabajar con datos de exoplanetas y satélites.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class ExoplanetData:
    """
    Modelo de datos para exoplanetas
    """
    object_name: str
    period: float                    # Período orbital en días
    radius: float                    # Radio planetario en radios terrestres
    temperature: float               # Temperatura de equilibrio en Kelvin
    star_radius: float              # Radio estelar en radios solares
    star_mass: float                # Masa estelar en masas solares
    star_temperature: float         # Temperatura estelar en Kelvin
    transit_depth: float            # Profundidad de tránsito en ppm
    transit_duration: float         # Duración de tránsito en horas
    signal_noise_ratio: float       # Relación señal-ruido
    mission_source: str             # Misión origen (Kepler, TESS, K2)
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        """Inicializar timestamps"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para MongoDB"""
        data = asdict(self)
        # Convertir datetime a ISO string para MongoDB
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExoplanetData':
        """Crear instancia desde diccionario de MongoDB"""
        # Convertir strings ISO a datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)


@dataclass
class SatelliteData:
    """
    Modelo de datos para información de satélites
    """
    satellite_id: str
    name: str
    mission: str                    # Kepler, TESS, K2, etc.
    launch_date: str
    status: str                     # Active, Inactive, etc.
    observations_count: int = 0
    last_observation: datetime = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        """Inicializar valores por defecto"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para MongoDB"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.last_observation:
            data['last_observation'] = self.last_observation.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SatelliteData':
        """Crear instancia desde diccionario de MongoDB"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'last_observation' in data and isinstance(data['last_observation'], str):
            data['last_observation'] = datetime.fromisoformat(data['last_observation'])
        
        return cls(**data)


@dataclass
class ProcessingResult:
    """
    Resultado de procesamiento de datos
    """
    processing_id: str
    input_data: Dict[str, Any]
    processed_data: Dict[str, Any]
    processing_time_ms: float
    status: str                     # success, error, warning
    error_message: str = None
    warnings: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        """Inicializar valores por defecto"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para MongoDB"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


class DatabaseHelper:
    """
    Clase auxiliar para operaciones de base de datos
    """
    
    @staticmethod
    def prepare_for_mongodb(data: Dict[str, Any]) -> Dict[str, Any]:
        """Preparar datos para insertar en MongoDB"""
        # Remover None values
        clean_data = {k: v for k, v in data.items() if v is not None}
        
        # Agregar timestamp si no existe
        if 'created_at' not in clean_data:
            clean_data['created_at'] = datetime.now().isoformat()
        
        return clean_data
    
    @staticmethod
    def clean_mongodb_result(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Limpiar resultado de MongoDB removiendo _id"""
        if '_id' in doc:
            doc.pop('_id')
        return doc
    
    @staticmethod
    def validate_exoplanet_data(data: Dict[str, Any]) -> bool:
        """Validación básica de datos de exoplanetas"""
        required_fields = [
            'object_name', 'period', 'radius', 'temperature',
            'star_radius', 'star_mass', 'star_temperature',
            'transit_depth', 'transit_duration', 'signal_noise_ratio'
        ]
        
        # Verificar campos requeridos
        for field in required_fields:
            if field not in data or data[field] is None:
                return False
        
        # Verificar tipos numéricos
        numeric_fields = [
            'period', 'radius', 'temperature', 'star_radius',
            'star_mass', 'star_temperature', 'transit_depth',
            'transit_duration', 'signal_noise_ratio'
        ]
        
        for field in numeric_fields:
            try:
                float(data[field])
            except (ValueError, TypeError):
                return False
        
        return True