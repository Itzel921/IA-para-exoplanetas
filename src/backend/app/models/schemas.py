"""
Pydantic models for Exoplanet Detection Backend

Data models for request/response validation and internal data structures.
Based on NASA Exoplanet Archive standards and research requirements.
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from enum import Enum
try:
    from pydantic import BaseModel, Field, validator, root_validator
except ImportError:
    # Fallback for development without pydantic
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def Field(*args, **kwargs):
        return None
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def root_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class DispositionType(str, Enum):
    """Exoplanet disposition classifications"""
    CONFIRMED = "CONFIRMED"
    CANDIDATE = "CANDIDATE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    
    # TESS specific dispositions
    KP = "KP"  # Known Planet
    PC = "PC"  # Planet Candidate
    FP = "FP"  # False Positive
    APC = "APC"  # Ambiguous Planet Candidate


class PlanetSize(str, Enum):
    """Planet size classifications based on radius"""
    SUB_EARTH = "Sub-Earth"
    EARTH_SIZE = "Earth-size"
    SUPER_EARTH = "Super-Earth"
    SUB_NEPTUNE = "Sub-Neptune"
    NEPTUNE_SIZE = "Neptune-size"
    JUPITER_SIZE = "Jupiter-size"


class StellarType(str, Enum):
    """Stellar classification by temperature"""
    M_DWARF = "M dwarf"
    K_DWARF = "K dwarf"
    G_DWARF = "G dwarf"
    F_DWARF = "F dwarf"
    A_TYPE = "A dwarf or hotter"


class ExoplanetFeatures(BaseModel):
    """
    Input features for exoplanet detection
    All parameters are required for prediction
    """
    
    # Orbital parameters
    period: float = Field(
        ...,
        description="Orbital period in days",
        gt=0.1,
        lt=5000.0
    )
    
    # Planetary parameters
    radius: float = Field(
        ...,
        description="Planetary radius in Earth radii",
        gt=0.1,
        lt=50.0
    )
    
    temp: float = Field(
        ...,
        description="Equilibrium temperature in Kelvin",
        gt=100.0,
        lt=10000.0
    )
    
    # Stellar parameters
    starRadius: float = Field(
        ...,
        description="Stellar radius in solar radii",
        gt=0.1,
        lt=10.0,
        alias="star_radius"
    )
    
    starMass: float = Field(
        ...,
        description="Stellar mass in solar masses",
        gt=0.1,
        lt=10.0,
        alias="star_mass"
    )
    
    starTemp: float = Field(
        ...,
        description="Stellar temperature in Kelvin",
        gt=2000.0,
        lt=50000.0,
        alias="star_temp"
    )
    
    # Transit parameters
    depth: float = Field(
        ...,
        description="Transit depth in parts per million (ppm)",
        gt=0.0
    )
    
    duration: float = Field(
        ...,
        description="Transit duration in hours",
        gt=0.1,
        lt=100.0
    )
    
    snr: float = Field(
        ...,
        description="Signal-to-noise ratio",
        gt=0.0
    )
    
    # Optional metadata
    source_mission: Optional[str] = Field(
        None,
        description="Source mission (Kepler, TESS, K2)"
    )
    
    object_name: Optional[str] = Field(
        None,
        description="Object identifier"
    )
    
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "period": 365.25,
                "radius": 1.0,
                "temp": 288.0,
                "starRadius": 1.0,
                "starMass": 1.0,
                "starTemp": 5778.0,
                "depth": 84.0,
                "duration": 6.5,
                "snr": 15.0,
                "source_mission": "Kepler",
                "object_name": "Earth-analog"
            }
        }


class EnhancedFeatures(BaseModel):
    """
    Enhanced features after feature engineering
    Includes derived astronomical parameters
    """
    
    # Original features
    original_features: ExoplanetFeatures
    
    # Derived physical ratios
    planet_star_radius_ratio: float = Field(
        ...,
        description="Ratio of planetary to stellar radius"
    )
    
    equilibrium_temp_ratio: float = Field(
        ...,
        description="Ratio of planetary to stellar temperature"
    )
    
    # Expected vs observed
    transit_depth_expected: float = Field(
        ...,
        description="Expected transit depth from geometry (ppm)"
    )
    
    # Orbital characteristics
    orbital_velocity: float = Field(
        ...,
        description="Approximate orbital velocity"
    )
    
    # Habitability metrics
    hz_distance: float = Field(
        ...,
        description="Distance from habitable zone center (scaled)"
    )
    
    # Signal quality
    depth_snr_ratio: float = Field(
        ...,
        description="Ratio of transit depth to SNR"
    )
    
    duration_period_ratio: float = Field(
        ...,
        description="Fraction of orbit spent in transit"
    )
    
    # Classifications
    planet_type: PlanetSize = Field(
        ...,
        description="Planet size classification"
    )
    
    stellar_type: StellarType = Field(
        ...,
        description="Stellar type classification"
    )
    
    is_habitable_zone: bool = Field(
        ...,
        description="Whether planet is in conservative habitable zone"
    )


class PredictionResult(BaseModel):
    """Single prediction result"""
    
    prediction: DispositionType = Field(
        ...,
        description="Primary prediction result"
    )
    
    confidence: float = Field(
        ...,
        description="Model confidence (0-1)",
        ge=0.0,
        le=1.0
    )
    
    probabilities: Dict[str, float] = Field(
        ...,
        description="Class probabilities"
    )
    
    feature_importance: Optional[Dict[str, float]] = Field(
        None,
        description="Feature importance scores"
    )
    
    enhanced_features: Optional[EnhancedFeatures] = Field(
        None,
        description="Enhanced features used for prediction"
    )
    
    analysis_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of analysis"
    )
    
    processing_time_ms: Optional[float] = Field(
        None,
        description="Processing time in milliseconds"
    )


class BatchResult(BaseModel):
    """Batch processing result"""
    
    total_processed: int = Field(
        ...,
        description="Total number of objects processed",
        ge=0
    )
    
    confirmed_planets: int = Field(
        ...,
        description="Number classified as confirmed planets",
        ge=0
    )
    
    candidates: int = Field(
        ...,
        description="Number classified as candidates",
        ge=0
    )
    
    false_positives: int = Field(
        ...,
        description="Number classified as false positives",
        ge=0
    )
    
    results: List[PredictionResult] = Field(
        ...,
        description="Individual prediction results"
    )
    
    processing_time_seconds: float = Field(
        ...,
        description="Total processing time in seconds"
    )
    
    analysis_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of batch analysis"
    )
    
    file_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Information about processed file"
    )
    
    @validator('total_processed')
    def validate_total(cls, v, values):
        """Validate that total matches sum of classifications"""
        confirmed = values.get('confirmed_planets', 0)
        candidates = values.get('candidates', 0)
        false_positives = values.get('false_positives', 0)
        
        if v != confirmed + candidates + false_positives:
            raise ValueError(
                f"Total processed ({v}) must equal sum of classifications "
                f"({confirmed + candidates + false_positives})"
            )
        return v


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    
    model_type: str = Field(
        ...,
        description="Type of ML model"
    )
    
    base_models: List[str] = Field(
        ...,
        description="List of base models in ensemble"
    )
    
    training_accuracy: float = Field(
        ...,
        description="Training accuracy score",
        ge=0.0,
        le=1.0
    )
    
    validation_accuracy: Optional[float] = Field(
        None,
        description="Validation accuracy score",
        ge=0.0,
        le=1.0
    )
    
    precision: Optional[float] = Field(
        None,
        description="Precision score",
        ge=0.0,
        le=1.0
    )
    
    recall: Optional[float] = Field(
        None,
        description="Recall score",
        ge=0.0,
        le=1.0
    )
    
    f1_score: Optional[float] = Field(
        None,
        description="F1 score",
        ge=0.0,
        le=1.0
    )
    
    roc_auc: Optional[float] = Field(
        None,
        description="ROC AUC score",
        ge=0.0,
        le=1.0
    )
    
    features_used: int = Field(
        ...,
        description="Number of features used by model",
        gt=0
    )
    
    target_accuracy: float = Field(
        0.83,
        description="Target accuracy from research",
        ge=0.0,
        le=1.0
    )
    
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="When model was last updated"
    )


class HealthStatus(BaseModel):
    """System health status"""
    
    status: str = Field(
        ...,
        description="Overall system status"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Health check timestamp"
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether ML model is loaded"
    )
    
    api_version: str = Field(
        ...,
        description="API version"
    )
    
    uptime_seconds: Optional[float] = Field(
        None,
        description="System uptime in seconds"
    )
    
    memory_usage_mb: Optional[float] = Field(
        None,
        description="Current memory usage in MB"
    )
    
    disk_space_gb: Optional[float] = Field(
        None,
        description="Available disk space in GB"
    )


class ErrorResponse(BaseModel):
    """Standardized error response"""
    
    error: str = Field(
        ...,
        description="Error type"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    error_code: Optional[str] = Field(
        None,
        description="Machine-readable error code"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Request identifier for debugging"
    )


# Export all models
__all__ = [
    "DispositionType",
    "PlanetSize", 
    "StellarType",
    "ExoplanetFeatures",
    "EnhancedFeatures",
    "PredictionResult",
    "BatchResult", 
    "ModelMetrics",
    "HealthStatus",
    "ErrorResponse"
]