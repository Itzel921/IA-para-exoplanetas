"""
MongoDB Document Models for Exoplanet Detection System
NASA Space Apps Challenge 2025

Pydantic models with Beanie ODM for async MongoDB operations
Following the data structure from implementation-methodology.md
"""

from beanie import Document
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

class PredictionStatus(str, Enum):
    """Prediction classification results"""
    CONFIRMED = "CONFIRMED"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    CANDIDATE = "CANDIDATE"

class ExoplanetFeatures(BaseModel):
    """Input features for exoplanet prediction"""
    # Planetary parameters
    period: float = Field(..., description="Orbital period (days)")
    radius: float = Field(..., description="Planetary radius (Earth radii)")
    temp: float = Field(..., description="Equilibrium temperature (K)")
    
    # Stellar parameters  
    star_radius: float = Field(..., description="Stellar radius (Solar radii)")
    star_mass: float = Field(..., description="Stellar mass (Solar masses)")
    star_temp: float = Field(..., description="Stellar temperature (K)")
    
    # Transit metrics
    depth: float = Field(..., description="Transit depth (ppm)")
    duration: float = Field(..., description="Transit duration (hours)")
    snr: float = Field(..., description="Signal-to-noise ratio")
    
    # Derived features (calculated during feature engineering)
    planet_star_radius_ratio: Optional[float] = None
    equilibrium_temp_ratio: Optional[float] = None
    habitable_zone_distance: Optional[float] = None

class PredictionResult(Document):
    """Individual prediction result stored in MongoDB"""
    
    # Input data
    features: ExoplanetFeatures
    
    # Prediction results
    prediction: PredictionStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    
    # Model information
    model_version: str = "stacking_ensemble_v1.0"
    accuracy_achieved: float = 0.8308  # Target from ensemble-algorithms.md
    
    # Feature importance (if available)
    feature_importance: Optional[Dict[str, float]] = None
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = None
    user_session: Optional[str] = None
    
    class Settings:
        name = "prediction_results"
        indexes = [
            "timestamp",
            "prediction", 
            "confidence",
            "user_session"
        ]

class BatchAnalysis(Document):
    """Batch processing analysis results"""
    
    # Analysis metadata
    batch_id: str = Field(..., unique=True)
    filename: str
    total_objects: int
    
    # Results summary
    confirmed_planets: int = 0
    candidates: int = 0
    false_positives: int = 0
    
    # Processing information
    processing_start: datetime
    processing_end: Optional[datetime] = None
    status: str = "processing"  # processing, completed, failed
    
    # Detailed results
    results: List[Dict] = []  # Individual predictions
    
    # Statistics
    average_confidence: Optional[float] = None
    high_confidence_detections: Optional[int] = None  # confidence > 0.8
    
    class Settings:
        name = "batch_analyses"
        indexes = [
            "batch_id",
            "processing_start",
            "status"
        ]

class ModelMetrics(Document):
    """Model performance metrics over time"""
    
    # Model identification
    model_name: str = "stacking_ensemble"
    model_version: str = "v1.0"
    
    # Performance metrics (from metrics-evaluation.md)
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    
    # Astronomical metrics (specific to exoplanets)
    completeness: float  # fraction of real planets detected
    reliability: float   # fraction of detections that are real planets
    false_discovery_rate: float
    
    # Evaluation metadata
    test_set_size: int
    positive_class_proportion: float
    evaluation_date: datetime = Field(default_factory=datetime.utcnow)
    
    # Configuration used
    hyperparameters: Optional[Dict] = None
    feature_count: Optional[int] = None
    
    class Settings:
        name = "model_metrics"
        indexes = [
            "model_version",
            "evaluation_date",
            "accuracy"
        ]

class UserSession(Document):
    """User session tracking for analytics"""
    
    session_id: str = Field(..., unique=True)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    
    # Usage statistics
    predictions_count: int = 0
    batch_analyses_count: int = 0
    
    # User preferences (optional)
    preferred_confidence_threshold: float = 0.8
    
    class Settings:
        name = "user_sessions"
        indexes = [
            "session_id",
            "start_time",
            "last_activity"
        ]