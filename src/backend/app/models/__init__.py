"""
Models module initialization
"""

try:
    from .schemas import (
        DispositionType,
        PlanetSize,
        StellarType,
        ExoplanetFeatures,
        EnhancedFeatures,
        PredictionResult,
        BatchResult,
        ModelMetrics,
        HealthStatus,
        ErrorResponse
    )
    
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
except ImportError:
    # Graceful fallback if pydantic is not available
    __all__ = []