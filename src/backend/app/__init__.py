"""
Main application initialization for Exoplanet Detection Backend

Modular backend architecture with clean separation of concerns.
This module provides the main app factory and configuration setup.
"""

from .core import (
    ExoplanetBackendError,
    ValidationError,
    DataProcessingError,
    FeatureEngineeringError,
    ModelError,
    ConfigurationError,
    FileProcessingError,
    AstronomicalDataError,
    ServiceUnavailableError,
    get_exception_status_code
)

from .services import (
    DataProcessor,
    FileProcessor,
    LoggingService
)

from .utils import (
    generate_request_id,
    get_config,
    init_config,
    get_setting,
    Timer,
    create_error_response
)

try:
    from .models import (
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
except ImportError:
    # Graceful fallback if pydantic is not available
    DispositionType = None
    PlanetSize = None
    StellarType = None
    ExoplanetFeatures = None
    EnhancedFeatures = None
    PredictionResult = None
    BatchResult = None
    ModelMetrics = None
    HealthStatus = None
    ErrorResponse = None

__version__ = "1.0.0"
__title__ = "Exoplanet Detection Backend"
__description__ = "NASA Space Apps Challenge 2025 - Modular backend for exoplanet detection"

# Export commonly used components
__all__ = [
    # Core exceptions
    "ExoplanetBackendError",
    "ValidationError",
    "DataProcessingError", 
    "FeatureEngineeringError",
    "ModelError",
    "ConfigurationError",
    "FileProcessingError",
    "AstronomicalDataError",
    "ServiceUnavailableError",
    "get_exception_status_code",
    
    # Services
    "DataProcessor",
    "FileProcessor", 
    "LoggingService",
    
    # Utilities
    "generate_request_id",
    "get_config",
    "init_config",
    "get_setting",
    "Timer",
    "create_error_response",
    
    # Models (if available)
    "DispositionType",
    "PlanetSize",
    "StellarType",
    "ExoplanetFeatures",
    "EnhancedFeatures",
    "PredictionResult",
    "BatchResult",
    "ModelMetrics",
    "HealthStatus",
    "ErrorResponse",
    
    # Metadata
    "__version__",
    "__title__",
    "__description__"
]