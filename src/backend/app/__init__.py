"""
Main application initialization for Exoplanet Detection Backend

Modular backend architecture with clean separation of concerns.
This module provides the main app factory and configuration setup.
"""

# Database imports
from .database import (
    MongoDBConnection,
    ExoplanetData,
    SatelliteData,
    ProcessingResult,
    DatabaseHelper,
    ExoplanetService,
    SatelliteService,
    ProcessingService,
    DataManagerService
)

# Core exceptions (if they exist)
try:
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
except ImportError:
    # Graceful fallback if core module is not available
    ExoplanetBackendError = Exception
    ValidationError = ValueError
    DataProcessingError = Exception
    FeatureEngineeringError = Exception
    ModelError = Exception
    ConfigurationError = Exception
    FileProcessingError = Exception
    AstronomicalDataError = Exception
    ServiceUnavailableError = Exception
    get_exception_status_code = lambda x: 500

# Services (if they exist)
try:
    from .services import (
        DataProcessor,
        FileProcessor,
        LoggingService
    )
except ImportError:
    DataProcessor = None
    FileProcessor = None
    LoggingService = None

# Utils (if they exist)
try:
    from .utils import (
        generate_request_id,
        get_config,
        init_config,
        get_setting,
        Timer,
        create_error_response
    )
except ImportError:
    generate_request_id = lambda: "default_id"
    get_config = lambda: {}
    init_config = lambda: None
    get_setting = lambda x, default=None: default
    Timer = None
    create_error_response = lambda x: {"error": x}

__version__ = "1.0.0"
__title__ = "Exoplanet Detection Backend"
__description__ = "NASA Space Apps Challenge 2025 - MongoDB backend for exoplanet data processing"

# Export commonly used components
__all__ = [
    # Database components
    "MongoDBConnection",
    "ExoplanetData",
    "SatelliteData", 
    "ProcessingResult",
    "DatabaseHelper",
    "ExoplanetService",
    "SatelliteService",
    "ProcessingService",
    "DataManagerService",
    
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
    
    # Services (if available)
    "DataProcessor",
    "FileProcessor", 
    "LoggingService",
    
    # Utilities (if available)
    "generate_request_id",
    "get_config",
    "init_config",
    "get_setting",
    "Timer",
    "create_error_response",
    
    # Metadata
    "__version__",
    "__title__",
    "__description__"
]