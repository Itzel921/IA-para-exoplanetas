"""
Core module initialization for Exoplanet Detection Backend
"""

from .exceptions import (
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

__all__ = [
    "ExoplanetBackendError",
    "ValidationError", 
    "DataProcessingError",
    "FeatureEngineeringError",
    "ModelError",
    "ConfigurationError",
    "FileProcessingError",
    "AstronomicalDataError",
    "ServiceUnavailableError",
    "get_exception_status_code"
]