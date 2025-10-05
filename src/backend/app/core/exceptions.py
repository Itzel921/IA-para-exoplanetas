"""
Custom exceptions for Exoplanet Detection Backend

Centralized exception handling for better error management
and debugging throughout the backend system.
"""

from typing import Optional, Dict, Any


class ExoplanetBackendError(Exception):
    """Base exception for all backend errors"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class ValidationError(ExoplanetBackendError):
    """Raised when data validation fails"""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected_range: Optional[tuple] = None
    ):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["provided_value"] = value
        if expected_range:
            details["expected_range"] = expected_range
            
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )


class DataProcessingError(ExoplanetBackendError):
    """Raised when data processing operations fail"""
    
    def __init__(
        self, 
        message: str, 
        operation: Optional[str] = None,
        data_info: Optional[Dict[str, Any]] = None
    ):
        details = {}
        if operation:
            details["operation"] = operation
        if data_info:
            details.update(data_info)
            
        super().__init__(
            message=message,
            error_code="DATA_PROCESSING_ERROR", 
            details=details
        )


class FeatureEngineeringError(ExoplanetBackendError):
    """Raised when feature engineering operations fail"""
    
    def __init__(
        self, 
        message: str, 
        feature_name: Optional[str] = None,
        input_data_shape: Optional[tuple] = None
    ):
        details = {}
        if feature_name:
            details["feature_name"] = feature_name
        if input_data_shape:
            details["input_data_shape"] = input_data_shape
            
        super().__init__(
            message=message,
            error_code="FEATURE_ENGINEERING_ERROR",
            details=details
        )


class ModelError(ExoplanetBackendError):
    """Raised when ML model operations fail"""
    
    def __init__(
        self, 
        message: str, 
        model_type: Optional[str] = None,
        operation: Optional[str] = None
    ):
        details = {}
        if model_type:
            details["model_type"] = model_type
        if operation:
            details["operation"] = operation
            
        super().__init__(
            message=message,
            error_code="MODEL_ERROR",
            details=details
        )


class ConfigurationError(ExoplanetBackendError):
    """Raised when configuration is invalid or missing"""
    
    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        config_file: Optional[str] = None
    ):
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_file:
            details["config_file"] = config_file
            
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details
        )


class FileProcessingError(ExoplanetBackendError):
    """Raised when file operations fail"""
    
    def __init__(
        self, 
        message: str, 
        filename: Optional[str] = None,
        file_type: Optional[str] = None,
        file_size: Optional[int] = None
    ):
        details = {}
        if filename:
            details["filename"] = filename
        if file_type:
            details["file_type"] = file_type
        if file_size:
            details["file_size"] = file_size
            
        super().__init__(
            message=message,
            error_code="FILE_PROCESSING_ERROR",
            details=details
        )


class AstronomicalDataError(ExoplanetBackendError):
    """Raised when astronomical data is invalid or inconsistent"""
    
    def __init__(
        self, 
        message: str, 
        parameter: Optional[str] = None,
        value: Optional[float] = None,
        astronomical_constraint: Optional[str] = None
    ):
        details = {}
        if parameter:
            details["parameter"] = parameter
        if value is not None:
            details["value"] = value
        if astronomical_constraint:
            details["astronomical_constraint"] = astronomical_constraint
            
        super().__init__(
            message=message,
            error_code="ASTRONOMICAL_DATA_ERROR",
            details=details
        )


class ServiceUnavailableError(ExoplanetBackendError):
    """Raised when a service is temporarily unavailable"""
    
    def __init__(
        self, 
        message: str, 
        service_name: Optional[str] = None,
        retry_after: Optional[int] = None
    ):
        details = {}
        if service_name:
            details["service_name"] = service_name
        if retry_after:
            details["retry_after"] = retry_after
            
        super().__init__(
            message=message,
            error_code="SERVICE_UNAVAILABLE",
            details=details
        )


# Exception mapping for HTTP status codes
EXCEPTION_STATUS_CODE_MAP = {
    ValidationError: 400,
    DataProcessingError: 422,
    FeatureEngineeringError: 422,
    FileProcessingError: 400,
    AstronomicalDataError: 400,
    ModelError: 500,
    ConfigurationError: 500,
    ServiceUnavailableError: 503,
    ExoplanetBackendError: 500,
}


def get_exception_status_code(exception: Exception) -> int:
    """Get appropriate HTTP status code for exception"""
    for exc_type, status_code in EXCEPTION_STATUS_CODE_MAP.items():
        if isinstance(exception, exc_type):
            return status_code
    return 500  # Default to internal server error