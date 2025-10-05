"""
Logging Service for Exoplanet Detection Backend

Centralized logging configuration and utilities for the backend system.
Provides structured logging with different levels, file rotation, and 
formatters optimized for exoplanet detection operations.
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured logging
    Outputs logs in JSON format for better parsing and analysis
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        
        # Extract exception info if present
        exc_info = None
        if record.exc_info:
            exc_info = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Build structured log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'thread_id': record.thread,
            'process_id': record.process
        }
        
        # Add exception info if present
        if exc_info:
            log_entry['exception'] = exc_info
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, default=str)


class ExoplanetLogFilter(logging.Filter):
    """
    Custom filter for exoplanet-specific logging
    Adds context and filters sensitive information
    """
    
    def __init__(self, component: Optional[str] = None):
        super().__init__()
        self.component = component
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and enhance log records"""
        
        # Add component context
        if self.component:
            record.component = self.component
        
        # Add request context if available (would be set by middleware)
        if hasattr(record, 'request_id'):
            record.request_id = getattr(record, 'request_id', 'unknown')
        
        # Filter sensitive information
        if hasattr(record, 'args') and record.args:
            # Replace any potential sensitive data in args
            record.args = self._sanitize_args(record.args)
        
        return True
    
    def _sanitize_args(self, args: tuple) -> tuple:
        """Remove sensitive information from log arguments"""
        # In a real implementation, you'd filter API keys, passwords, etc.
        return args


class LoggingService:
    """
    Centralized logging service for the exoplanet detection backend
    
    Provides:
    - Structured JSON logging
    - File rotation
    - Different log levels for different components
    - Context injection
    - Performance logging
    """
    
    def __init__(
        self, 
        log_level: str = "INFO",
        log_dir: Optional[Path] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True,
        enable_structured: bool = False
    ):
        """
        Initialize logging service
        
        Args:
            log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files (None = no file logging)
            max_file_size: Maximum size per log file in bytes
            backup_count: Number of backup files to keep
            enable_console: Whether to log to console
            enable_structured: Whether to use structured JSON logging
        """
        
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = log_dir
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_structured = enable_structured
        
        # Create log directory if specified
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Track configured loggers
        self._configured_loggers = set()
        
        # Setup main application logger
        self.setup_logger('exoplanet_backend')
    
    def setup_logger(
        self, 
        logger_name: str, 
        component: Optional[str] = None,
        log_file: Optional[str] = None
    ) -> logging.Logger:
        """
        Setup a logger with consistent configuration
        
        Args:
            logger_name: Name of the logger
            component: Component name for context
            log_file: Custom log file name (optional)
            
        Returns:
            Configured logger instance
        """
        
        logger = logging.getLogger(logger_name)
        
        # Avoid duplicate configuration
        if logger_name in self._configured_loggers:
            return logger
        
        logger.setLevel(self.log_level)
        logger.handlers.clear()  # Remove any existing handlers
        
        # Setup formatters
        if self.enable_structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s'
            )
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            
            # Add filter for component context
            if component:
                console_handler.addFilter(ExoplanetLogFilter(component))
            
            logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.log_dir:
            log_filename = log_file or f"{logger_name}.log"
            log_path = self.log_dir / log_filename
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            
            # Add filter for component context
            if component:
                file_handler.addFilter(ExoplanetLogFilter(component))
            
            logger.addHandler(file_handler)
        
        # Mark as configured
        self._configured_loggers.add(logger_name)
        
        logger.info(f"Logger '{logger_name}' configured successfully")
        return logger
    
    def get_logger(self, name: str, component: Optional[str] = None) -> logging.Logger:
        """
        Get a logger instance
        
        Args:
            name: Logger name
            component: Component name for context
            
        Returns:
            Logger instance
        """
        if name not in self._configured_loggers:
            return self.setup_logger(name, component)
        
        return logging.getLogger(name)
    
    def log_performance(
        self, 
        logger: logging.Logger, 
        operation: str, 
        duration_ms: float,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """
        Log performance metrics
        
        Args:
            logger: Logger instance
            operation: Operation name
            duration_ms: Duration in milliseconds
            additional_data: Additional metrics
        """
        
        perf_data = {
            'operation': operation,
            'duration_ms': duration_ms,
            'performance_log': True
        }
        
        if additional_data:
            perf_data.update(additional_data)
        
        logger.info(f"Performance: {operation} completed in {duration_ms:.2f}ms", extra=perf_data)
    
    def log_api_request(
        self,
        logger: logging.Logger,
        method: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        request_id: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Log API request information
        
        Args:
            logger: Logger instance
            method: HTTP method
            endpoint: API endpoint
            status_code: Response status code
            duration_ms: Request duration in milliseconds
            request_id: Unique request identifier
            user_agent: User agent string
        """
        
        request_data = {
            'api_request': True,
            'method': method,
            'endpoint': endpoint,
            'status_code': status_code,
            'duration_ms': duration_ms,
            'request_id': request_id,
            'user_agent': user_agent
        }
        
        # Filter None values
        request_data = {k: v for k, v in request_data.items() if v is not None}
        
        logger.info(f"API {method} {endpoint} - {status_code} ({duration_ms:.2f}ms)", extra=request_data)
    
    def log_model_prediction(
        self,
        logger: logging.Logger,
        prediction_type: str,
        input_features: Dict[str, Any],
        prediction_result: str,
        confidence: float,
        processing_time_ms: float
    ):
        """
        Log ML model prediction information
        
        Args:
            logger: Logger instance
            prediction_type: Type of prediction (single, batch)
            input_features: Input feature summary
            prediction_result: Prediction result
            confidence: Model confidence
            processing_time_ms: Processing time in milliseconds
        """
        
        prediction_data = {
            'model_prediction': True,
            'prediction_type': prediction_type,
            'prediction_result': prediction_result,
            'confidence': confidence,
            'processing_time_ms': processing_time_ms,
            'feature_count': len(input_features) if input_features else 0
        }
        
        logger.info(
            f"Model prediction: {prediction_result} (confidence: {confidence:.3f}, "
            f"time: {processing_time_ms:.2f}ms)",
            extra=prediction_data
        )
    
    def log_data_processing(
        self,
        logger: logging.Logger,
        operation: str,
        input_count: int,
        output_count: int,
        processing_time_ms: float,
        errors: Optional[List[str]] = None
    ):
        """
        Log data processing operations
        
        Args:
            logger: Logger instance
            operation: Processing operation name
            input_count: Number of input records
            output_count: Number of output records
            processing_time_ms: Processing time in milliseconds
            errors: List of error messages (if any)
        """
        
        processing_data = {
            'data_processing': True,
            'operation': operation,
            'input_count': input_count,
            'output_count': output_count,
            'success_rate': output_count / input_count if input_count > 0 else 0,
            'processing_time_ms': processing_time_ms,
            'error_count': len(errors) if errors else 0
        }
        
        if errors:
            processing_data['errors'] = errors
        
        logger.info(
            f"Data processing: {operation} - {output_count}/{input_count} records "
            f"({processing_time_ms:.2f}ms)",
            extra=processing_data
        )
    
    def create_request_logger(self, request_id: str) -> logging.LoggerAdapter:
        """
        Create a logger adapter with request context
        
        Args:
            request_id: Unique request identifier
            
        Returns:
            LoggerAdapter with request context
        """
        
        base_logger = self.get_logger('exoplanet_backend.requests')
        return logging.LoggerAdapter(base_logger, {'request_id': request_id})
    
    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get logging statistics
        
        Returns:
            Dictionary with logging statistics
        """
        
        stats = {
            'configured_loggers': len(self._configured_loggers),
            'logger_names': list(self._configured_loggers),
            'log_level': logging.getLevelName(self.log_level),
            'log_directory': str(self.log_dir) if self.log_dir else None,
            'console_logging': self.enable_console,
            'structured_logging': self.enable_structured,
            'max_file_size_mb': self.max_file_size / (1024 * 1024),
            'backup_count': self.backup_count
        }
        
        # Add log file information if logging to files
        if self.log_dir and self.log_dir.exists():
            log_files = []
            for log_file in self.log_dir.glob('*.log*'):
                try:
                    stat = log_file.stat()
                    log_files.append({
                        'name': log_file.name,
                        'size_bytes': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except Exception:
                    pass
            
            stats['log_files'] = log_files
        
        return stats


# Global logging service instance
_logging_service: Optional[LoggingService] = None


def get_logging_service() -> LoggingService:
    """Get global logging service instance"""
    global _logging_service
    if _logging_service is None:
        # Initialize with default settings
        _logging_service = LoggingService(
            log_level="INFO",
            log_dir=Path("logs"),
            enable_console=True,
            enable_structured=False
        )
    return _logging_service


def configure_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_console: bool = True,
    enable_structured: bool = False,
    max_file_size: int = 10 * 1024 * 1024,
    backup_count: int = 5
) -> LoggingService:
    """
    Configure global logging service
    
    Args:
        log_level: Minimum log level
        log_dir: Log directory path
        enable_console: Enable console logging
        enable_structured: Enable structured JSON logging
        max_file_size: Maximum file size in bytes
        backup_count: Number of backup files
        
    Returns:
        Configured logging service
    """
    global _logging_service
    
    log_path = Path(log_dir) if log_dir else None
    
    _logging_service = LoggingService(
        log_level=log_level,
        log_dir=log_path,
        enable_console=enable_console,
        enable_structured=enable_structured,
        max_file_size=max_file_size,
        backup_count=backup_count
    )
    
    return _logging_service


# Convenience functions
def get_logger(name: str, component: Optional[str] = None) -> logging.Logger:
    """Get a configured logger"""
    return get_logging_service().get_logger(name, component)


def log_performance(operation: str, duration_ms: float, **kwargs):
    """Log performance metrics using default logger"""
    logger = get_logger('exoplanet_backend.performance')
    get_logging_service().log_performance(logger, operation, duration_ms, kwargs)