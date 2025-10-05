"""
Utility functions for Exoplanet Detection Backend

Collection of helper functions for common operations:
- Data validation
- File operations
- Mathematical calculations
- String formatting
- Type conversions
"""

import os
import hashlib
import uuid
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timezone
from pathlib import Path
import re

# Handle imports with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


def generate_request_id() -> str:
    """
    Generate a unique request ID for tracking
    
    Returns:
        Unique request identifier string
    """
    return str(uuid.uuid4())


def generate_file_hash(content: bytes) -> str:
    """
    Generate SHA-256 hash of file content
    
    Args:
        content: File content as bytes
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(content).hexdigest()


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float with fallback
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    try:
        if value is None:
            return default
        
        if isinstance(value, (int, float)):
            if NUMPY_AVAILABLE and np and (np.isnan(value) or np.isinf(value)):
                return default
            return float(value)
        
        if isinstance(value, str):
            # Clean string (remove whitespace, handle common formatting)
            cleaned = value.strip().replace(',', '')
            if not cleaned:
                return default
            
            # Handle scientific notation
            try:
                return float(cleaned)
            except ValueError:
                return default
        
        return default
        
    except Exception:
        return default


def safe_int_conversion(value: Any, default: int = 0) -> int:
    """
    Safely convert value to integer with fallback
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Integer value or default
    """
    try:
        if value is None:
            return default
        
        if isinstance(value, bool):
            return int(value)
        
        if isinstance(value, (int, float)):
            if NUMPY_AVAILABLE and np and (np.isnan(value) or np.isinf(value)):
                return default
            return int(value)
        
        if isinstance(value, str):
            cleaned = value.strip().replace(',', '')
            if not cleaned:
                return default
            
            try:
                return int(float(cleaned))  # Handle decimal strings
            except ValueError:
                return default
        
        return default
        
    except Exception:
        return default


def clean_string(value: Any, max_length: Optional[int] = None) -> str:
    """
    Clean and sanitize string value
    
    Args:
        value: Value to clean
        max_length: Maximum length (optional)
        
    Returns:
        Cleaned string
    """
    try:
        if value is None:
            return ""
        
        # Convert to string
        if not isinstance(value, str):
            value = str(value)
        
        # Basic cleaning
        cleaned = value.strip()
        
        # Remove control characters
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
        
        # Limit length if specified
        if max_length and len(cleaned) > max_length:
            cleaned = cleaned[:max_length].strip()
        
        return cleaned
        
    except Exception:
        return ""


def validate_astronomical_range(
    value: float, 
    param_name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate if value is within astronomical reasonable range
    
    Args:
        value: Value to validate
        param_name: Parameter name for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check for NaN or infinite values
        if NUMPY_AVAILABLE and np:
            if np.isnan(value) or np.isinf(value):
                return False, f"{param_name} contains invalid value (NaN or infinity)"
        
        # Check range
        if min_val is not None and value < min_val:
            return False, f"{param_name} value {value} below minimum {min_val}"
        
        if max_val is not None and value > max_val:
            return False, f"{param_name} value {value} above maximum {max_val}"
        
        return True, None
        
    except Exception as e:
        return False, f"Validation error for {param_name}: {str(e)}"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    try:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    except Exception:
        return f"{size_bytes} B"


def format_duration(duration_seconds: float) -> str:
    """
    Format duration in human-readable format
    
    Args:
        duration_seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    try:
        if duration_seconds < 1:
            return f"{duration_seconds * 1000:.1f} ms"
        elif duration_seconds < 60:
            return f"{duration_seconds:.2f} s"
        elif duration_seconds < 3600:
            minutes = int(duration_seconds // 60)
            seconds = duration_seconds % 60
            return f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    except Exception:
        return f"{duration_seconds:.2f} s"


def get_timestamp() -> str:
    """
    Get current timestamp in ISO format
    
    Returns:
        ISO formatted timestamp string
    """
    return datetime.now(timezone.utc).isoformat()


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def safe_dict_get(data: Dict[str, Any], key: str, default: Any = None, type_cast: Optional[type] = None) -> Any:
    """
    Safely get value from dictionary with type casting
    
    Args:
        data: Dictionary to get value from
        key: Key to look for
        default: Default value if key not found
        type_cast: Type to cast value to
        
    Returns:
        Value with optional type casting
    """
    try:
        value = data.get(key, default)
        
        if value is None:
            return default
        
        if type_cast is not None:
            if type_cast == float:
                return safe_float_conversion(value, default)
            elif type_cast == int:
                return safe_int_conversion(value, default)
            elif type_cast == str:
                return clean_string(value)
            else:
                return type_cast(value)
        
        return value
        
    except Exception:
        return default


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries, later ones override earlier ones
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


def filter_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter out None values from dictionary
    
    Args:
        data: Dictionary to filter
        
    Returns:
        Dictionary without None values
    """
    return {k: v for k, v in data.items() if v is not None}


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with statistics
    """
    try:
        if not values:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        # Filter out None values and convert to float
        clean_values = [safe_float_conversion(v) for v in values if v is not None]
        
        if not clean_values:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        count = len(clean_values)
        mean_val = sum(clean_values) / count
        min_val = min(clean_values)
        max_val = max(clean_values)
        
        # Calculate median
        sorted_values = sorted(clean_values)
        if count % 2 == 0:
            median_val = (sorted_values[count // 2 - 1] + sorted_values[count // 2]) / 2
        else:
            median_val = sorted_values[count // 2]
        
        # Calculate standard deviation
        if count > 1:
            variance = sum((x - mean_val) ** 2 for x in clean_values) / (count - 1)
            std_val = variance ** 0.5
        else:
            std_val = 0.0
        
        return {
            'count': count,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'min': min_val,
            'max': max_val
        }
        
    except Exception as e:
        return {
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'error': str(e)
        }


def chunk_list(data: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size
    
    Args:
        data: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    return chunks


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Validate file extension
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions (with or without dots)
        
    Returns:
        True if extension is allowed
    """
    try:
        file_path = Path(filename)
        file_ext = file_path.suffix.lower()
        
        # Normalize extensions (ensure they start with dot)
        normalized_extensions = []
        for ext in allowed_extensions:
            if not ext.startswith('.'):
                ext = '.' + ext
            normalized_extensions.append(ext.lower())
        
        return file_ext in normalized_extensions
        
    except Exception:
        return False


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename
    """
    try:
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
        
        # Limit length
        if len(sanitized) > max_length:
            name_part = Path(sanitized).stem
            ext_part = Path(sanitized).suffix
            
            # Keep extension, truncate name
            max_name_length = max_length - len(ext_part)
            if max_name_length > 0:
                sanitized = name_part[:max_name_length] + ext_part
            else:
                sanitized = sanitized[:max_length]
        
        return sanitized.strip()
        
    except Exception:
        return "unknown_file"


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information
    
    Returns:
        Dictionary with memory usage in MB
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Physical memory
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual memory
            'percent': process.memory_percent()
        }
        
    except ImportError:
        # psutil not available
        return {
            'rss_mb': 0.0,
            'vms_mb': 0.0,
            'percent': 0.0,
            'error': 'psutil not available'
        }
    except Exception as e:
        return {
            'rss_mb': 0.0,
            'vms_mb': 0.0,
            'percent': 0.0,
            'error': str(e)
        }


class Timer:
    """
    Context manager for timing operations
    
    Usage:
        with Timer() as timer:
            # do something
        print(f"Operation took {timer.elapsed_ms:.2f} ms")
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        return self.elapsed_seconds * 1000
    
    @property
    def elapsed_formatted(self) -> str:
        """Get formatted elapsed time"""
        return format_duration(self.elapsed_seconds)


def create_error_response(
    error_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    error_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        error_type: Type of error
        message: Error message
        details: Additional error details
        error_code: Machine-readable error code
        
    Returns:
        Standardized error dictionary
    """
    response = {
        'error': error_type,
        'message': message,
        'timestamp': get_timestamp()
    }
    
    if error_code:
        response['error_code'] = error_code
    
    if details:
        response['details'] = details
    
    return response