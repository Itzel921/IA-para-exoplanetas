"""
Configuration utilities for Exoplanet Detection Backend

Utilities for loading and managing configuration from various sources:
- Environment variables
- Configuration files
- Default settings
"""

import os
import json
from typing import Any, Dict, Optional, Union
from pathlib import Path


class ConfigurationManager:
    """
    Centralized configuration management
    
    Handles loading configuration from multiple sources with priority:
    1. Environment variables (highest priority)
    2. Configuration files
    3. Default values (lowest priority)
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = Path(config_file) if config_file else None
        self._config = {}
        self._defaults = self._get_default_config()
        self._load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            # Application settings
            'app': {
                'name': 'Exoplanet Detection Backend',
                'version': '1.0.0',
                'debug': False,
                'environment': 'development'
            },
            
            # Server settings
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'reload': True,
                'workers': 1
            },
            
            # Logging settings
            'logging': {
                'level': 'INFO',
                'max_file_size': 10 * 1024 * 1024,  # 10MB
                'backup_count': 5,
                'enable_console': True,
                'enable_structured': False
            },
            
            # File processing settings
            'file_processing': {
                'max_file_size': 50 * 1024 * 1024,  # 50MB
                'allowed_extensions': ['.csv', '.txt'],
                'max_batch_size': 10000
            },
            
            # Data validation settings
            'validation': {
                'ranges': {
                    'period': [0.1, 5000.0],
                    'radius': [0.1, 50.0], 
                    'temp': [100.0, 10000.0],
                    'star_radius': [0.1, 10.0],
                    'star_mass': [0.1, 10.0],
                    'star_temp': [2000.0, 50000.0],
                    'depth': [0.0, 1000000.0],
                    'duration': [0.1, 100.0],
                    'snr': [0.0, 1000.0]
                }
            },
            
            # Model settings
            'model': {
                'ensemble_type': 'stacking',
                'target_accuracy': 0.83,
                'enable_feature_engineering': True,
                'feature_scaling_method': 'robust'
            },
            
            # Security settings
            'security': {
                'cors_origins': ['*'],
                'api_rate_limit': '100/minute'
            },
            
            # Paths
            'paths': {
                'models': 'models',
                'data': 'data',
                'logs': 'logs',
                'temp': 'temp'
            }
        }
    
    def _load_config(self):
        """Load configuration from all sources"""
        # Start with defaults
        self._config = self._defaults.copy()
        
        # Load from file if specified
        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    self._deep_update(self._config, file_config)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
        
        # Override with environment variables
        self._load_from_environment()
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Deep update of nested dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            # App settings
            'DEBUG': ('app', 'debug', bool),
            'ENVIRONMENT': ('app', 'environment', str),
            
            # Server settings
            'HOST': ('server', 'host', str),
            'PORT': ('server', 'port', int),
            'RELOAD': ('server', 'reload', bool),
            'WORKERS': ('server', 'workers', int),
            
            # Logging settings
            'LOG_LEVEL': ('logging', 'level', str),
            'LOG_MAX_SIZE': ('logging', 'max_file_size', int),
            'LOG_BACKUP_COUNT': ('logging', 'backup_count', int),
            
            # File processing
            'MAX_FILE_SIZE': ('file_processing', 'max_file_size', int),
            'MAX_BATCH_SIZE': ('file_processing', 'max_batch_size', int),
            
            # Model settings
            'ENABLE_FEATURE_ENGINEERING': ('model', 'enable_feature_engineering', bool),
            'FEATURE_SCALING_METHOD': ('model', 'feature_scaling_method', str),
            
            # Paths
            'MODELS_DIR': ('paths', 'models', str),
            'DATA_DIR': ('paths', 'data', str),
            'LOGS_DIR': ('paths', 'logs', str),
        }
        
        for env_var, (section, key, var_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Convert to appropriate type
                    if var_type == bool:
                        value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif var_type == int:
                        value = int(env_value)
                    elif var_type == float:
                        value = float(env_value)
                    else:
                        value = env_value
                    
                    # Set the value
                    if section not in self._config:
                        self._config[section] = {}
                    self._config[section][key] = value
                    
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {env_var}: {env_value} ({e})")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'app.debug', 'server.port')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'app.debug')
            value: Value to set
        """
        try:
            keys = key.split('.')
            current = self._config
            
            # Navigate to the parent dictionary
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the final value
            current[keys[-1]] = value
            
        except Exception as e:
            print(f"Warning: Could not set config value {key}: {e}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section
        
        Args:
            section: Section name
            
        Returns:
            Configuration section dictionary
        """
        return self._config.get(section, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary"""
        return self._config.copy()
    
    def save_to_file(self, file_path: Union[str, Path]):
        """
        Save current configuration to file
        
        Args:
            file_path: Path to save configuration file
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Warning: Could not save config to {file_path}: {e}")
    
    def reload(self):
        """Reload configuration from all sources"""
        self._load_config()
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Validate required sections
        required_sections = ['app', 'server', 'logging', 'file_processing', 'validation']
        for section in required_sections:
            if section not in self._config:
                errors.append(f"Missing required configuration section: {section}")
        
        # Validate specific values
        try:
            port = self.get('server.port')
            if not isinstance(port, int) or port < 1 or port > 65535:
                errors.append("Invalid server port (must be 1-65535)")
        except Exception:
            errors.append("Invalid server port configuration")
        
        try:
            log_level = self.get('logging.level', '').upper()
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if log_level not in valid_levels:
                errors.append(f"Invalid log level (must be one of: {valid_levels})")
        except Exception:
            errors.append("Invalid logging level configuration")
        
        # Validate file size limits
        try:
            max_file_size = self.get('file_processing.max_file_size')
            if not isinstance(max_file_size, int) or max_file_size < 0:
                errors.append("Invalid max file size (must be positive integer)")
        except Exception:
            errors.append("Invalid file processing configuration")
        
        return len(errors) == 0, errors


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config() -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def init_config(config_file: Optional[Union[str, Path]] = None) -> ConfigurationManager:
    """
    Initialize global configuration manager
    
    Args:
        config_file: Path to configuration file (optional)
        
    Returns:
        Configuration manager instance
    """
    global _config_manager
    _config_manager = ConfigurationManager(config_file)
    return _config_manager


def get_setting(key: str, default: Any = None) -> Any:
    """
    Get configuration setting using dot notation
    
    Args:
        key: Setting key (e.g., 'app.debug')
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    return get_config().get(key, default)


def create_sample_config_file(file_path: Union[str, Path]):
    """
    Create a sample configuration file
    
    Args:
        file_path: Path where to create the sample config
    """
    config_manager = ConfigurationManager()
    config_manager.save_to_file(file_path)