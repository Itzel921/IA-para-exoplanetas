"""
Core configuration for Exoplanet Detection Backend
NASA Space Apps Challenge 2025

Centralized configuration management for all backend components
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with environment variable support
    All settings can be overridden via environment variables
    """
    
    # Application Info
    app_name: str = "Exoplanet Detection Backend"
    app_version: str = "1.0.0"
    debug: bool = Field(False, env="DEBUG")
    environment: str = Field("development", env="ENVIRONMENT")
    
    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    reload: bool = Field(True, env="RELOAD")
    
    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent.parent
    models_dir: Path = base_dir / "models"
    data_dir: Path = base_dir / "data"
    logs_dir: Path = base_dir / "src" / "backend" / "logs"
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = "exoplanet_backend.log"
    log_max_size: int = Field(10485760, env="LOG_MAX_SIZE")  # 10MB
    log_backup_count: int = Field(5, env="LOG_BACKUP_COUNT")
    
    # ML Model Configuration
    model_path: Optional[str] = Field(None, env="MODEL_PATH")
    preprocessor_path: Optional[str] = Field(None, env="PREPROCESSOR_PATH")
    feature_names_path: Optional[str] = Field(None, env="FEATURE_NAMES_PATH")
    
    # Data Processing
    max_file_size: int = Field(50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    allowed_file_types: List[str] = ["csv", "txt"]
    max_batch_size: int = Field(10000, env="MAX_BATCH_SIZE")
    
    # Feature Engineering Parameters
    enable_feature_engineering: bool = Field(True, env="ENABLE_FEATURE_ENGINEERING")
    feature_scaling_method: str = Field("robust", env="FEATURE_SCALING_METHOD")
    outlier_detection_method: str = Field("iqr", env="OUTLIER_DETECTION_METHOD")
    
    # Astronomical Constants
    solar_temperature: float = 5778.0  # Kelvin
    earth_radius: float = 1.0  # Earth radii
    solar_radius: float = 1.0  # Solar radii
    solar_mass: float = 1.0  # Solar masses
    
    # Data Validation Ranges
    min_period: float = 0.1  # days
    max_period: float = 5000.0  # days
    min_radius: float = 0.1  # Earth radii
    max_radius: float = 50.0  # Earth radii
    min_temperature: float = 100.0  # Kelvin
    max_temperature: float = 10000.0  # Kelvin
    min_stellar_mass: float = 0.1  # Solar masses
    max_stellar_mass: float = 10.0  # Solar masses
    
    # Performance Configuration
    enable_caching: bool = Field(True, env="ENABLE_CACHING")
    cache_ttl: int = Field(3600, env="CACHE_TTL")  # seconds
    enable_async_processing: bool = Field(True, env="ENABLE_ASYNC_PROCESSING")
    
    # Security
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")
    api_rate_limit: str = Field("100/minute", env="API_RATE_LIMIT")
    
    # Database (optional for future use)
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.create_directories()
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        directories = [
            self.models_dir,
            self.data_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"
    
    def get_model_paths(self) -> dict:
        """Get full paths for model files"""
        return {
            "model": self.models_dir / "best_ensemble_model.pkl",
            "preprocessor": self.models_dir / "preprocessor.pkl", 
            "feature_names": self.models_dir / "feature_names.pkl"
        }


# Global settings instance
settings = Settings()


class ExoplanetConstants:
    """
    Astronomical and physical constants for exoplanet detection
    Based on NASA Exoplanet Archive standards
    """
    
    # Physical Constants
    SOLAR_RADIUS_KM = 695700  # km
    EARTH_RADIUS_KM = 6371  # km
    SOLAR_MASS_KG = 1.989e30  # kg
    EARTH_MASS_KG = 5.972e24  # kg
    
    # Transit Detection Thresholds
    MIN_TRANSIT_DEPTH = 1e-6  # Minimum detectable transit depth
    MAX_TRANSIT_DEPTH = 0.1   # Maximum reasonable transit depth
    MIN_TRANSIT_DURATION = 0.1  # hours
    MAX_TRANSIT_DURATION = 100.0  # hours
    
    # Signal Quality Thresholds
    MIN_SNR = 3.0  # Minimum signal-to-noise ratio
    DETECTION_THRESHOLD_SNR = 7.5  # NASA standard for detection
    
    # Habitability Zone Boundaries (in AU)
    HZ_INNER_CONSERVATIVE = 0.95
    HZ_OUTER_CONSERVATIVE = 1.37
    HZ_INNER_OPTIMISTIC = 0.75
    HZ_OUTER_OPTIMISTIC = 1.77
    
    # Planet Classification Boundaries
    EARTH_SIZE_MIN = 0.8  # Earth radii
    EARTH_SIZE_MAX = 1.25  # Earth radii
    SUPER_EARTH_MAX = 2.0  # Earth radii
    NEPTUNE_MIN = 2.0  # Earth radii
    JUPITER_MIN = 10.0  # Earth radii
    
    # Stellar Classification
    M_DWARF_TEMP_MAX = 3800  # Kelvin
    K_DWARF_TEMP_MAX = 5200  # Kelvin
    G_DWARF_TEMP_MAX = 6000  # Kelvin
    F_DWARF_TEMP_MAX = 7500  # Kelvin
    
    @classmethod
    def get_planet_type(cls, radius: float) -> str:
        """Classify planet by radius"""
        if radius < cls.EARTH_SIZE_MIN:
            return "Sub-Earth"
        elif radius <= cls.EARTH_SIZE_MAX:
            return "Earth-size"
        elif radius <= cls.SUPER_EARTH_MAX:
            return "Super-Earth"
        elif radius <= cls.NEPTUNE_MIN:
            return "Sub-Neptune"
        elif radius < cls.JUPITER_MIN:
            return "Neptune-size"
        else:
            return "Jupiter-size"
    
    @classmethod
    def get_stellar_type(cls, temperature: float) -> str:
        """Classify star by temperature"""
        if temperature <= cls.M_DWARF_TEMP_MAX:
            return "M dwarf"
        elif temperature <= cls.K_DWARF_TEMP_MAX:
            return "K dwarf"
        elif temperature <= cls.G_DWARF_TEMP_MAX:
            return "G dwarf"
        elif temperature <= cls.F_DWARF_TEMP_MAX:
            return "F dwarf"
        else:
            return "A dwarf or hotter"
    
    @classmethod
    def is_in_habitable_zone(cls, distance_au: float, stellar_temp: float) -> bool:
        """Check if planet is in conservative habitable zone"""
        # Scaled habitable zone based on stellar temperature
        temp_factor = (stellar_temp / settings.solar_temperature) ** 0.5
        hz_inner = cls.HZ_INNER_CONSERVATIVE * temp_factor
        hz_outer = cls.HZ_OUTER_CONSERVATIVE * temp_factor
        
        return hz_inner <= distance_au <= hz_outer


# Export commonly used objects
__all__ = ["settings", "Settings", "ExoplanetConstants"]