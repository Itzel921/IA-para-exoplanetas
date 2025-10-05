"""
Data Processing Service for Exoplanet Detection Backend

Handles all data processing operations including:
- Data validation and cleaning
- Feature engineering
- Astronomical calculations
- Data format conversions

Based on NASA Exoplanet Archive standards and research requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime

# Local imports with fallback
try:
    from ..core.exceptions import (
        DataProcessingError,
        FeatureEngineeringError,
        AstronomicalDataError,
        ValidationError
    )
    from ..models.schemas import ExoplanetFeatures, EnhancedFeatures, PlanetSize, StellarType
except ImportError:
    # Fallback classes for development
    class DataProcessingError(Exception): pass
    class FeatureEngineeringError(Exception): pass
    class AstronomicalDataError(Exception): pass
    class ValidationError(Exception): pass
    
    class ExoplanetFeatures:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class EnhancedFeatures:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class PlanetSize:
        SUB_EARTH = "Sub-Earth"
        EARTH_SIZE = "Earth-size"
        SUPER_EARTH = "Super-Earth"
        SUB_NEPTUNE = "Sub-Neptune"
        NEPTUNE_SIZE = "Neptune-size"
        JUPITER_SIZE = "Jupiter-size"
    
    class StellarType:
        M_DWARF = "M dwarf"
        K_DWARF = "K dwarf"
        G_DWARF = "G dwarf"
        F_DWARF = "F dwarf"
        A_TYPE = "A dwarf or hotter"


logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Main data processing service for exoplanet detection
    
    Handles data validation, cleaning, feature engineering, and astronomical calculations
    """
    
    def __init__(self):
        """Initialize data processor with astronomical constants"""
        # Astronomical constants
        self.SOLAR_TEMP = 5778.0  # Kelvin
        self.EARTH_RADIUS = 1.0   # Earth radii
        self.SOLAR_RADIUS = 1.0   # Solar radii
        self.SOLAR_MASS = 1.0     # Solar masses
        
        # Validation ranges
        self.VALIDATION_RANGES = {
            'period': (0.1, 5000.0),      # days
            'radius': (0.1, 50.0),        # Earth radii
            'temp': (100.0, 10000.0),     # Kelvin
            'starRadius': (0.1, 10.0),    # Solar radii
            'starMass': (0.1, 10.0),      # Solar masses
            'starTemp': (2000.0, 50000.0), # Kelvin
            'depth': (0.0, 1000000.0),    # ppm
            'duration': (0.1, 100.0),     # hours
            'snr': (0.0, 1000.0)          # ratio
        }
        
        # Planet classification boundaries (Earth radii)
        self.PLANET_BOUNDARIES = {
            'earth_min': 0.8,
            'earth_max': 1.25,
            'super_earth_max': 2.0,
            'neptune_min': 2.0,
            'jupiter_min': 10.0
        }
        
        # Stellar temperature boundaries (Kelvin)
        self.STELLAR_BOUNDARIES = {
            'm_dwarf_max': 3800,
            'k_dwarf_max': 5200,
            'g_dwarf_max': 6000,
            'f_dwarf_max': 7500
        }
        
        # Habitable zone boundaries (conservative, in AU)
        self.HZ_INNER = 0.95
        self.HZ_OUTER = 1.37
    
    def validate_single_input(self, data: Dict) -> ExoplanetFeatures:
        """
        Validate single exoplanet input data
        
        Args:
            data: Dictionary with exoplanet parameters
            
        Returns:
            ExoplanetFeatures object with validated data
            
        Raises:
            ValidationError: If data is invalid
        """
        try:
            # Check required fields
            required_fields = [
                'period', 'radius', 'temp', 'starRadius', 'starMass',
                'starTemp', 'depth', 'duration', 'snr'
            ]
            
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValidationError(
                    f"Missing required fields: {missing_fields}",
                    details={"missing_fields": missing_fields}
                )
            
            # Validate ranges
            for field, (min_val, max_val) in self.VALIDATION_RANGES.items():
                if field in data:
                    value = data[field]
                    if not isinstance(value, (int, float)) or np.isnan(value):
                        raise ValidationError(
                            f"Field '{field}' must be a valid number",
                            field=field,
                            value=value
                        )
                    
                    if not (min_val <= value <= max_val):
                        raise ValidationError(
                            f"Field '{field}' value {value} outside valid range [{min_val}, {max_val}]",
                            field=field,
                            value=value,
                            expected_range=(min_val, max_val)
                        )
            
            # Validate astronomical consistency
            self._validate_astronomical_consistency(data)
            
            # Create validated object
            return ExoplanetFeatures(**data)
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            else:
                raise ValidationError(f"Validation failed: {str(e)}")
    
    def validate_batch_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate batch data from DataFrame
        
        Args:
            df: DataFrame with exoplanet data
            
        Returns:
            Cleaned and validated DataFrame
            
        Raises:
            DataProcessingError: If batch validation fails
        """
        try:
            logger.info(f"Validating batch data with {len(df)} rows")
            
            # Check required columns
            required_columns = list(self.VALIDATION_RANGES.keys())
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise DataProcessingError(
                    f"Missing required columns: {missing_columns}",
                    operation="batch_validation",
                    data_info={"missing_columns": list(missing_columns)}
                )
            
            # Remove rows with any NaN values in required columns
            initial_count = len(df)
            df_clean = df.dropna(subset=required_columns)
            dropped_count = initial_count - len(df_clean)
            
            if dropped_count > 0:
                logger.warning(f"Dropped {dropped_count} rows with missing values")
            
            # Validate ranges for each column
            validation_errors = []
            for column, (min_val, max_val) in self.VALIDATION_RANGES.items():
                if column in df_clean.columns:
                    # Check for values outside range
                    mask = (df_clean[column] < min_val) | (df_clean[column] > max_val)
                    invalid_count = mask.sum()
                    
                    if invalid_count > 0:
                        validation_errors.append(
                            f"Column '{column}': {invalid_count} values outside range [{min_val}, {max_val}]"
                        )
                        # Remove invalid rows
                        df_clean = df_clean[~mask]
            
            if validation_errors:
                logger.warning(f"Validation issues found: {validation_errors}")
            
            final_count = len(df_clean)
            logger.info(f"Validation complete: {final_count}/{initial_count} rows passed validation")
            
            return df_clean
            
        except Exception as e:
            raise DataProcessingError(f"Batch validation failed: {str(e)}")
    
    def _validate_astronomical_consistency(self, data: Dict) -> None:
        """
        Validate astronomical consistency of the data
        
        Args:
            data: Dictionary with exoplanet parameters
            
        Raises:
            AstronomicalDataError: If data is astronomically inconsistent
        """
        try:
            # Check if transit depth is reasonable given planet/star radii
            expected_depth = (data['radius'] / data['starRadius']) ** 2 * 1e6  # ppm
            observed_depth = data['depth']
            
            # Allow factor of 10 difference (limb darkening, inclination effects)
            if observed_depth > expected_depth * 10 or observed_depth < expected_depth / 10:
                raise AstronomicalDataError(
                    f"Transit depth ({observed_depth} ppm) inconsistent with planet/star radius ratio "
                    f"(expected ~{expected_depth:.1f} ppm)",
                    parameter="depth",
                    value=observed_depth,
                    astronomical_constraint=f"Expected depth ~{expected_depth:.1f} ppm from geometry"
                )
            
            # Check if equilibrium temperature is reasonable
            # Simple approximation: T_eq ≈ T_star * sqrt(R_star / (2 * distance))
            # For circular orbit: distance ≈ (period^2 * star_mass)^(1/3) (in AU, simplified)
            period_years = data['period'] / 365.25
            approx_distance = (period_years ** 2 * data['starMass']) ** (1/3)  # Very rough approximation
            expected_temp = data['starTemp'] * np.sqrt(data['starRadius'] / (2 * approx_distance))
            
            temp_ratio = data['temp'] / expected_temp
            if temp_ratio > 2.0 or temp_ratio < 0.5:
                logger.warning(
                    f"Equilibrium temperature ({data['temp']} K) may be inconsistent "
                    f"with orbital parameters (expected ~{expected_temp:.0f} K)"
                )
            
        except Exception as e:
            if isinstance(e, AstronomicalDataError):
                raise
            else:
                logger.warning(f"Could not validate astronomical consistency: {str(e)}")
    
    def apply_feature_engineering(self, data: Union[Dict, pd.DataFrame]) -> Union[EnhancedFeatures, pd.DataFrame]:
        """
        Apply feature engineering to create derived astronomical features
        
        Args:
            data: Single data dict or DataFrame with exoplanet parameters
            
        Returns:
            Enhanced features object or DataFrame with additional columns
            
        Raises:
            FeatureEngineeringError: If feature engineering fails
        """
        try:
            if isinstance(data, dict):
                return self._engineer_single_features(data)
            else:
                return self._engineer_batch_features(data)
                
        except Exception as e:
            raise FeatureEngineeringError(f"Feature engineering failed: {str(e)}")
    
    def _engineer_single_features(self, data: Dict) -> EnhancedFeatures:
        """Engineer features for single data point"""
        try:
            # Original features
            original = ExoplanetFeatures(**data)
            
            # Physical ratios
            planet_star_radius_ratio = data['radius'] / data['starRadius']
            equilibrium_temp_ratio = data['temp'] / data['starTemp']
            
            # Expected transit depth from geometry
            transit_depth_expected = (data['radius'] / data['starRadius']) ** 2 * 1e6  # ppm
            
            # Orbital characteristics (simplified)
            orbital_velocity = 2 * np.pi / data['period']  # Relative units
            
            # Habitable zone distance
            temp_factor = (data['starTemp'] / self.SOLAR_TEMP) ** 0.5
            hz_center = (self.HZ_INNER + self.HZ_OUTER) / 2 * temp_factor
            # Rough distance estimate (would need more orbital mechanics for accuracy)
            period_years = data['period'] / 365.25
            approx_distance = (period_years ** 2 * data['starMass']) ** (1/3)
            hz_distance = (approx_distance - hz_center) / (self.HZ_OUTER - self.HZ_INNER)
            
            # Signal quality metrics
            depth_snr_ratio = data['depth'] / data['snr'] if data['snr'] > 0 else 0
            duration_period_ratio = data['duration'] / (data['period'] * 24)  # Fraction of orbit
            
            # Classifications
            planet_type = self._classify_planet_size(data['radius'])
            stellar_type = self._classify_stellar_type(data['starTemp'])
            is_habitable_zone = self._is_in_habitable_zone(approx_distance, data['starTemp'])
            
            return EnhancedFeatures(
                original_features=original,
                planet_star_radius_ratio=planet_star_radius_ratio,
                equilibrium_temp_ratio=equilibrium_temp_ratio,
                transit_depth_expected=transit_depth_expected,
                orbital_velocity=orbital_velocity,
                hz_distance=hz_distance,
                depth_snr_ratio=depth_snr_ratio,
                duration_period_ratio=duration_period_ratio,
                planet_type=planet_type,
                stellar_type=stellar_type,
                is_habitable_zone=is_habitable_zone
            )
            
        except Exception as e:
            raise FeatureEngineeringError(f"Single feature engineering failed: {str(e)}")
    
    def _engineer_batch_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for batch DataFrame"""
        try:
            logger.info(f"Engineering features for {len(df)} objects")
            
            df_enhanced = df.copy()
            
            # Physical ratios
            df_enhanced['planet_star_radius_ratio'] = df['radius'] / df['starRadius']
            df_enhanced['equilibrium_temp_ratio'] = df['temp'] / df['starTemp']
            
            # Expected transit depth
            df_enhanced['transit_depth_expected'] = (df['radius'] / df['starRadius']) ** 2 * 1e6
            
            # Orbital characteristics
            df_enhanced['orbital_velocity'] = 2 * np.pi / df['period']
            
            # Habitable zone calculations
            temp_factor = (df['starTemp'] / self.SOLAR_TEMP) ** 0.5
            hz_center = (self.HZ_INNER + self.HZ_OUTER) / 2 * temp_factor
            period_years = df['period'] / 365.25
            approx_distance = (period_years ** 2 * df['starMass']) ** (1/3)
            df_enhanced['hz_distance'] = (approx_distance - hz_center) / (self.HZ_OUTER - self.HZ_INNER)
            
            # Signal quality metrics
            df_enhanced['depth_snr_ratio'] = np.where(
                df['snr'] > 0,
                df['depth'] / df['snr'],
                0
            )
            df_enhanced['duration_period_ratio'] = df['duration'] / (df['period'] * 24)
            
            # Classifications
            df_enhanced['planet_type'] = df['radius'].apply(self._classify_planet_size)
            df_enhanced['stellar_type'] = df['starTemp'].apply(self._classify_stellar_type)
            df_enhanced['is_habitable_zone'] = [
                self._is_in_habitable_zone(dist, temp)
                for dist, temp in zip(approx_distance, df['starTemp'])
            ]
            
            logger.info(f"Feature engineering complete. Added {len(df_enhanced.columns) - len(df.columns)} new features")
            
            return df_enhanced
            
        except Exception as e:
            raise FeatureEngineeringError(f"Batch feature engineering failed: {str(e)}")
    
    def _classify_planet_size(self, radius: float) -> str:
        """Classify planet by radius"""
        boundaries = self.PLANET_BOUNDARIES
        
        if radius < boundaries['earth_min']:
            return PlanetSize.SUB_EARTH
        elif radius <= boundaries['earth_max']:
            return PlanetSize.EARTH_SIZE
        elif radius <= boundaries['super_earth_max']:
            return PlanetSize.SUPER_EARTH
        elif radius < boundaries['jupiter_min']:
            return PlanetSize.NEPTUNE_SIZE
        else:
            return PlanetSize.JUPITER_SIZE
    
    def _classify_stellar_type(self, temperature: float) -> str:
        """Classify star by temperature"""
        boundaries = self.STELLAR_BOUNDARIES
        
        if temperature <= boundaries['m_dwarf_max']:
            return StellarType.M_DWARF
        elif temperature <= boundaries['k_dwarf_max']:
            return StellarType.K_DWARF
        elif temperature <= boundaries['g_dwarf_max']:
            return StellarType.G_DWARF
        elif temperature <= boundaries['f_dwarf_max']:
            return StellarType.F_DWARF
        else:
            return StellarType.A_TYPE
    
    def _is_in_habitable_zone(self, distance_au: float, stellar_temp: float) -> bool:
        """Check if planet is in conservative habitable zone"""
        temp_factor = (stellar_temp / self.SOLAR_TEMP) ** 0.5
        hz_inner = self.HZ_INNER * temp_factor
        hz_outer = self.HZ_OUTER * temp_factor
        
        return hz_inner <= distance_au <= hz_outer
    
    def get_processing_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get processing statistics for a dataset
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with processing statistics
        """
        try:
            stats = {
                'total_objects': len(df),
                'parameter_stats': {},
                'classifications': {},
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Basic parameter statistics
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in self.VALIDATION_RANGES:
                    stats['parameter_stats'][col] = {
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    }
            
            # Classifications (if available)
            if 'planet_type' in df.columns:
                stats['classifications']['planet_types'] = df['planet_type'].value_counts().to_dict()
            
            if 'stellar_type' in df.columns:
                stats['classifications']['stellar_types'] = df['stellar_type'].value_counts().to_dict()
            
            if 'is_habitable_zone' in df.columns:
                hz_count = df['is_habitable_zone'].sum()
                stats['classifications']['habitable_zone_candidates'] = int(hz_count)
                stats['classifications']['habitable_zone_percentage'] = float(hz_count / len(df) * 100)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to generate processing stats: {str(e)}")
            return {
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }