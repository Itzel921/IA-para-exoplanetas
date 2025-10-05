"""
File Processing Service for Exoplanet Detection Backend

Handles file upload, validation, and processing operations.
Supports CSV files with exoplanet data from various NASA missions.
"""

import io
import csv
from typing import Dict, List, Optional, Tuple, Any, BinaryIO
import logging
from pathlib import Path
from datetime import datetime

# Handle imports with fallback
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

# Local imports with fallback
try:
    from ..core.exceptions import FileProcessingError, ValidationError
    from .data_processing import DataProcessor
except ImportError:
    class FileProcessingError(Exception): pass
    class ValidationError(Exception): pass
    
    class DataProcessor:
        def __init__(self):
            pass
        
        def validate_batch_data(self, df):
            return df


logger = logging.getLogger(__name__)


class FileProcessor:
    """
    Service for handling file operations in the exoplanet detection system
    
    Supports:
    - CSV file validation and parsing
    - File format detection
    - Batch processing of exoplanet data
    - Error handling and reporting
    """
    
    def __init__(self, max_file_size: int = 50 * 1024 * 1024):
        """
        Initialize file processor
        
        Args:
            max_file_size: Maximum file size in bytes (default 50MB)
        """
        self.max_file_size = max_file_size
        self.allowed_extensions = {'.csv', '.txt'}
        self.required_columns = [
            'period', 'radius', 'temp', 'starRadius', 'starMass',
            'starTemp', 'depth', 'duration', 'snr'
        ]
        
        # Column name mappings for different NASA datasets
        self.column_mappings = {
            # Kepler mappings
            'koi_period': 'period',
            'koi_prad': 'radius',
            'koi_teq': 'temp',
            'koi_srad': 'starRadius',
            'koi_smass': 'starMass',
            'koi_steff': 'starTemp',
            'koi_depth': 'depth',
            'koi_duration': 'duration',
            'koi_model_snr': 'snr',
            
            # TESS mappings
            'pl_orbper': 'period',
            'pl_rade': 'radius',
            'pl_eqt': 'temp',
            'st_rad': 'starRadius',
            'st_mass': 'starMass',
            'st_teff': 'starTemp',
            'depth_ppm': 'depth',
            'duration_hr': 'duration',
            'tess_snr': 'snr',
            
            # Alternative names
            'orbital_period': 'period',
            'planet_radius': 'radius',
            'equilibrium_temperature': 'temp',
            'stellar_radius': 'starRadius',
            'stellar_mass': 'starMass',
            'stellar_temperature': 'starTemp',
            'transit_depth': 'depth',
            'transit_duration': 'duration',
            'signal_to_noise': 'snr'
        }
        
        self.data_processor = DataProcessor()
    
    def validate_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Validate uploaded file
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Dictionary with validation results
            
        Raises:
            FileProcessingError: If file validation fails
        """
        try:
            validation_result = {
                'valid': False,
                'filename': filename,
                'file_size': len(file_content),
                'errors': [],
                'warnings': [],
                'detected_format': None,
                'estimated_rows': 0
            }
            
            # Check file size
            if len(file_content) > self.max_file_size:
                raise FileProcessingError(
                    f"File size ({len(file_content)} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)",
                    filename=filename,
                    file_size=len(file_content)
                )
            
            # Check file extension
            file_path = Path(filename)
            if file_path.suffix.lower() not in self.allowed_extensions:
                raise FileProcessingError(
                    f"File extension '{file_path.suffix}' not supported. Allowed: {self.allowed_extensions}",
                    filename=filename,
                    file_type=file_path.suffix
                )
            
            # Validate content format
            try:
                content_str = file_content.decode('utf-8')
                validation_result['detected_format'] = 'UTF-8 CSV'
            except UnicodeDecodeError:
                try:
                    content_str = file_content.decode('latin-1')
                    validation_result['detected_format'] = 'Latin-1 CSV'
                    validation_result['warnings'].append("File encoded in Latin-1, converted to UTF-8")
                except UnicodeDecodeError:
                    raise FileProcessingError(
                        "File encoding not supported. Please use UTF-8 or Latin-1 encoding.",
                        filename=filename
                    )
            
            # Basic CSV validation
            try:
                # Try to read first few lines to validate CSV format
                csv_reader = csv.reader(io.StringIO(content_str))
                header = next(csv_reader)
                
                # Count estimated rows (quick estimate)
                line_count = content_str.count('\n')
                validation_result['estimated_rows'] = max(0, line_count - 1)  # Subtract header
                
                # Validate header
                header_validation = self._validate_header(header)
                validation_result.update(header_validation)
                
                if header_validation['valid']:
                    validation_result['valid'] = True
                else:
                    validation_result['errors'].extend(header_validation['errors'])
                
            except Exception as e:
                raise FileProcessingError(
                    f"Invalid CSV format: {str(e)}",
                    filename=filename
                )
            
            logger.info(f"File validation completed for {filename}: {validation_result['valid']}")
            return validation_result
            
        except FileProcessingError:
            raise
        except Exception as e:
            raise FileProcessingError(f"File validation failed: {str(e)}", filename=filename)
    
    def _validate_header(self, header: List[str]) -> Dict[str, Any]:
        """
        Validate CSV header and detect column mappings
        
        Args:
            header: List of column names from CSV header
            
        Returns:
            Dictionary with header validation results
        """
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'mapped_columns': {},
            'missing_columns': [],
            'extra_columns': []
        }
        
        # Clean header (strip whitespace, handle case)
        cleaned_header = [col.strip().lower().replace(' ', '_') for col in header]
        
        # Map columns to standard names
        mapped_cols = {}
        for i, col in enumerate(cleaned_header):
            original_col = header[i]
            
            if col in [c.lower() for c in self.required_columns]:
                # Direct match
                standard_name = next(c for c in self.required_columns if c.lower() == col)
                mapped_cols[original_col] = standard_name
            elif col in [k.lower() for k in self.column_mappings.keys()]:
                # Mapped column
                mapping_key = next(k for k in self.column_mappings.keys() if k.lower() == col)
                standard_name = self.column_mappings[mapping_key]
                mapped_cols[original_col] = standard_name
                result['warnings'].append(f"Mapped column '{original_col}' to '{standard_name}'")
        
        result['mapped_columns'] = mapped_cols
        
        # Check for missing required columns
        mapped_standard_names = set(mapped_cols.values())
        missing = set(self.required_columns) - mapped_standard_names
        result['missing_columns'] = list(missing)
        
        # Identify extra columns
        original_names = set(header)
        mapped_originals = set(mapped_cols.keys())
        result['extra_columns'] = list(original_names - mapped_originals)
        
        if result['extra_columns']:
            result['warnings'].append(f"Extra columns will be ignored: {result['extra_columns']}")
        
        # Determine if header is valid
        if not result['missing_columns']:
            result['valid'] = True
        else:
            result['errors'].append(f"Missing required columns: {result['missing_columns']}")
        
        return result
    
    def process_csv_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process CSV file with exoplanet data
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Dictionary with processing results
            
        Raises:
            FileProcessingError: If file processing fails
        """
        try:
            start_time = datetime.now()
            logger.info(f"Processing CSV file: {filename}")
            
            # Validate file first
            validation_result = self.validate_file(file_content, filename)
            if not validation_result['valid']:
                raise FileProcessingError(
                    f"File validation failed: {validation_result['errors']}",
                    filename=filename
                )
            
            # Load data
            if not PANDAS_AVAILABLE:
                raise FileProcessingError(
                    "Pandas not available. Cannot process CSV files.",
                    filename=filename
                )
            
            try:
                content_str = file_content.decode('utf-8')
            except UnicodeDecodeError:
                content_str = file_content.decode('latin-1')
            
            df = pd.read_csv(io.StringIO(content_str))
            
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Apply column mapping
            if validation_result.get('mapped_columns'):
                df_mapped = df.rename(columns=validation_result['mapped_columns'])
                # Keep only required columns
                df_processed = df_mapped[self.required_columns].copy()
            else:
                df_processed = df[self.required_columns].copy()
            
            # Validate and clean data
            df_clean = self.data_processor.validate_batch_data(df_processed)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'filename': filename,
                'original_rows': len(df),
                'processed_rows': len(df_clean),
                'columns_processed': list(df_clean.columns),
                'processing_time_seconds': processing_time,
                'data': df_clean,
                'validation_warnings': validation_result.get('warnings', []),
                'processing_timestamp': start_time.isoformat()
            }
            
            logger.info(f"CSV processing completed: {len(df_clean)}/{len(df)} rows processed successfully")
            return result
            
        except FileProcessingError:
            raise
        except Exception as e:
            raise FileProcessingError(f"CSV processing failed: {str(e)}", filename=filename)
    
    def convert_to_dict_list(self, df) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to list of dictionaries
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            List of dictionaries with row data
        """
        try:
            if not PANDAS_AVAILABLE or df is None:
                return []
            
            # Convert DataFrame to list of dicts, handling NaN values
            records = []
            for _, row in df.iterrows():
                record = {}
                for col, value in row.items():
                    if pd.isna(value):
                        record[col] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        record[col] = float(value)
                    else:
                        record[col] = value
                records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to convert DataFrame to dict list: {str(e)}")
            return []
    
    def get_file_info(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Get detailed information about a file
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Dictionary with file information
        """
        try:
            file_path = Path(filename)
            
            info = {
                'filename': filename,
                'extension': file_path.suffix.lower(),
                'size_bytes': len(file_content),
                'size_kb': round(len(file_content) / 1024, 2),
                'size_mb': round(len(file_content) / (1024 * 1024), 2),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Try to get more details if it's a CSV
            if file_path.suffix.lower() == '.csv':
                try:
                    content_str = file_content.decode('utf-8')
                    line_count = content_str.count('\n')
                    
                    # Quick column detection
                    first_line = content_str.split('\n')[0] if content_str else ""
                    estimated_columns = len(first_line.split(',')) if first_line else 0
                    
                    info.update({
                        'estimated_rows': max(0, line_count - 1),
                        'estimated_columns': estimated_columns,
                        'encoding': 'UTF-8'
                    })
                    
                except UnicodeDecodeError:
                    try:
                        content_str = file_content.decode('latin-1')
                        info['encoding'] = 'Latin-1'
                    except UnicodeDecodeError:
                        info['encoding'] = 'Unknown'
                        info['encoding_error'] = True
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to analyze file {filename}: {str(e)}")
            return {
                'filename': filename,
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def create_sample_csv(self) -> str:
        """
        Create a sample CSV content for download/reference
        
        Returns:
            CSV content as string
        """
        sample_data = [
            {
                'period': 365.25,
                'radius': 1.0,
                'temp': 288.0,
                'starRadius': 1.0,
                'starMass': 1.0,
                'starTemp': 5778.0,
                'depth': 84.0,
                'duration': 6.5,
                'snr': 15.0
            },
            {
                'period': 88.0,
                'radius': 0.95,
                'temp': 700.0,
                'starRadius': 1.1,
                'starMass': 1.05,
                'starTemp': 5950.0,
                'depth': 79.0,
                'duration': 3.2,
                'snr': 22.0
            },
            {
                'period': 225.0,
                'radius': 0.85,
                'temp': 464.0,
                'starRadius': 0.9,
                'starMass': 0.95,
                'starTemp': 5200.0,
                'depth': 91.0,
                'duration': 5.1,
                'snr': 18.5
            }
        ]
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=self.required_columns)
        writer.writeheader()
        writer.writerows(sample_data)
        
        return output.getvalue()