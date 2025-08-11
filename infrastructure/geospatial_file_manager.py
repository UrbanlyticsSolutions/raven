"""
Unified GeoSpatial File Manager - GeoJSON-First Design

Centralizes all geospatial file I/O operations with automatic format conversion,
data provenance tracking, and consistent coordinate handling.

Key Features:
- GeoJSON-first storage standard
- Automatic SHP→GeoJSON conversion on read
- Centralized file naming conventions
- Data provenance and quality tracking
- Single source of truth for each dataset
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
import uuid
import hashlib

logger = logging.getLogger(__name__)


class GeospatialFileError(Exception):
    """Base exception for geospatial file operations"""
    pass


class FormatConversionError(GeospatialFileError):
    """Raised when format conversion fails"""
    pass


class DataValidationError(GeospatialFileError):
    """Raised when data validation fails"""
    pass


class GeospatialFileManager:
    """
    Unified manager for all geospatial file operations in the RAVEN workflow.
    
    Design Principles:
    1. GeoJSON is the single storage format
    2. Consistent folder structure across all steps
    3. Each step reads from previous step's output folder
    4. Automatic conversion from legacy formats
    5. Data provenance tracking
    6. Consistent coordinate system handling (WGS84 for storage)
    """
    
    # Standard folder structure for workflow steps
    STEP_FOLDERS = {
        'step1': 'step1_data_preparation',
        'step2': 'step2_watershed_delineation', 
        'step3': 'step3_lake_processing',
        'step4': 'step4_hru_generation',
        'step5': 'step5_raven_model',
        'step6': 'step6_model_validation'
    }
    
    # Standard files by step - each step outputs to its folder
    STEP_OUTPUTS = {
        'step1': {
            'dem': 'dem.tif',
            'landcover': 'landcover.tif', 
            'soil': 'soil.tif',
            'study_area': 'study_area.geojson'
        },
        'step2': {
            'watershed': 'watershed.geojson',
            'streams': 'streams.geojson',
            'subbasins': 'subbasins.geojson'
        },
        'step3': {
            'lakes': 'lakes.geojson',
            'subbasins_with_lakes': 'subbasins_with_lakes.geojson',
            'routing_network': 'routing_network.geojson'
        },
        'step4': {
            'hrus': 'hrus.geojson',
            'hru_attributes': 'hru_attributes.geojson'
        },
        'step5': {
            'raven_files': 'raven_model_files',  # Directory
            'model_inputs': 'model_inputs.geojson'
        },
        'step6': {
            'validation_results': 'validation_results.geojson',
            'calibrated_model': 'calibrated_model_files'  # Directory
        }
    }
    
    # Step dependencies - where each step reads its inputs from
    STEP_INPUTS = {
        'step2': ['step1'],  # Step 2 reads from Step 1 outputs
        'step3': ['step1', 'step2'],  # Step 3 reads from Step 1 & 2
        'step4': ['step1', 'step2', 'step3'],  # Step 4 reads from 1, 2, 3
        'step5': ['step1', 'step2', 'step3', 'step4'],
        'step6': ['step1', 'step2', 'step3', 'step4', 'step5']
    }
    
    LEGACY_FORMATS = ['.shp', '.gpkg', '.kml']
    SUPPORTED_RASTER_FORMATS = ['.tif', '.tiff', '.nc']
    
    def __init__(self, workspace_root: Union[str, Path]):
        """
        Initialize the file manager with workspace root.
        
        Args:
            workspace_root: Root directory for all spatial data
        """
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(exist_ok=True, parents=True)
        
        # Initialize metadata tracking
        self.metadata_file = self.workspace_root / "spatial_data_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"GeoSpatialFileManager initialized: {self.workspace_root}")
    
    def write_step_output(self, 
                         data: gpd.GeoDataFrame, 
                         step: str,
                         file_type: str,
                         metadata: Optional[Dict] = None) -> Path:
        """
        Write geospatial data to step's output folder as GeoJSON.
        
        Args:
            data: GeoDataFrame to write
            step: Step name ('step1', 'step2', etc.)
            file_type: File type for this step ('watershed', 'streams', etc.)
            metadata: Additional metadata to store
            
        Returns:
            Path to written GeoJSON file
            
        Raises:
            GeospatialFileError: If write operation fails
        """
        if step not in self.STEP_FOLDERS:
            raise GeospatialFileError(f"Unknown step: {step}")
        
        if step not in self.STEP_OUTPUTS or file_type not in self.STEP_OUTPUTS[step]:
            raise GeospatialFileError(f"File type '{file_type}' not defined for {step}")
        
        # Create step output directory
        step_dir = self.workspace_root / self.STEP_FOLDERS[step]
        step_dir.mkdir(exist_ok=True, parents=True)
        
        # Get filename for this file type
        filename = self.STEP_OUTPUTS[step][file_type]
        output_path = step_dir / filename
        
        try:
            # Ensure consistent coordinate system (WGS84 for storage)
            if data.crs is None:
                logger.warning(f"No CRS specified for {file_type}, assuming WGS84")
                data.crs = 'EPSG:4326'
            elif data.crs != 'EPSG:4326':
                logger.info(f"Converting {file_type} from {data.crs} to WGS84 for storage")
                data = data.to_crs('EPSG:4326')
            
            # Validate data before writing
            validation_result = self._validate_geospatial_data(data, file_type)
            if not validation_result['is_valid']:
                raise DataValidationError(f"Data validation failed for {file_type}: {validation_result['errors']}")
            
            # Write GeoJSON with high precision
            data.to_file(output_path, driver='GeoJSON', index=False)
            
            # Update metadata with step-specific key
            metadata_key = f"{step}_{file_type}"
            file_metadata = {
                'step': step,
                'file_type': file_type,
                'file_path': str(output_path),
                'created_at': datetime.now().isoformat(),
                'record_count': len(data),
                'crs': str(data.crs),
                'bounds': list(data.total_bounds),
                'columns': list(data.columns),
                'geometry_types': list(data.geometry.type.unique()),
                'data_hash': self._calculate_data_hash(data),
                'validation_score': validation_result['score'],
                'custom_metadata': metadata or {}
            }
            
            self.metadata[metadata_key] = file_metadata
            self._save_metadata()
            
            logger.info(f"Written {file_type}: {output_path} ({len(data)} records)")
            return output_path
            
        except Exception as e:
            raise GeospatialFileError(f"Failed to write {file_type}: {str(e)}")
    
    def read_step_input(self, 
                       current_step: str,
                       file_type: str,
                       target_crs: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Read geospatial data from previous steps with automatic format conversion.
        
        Args:
            current_step: Current step requesting the data ('step2', 'step3', etc.)
            file_type: File type to read ('watershed', 'streams', etc.)
            target_crs: Optional target CRS for output (default: keep storage CRS)
            
        Returns:
            GeoDataFrame with requested data
            
        Raises:
            GeospatialFileError: If file not found or read fails
        """
        if current_step not in self.STEP_INPUTS:
            raise GeospatialFileError(f"Unknown step: {current_step}")
        
        # Search previous steps in reverse order (most recent first)
        input_steps = self.STEP_INPUTS[current_step]
        
        for input_step in reversed(input_steps):
            if input_step in self.STEP_OUTPUTS and file_type in self.STEP_OUTPUTS[input_step]:
                step_dir = self.workspace_root / self.STEP_FOLDERS[input_step]
                filename = self.STEP_OUTPUTS[input_step][file_type]
                geojson_path = step_dir / filename
                
                if geojson_path.exists():
                    logger.info(f"Reading {file_type} from {input_step}: {geojson_path}")
                    return self._read_geojson(geojson_path, file_type, target_crs)
                
                # Check for legacy formats in this step directory
                legacy_path = self._find_legacy_file(step_dir, file_type)
                if legacy_path:
                    logger.info(f"Converting legacy format from {input_step}: {legacy_path}")
                    return self._convert_and_read_legacy(legacy_path, geojson_path, file_type, target_crs)
        
        # If not found in expected steps, search all step directories as fallback
        for step_name, step_folder in self.STEP_FOLDERS.items():
            step_dir = self.workspace_root / step_folder
            if not step_dir.exists():
                continue
                
            # Look for file with expected name pattern
            for output_files in self.STEP_OUTPUTS.values():
                if file_type in output_files:
                    filename = output_files[file_type]
                    potential_path = step_dir / filename
                    
                    if potential_path.exists():
                        logger.warning(f"Found {file_type} in unexpected location: {potential_path}")
                        return self._read_geojson(potential_path, file_type, target_crs)
        
        # Also search in central data directory (workflow-specific)
        data_dir = self.workspace_root / "data"
        if data_dir.exists():
            for output_files in self.STEP_OUTPUTS.values():
                if file_type in output_files:
                    filename = output_files[file_type]
                    potential_path = data_dir / filename
                    
                    if potential_path.exists():
                        logger.warning(f"Found {file_type} in central data directory: {potential_path}")
                        return self._read_geojson(potential_path, file_type, target_crs)
        
        raise GeospatialFileError(f"No {file_type} file found for {current_step}. Searched steps: {input_steps}")
    
    def get_step_directory(self, step: str) -> Path:
        """
        Get the directory path for a specific step.
        
        Args:
            step: Step name ('step1', 'step2', etc.)
            
        Returns:
            Path to step directory (creates if doesn't exist)
        """
        if step not in self.STEP_FOLDERS:
            raise GeospatialFileError(f"Unknown step: {step}")
        
        step_dir = self.workspace_root / self.STEP_FOLDERS[step]
        step_dir.mkdir(exist_ok=True, parents=True)
        return step_dir
    
    def get_expected_outputs(self, step: str) -> Dict[str, str]:
        """
        Get expected output files for a step.
        
        Args:
            step: Step name
            
        Returns:
            Dictionary mapping file types to filenames
        """
        if step not in self.STEP_OUTPUTS:
            return {}
        return self.STEP_OUTPUTS[step].copy()
    
    def list_step_files(self, step: str) -> Dict[str, Dict]:
        """
        List all available files for a specific step.
        
        Args:
            step: Step name to list files for
            
        Returns:
            Dictionary mapping file types to file information
        """
        if step not in self.STEP_FOLDERS:
            return {}
        
        step_dir = self.workspace_root / self.STEP_FOLDERS[step]
        if not step_dir.exists():
            return {}
        
        available = {}
        expected_outputs = self.get_expected_outputs(step)
        
        for file_type, filename in expected_outputs.items():
            if filename.endswith('.tif'):  # Skip raster files for now
                continue
                
            file_path = step_dir / filename
            
            if file_path.exists():
                available[file_type] = {
                    'path': str(file_path),
                    'format': 'GeoJSON',
                    'exists': True,
                    'step': step,
                    'metadata': self.metadata.get(f"{step}_{file_type}", {})
                }
            else:
                # Check for legacy formats
                legacy_path = self._find_legacy_file(step_dir, file_type)
                if legacy_path:
                    available[file_type] = {
                        'path': str(legacy_path),
                        'format': legacy_path.suffix.upper(),
                        'exists': True,
                        'step': step,
                        'needs_conversion': True
                    }
        
        return available
    
    def list_all_files(self) -> Dict[str, Dict[str, Dict]]:
        """
        List all available files across all steps.
        
        Returns:
            Dictionary mapping steps to their file information
        """
        all_files = {}
        
        for step in self.STEP_FOLDERS:
            step_files = self.list_step_files(step)
            if step_files:
                all_files[step] = step_files
        
        return all_files
    
    def migrate_step_legacy_files(self, step: str) -> Dict[str, str]:
        """
        Convert all legacy format files to GeoJSON standard for a specific step.
        
        Args:
            step: Step name to migrate
            
        Returns:
            Dictionary of migration results
        """
        if step not in self.STEP_FOLDERS:
            return {'error': f'Unknown step: {step}'}
        
        step_dir = self.workspace_root / self.STEP_FOLDERS[step]
        if not step_dir.exists():
            return {'error': f'Step directory does not exist: {step_dir}'}
        
        results = {}
        expected_outputs = self.get_expected_outputs(step)
        
        for file_type, filename in expected_outputs.items():
            if filename.endswith('.tif'):
                continue  # Skip raster files
                
            geojson_path = step_dir / filename
            
            if not geojson_path.exists():
                legacy_path = self._find_legacy_file(step_dir, file_type)
                if legacy_path:
                    try:
                        self._convert_legacy_to_geojson(legacy_path, geojson_path, file_type, step)
                        results[file_type] = f"Converted: {legacy_path.name} → {filename}"
                        logger.info(f"Migrated {step}/{file_type}: {legacy_path} → {geojson_path}")
                    except Exception as e:
                        results[file_type] = f"Failed: {str(e)}"
                        logger.error(f"Migration failed for {step}/{file_type}: {e}")
                else:
                    results[file_type] = "No legacy file found"
            else:
                results[file_type] = "GeoJSON already exists"
        
        return results
    
    def migrate_all_legacy_files(self) -> Dict[str, Dict[str, str]]:
        """
        Convert all legacy format files across all steps.
        
        Returns:
            Dictionary mapping steps to their migration results
        """
        all_results = {}
        
        for step in self.STEP_FOLDERS:
            step_results = self.migrate_step_legacy_files(step)
            if step_results and step_results != {'error': f'Step directory does not exist: {self.workspace_root / self.STEP_FOLDERS[step]}'}:
                all_results[step] = step_results
        
        return all_results
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """
        Validate integrity of all managed geospatial files.
        
        Returns:
            Validation results with scores and recommendations
        """
        validation_results = {
            'overall_score': 0.0,
            'file_validations': {},
            'recommendations': [],
            'timestamp': datetime.now().isoformat()
        }
        
        valid_files = 0
        total_files = 0
        
        for file_type in self.STANDARD_FILES:
            if self.STANDARD_FILES[file_type].endswith('.tif'):
                continue
                
            try:
                data = self.read_geospatial(file_type)
                validation = self._validate_geospatial_data(data, file_type)
                validation_results['file_validations'][file_type] = validation
                
                if validation['is_valid']:
                    valid_files += 1
                total_files += 1
                
            except GeospatialFileError:
                validation_results['file_validations'][file_type] = {
                    'is_valid': False,
                    'score': 0.0,
                    'errors': ['File not found or unreadable']
                }
                total_files += 1
        
        if total_files > 0:
            validation_results['overall_score'] = (valid_files / total_files) * 100
        
        # Generate recommendations
        if validation_results['overall_score'] < 80:
            validation_results['recommendations'].append("Consider running migrate_legacy_files() to improve data consistency")
        
        if any(not v['is_valid'] for v in validation_results['file_validations'].values()):
            validation_results['recommendations'].append("Some files have validation errors - check individual file reports")
        
        return validation_results
    
    def _read_geojson(self, file_path: Path, file_type: str, target_crs: Optional[str]) -> gpd.GeoDataFrame:
        """Read GeoJSON file with error handling"""
        try:
            data = gpd.read_file(file_path)
            
            # Apply target CRS if specified
            if target_crs and data.crs != target_crs:
                data = data.to_crs(target_crs)
            
            logger.info(f"Read {file_type}: {file_path} ({len(data)} records)")
            return data
            
        except Exception as e:
            raise GeospatialFileError(f"Failed to read {file_path}: {str(e)}")
    
    def _find_legacy_file(self, search_dir: Path, file_type: str) -> Optional[Path]:
        """Find legacy format file for given type"""
        # Get base name by looking for expected GeoJSON filename across all steps
        base_name = file_type  # Default to file_type itself
        
        for step_outputs in self.STEP_OUTPUTS.values():
            if file_type in step_outputs:
                filename = step_outputs[file_type]
                if filename.endswith('.geojson'):
                    base_name = filename.replace('.geojson', '')
                    break
        
        # Search for legacy format with base name
        for ext in self.LEGACY_FORMATS:
            legacy_path = search_dir / f"{base_name}{ext}"
            if legacy_path.exists():
                return legacy_path
        
        # Also search for common variations
        common_variations = {
            'subbasins': ['finalcat_info', 'catchments', 'basins', 'subbasins_merged'],
            'streams': ['finalcat_info_riv', 'rivers', 'network', 'stream_network'],
            'lakes': ['waterbodies', 'reservoirs', 'all_lakes'],
            'watershed': ['watershed_boundary', 'basin', 'catchment'],
            'hrus': ['final_hrus', 'hru_polygons', 'hru_data'],
            'subbasins_with_lakes': ['subbasins_enhanced', 'catchments_with_lakes']
        }
        
        if file_type in common_variations:
            for variation in common_variations[file_type]:
                for ext in self.LEGACY_FORMATS:
                    variant_path = search_dir / f"{variation}{ext}"
                    if variant_path.exists():
                        return variant_path
        
        return None
    
    def _convert_and_read_legacy(self, legacy_path: Path, geojson_path: Path, 
                                file_type: str, target_crs: Optional[str]) -> gpd.GeoDataFrame:
        """Convert legacy file to GeoJSON and read"""
        self._convert_legacy_to_geojson(legacy_path, geojson_path, file_type)
        return self._read_geojson(geojson_path, file_type, target_crs)
    
    def _convert_legacy_to_geojson(self, legacy_path: Path, geojson_path: Path, file_type: str, step: Optional[str] = None):
        """Convert legacy format file to GeoJSON"""
        try:
            # Read legacy file
            data = gpd.read_file(legacy_path)
            
            # Ensure WGS84 for storage
            if data.crs is None:
                logger.warning(f"No CRS in legacy file {legacy_path}, assuming WGS84")
                data.crs = 'EPSG:4326'
            elif data.crs != 'EPSG:4326':
                data = data.to_crs('EPSG:4326')
            
            # Write as GeoJSON
            data.to_file(geojson_path, driver='GeoJSON', index=False)
            
            # Update metadata with step-specific key
            metadata_key = f"{step}_{file_type}" if step else file_type
            file_metadata = {
                'step': step,
                'file_type': file_type,
                'file_path': str(geojson_path),
                'created_at': datetime.now().isoformat(),
                'converted_from': str(legacy_path),
                'record_count': len(data),
                'crs': str(data.crs),
                'bounds': list(data.total_bounds),
                'columns': list(data.columns),
                'geometry_types': list(data.geometry.type.unique()),
                'data_hash': self._calculate_data_hash(data)
            }
            
            self.metadata[metadata_key] = file_metadata
            self._save_metadata()
            
        except Exception as e:
            raise FormatConversionError(f"Failed to convert {legacy_path}: {str(e)}")
    
    def _validate_geospatial_data(self, data: gpd.GeoDataFrame, file_type: str) -> Dict[str, Any]:
        """Validate geospatial data quality"""
        errors = []
        warnings = []
        score = 100.0
        
        # Basic validation
        if data.empty:
            errors.append("Dataset is empty")
            score -= 50
        
        if data.crs is None:
            errors.append("No coordinate reference system defined")
            score -= 20
        
        # Geometry validation
        invalid_geoms = data.geometry.isnull().sum()
        if invalid_geoms > 0:
            errors.append(f"{invalid_geoms} null geometries found")
            score -= min(30, invalid_geoms * 5)
        
        # Check for geometry validity
        if hasattr(data.geometry, 'is_valid'):
            invalid_count = (~data.geometry.is_valid).sum()
            if invalid_count > 0:
                warnings.append(f"{invalid_count} invalid geometries found")
                score -= min(20, invalid_count * 2)
        
        # Type-specific validation
        if file_type == 'subbasins':
            required_fields = ['SubId', 'DowSubId']
            for field in required_fields:
                if field not in data.columns:
                    errors.append(f"Missing required field: {field}")
                    score -= 15
        
        elif file_type == 'streams':
            if 'geometry' in data.columns:
                non_line_geoms = data.geometry.type.isin(['Point', 'Polygon']).sum()
                if non_line_geoms > 0:
                    warnings.append(f"{non_line_geoms} non-linear geometries in stream data")
                    score -= min(10, non_line_geoms)
        
        elif file_type == 'lakes':
            if 'Lake_ID' in data.columns:
                duplicate_ids = data['Lake_ID'].duplicated().sum()
                if duplicate_ids > 0:
                    errors.append(f"{duplicate_ids} duplicate Lake_IDs found")
                    score -= min(25, duplicate_ids * 5)
        
        # Ensure score bounds
        score = max(0.0, min(100.0, score))
        
        return {
            'is_valid': len(errors) == 0,
            'score': score,
            'errors': errors,
            'warnings': warnings,
            'record_count': len(data),
            'geometry_types': list(data.geometry.type.unique()) if not data.empty else []
        }
    
    def _calculate_data_hash(self, data: gpd.GeoDataFrame) -> str:
        """Calculate hash of geospatial data for change detection"""
        # Create a string representation of key data characteristics
        hash_data = {
            'record_count': len(data),
            'columns': sorted(data.columns.tolist()),
            'bounds': list(data.total_bounds) if not data.empty else [],
            'geometry_types': sorted(data.geometry.type.unique().tolist()) if not data.empty else []
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        return {}
    
    def _save_metadata(self):
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")


def create_file_manager(workspace_root: Union[str, Path]) -> GeospatialFileManager:
    """
    Factory function to create a configured GeospatialFileManager.
    
    Args:
        workspace_root: Root directory for spatial data management
        
    Returns:
        Configured GeospatialFileManager instance
    """
    return GeospatialFileManager(workspace_root)