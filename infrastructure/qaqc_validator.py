"""
QAQCValidator for automated data quality checks and validation scripts.
"""

import json
import pandas as pd
import geopandas as gpd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Set
from dataclasses import dataclass, asdict
import logging

from .path_manager import AbsolutePathManager, PathResolutionError, FileAccessError

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    validation_type: str
    file_path: str  # Absolute path
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO timestamp"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """Create from dictionary with ISO timestamp parsing"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class QAQCValidator:
    """
    Automated validation scripts for data quality checks.
    Checks for missing HRU fields and verifies routing table connectivity.
    """
    
    # Required HRU fields and their expected data types
    REQUIRED_HRU_FIELDS = {
        'HRU_ID': ['int64', 'int32', 'object'],  # Unique identifier
        'Area_km2': ['float64', 'float32'],      # HRU area
        'Landcover': ['object', 'category'],     # Land cover classification
        'Soil_Type': ['object', 'category'],     # Soil classification
        'Elevation_m': ['float64', 'float32'],   # Mean elevation
        'Slope': ['float64', 'float32']          # Mean slope
    }
    
    # Required routing table fields
    REQUIRED_ROUTING_FIELDS = {
        'from_node': ['int64', 'int32', 'object'],
        'to_node': ['int64', 'int32', 'object'],
        'length_m': ['float64', 'float32'],
        'slope': ['float64', 'float32']
    }
    
    # Required BasinMaker catchment fields (finalcat_info.shp)
    REQUIRED_BASINMAKER_CATCHMENT_FIELDS = {
        'SubId': ['int64', 'int32', 'object'],
        'DowSubId': ['int64', 'int32', 'object'],
        'IsLake': ['int64', 'int32'],
        'HyLakeId': ['int64', 'int32', 'object']  # Can be null for non-lake subbasins
    }
    
    # Required BasinMaker river fields (finalcat_info_riv.shp)
    REQUIRED_BASINMAKER_RIVER_FIELDS = {
        'SubId': ['int64', 'int32', 'object'],
        'DowSubId': ['int64', 'int32', 'object'],
        'RivLength': ['float64', 'float32'],
        'RivSlope': ['float64', 'float32'],
        'Ch_n': ['float64', 'float32'],
        'FloodP_n': ['float64', 'float32'],
        'BkfWidth': ['float64', 'float32'],
        'BkfDepth': ['float64', 'float32']
    }
    
    # Required enhanced HRU fields (with BasinMaker attributes)
    REQUIRED_ENHANCED_HRU_FIELDS = {
        # Subbasin attributes from BasinMaker
        'SubId': ['int64', 'int32', 'object'],
        'DowSubId': ['int64', 'int32', 'object'],
        'IsLake': ['int64', 'int32'],
        'HyLakeId': ['int64', 'int32', 'object'],
        'RivLength': ['float64', 'float32'],
        'RivSlope': ['float64', 'float32'],
        'Ch_n': ['float64', 'float32'],
        'FloodP_n': ['float64', 'float32'],
        'BkfWidth': ['float64', 'float32'],
        'BkfDepth': ['float64', 'float32'],
        
        # HRU attributes for Raven
        'HRU_ID': ['int64', 'int32', 'object'],
        'HRU_Area': ['float64', 'float32'],
        'HRU_E_mean': ['float64', 'float32'],
        'HRU_S_mean': ['float64', 'float32'],
        'HRU_A_mean': ['float64', 'float32'],
        'HRU_IsLake': ['int64', 'int32'],
        'LAND_USE_C': ['object', 'category'],
        'VEG_C': ['object', 'category'],
        'SOIL_PROF': ['object', 'category'],
        'HRU_CenX': ['float64', 'float32'],
        'HRU_CenY': ['float64', 'float32']
    }
    
    def __init__(self, path_manager: AbsolutePathManager):
        """
        Initialize QA/QC validator.
        
        Args:
            path_manager: AbsolutePathManager instance for path operations
        """
        self.path_manager = path_manager
        logger.info("Initialized QAQCValidator")
    
    def validate_hru_fields(self, hru_file: Union[str, Path]) -> ValidationResult:
        """
        Check for missing required fields in HRU data.
        
        Args:
            hru_file: Path to HRU data file (CSV, shapefile, or GeoJSON)
            
        Returns:
            ValidationResult with field validation results
            
        Raises:
            PathResolutionError: If file path cannot be resolved
            FileAccessError: If file cannot be read
        """
        abs_path = self.path_manager.resolve_path(hru_file)
        self.path_manager.validate_path(abs_path, must_exist=True, must_be_file=True)
        
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Load data based on file extension
            file_ext = abs_path.suffix.lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(abs_path)
            elif file_ext in ['.shp', '.geojson', '.gpkg']:
                gdf = gpd.read_file(abs_path)
                df = pd.DataFrame(gdf.drop(columns='geometry'))
            else:
                errors.append(f"Unsupported file format: {file_ext}")
                return ValidationResult(
                    is_valid=False,
                    validation_type="hru_fields",
                    file_path=str(abs_path),
                    errors=errors,
                    warnings=warnings,
                    metrics=metrics,
                    timestamp=datetime.now()
                )
            
            metrics['total_records'] = len(df)
            metrics['total_columns'] = len(df.columns)
            
            # Check for required fields
            missing_fields = []
            invalid_types = []
            
            for field, expected_types in self.REQUIRED_HRU_FIELDS.items():
                if field not in df.columns:
                    missing_fields.append(field)
                else:
                    # Check data type
                    actual_type = str(df[field].dtype)
                    if actual_type not in expected_types:
                        invalid_types.append(f"{field}: expected {expected_types}, got {actual_type}")
                    
                    # Check for null values
                    null_count = df[field].isnull().sum()
                    if null_count > 0:
                        warnings.append(f"{field} has {null_count} null values")
                    
                    metrics[f'{field}_null_count'] = int(null_count)
            
            if missing_fields:
                errors.extend([f"Missing required field: {field}" for field in missing_fields])
            
            if invalid_types:
                warnings.extend([f"Invalid data type for {issue}" for issue in invalid_types])
            
            # Additional HRU-specific validations
            if 'Area_km2' in df.columns:
                negative_areas = (df['Area_km2'] <= 0).sum()
                if negative_areas > 0:
                    errors.append(f"{negative_areas} HRUs have non-positive areas")
                metrics['negative_areas'] = int(negative_areas)
                metrics['min_area_km2'] = float(df['Area_km2'].min())
                metrics['max_area_km2'] = float(df['Area_km2'].max())
                metrics['mean_area_km2'] = float(df['Area_km2'].mean())
            
            if 'Elevation_m' in df.columns:
                metrics['min_elevation_m'] = float(df['Elevation_m'].min())
                metrics['max_elevation_m'] = float(df['Elevation_m'].max())
                metrics['mean_elevation_m'] = float(df['Elevation_m'].mean())
            
            if 'Slope' in df.columns:
                negative_slopes = (df['Slope'] < 0).sum()
                if negative_slopes > 0:
                    warnings.append(f"{negative_slopes} HRUs have negative slopes")
                metrics['negative_slopes'] = int(negative_slopes)
                metrics['min_slope'] = float(df['Slope'].min())
                metrics['max_slope'] = float(df['Slope'].max())
                metrics['mean_slope'] = float(df['Slope'].mean())
            
            # Check for duplicate HRU_IDs
            if 'HRU_ID' in df.columns:
                duplicate_ids = df['HRU_ID'].duplicated().sum()
                if duplicate_ids > 0:
                    errors.append(f"{duplicate_ids} duplicate HRU_IDs found")
                metrics['duplicate_hru_ids'] = int(duplicate_ids)
            
            is_valid = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Error reading HRU file: {str(e)}")
            is_valid = False
        
        return ValidationResult(
            is_valid=is_valid,
            validation_type="hru_fields",
            file_path=str(abs_path),
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    def validate_routing_connectivity(self, routing_table: Union[str, Path]) -> ValidationResult:
        """
        Verify routing table connectivity and structure.
        
        Args:
            routing_table: Path to routing table file (CSV or similar)
            
        Returns:
            ValidationResult with connectivity validation results
            
        Raises:
            PathResolutionError: If file path cannot be resolved
            FileAccessError: If file cannot be read
        """
        abs_path = self.path_manager.resolve_path(routing_table)
        self.path_manager.validate_path(abs_path, must_exist=True, must_be_file=True)
        
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Load routing table
            file_ext = abs_path.suffix.lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(abs_path)
            else:
                errors.append(f"Unsupported routing table format: {file_ext}")
                return ValidationResult(
                    is_valid=False,
                    validation_type="routing_connectivity",
                    file_path=str(abs_path),
                    errors=errors,
                    warnings=warnings,
                    metrics=metrics,
                    timestamp=datetime.now()
                )
            
            metrics['total_connections'] = len(df)
            
            # Check for required fields
            missing_fields = []
            for field, expected_types in self.REQUIRED_ROUTING_FIELDS.items():
                if field not in df.columns:
                    missing_fields.append(field)
                else:
                    # Check data type
                    actual_type = str(df[field].dtype)
                    if actual_type not in expected_types:
                        warnings.append(f"{field}: expected {expected_types}, got {actual_type}")
                    
                    # Check for null values
                    null_count = df[field].isnull().sum()
                    if null_count > 0:
                        errors.append(f"{field} has {null_count} null values")
                    
                    metrics[f'{field}_null_count'] = int(null_count)
            
            if missing_fields:
                errors.extend([f"Missing required field: {field}" for field in missing_fields])
                return ValidationResult(
                    is_valid=False,
                    validation_type="routing_connectivity",
                    file_path=str(abs_path),
                    errors=errors,
                    warnings=warnings,
                    metrics=metrics,
                    timestamp=datetime.now()
                )
            
            # Connectivity analysis
            from_nodes = set(df['from_node'].dropna())
            to_nodes = set(df['to_node'].dropna())
            all_nodes = from_nodes.union(to_nodes)
            
            metrics['unique_from_nodes'] = len(from_nodes)
            metrics['unique_to_nodes'] = len(to_nodes)
            metrics['total_unique_nodes'] = len(all_nodes)
            
            # Find outlet nodes (nodes that appear in from_node but not in to_node)
            outlet_nodes = from_nodes - to_nodes
            metrics['outlet_nodes'] = len(outlet_nodes)
            
            if len(outlet_nodes) == 0:
                errors.append("No outlet nodes found - all from_nodes have downstream connections")
            elif len(outlet_nodes) > 1:
                warnings.append(f"Multiple outlet nodes found: {len(outlet_nodes)}")
            
            # Find orphaned nodes (nodes that appear in to_node but not in from_node)
            orphaned_nodes = to_nodes - from_nodes
            metrics['orphaned_nodes'] = len(orphaned_nodes)
            
            if len(orphaned_nodes) > 0:
                warnings.append(f"{len(orphaned_nodes)} orphaned nodes found (appear as to_node but not from_node)")
            
            # Check for circular routing (simplified check)
            circular_routes = []
            for _, row in df.iterrows():
                if pd.notna(row['from_node']) and pd.notna(row['to_node']):
                    if row['from_node'] == row['to_node']:
                        circular_routes.append(row['from_node'])
            
            if circular_routes:
                errors.append(f"Circular routing detected for nodes: {circular_routes}")
            metrics['circular_routes'] = len(circular_routes)
            
            # Validate channel properties
            if 'length_m' in df.columns:
                negative_lengths = (df['length_m'] <= 0).sum()
                if negative_lengths > 0:
                    errors.append(f"{negative_lengths} connections have non-positive lengths")
                metrics['negative_lengths'] = int(negative_lengths)
                metrics['min_length_m'] = float(df['length_m'].min())
                metrics['max_length_m'] = float(df['length_m'].max())
                metrics['mean_length_m'] = float(df['length_m'].mean())
            
            if 'slope' in df.columns:
                negative_slopes = (df['slope'] < 0).sum()
                if negative_slopes > 0:
                    warnings.append(f"{negative_slopes} connections have negative slopes")
                metrics['negative_slopes'] = int(negative_slopes)
                metrics['min_slope'] = float(df['slope'].min())
                metrics['max_slope'] = float(df['slope'].max())
                metrics['mean_slope'] = float(df['slope'].mean())
            
            is_valid = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Error analyzing routing table: {str(e)}")
            is_valid = False
        
        return ValidationResult(
            is_valid=is_valid,
            validation_type="routing_connectivity",
            file_path=str(abs_path),
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    def validate_spatial_data_integrity(self, spatial_file: Union[str, Path]) -> ValidationResult:
        """
        Validate spatial data integrity (geometry validity, CRS, etc.).
        
        Args:
            spatial_file: Path to spatial data file
            
        Returns:
            ValidationResult with spatial validation results
        """
        abs_path = self.path_manager.resolve_path(spatial_file)
        self.path_manager.validate_path(abs_path, must_exist=True, must_be_file=True)
        
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Load spatial data
            gdf = gpd.read_file(abs_path)
            
            metrics['total_features'] = len(gdf)
            metrics['geometry_type'] = str(gdf.geometry.geom_type.iloc[0]) if len(gdf) > 0 else "None"
            
            # Check CRS
            if gdf.crs is None:
                errors.append("No coordinate reference system (CRS) defined")
            else:
                metrics['crs'] = str(gdf.crs)
            
            # Check for invalid geometries
            invalid_geoms = (~gdf.geometry.is_valid).sum()
            if invalid_geoms > 0:
                errors.append(f"{invalid_geoms} invalid geometries found")
            metrics['invalid_geometries'] = int(invalid_geoms)
            
            # Check for empty geometries
            empty_geoms = gdf.geometry.is_empty.sum()
            if empty_geoms > 0:
                warnings.append(f"{empty_geoms} empty geometries found")
            metrics['empty_geometries'] = int(empty_geoms)
            
            # Calculate spatial extent
            if len(gdf) > 0 and not gdf.geometry.is_empty.all():
                bounds = gdf.total_bounds
                metrics['spatial_extent'] = {
                    'min_x': float(bounds[0]),
                    'min_y': float(bounds[1]),
                    'max_x': float(bounds[2]),
                    'max_y': float(bounds[3])
                }
            
            is_valid = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Error validating spatial data: {str(e)}")
            is_valid = False
        
        return ValidationResult(
            is_valid=is_valid,
            validation_type="spatial_integrity",
            file_path=str(abs_path),
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    def validate_basinmaker_catchments(self, catchment_file: Union[str, Path]) -> ValidationResult:
        """
        Validate BasinMaker catchment data (finalcat_info.shp).
        
        Args:
            catchment_file: Path to BasinMaker catchment shapefile
            
        Returns:
            ValidationResult with BasinMaker catchment validation results
        """
        abs_path = self.path_manager.resolve_path(catchment_file)
        self.path_manager.validate_path(abs_path, must_exist=True, must_be_file=True)
        
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Load catchment data
            gdf = gpd.read_file(abs_path)
            df = pd.DataFrame(gdf.drop(columns='geometry'))
            
            metrics['total_catchments'] = len(df)
            metrics['total_columns'] = len(df.columns)
            
            # Check for required fields
            missing_fields = []
            for field, expected_types in self.REQUIRED_BASINMAKER_CATCHMENT_FIELDS.items():
                if field not in df.columns:
                    missing_fields.append(field)
                else:
                    # Check data type
                    actual_type = str(df[field].dtype)
                    if actual_type not in expected_types:
                        warnings.append(f"{field}: expected {expected_types}, got {actual_type}")
                    
                    # Check for null values (except HyLakeId which can be null)
                    null_count = df[field].isnull().sum()
                    if field != 'HyLakeId' and null_count > 0:
                        errors.append(f"{field} has {null_count} null values")
                    elif field == 'HyLakeId' and null_count > 0:
                        warnings.append(f"{field} has {null_count} null values (expected for non-lake subbasins)")
                    
                    metrics[f'{field}_null_count'] = int(null_count)
            
            if missing_fields:
                errors.extend([f"Missing required field: {field}" for field in missing_fields])
            
            # BasinMaker-specific validations
            if 'SubId' in df.columns and 'DowSubId' in df.columns:
                # Check for duplicate SubIds
                duplicate_subids = df['SubId'].duplicated().sum()
                if duplicate_subids > 0:
                    errors.append(f"{duplicate_subids} duplicate SubIds found")
                metrics['duplicate_subids'] = int(duplicate_subids)
                
                # Check routing topology (acyclic)
                subids = set(df['SubId'].dropna())
                dowsubids = set(df['DowSubId'].dropna())
                
                # Find outlets (SubIds not in DowSubId)
                outlets = subids - dowsubids
                metrics['outlet_count'] = len(outlets)
                
                if len(outlets) == 0:
                    errors.append("No outlet subbasins found - routing may be circular")
                elif len(outlets) > 1:
                    warnings.append(f"Multiple outlets found: {len(outlets)}")
                
                # Check for self-referencing subbasins
                self_refs = (df['SubId'] == df['DowSubId']).sum()
                if self_refs > 0:
                    errors.append(f"{self_refs} self-referencing subbasins found")
                metrics['self_referencing'] = int(self_refs)
            
            # Lake consistency validation
            if 'IsLake' in df.columns and 'HyLakeId' in df.columns:
                lake_subbasins = df[df['IsLake'] == 1]
                non_lake_subbasins = df[df['IsLake'] != 1]
                
                metrics['lake_subbasins'] = len(lake_subbasins)
                metrics['non_lake_subbasins'] = len(non_lake_subbasins)
                
                # Check that lake subbasins have HyLakeId
                lake_without_id = lake_subbasins['HyLakeId'].isnull().sum()
                if lake_without_id > 0:
                    errors.append(f"{lake_without_id} lake subbasins (IsLake=1) missing HyLakeId")
                
                # Check that non-lake subbasins don't have HyLakeId
                non_lake_with_id = non_lake_subbasins['HyLakeId'].notnull().sum()
                if non_lake_with_id > 0:
                    warnings.append(f"{non_lake_with_id} non-lake subbasins have HyLakeId")
            
            is_valid = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Error validating BasinMaker catchments: {str(e)}")
            is_valid = False
        
        return ValidationResult(
            is_valid=is_valid,
            validation_type="basinmaker_catchments",
            file_path=str(abs_path),
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    def validate_basinmaker_rivers(self, river_file: Union[str, Path]) -> ValidationResult:
        """
        Validate BasinMaker river data (finalcat_info_riv.shp).
        
        Args:
            river_file: Path to BasinMaker river shapefile
            
        Returns:
            ValidationResult with BasinMaker river validation results
        """
        abs_path = self.path_manager.resolve_path(river_file)
        self.path_manager.validate_path(abs_path, must_exist=True, must_be_file=True)
        
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Load river data
            gdf = gpd.read_file(abs_path)
            df = pd.DataFrame(gdf.drop(columns='geometry'))
            
            metrics['total_rivers'] = len(df)
            metrics['total_columns'] = len(df.columns)
            
            # Check for required fields
            missing_fields = []
            for field, expected_types in self.REQUIRED_BASINMAKER_RIVER_FIELDS.items():
                if field not in df.columns:
                    missing_fields.append(field)
                else:
                    # Check data type
                    actual_type = str(df[field].dtype)
                    if actual_type not in expected_types:
                        warnings.append(f"{field}: expected {expected_types}, got {actual_type}")
                    
                    # Check for null values
                    null_count = df[field].isnull().sum()
                    if null_count > 0:
                        errors.append(f"{field} has {null_count} null values")
                    
                    metrics[f'{field}_null_count'] = int(null_count)
            
            if missing_fields:
                errors.extend([f"Missing required field: {field}" for field in missing_fields])
            
            # Channel properties validation
            if 'RivLength' in df.columns:
                negative_lengths = (df['RivLength'] <= 0).sum()
                if negative_lengths > 0:
                    errors.append(f"{negative_lengths} rivers have non-positive lengths")
                metrics['negative_lengths'] = int(negative_lengths)
                metrics['min_length'] = float(df['RivLength'].min())
                metrics['max_length'] = float(df['RivLength'].max())
                metrics['mean_length'] = float(df['RivLength'].mean())
            
            if 'RivSlope' in df.columns:
                negative_slopes = (df['RivSlope'] < 0).sum()
                if negative_slopes > 0:
                    warnings.append(f"{negative_slopes} rivers have negative slopes")
                metrics['negative_slopes'] = int(negative_slopes)
                metrics['min_slope'] = float(df['RivSlope'].min())
                metrics['max_slope'] = float(df['RivSlope'].max())
                metrics['mean_slope'] = float(df['RivSlope'].mean())
            
            # Manning's coefficient validation
            for field in ['Ch_n', 'FloodP_n']:
                if field in df.columns:
                    invalid_manning = ((df[field] <= 0) | (df[field] > 1)).sum()
                    if invalid_manning > 0:
                        warnings.append(f"{invalid_manning} rivers have invalid {field} values (should be 0-1)")
                    metrics[f'invalid_{field.lower()}'] = int(invalid_manning)
            
            # Channel geometry validation
            for field in ['BkfWidth', 'BkfDepth']:
                if field in df.columns:
                    negative_values = (df[field] <= 0).sum()
                    if negative_values > 0:
                        errors.append(f"{negative_values} rivers have non-positive {field}")
                    metrics[f'negative_{field.lower()}'] = int(negative_values)
            
            is_valid = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Error validating BasinMaker rivers: {str(e)}")
            is_valid = False
        
        return ValidationResult(
            is_valid=is_valid,
            validation_type="basinmaker_rivers",
            file_path=str(abs_path),
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    def validate_enhanced_hru_fields(self, hru_file: Union[str, Path]) -> ValidationResult:
        """
        Validate enhanced HRU data with BasinMaker attributes.
        
        Args:
            hru_file: Path to enhanced HRU data file
            
        Returns:
            ValidationResult with enhanced HRU validation results
        """
        abs_path = self.path_manager.resolve_path(hru_file)
        self.path_manager.validate_path(abs_path, must_exist=True, must_be_file=True)
        
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Load data based on file extension
            file_ext = abs_path.suffix.lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(abs_path)
            elif file_ext in ['.shp', '.geojson', '.gpkg']:
                gdf = gpd.read_file(abs_path)
                df = pd.DataFrame(gdf.drop(columns='geometry'))
            else:
                errors.append(f"Unsupported file format: {file_ext}")
                return ValidationResult(
                    is_valid=False,
                    validation_type="enhanced_hru_fields",
                    file_path=str(abs_path),
                    errors=errors,
                    warnings=warnings,
                    metrics=metrics,
                    timestamp=datetime.now()
                )
            
            metrics['total_hrus'] = len(df)
            metrics['total_columns'] = len(df.columns)
            
            # Check for required fields
            missing_fields = []
            for field, expected_types in self.REQUIRED_ENHANCED_HRU_FIELDS.items():
                if field not in df.columns:
                    missing_fields.append(field)
                else:
                    # Check data type
                    actual_type = str(df[field].dtype)
                    if actual_type not in expected_types:
                        warnings.append(f"{field}: expected {expected_types}, got {actual_type}")
                    
                    # Check for null values (except HyLakeId which can be null)
                    null_count = df[field].isnull().sum()
                    if field != 'HyLakeId' and null_count > 0:
                        errors.append(f"{field} has {null_count} null values")
                    elif field == 'HyLakeId' and null_count > 0:
                        warnings.append(f"{field} has {null_count} null values (expected for non-lake HRUs)")
                    
                    metrics[f'{field}_null_count'] = int(null_count)
            
            if missing_fields:
                errors.extend([f"Missing required field: {field}" for field in missing_fields])
            
            # Area conservation validation by SubId
            if 'SubId' in df.columns and 'HRU_Area' in df.columns:
                subbasin_areas = df.groupby('SubId')['HRU_Area'].sum()
                metrics['unique_subbasins'] = len(subbasin_areas)
                metrics['min_subbasin_area'] = float(subbasin_areas.min())
                metrics['max_subbasin_area'] = float(subbasin_areas.max())
                metrics['mean_subbasin_area'] = float(subbasin_areas.mean())
            
            # Lake HRU validation
            if 'IsLake' in df.columns and 'HRU_IsLake' in df.columns:
                # Check consistency between subbasin IsLake and HRU_IsLake
                lake_subbasins = df[df['IsLake'] == 1]['SubId'].unique()
                for subid in lake_subbasins:
                    subbasin_hrus = df[df['SubId'] == subid]
                    lake_hrus = subbasin_hrus[subbasin_hrus['HRU_IsLake'] == 1]
                    non_lake_hrus = subbasin_hrus[subbasin_hrus['HRU_IsLake'] != 1]
                    
                    if len(lake_hrus) == 0:
                        errors.append(f"Lake subbasin {subid} has no lake HRUs")
                    if len(non_lake_hrus) == 0:
                        warnings.append(f"Lake subbasin {subid} has no non-lake HRUs")
                
                metrics['lake_hrus'] = int((df['HRU_IsLake'] == 1).sum())
                metrics['non_lake_hrus'] = int((df['HRU_IsLake'] != 1).sum())
            
            # Duplicate HRU_ID check
            if 'HRU_ID' in df.columns:
                duplicate_ids = df['HRU_ID'].duplicated().sum()
                if duplicate_ids > 0:
                    errors.append(f"{duplicate_ids} duplicate HRU_IDs found")
                metrics['duplicate_hru_ids'] = int(duplicate_ids)
            
            # HRU area validation
            if 'HRU_Area' in df.columns:
                negative_areas = (df['HRU_Area'] <= 0).sum()
                if negative_areas > 0:
                    errors.append(f"{negative_areas} HRUs have non-positive areas")
                metrics['negative_areas'] = int(negative_areas)
                metrics['min_hru_area'] = float(df['HRU_Area'].min())
                metrics['max_hru_area'] = float(df['HRU_Area'].max())
                metrics['mean_hru_area'] = float(df['HRU_Area'].mean())
            
            is_valid = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Error validating enhanced HRU fields: {str(e)}")
            is_valid = False
        
        return ValidationResult(
            is_valid=is_valid,
            validation_type="enhanced_hru_fields",
            file_path=str(abs_path),
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    def validate_lake_connectivity(self, connected_lakes: Union[str, Path], 
                                 non_connected_lakes: Union[str, Path],
                                 all_lakes: Union[str, Path]) -> ValidationResult:
        """
        Validate lake connectivity classification (Step 3.5 gate).
        
        Args:
            connected_lakes: Path to connected lakes shapefile
            non_connected_lakes: Path to non-connected lakes shapefile
            all_lakes: Path to all lakes shapefile
            
        Returns:
            ValidationResult with lake connectivity validation results
        """
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Load all lake files
            connected_gdf = gpd.read_file(self.path_manager.resolve_path(connected_lakes))
            non_connected_gdf = gpd.read_file(self.path_manager.resolve_path(non_connected_lakes))
            all_lakes_gdf = gpd.read_file(self.path_manager.resolve_path(all_lakes))
            
            metrics['connected_lakes'] = len(connected_gdf)
            metrics['non_connected_lakes'] = len(non_connected_gdf)
            metrics['total_input_lakes'] = len(all_lakes_gdf)
            metrics['total_output_lakes'] = len(connected_gdf) + len(non_connected_gdf)
            
            # Completeness gate: connected ⊕ non_connected = all lakes
            if metrics['total_output_lakes'] != metrics['total_input_lakes']:
                errors.append(f"Lake count mismatch: {metrics['total_input_lakes']} input lakes, "
                            f"{metrics['total_output_lakes']} output lakes")
            
            # Uniqueness gate: No lake appears in both outputs
            if 'HyLakeId' in connected_gdf.columns and 'HyLakeId' in non_connected_gdf.columns:
                connected_ids = set(connected_gdf['HyLakeId'].dropna())
                non_connected_ids = set(non_connected_gdf['HyLakeId'].dropna())
                
                overlap = connected_ids.intersection(non_connected_ids)
                if overlap:
                    errors.append(f"{len(overlap)} lakes appear in both connected and non-connected outputs")
                    metrics['overlapping_lakes'] = len(overlap)
                else:
                    metrics['overlapping_lakes'] = 0
            
            # Geometry gate: All geometries are valid
            invalid_connected = (~connected_gdf.geometry.is_valid).sum()
            invalid_non_connected = (~non_connected_gdf.geometry.is_valid).sum()
            
            if invalid_connected > 0:
                errors.append(f"{invalid_connected} invalid geometries in connected lakes")
            if invalid_non_connected > 0:
                errors.append(f"{invalid_non_connected} invalid geometries in non-connected lakes")
            
            metrics['invalid_connected_geoms'] = int(invalid_connected)
            metrics['invalid_non_connected_geoms'] = int(invalid_non_connected)
            
            is_valid = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Error validating lake connectivity: {str(e)}")
            is_valid = False
        
        return ValidationResult(
            is_valid=is_valid,
            validation_type="lake_connectivity",
            file_path=f"{connected_lakes}, {non_connected_lakes}, {all_lakes}",
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    def generate_validation_report(self, results: List[ValidationResult], 
                                 output_file: Optional[Union[str, Path]] = None) -> Path:
        """
        Generate comprehensive validation report from multiple validation results.
        
        Args:
            results: List of ValidationResult objects
            output_file: Optional path for output file (defaults to workspace/validation_report.json)
            
        Returns:
            Path to generated report file
            
        Raises:
            PathResolutionError: If output path cannot be resolved
            FileAccessError: If report cannot be written
        """
        if output_file is None:
            output_file = self.path_manager.workspace_root / "validation_report.json"
        
        abs_output_path = self.path_manager.ensure_file_writable(output_file)
        
        # Compile report data
        report_data = {
            'report_timestamp': datetime.now().isoformat(),
            'total_validations': len(results),
            'passed_validations': sum(1 for r in results if r.is_valid),
            'failed_validations': sum(1 for r in results if not r.is_valid),
            'validation_results': [result.to_dict() for result in results],
            'summary': {
                'validation_types': {},
                'total_errors': 0,
                'total_warnings': 0
            }
        }
        
        # Generate summary statistics
        for result in results:
            val_type = result.validation_type
            if val_type not in report_data['summary']['validation_types']:
                report_data['summary']['validation_types'][val_type] = {
                    'total': 0,
                    'passed': 0,
                    'failed': 0
                }
            
            report_data['summary']['validation_types'][val_type]['total'] += 1
            if result.is_valid:
                report_data['summary']['validation_types'][val_type]['passed'] += 1
            else:
                report_data['summary']['validation_types'][val_type]['failed'] += 1
            
            report_data['summary']['total_errors'] += len(result.errors)
            report_data['summary']['total_warnings'] += len(result.warnings)
        
        # Write report
        try:
            with open(abs_output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated validation report: {abs_output_path}")
            
        except (OSError, IOError, json.JSONEncodeError) as e:
            raise FileAccessError(str(abs_output_path), "write validation report", str(e))
        
        return abs_output_path
    
    def run_comprehensive_validation(self, files_to_validate: Dict[str, Union[str, Path]]) -> List[ValidationResult]:
        """
        Run comprehensive validation on multiple files.
        
        Args:
            files_to_validate: Dict mapping validation types to file paths
                              Supported types: 'hru', 'routing', 'spatial', 'basinmaker_catchments', 
                              'basinmaker_rivers', 'enhanced_hru', 'lake_connectivity'
        
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        for val_type, file_path in files_to_validate.items():
            try:
                if val_type == 'hru':
                    result = self.validate_hru_fields(file_path)
                elif val_type == 'routing':
                    result = self.validate_routing_connectivity(file_path)
                elif val_type == 'spatial':
                    result = self.validate_spatial_data_integrity(file_path)
                elif val_type == 'basinmaker_catchments':
                    result = self.validate_basinmaker_catchments(file_path)
                elif val_type == 'basinmaker_rivers':
                    result = self.validate_basinmaker_rivers(file_path)
                elif val_type == 'enhanced_hru':
                    result = self.validate_enhanced_hru_fields(file_path)
                elif val_type == 'lake_connectivity':
                    # Special case - expects tuple of (connected, non_connected, all_lakes)
                    if isinstance(file_path, (list, tuple)) and len(file_path) == 3:
                        result = self.validate_lake_connectivity(file_path[0], file_path[1], file_path[2])
                    else:
                        logger.error(f"lake_connectivity validation requires tuple of 3 file paths")
                        continue
                else:
                    logger.warning(f"Unknown validation type: {val_type}")
                    continue
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Validation failed for {val_type} file {file_path}: {e}")
                # Create error result
                if val_type == 'lake_connectivity' and isinstance(file_path, (list, tuple)):
                    abs_path_str = f"{file_path[0]}, {file_path[1]}, {file_path[2]}"
                else:
                    abs_path = self.path_manager.resolve_path(file_path)
                    abs_path_str = str(abs_path)
                
                error_result = ValidationResult(
                    is_valid=False,
                    validation_type=val_type,
                    file_path=abs_path_str,
                    errors=[f"Validation failed: {str(e)}"],
                    warnings=[],
                    metrics={},
                    timestamp=datetime.now()
                )
                results.append(error_result)
        
        return results