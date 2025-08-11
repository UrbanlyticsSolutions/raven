"""
Lake Connectivity Step for BasinMaker Workflow Integration

This module implements Step 3.5: Lake Connectivity Analysis
Classifies lakes as connected or non-connected based on spatial intersection with river network.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional
import logging
import geopandas as gpd
import pandas as pd
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from workflows.base_workflow_step import BaseWorkflowStep
from infrastructure.qaqc_validator import ValidationResult

logger = logging.getLogger(__name__)


class LakeConnectivityStep(BaseWorkflowStep):
    """
    Step 3.5: Lake Connectivity Analysis
    
    Classifies lakes into connected and non-connected categories based on spatial
    intersection analysis with the river network. This step prepares lake data
    for BasinMaker integration in Step 4.
    
    Requirements addressed: 2.4, 8.5
    """
    
    def __init__(self, workspace_dir: Union[str, Path], config=None):
        """
        Initialize Lake Connectivity Step.
        
        Args:
            workspace_dir: Workspace directory for this step
            config: Optional workflow configuration
        """
        super().__init__(workspace_dir, config, step_name="lake_connectivity")
        
        # Configuration parameters
        self.buffer_distance = 0.001  # degrees (~100m buffer for intersection analysis)
        if self.step_config:
            self.buffer_distance = self.step_config.get('buffer_distance', self.buffer_distance)
    
    def _execute_step(self, **kwargs) -> Dict[str, Any]:
        """
        Execute lake connectivity analysis.
        
        Args:
            lakes_file: Path to lakes_all.shp from Step 3
            rivers_file: Path to river_without_merging_lakes.shp from Step 3
            
        Returns:
            Dictionary with execution results including connected/non-connected lake files
        """
        try:
            # Validate required inputs
            lakes_file = kwargs.get('lakes_file')
            rivers_file = kwargs.get('rivers_file')
            
            if not lakes_file or not rivers_file:
                return {
                    'success': False,
                    'error': 'Required inputs missing: lakes_file and rivers_file must be provided'
                }
            
            # Resolve and validate input paths
            lakes_path = self.file_ops.read_file(lakes_file, must_exist=True)
            rivers_path = self.file_ops.read_file(rivers_file, must_exist=True)
            
            logger.info(f"Starting lake connectivity analysis")
            logger.info(f"Lakes file: {lakes_path}")
            logger.info(f"Rivers file: {rivers_path}")
            
            # Load spatial data
            lakes_gdf = gpd.read_file(lakes_path)
            rivers_gdf = gpd.read_file(rivers_path)
            
            logger.info(f"Loaded {len(lakes_gdf)} lakes and {len(rivers_gdf)} river segments")
            
            # Ensure consistent CRS
            if lakes_gdf.crs != rivers_gdf.crs:
                logger.info(f"Reprojecting lakes from {lakes_gdf.crs} to {rivers_gdf.crs}")
                lakes_gdf = lakes_gdf.to_crs(rivers_gdf.crs)
            
            # Perform connectivity analysis
            connected_lakes, non_connected_lakes = self._classify_lake_connectivity(
                lakes_gdf, rivers_gdf
            )
            
            # Prepare output files
            output_files = {}
            
            # Save connected lakes
            if len(connected_lakes) > 0:
                connected_file = self.path_manager.workspace_root / "sl_connected_lake.shp"
                connected_path = self.file_ops.write_file(
                    connected_file,
                    source_info={'description': 'Connected lakes from connectivity analysis'}
                )
                
                connected_gdf = gpd.GeoDataFrame(connected_lakes, crs=lakes_gdf.crs)
                connected_gdf.to_file(connected_path)
                
                # Track output metadata
                self.file_ops.track_output(
                    connected_path,
                    source_info={'description': 'Lakes connected to river network'},
                    processing_step={
                        'step_name': self.step_name,
                        'timestamp': datetime.now(),
                        'parameters': {'buffer_distance': self.buffer_distance},
                        'input_files': [str(lakes_path), str(rivers_path)],
                        'output_files': [str(connected_path)],
                        'processing_time_seconds': 0.0,
                        'software_version': 'LakeConnectivityStep'
                    },
                    coordinate_system=str(lakes_gdf.crs),
                    spatial_extent=self._get_spatial_extent(connected_gdf)
                )
                
                output_files['sl_connected_lake'] = str(connected_path)
                logger.info(f"Saved {len(connected_lakes)} connected lakes to: {connected_path}")
            else:
                logger.info("No connected lakes found")
                output_files['sl_connected_lake'] = None
            
            # Save non-connected lakes
            if len(non_connected_lakes) > 0:
                non_connected_file = self.path_manager.workspace_root / "sl_non_connected_lake.shp"
                non_connected_path = self.file_ops.write_file(
                    non_connected_file,
                    source_info={'description': 'Non-connected lakes from connectivity analysis'}
                )
                
                non_connected_gdf = gpd.GeoDataFrame(non_connected_lakes, crs=lakes_gdf.crs)
                non_connected_gdf.to_file(non_connected_path)
                
                # Track output metadata
                self.file_ops.track_output(
                    non_connected_path,
                    source_info={'description': 'Lakes not connected to river network'},
                    processing_step={
                        'step_name': self.step_name,
                        'timestamp': datetime.now(),
                        'parameters': {'buffer_distance': self.buffer_distance},
                        'input_files': [str(lakes_path), str(rivers_path)],
                        'output_files': [str(non_connected_path)],
                        'processing_time_seconds': 0.0,
                        'software_version': 'LakeConnectivityStep'
                    },
                    coordinate_system=str(lakes_gdf.crs),
                    spatial_extent=self._get_spatial_extent(non_connected_gdf)
                )
                
                output_files['sl_non_connected_lake'] = str(non_connected_path)
                logger.info(f"Saved {len(non_connected_lakes)} non-connected lakes to: {non_connected_path}")
            else:
                logger.info("No non-connected lakes found")
                output_files['sl_non_connected_lake'] = None
            
            # Perform validation gates
            validation_results = self._validate_connectivity_results(
                lakes_gdf, connected_lakes, non_connected_lakes, output_files
            )
            
            # Check if validation passed
            failed_validations = [r for r in validation_results if not r.is_valid]
            if failed_validations:
                error_messages = []
                for result in failed_validations:
                    error_messages.extend(result.errors)
                
                return {
                    'success': False,
                    'error': f"Validation gates failed: {'; '.join(error_messages)}",
                    'validation_results': [r.to_dict() for r in validation_results],
                    'files': output_files
                }
            
            # Success
            return {
                'success': True,
                'files': output_files,
                'metrics': {
                    'total_lakes': len(lakes_gdf),
                    'connected_lakes': len(connected_lakes),
                    'non_connected_lakes': len(non_connected_lakes),
                    'connectivity_ratio': len(connected_lakes) / len(lakes_gdf) if len(lakes_gdf) > 0 else 0.0
                },
                'custom_validation_results': [r.to_dict() for r in validation_results]
            }
            
        except Exception as e:
            error_msg = f"Lake connectivity analysis failed: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def _classify_lake_connectivity(self, lakes_gdf: gpd.GeoDataFrame, 
                                  rivers_gdf: gpd.GeoDataFrame) -> tuple:
        """
        Classify lakes as connected or non-connected based on spatial intersection.
        
        Args:
            lakes_gdf: GeoDataFrame of lakes
            rivers_gdf: GeoDataFrame of rivers
            
        Returns:
            Tuple of (connected_lakes_list, non_connected_lakes_list)
        """
        logger.info("Performing spatial intersection analysis...")
        
        # Create buffer around rivers for intersection analysis
        river_buffer = rivers_gdf.geometry.buffer(self.buffer_distance).union_all()
        
        connected_lakes = []
        non_connected_lakes = []
        
        for idx, lake in lakes_gdf.iterrows():
            # Check if lake intersects with buffered river network
            if lake.geometry.intersects(river_buffer):
                connected_lakes.append(lake)
            else:
                non_connected_lakes.append(lake)
        
        logger.info(f"Classification complete: {len(connected_lakes)} connected, {len(non_connected_lakes)} non-connected")
        
        return connected_lakes, non_connected_lakes
    
    def _validate_connectivity_results(self, original_lakes: gpd.GeoDataFrame,
                                     connected_lakes: list, non_connected_lakes: list,
                                     output_files: Dict[str, str]) -> list:
        """
        Validate connectivity analysis results with validation gates.
        
        Args:
            original_lakes: Original lakes GeoDataFrame
            connected_lakes: List of connected lake features
            non_connected_lakes: List of non-connected lake features
            output_files: Dictionary of output file paths
            
        Returns:
            List of ValidationResult objects
        """
        validation_results = []
        
        # Gate 1: Completeness Gate - connected ⊕ non_connected = all lakes
        total_classified = len(connected_lakes) + len(non_connected_lakes)
        total_original = len(original_lakes)
        
        completeness_errors = []
        if total_classified != total_original:
            completeness_errors.append(
                f"Lake count mismatch: {total_original} original lakes, "
                f"{total_classified} classified lakes"
            )
        
        completeness_result = ValidationResult(
            is_valid=len(completeness_errors) == 0,
            validation_type="lake_connectivity_completeness",
            file_path=str(self.path_manager.workspace_root),
            errors=completeness_errors,
            warnings=[],
            metrics={
                'original_lake_count': total_original,
                'connected_lake_count': len(connected_lakes),
                'non_connected_lake_count': len(non_connected_lakes),
                'total_classified': total_classified
            },
            timestamp=datetime.now()
        )
        validation_results.append(completeness_result)
        
        # Gate 2: Uniqueness Gate - no lake appears in both categories
        uniqueness_errors = []
        if connected_lakes and non_connected_lakes:
            # Check for duplicate lake IDs if available
            connected_ids = set()
            non_connected_ids = set()
            
            # Try to get lake IDs from various possible field names
            id_fields = ['lake_id', 'LAKE_ID', 'HyLakeId', 'id', 'ID', 'FID']
            
            for lake in connected_lakes:
                for field in id_fields:
                    if field in lake.index and pd.notna(lake[field]):
                        connected_ids.add(lake[field])
                        break
            
            for lake in non_connected_lakes:
                for field in id_fields:
                    if field in lake.index and pd.notna(lake[field]):
                        non_connected_ids.add(lake[field])
                        break
            
            # Check for overlaps
            overlapping_ids = connected_ids.intersection(non_connected_ids)
            if overlapping_ids:
                uniqueness_errors.append(
                    f"Lakes appear in both connected and non-connected categories: {overlapping_ids}"
                )
        
        uniqueness_result = ValidationResult(
            is_valid=len(uniqueness_errors) == 0,
            validation_type="lake_connectivity_uniqueness",
            file_path=str(self.path_manager.workspace_root),
            errors=uniqueness_errors,
            warnings=[],
            metrics={
                'connected_unique_ids': len(connected_ids) if 'connected_ids' in locals() else 0,
                'non_connected_unique_ids': len(non_connected_ids) if 'non_connected_ids' in locals() else 0,
                'overlapping_ids': len(overlapping_ids) if 'overlapping_ids' in locals() else 0
            },
            timestamp=datetime.now()
        )
        validation_results.append(uniqueness_result)
        
        # Gate 3: Geometry Gate - all lake geometries are valid polygons
        geometry_errors = []
        geometry_warnings = []
        
        all_classified_lakes = connected_lakes + non_connected_lakes
        invalid_geom_count = 0
        empty_geom_count = 0
        
        for lake in all_classified_lakes:
            if not lake.geometry.is_valid:
                invalid_geom_count += 1
            if lake.geometry.is_empty:
                empty_geom_count += 1
        
        if invalid_geom_count > 0:
            geometry_errors.append(f"{invalid_geom_count} lakes have invalid geometries")
        
        if empty_geom_count > 0:
            geometry_warnings.append(f"{empty_geom_count} lakes have empty geometries")
        
        geometry_result = ValidationResult(
            is_valid=len(geometry_errors) == 0,
            validation_type="lake_connectivity_geometry",
            file_path=str(self.path_manager.workspace_root),
            errors=geometry_errors,
            warnings=geometry_warnings,
            metrics={
                'total_classified_lakes': len(all_classified_lakes),
                'invalid_geometries': invalid_geom_count,
                'empty_geometries': empty_geom_count
            },
            timestamp=datetime.now()
        )
        validation_results.append(geometry_result)
        
        # Gate 4: Output File Validation
        for file_key, file_path in output_files.items():
            if file_path is not None:
                # Validate that output files were created and are readable
                file_result = self.validator.validate_spatial_data_integrity(file_path)
                validation_results.append(file_result)
        
        return validation_results
    
    def _get_spatial_extent(self, gdf: gpd.GeoDataFrame) -> Optional[Dict[str, float]]:
        """
        Get spatial extent of GeoDataFrame.
        
        Args:
            gdf: GeoDataFrame to get extent for
            
        Returns:
            Dictionary with spatial extent or None if empty
        """
        if len(gdf) == 0 or gdf.geometry.is_empty.all():
            return None
        
        bounds = gdf.total_bounds
        return {
            'min_x': float(bounds[0]),
            'min_y': float(bounds[1]),
            'max_x': float(bounds[2]),
            'max_y': float(bounds[3])
        }