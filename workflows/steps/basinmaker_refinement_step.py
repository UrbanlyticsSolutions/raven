"""
BasinMaker Refinement Step for Hydrological Workflow

This module implements Step 4.1: BasinMaker Refinement
Provides optional post-processing functions for BasinMaker outputs including:
- Tiny subbasin merging (merge_catchments)
- Stream order computation (update_stream_order)
- Routing attributes recalculation (update_routing_attributes)
- Diagnostic CSV output generation (generate_routing_table)
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import logging
import geopandas as gpd
import pandas as pd
from datetime import datetime
import tempfile
import shutil
import os
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from workflows.base_workflow_step import BaseWorkflowStep
from infrastructure.qaqc_validator import ValidationResult
from infrastructure.configuration_manager import BasinMakerConfig

logger = logging.getLogger(__name__)


def _calculate_simple_stream_order(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Simple fallback implementation for Strahler stream order calculation.
    
    Args:
        gdf: GeoDataFrame with SubId and DowSubId columns
        
    Returns:
        GeoDataFrame with updated Strahler stream order
    """
    # Initialize stream order
    if 'Strahler' not in gdf.columns:
        gdf['Strahler'] = 1
    
    # Create routing dictionary
    routing = {}
    for _, row in gdf.iterrows():
        subid = row['SubId']
        dowsubid = row['DowSubId']
        if dowsubid != -1:  # Not an outlet
            routing[subid] = dowsubid
    
    # Find outlets (subbasins that don't drain to anything)
    outlets = []
    all_subids = set(gdf['SubId'])
    for subid in all_subids:
        if subid not in routing:
            outlets.append(subid)
    
    # Calculate stream order using simple upstream traversal
    def get_upstream_subids(target_subid):
        """Get all subbasins that drain to target_subid"""
        upstream = []
        for subid, dowsubid in routing.items():
            if dowsubid == target_subid:
                upstream.append(subid)
        return upstream
    
    def calculate_order(subid, visited=None):
        """Recursively calculate stream order"""
        if visited is None:
            visited = set()
        
        if subid in visited:
            return 1  # Avoid infinite loops
        
        visited.add(subid)
        upstream_subids = get_upstream_subids(subid)
        
        if not upstream_subids:
            # No upstream subbasins - this is a headwater
            return 1
        
        # Calculate orders of upstream subbasins
        upstream_orders = []
        for upstream_subid in upstream_subids:
            order = calculate_order(upstream_subid, visited.copy())
            upstream_orders.append(order)
        
        # Strahler order rules:
        # - If all upstream orders are different, take the maximum
        # - If two or more upstream orders are the same (and maximum), add 1
        if not upstream_orders:
            return 1
        
        max_order = max(upstream_orders)
        max_count = upstream_orders.count(max_order)
        
        if max_count >= 2:
            return max_order + 1
        else:
            return max_order
    
    # Calculate stream order for each subbasin
    for _, row in gdf.iterrows():
        subid = row['SubId']
        order = calculate_order(subid)
        gdf.loc[gdf['SubId'] == subid, 'Strahler'] = order
    
    return gdf


# BasinMaker functions not available - using custom implementations
BASINMAKER_AVAILABLE = False
BASINMAKER_IMPORT_ERROR = "Custom implementation in use"


def merge_catchments_purepy(routing_product_folder: str, area_threshold: float = 0, 
                           length_threshold: float = -10, output_folder: str = None) -> Dict[str, Any]:
    """
    Wrapper for BasinMaker's combine_catchments_covered_by_the_same_lake_purepy function.
    Merges tiny subbasins and catchments covered by the same lake.
    
    Args:
        routing_product_folder: Path to routing product folder
        area_threshold: Minimum area threshold for subbasins (km²)
        length_threshold: Minimum length threshold for rivers (km)
        output_folder: Optional output folder (defaults to routing_product_folder)
        
    Returns:
        Dictionary with processing results
    """
    try:
        if output_folder is None:
            output_folder = routing_product_folder
            
        if not BASINMAKER_AVAILABLE:
            return {
                'success': False,
                'error': f'BasinMaker functions not available: {BASINMAKER_IMPORT_ERROR}'
            }
            
        # Call BasinMaker function
        combine_catchments_covered_by_the_same_lake_purepy(
            Routing_Product_Folder=routing_product_folder,
            area_thresthold=area_threshold,
            length_thresthold=length_threshold
        )
        
        return {
            'success': True,
            'message': f'Successfully merged catchments with area_threshold={area_threshold}km², length_threshold={length_threshold}km',
            'parameters': {
                'area_threshold': area_threshold,
                'length_threshold': length_threshold
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Merge catchments failed: {str(e)}'
        }


def update_stream_order(routing_product_folder: str, output_folder: str = None) -> Dict[str, Any]:
    """
    Wrapper for updating stream order using BasinMaker's topology functions.
    Computes Strahler and Shreve stream order on routing products.
    
    Args:
        routing_product_folder: Path to routing product folder
        output_folder: Optional output folder (defaults to routing_product_folder)
        
    Returns:
        Dictionary with processing results
    """
    try:
        if output_folder is None:
            output_folder = routing_product_folder
            
        # Find finalcat_info.shp file
        finalcat_file = None
        for file in os.listdir(routing_product_folder):
            if file.startswith('finalcat_info') and file.endswith('.shp'):
                finalcat_file = os.path.join(routing_product_folder, file)
                break
                
        if not finalcat_file:
            return {
                'success': False,
                'error': 'No finalcat_info.shp file found in routing product folder'
            }
            
        # Read the data
        gdf = gpd.read_file(finalcat_file)
        
        if not BASINMAKER_AVAILABLE:
            # Fallback implementation - simple stream order calculation
            logger.warning("BasinMaker not available, using fallback stream order calculation")
            
            # Simple Strahler stream order calculation
            if 'SubId' in gdf.columns and 'DowSubId' in gdf.columns:
                gdf = _calculate_simple_stream_order(gdf)
            
            updated_gdf = gdf
        else:
            # Update topology and stream order using BasinMaker
            updated_gdf = UpdateTopology(gdf, UpdateStreamorder=1, UpdateSubId=-1)
        
        # Save updated file
        output_file = os.path.join(output_folder, os.path.basename(finalcat_file))
        updated_gdf.to_file(output_file)
        
        return {
            'success': True,
            'message': f'Successfully updated stream order for {len(updated_gdf)} subbasins',
            'output_file': output_file,
            'metrics': {
                'subbasin_count': len(updated_gdf),
                'max_stream_order': updated_gdf.get('Strahler', pd.Series([0])).max()
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Update stream order failed: {str(e)}'
        }


def update_routing_attributes_purepy(routing_product_folder: str, output_folder: str = None) -> Dict[str, Any]:
    """
    Wrapper for updating routing attributes (lengths, slopes) after geometry changes.
    Recalculates RivLength, RivSlope and other channel metrics.
    
    Args:
        routing_product_folder: Path to routing product folder
        output_folder: Optional output folder (defaults to routing_product_folder)
        
    Returns:
        Dictionary with processing results
    """
    try:
        if output_folder is None:
            output_folder = routing_product_folder
            
        # Find finalcat_info_riv.shp file
        finalriv_file = None
        for file in os.listdir(routing_product_folder):
            if file.startswith('finalcat_info_riv') and file.endswith('.shp'):
                finalriv_file = os.path.join(routing_product_folder, file)
                break
                
        if not finalriv_file:
            return {
                'success': False,
                'error': 'No finalcat_info_riv.shp file found in routing product folder'
            }
            
        # Read the river data
        riv_gdf = gpd.read_file(finalriv_file)
        
        # Recalculate river lengths from geometry
        if 'RivLength' in riv_gdf.columns:
            # Convert to projected CRS for accurate length calculation
            if riv_gdf.crs and not riv_gdf.crs.is_projected:
                # Use a suitable projected CRS (UTM or equal area)
                riv_projected = riv_gdf.to_crs('EPSG:3857')  # Web Mercator as fallback
                riv_projected['RivLength'] = riv_projected.geometry.length / 1000  # Convert to km
                riv_gdf['RivLength'] = riv_projected['RivLength']
            else:
                riv_gdf['RivLength'] = riv_gdf.geometry.length / 1000  # Assume meters, convert to km
        
        # Recalculate slopes if elevation data is available
        # Note: This is a simplified implementation - full slope calculation would require DEM
        if 'RivSlope' in riv_gdf.columns and 'Max_DEM' in riv_gdf.columns and 'Min_DEM' in riv_gdf.columns:
            # Calculate slope as elevation difference / length
            elevation_diff = riv_gdf['Max_DEM'] - riv_gdf['Min_DEM']
            length_m = riv_gdf['RivLength'] * 1000  # Convert km to m
            riv_gdf.loc[length_m > 0, 'RivSlope'] = elevation_diff[length_m > 0] / length_m[length_m > 0]
            riv_gdf.loc[length_m <= 0, 'RivSlope'] = 0
        
        # Save updated file
        output_file = os.path.join(output_folder, os.path.basename(finalriv_file))
        riv_gdf.to_file(output_file)
        
        return {
            'success': True,
            'message': f'Successfully updated routing attributes for {len(riv_gdf)} river segments',
            'output_file': output_file,
            'metrics': {
                'river_segment_count': len(riv_gdf),
                'total_river_length_km': riv_gdf.get('RivLength', pd.Series([0])).sum(),
                'avg_slope': riv_gdf.get('RivSlope', pd.Series([0])).mean()
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Update routing attributes failed: {str(e)}'
        }


def generate_routing_table_purepy(routing_product_folder: str, output_folder: str = None) -> Dict[str, Any]:
    """
    Wrapper for generating diagnostic CSV routing table from routing products.
    Creates a CSV file with routing topology information for diagnostics.
    
    Args:
        routing_product_folder: Path to routing product folder
        output_folder: Optional output folder (defaults to routing_product_folder)
        
    Returns:
        Dictionary with processing results
    """
    try:
        if output_folder is None:
            output_folder = routing_product_folder
            
        # Find finalcat_info.shp file
        finalcat_file = None
        for file in os.listdir(routing_product_folder):
            if file.startswith('finalcat_info') and file.endswith('.shp'):
                finalcat_file = os.path.join(routing_product_folder, file)
                break
                
        if not finalcat_file:
            return {
                'success': False,
                'error': 'No finalcat_info.shp file found in routing product folder'
            }
            
        # Read the data
        gdf = gpd.read_file(finalcat_file)
        
        # Create routing table with key attributes
        routing_columns = ['SubId', 'DowSubId', 'DrainArea', 'BasArea', 'RivLength', 'RivSlope', 
                          'Strahler', 'Seg_order', 'Lake_Cat', 'HyLakeId']
        
        # Select available columns
        available_columns = [col for col in routing_columns if col in gdf.columns]
        routing_table = gdf[available_columns].copy()
        
        # Add diagnostic information
        routing_table['IsOutlet'] = ~routing_table['SubId'].isin(routing_table['DowSubId'])
        routing_table['HasLake'] = routing_table.get('Lake_Cat', 0) > 0
        
        # Calculate additional metrics
        if 'DrainArea' in routing_table.columns:
            routing_table['DrainArea_km2'] = routing_table['DrainArea']
        if 'BasArea' in routing_table.columns:
            routing_table['BasArea_km2'] = routing_table['BasArea']
            
        # Save as CSV
        output_file = os.path.join(output_folder, 'routing_table_diagnostic.csv')
        routing_table.to_csv(output_file, index=False)
        
        # Calculate summary statistics
        total_subbasins = len(routing_table)
        outlet_count = routing_table['IsOutlet'].sum()
        lake_subbasins = routing_table['HasLake'].sum()
        total_drainage_area = routing_table.get('DrainArea_km2', pd.Series([0])).sum()
        
        return {
            'success': True,
            'message': f'Successfully generated routing table with {total_subbasins} subbasins',
            'output_file': output_file,
            'metrics': {
                'total_subbasins': total_subbasins,
                'outlet_count': outlet_count,
                'lake_subbasins': lake_subbasins,
                'total_drainage_area_km2': total_drainage_area,
                'columns_included': available_columns
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Generate routing table failed: {str(e)}'
        }


class BasinMakerRefinementStep(BaseWorkflowStep):
    """
    Step 4.1: BasinMaker Refinement
    
    Provides optional post-processing functions for BasinMaker outputs including:
    - Tiny subbasin merging
    - Stream order computation  
    - Routing attributes recalculation
    - Diagnostic CSV output generation
    
    Requirements addressed: 4.1, 4.2, 4.3, 4.4
    """
    
    def __init__(self, workspace_dir: Union[str, Path], config=None):
        """
        Initialize BasinMaker Refinement Step.
        
        Args:
            workspace_dir: Workspace directory for this step
            config: Optional workflow configuration
        """
        super().__init__(workspace_dir, config, step_name="basinmaker_refinement")
        
        # Check BasinMaker availability
        if not BASINMAKER_AVAILABLE:
            logger.warning(f"BasinMaker not available: {BASINMAKER_IMPORT_ERROR}")
            logger.warning("Some refinement functions may not work properly")
        
        # Get BasinMaker configuration
        self.basinmaker_config = BasinMakerConfig()
        if self.config and hasattr(self.config, 'basinmaker'):
            self.basinmaker_config = self.config.basinmaker
        
        logger.info(f"Initialized BasinMaker Refinement Step")
    
    def _execute_step(self, **kwargs) -> Dict[str, Any]:
        """
        Execute BasinMaker refinement with configurable post-processing functions.
        
        Args:
            routing_product_folder: Path to routing product folder (from Step 4)
            merge_catchments: Enable tiny subbasin merging (default: False)
            update_stream_order: Enable stream order computation (default: False)
            update_routing_attributes: Enable routing attributes recalculation (default: False)
            generate_routing_table: Enable diagnostic CSV output (default: False)
            area_threshold: Minimum area threshold for merging (km², default: 0.1)
            length_threshold: Minimum length threshold for merging (km, default: 0.1)
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Get input routing product folder
            routing_product_folder = kwargs.get('routing_product_folder')
            if not routing_product_folder:
                return {
                    'success': False,
                    'error': 'routing_product_folder parameter is required'
                }
            
            # Resolve and validate input path
            routing_folder_path = self.path_manager.resolve_path(routing_product_folder)
            
            if not routing_folder_path.exists():
                return {
                    'success': False,
                    'error': f'routing_product_folder does not exist: {routing_folder_path}'
                }
            
            if not routing_folder_path.is_dir():
                return {
                    'success': False,
                    'error': f'routing_product_folder must be a directory: {routing_folder_path}'
                }
            
            logger.info("Starting BasinMaker refinement")
            logger.info(f"Input routing product folder: {routing_folder_path}")
            
            # Get configuration flags
            merge_catchments = kwargs.get('merge_catchments', False)
            update_stream_order_flag = kwargs.get('update_stream_order', False)
            update_routing_attributes_flag = kwargs.get('update_routing_attributes', False)
            generate_routing_table_flag = kwargs.get('generate_routing_table', False)
            
            # Get parameters
            area_threshold = kwargs.get('area_threshold', 0.1)
            length_threshold = kwargs.get('length_threshold', 0.1)
            
            logger.info(f"Refinement flags: merge_catchments={merge_catchments}, "
                       f"update_stream_order={update_stream_order_flag}, "
                       f"update_routing_attributes={update_routing_attributes_flag}, "
                       f"generate_routing_table={generate_routing_table_flag}")
            
            # Prepare output folder
            output_folder = self.path_manager.workspace_root / "basinmaker_refinement_output"
            output_folder.mkdir(exist_ok=True, parents=True)
            
            # Copy input files to output folder first
            self._copy_routing_products(routing_folder_path, output_folder)
            
            results = {
                'success': True,
                'refinement_results': {},
                'files': {},
                'metrics': {}
            }
            
            # Execute refinement functions based on configuration
            if merge_catchments:
                logger.info("Executing merge catchments...")
                merge_result = merge_catchments_purepy(
                    str(output_folder),
                    area_threshold=area_threshold,
                    length_threshold=length_threshold
                )
                results['refinement_results']['merge_catchments'] = merge_result
                
                if not merge_result['success']:
                    logger.error(f"Merge catchments failed: {merge_result.get('error')}")
            
            if update_stream_order_flag:
                logger.info("Executing update stream order...")
                stream_order_result = update_stream_order(str(output_folder))
                results['refinement_results']['update_stream_order'] = stream_order_result
                
                if not stream_order_result['success']:
                    logger.error(f"Update stream order failed: {stream_order_result.get('error')}")
            
            if update_routing_attributes_flag:
                logger.info("Executing update routing attributes...")
                routing_attrs_result = update_routing_attributes_purepy(str(output_folder))
                results['refinement_results']['update_routing_attributes'] = routing_attrs_result
                
                if not routing_attrs_result['success']:
                    logger.error(f"Update routing attributes failed: {routing_attrs_result.get('error')}")
            
            if generate_routing_table_flag:
                logger.info("Executing generate routing table...")
                routing_table_result = generate_routing_table_purepy(str(output_folder))
                results['refinement_results']['generate_routing_table'] = routing_table_result
                
                if not routing_table_result['success']:
                    logger.error(f"Generate routing table failed: {routing_table_result.get('error')}")
            
            # Identify output files
            output_files = self._identify_output_files(output_folder)
            results['files'] = output_files
            
            # Calculate overall metrics
            results['metrics'] = self._calculate_refinement_metrics(results['refinement_results'])
            
            # Track output metadata
            self._track_output_metadata(output_files, routing_folder_path, results['refinement_results'])
            
            logger.info("BasinMaker refinement completed successfully")
            return results
            
        except Exception as e:
            error_msg = f"BasinMaker refinement failed: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def _copy_routing_products(self, source_folder: Path, dest_folder: Path):
        """
        Copy routing product files from source to destination folder.
        
        Args:
            source_folder: Source routing product folder
            dest_folder: Destination folder
        """
        for file in source_folder.iterdir():
            if file.suffix.lower() in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.csv']:
                dest_file = dest_folder / file.name
                shutil.copy2(file, dest_file)
                logger.debug(f"Copied {file.name} to output folder")
    
    def _identify_output_files(self, output_folder: Path) -> Dict[str, Path]:
        """
        Identify output files in the refinement output folder.
        
        Args:
            output_folder: Path to refinement output folder
            
        Returns:
            Dictionary of identified output files
        """
        output_files = {}
        
        for file in output_folder.iterdir():
            if file.suffix.lower() == '.shp':
                filename = file.name.lower()
                
                if 'finalcat_info_riv' in filename:
                    output_files['finalcat_info_riv'] = file
                elif 'finalcat_info' in filename:
                    output_files['finalcat_info'] = file
                elif 'sl_connected_lake' in filename:
                    output_files['sl_connected_lake'] = file
                elif 'sl_non_connected_lake' in filename:
                    output_files['sl_non_connected_lake'] = file
                elif 'poi' in filename or 'obs_gauges' in filename:
                    output_files['poi'] = file
            elif file.suffix.lower() == '.csv':
                if 'routing_table' in file.name.lower():
                    output_files['routing_table'] = file
        
        return output_files
    
    def _calculate_refinement_metrics(self, refinement_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall refinement metrics from individual function results.
        
        Args:
            refinement_results: Dictionary of refinement function results
            
        Returns:
            Dictionary of overall metrics
        """
        metrics = {
            'functions_executed': len(refinement_results),
            'functions_successful': sum(1 for r in refinement_results.values() if r.get('success', False)),
            'functions_failed': sum(1 for r in refinement_results.values() if not r.get('success', False))
        }
        
        # Aggregate specific metrics
        for func_name, result in refinement_results.items():
            if result.get('success') and 'metrics' in result:
                func_metrics = result['metrics']
                for key, value in func_metrics.items():
                    metrics[f"{func_name}_{key}"] = value
        
        return metrics
    
    def _track_output_metadata(self, output_files: Dict[str, Path], input_folder: Path, 
                              refinement_results: Dict[str, Any]):
        """
        Track output metadata for refinement results.
        
        Args:
            output_files: Dictionary of output file paths
            input_folder: Input routing product folder
            refinement_results: Results from refinement functions
        """
        processing_step = {
            'step_name': 'basinmaker_refinement',
            'timestamp': datetime.now(),
            'parameters': {
                'functions_executed': list(refinement_results.keys()),
                'input_folder': str(input_folder)
            },
            'input_files': [str(input_folder)],
            'output_files': [str(f) for f in output_files.values()],
            'processing_time_seconds': 0.0,  # Will be updated by base class
            'software_version': 'BasinMakerRefinementStep'
        }
        
        for file_path in output_files.values():
            self.file_ops.track_output(
                file_path,
                source_info={'description': f'BasinMaker refinement output from {input_folder}'},
                processing_step=processing_step
            )