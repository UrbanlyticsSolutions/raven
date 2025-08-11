#!/usr/bin/env python3
"""
Step 1: Data Preparation for RAVEN Single Outlet Delineation
Downloads DEM, Landcover, and Soil data for watershed analysis with absolute path support
"""

import sys
from pathlib import Path
import argparse
import json
import math
from datetime import datetime
from typing import Dict, Any, Union, Optional

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # Project root
sys.path.append(str(Path(__file__).parent.parent.parent))  # workflows dir

from workflows.base_workflow_step import BaseWorkflowStep
from workflows.steps import DEMClippingStep, LandcoverExtractionStep, SoilExtractionStep
from clients.data_clients.spatial_client import SpatialLayersClient
from infrastructure.path_manager import PathResolutionError, FileAccessError
from infrastructure.configuration_manager import WorkflowConfiguration
from processors.coordinate_system_processor import CoordinateSystemProcessor
# Removed GeospatialFileManager - using ultra-simple data/ folder structure


class Step1DataPreparation(BaseWorkflowStep):
    """Step 1: Download spatial data (DEM, Landcover, Soil) for watershed with absolute path support"""
    
    def __init__(self, workspace_dir: Union[str, Path], config: Optional[WorkflowConfiguration] = None):
        """
        Initialize Step 1 with absolute path infrastructure.
        
        Args:
            workspace_dir: Workspace directory (required, no fallback)
            config: Optional workflow configuration
            
        Raises:
            ValueError: If workspace_dir is not provided
            PathResolutionError: If workspace_dir cannot be resolved
        """
        if not workspace_dir:
            raise ValueError("workspace_dir is required for Step1DataPreparation - no fallback paths allowed")
        
        # Initialize base class with absolute path infrastructure
        super().__init__(workspace_dir, config, 'step1_data_preparation')
        
        # Initialize clients and steps with absolute paths
        self.spatial_client = SpatialLayersClient()
        
        # ULTRA-SIMPLE: All files go to workspace/data/ folder
        self.data_dir = self.path_manager.workspace_root / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Use absolute workspace path for sub-steps
        abs_workspace = str(self.path_manager.workspace_root)
        self.dem_step = DEMClippingStep(workspace_dir=abs_workspace)
        self.landcover_step = LandcoverExtractionStep(workspace_dir=abs_workspace)
        self.soil_step = SoilExtractionStep(workspace_dir=abs_workspace)
        
        # Initialize coordinate system processor for BasinMaker-style CRS standardization
        self.crs_processor = CoordinateSystemProcessor(workspace_dir=self.data_dir)
    
    def get_watershed_extent(self, lat: float, lon: float, buffer_km: float = 2.0) -> Dict[str, Any]:
        """
        Get watershed extent using MGHydro API with explicit error handling.
        No fallback paths - fails explicitly if MGHydro is unavailable.
        """
        try:
            # Get watershed from MGHydro API
            watershed_result = self.spatial_client.get_watershed_from_mghydro(
                lat=lat, 
                lng=lon,
                precision="high"
            )
            
            if watershed_result['success']:
                # Calculate bbox from watershed_geojson
                watershed_geojson = watershed_result.get('watershed_geojson')
                if watershed_geojson and 'features' in watershed_geojson and watershed_geojson['features']:
                    # Extract coordinates from first feature
                    feature = watershed_geojson['features'][0]
                    coords = feature['geometry']['coordinates'][0]  # Get exterior ring
                    
                    # Calculate bbox from coordinates
                    lons = [coord[0] for coord in coords]
                    lats = [coord[1] for coord in coords]
                    watershed_bbox = (min(lons), min(lats), max(lons), max(lats))
                    
                    # Apply buffer
                    lat_buffer = buffer_km / 111.0
                    lon_buffer = buffer_km / (111.0 * abs(math.cos(math.radians(lat))))
                    
                    buffered_bbox = (
                        watershed_bbox[0] - lon_buffer,
                        watershed_bbox[1] - lat_buffer,
                        watershed_bbox[2] + lon_buffer,
                        watershed_bbox[3] + lat_buffer
                    )
                    
                    return {
                        'success': True,
                        'watershed_boundary': watershed_geojson,
                        'original_bbox': watershed_bbox,
                        'buffered_bbox': buffered_bbox,
                        'source': 'mghydro_hydrosheds'
                    }
                else:
                    raise FileAccessError("", "parse watershed data", "Invalid watershed geometry from MGHydro API")
            else:
                raise FileAccessError("", "fetch watershed data", f"MGHydro API failed: {watershed_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            # Explicit error handling - no silent failures or fallbacks
            error_msg = f"MGHydro watershed fetch failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            raise FileAccessError("", "fetch watershed data", error_msg)
    
    def _execute_step(self, latitude: float, longitude: float, buffer_km: float = 2.0, **kwargs) -> Dict[str, Any]:
        """
        Execute Step 1: Download all spatial data with comprehensive tracking and validation.
        No hardcoded fallback paths - requires explicit configuration for all paths.
        """
        print(f"STEP 1: Preparing spatial data for outlet ({latitude}, {longitude})")
        
        # Get parameters from configuration if available
        if self.step_config:
            buffer_km = self.step_config.parameters.get('buffer_km', buffer_km)
            resolution = self.step_config.parameters.get('resolution_m', 30)
            dem_source = self.step_config.parameters.get('dem_source', 'SRTM')
            landcover_source = self.step_config.parameters.get('landcover_source', 'ESA_WorldCover')
            soil_source = self.step_config.parameters.get('soil_source', 'SoilGrids')
        else:
            resolution = 30
            dem_source = 'SRTM'
            landcover_source = 'ESA_WorldCover'
            soil_source = 'SoilGrids'
        
        # Get watershed extent with explicit error handling
        print("Getting watershed extent from MGHydro...")
        try:
            extent_result = self.get_watershed_extent(latitude, longitude, buffer_km)
        except FileAccessError as e:
            return {
                'success': False,
                'error': f"Data download failed: {str(e)} - no fallback provided"
            }
            
        final_bounds = extent_result['buffered_bbox']
        print(f"Processing extent: {final_bounds}")
        
        # Initialize results first
        results = {
            'success': True,
            'extent_info': extent_result,
            'bounds': final_bounds,
            'files': {},
            'outlet_coordinates': [latitude, longitude],
            'data_directory': str(self.data_dir),  # Ultra-simple: just point to data folder
            'geospatial_files_created': []  # Track GeoJSON files created
        }
        
        # Save study area as GeoJSON using file manager
        if extent_result.get('watershed_boundary'):
            try:
                import geopandas as gpd
                from shapely.geometry import shape
                
                # Convert GeoJSON to GeoDataFrame
                watershed_geojson = extent_result['watershed_boundary']
                geometries = [shape(feature['geometry']) for feature in watershed_geojson['features']]
                study_area_gdf = gpd.GeoDataFrame(
                    {
                        'outlet_lat': [latitude],
                        'outlet_lon': [longitude], 
                        'buffer_km': [buffer_km],
                        'source': ['mghydro_hydrosheds'],
                        'created_at': [datetime.now().isoformat()]
                    },
                    geometry=geometries,
                    crs='EPSG:4326'
                )
                
                # Save using file manager
                # Save directly to data/ folder - ULTRA-SIMPLE STRUCTURE  
                study_area_path = self.data_dir / "study_area.geojson"
                study_area_gdf.to_file(study_area_path, driver="GeoJSON")
                results['geospatial_files_created'].append(str(study_area_path))
                print(f"SUCCESS: Study area saved: {study_area_path}")
                
            except Exception as e:
                print(f"Warning: Failed to save study area GeoJSON: {e}")
        
        # Download DEM with metadata tracking
        print("Downloading DEM data...")
        try:
            dem_result = self.dem_step.execute(
                bounds=final_bounds,
                resolution=resolution,
                output_filename="data/dem.tif"  # Save in data directory where Step 2 expects it
            )
            if not dem_result['success']:
                return {
                    'success': False,
                    'error': f"DEM download failed: {dem_result.get('error', 'Unknown error')} - no fallback provided"
                }
            
            # Convert to absolute path and track metadata
            dem_file = self.path_manager.resolve_path(dem_result['dem_file'])
            results['files']['dem'] = str(dem_file)
            
            # Track DEM metadata
            self.file_ops.track_output(
                dem_file,
                source_info={
                    'description': f'Digital Elevation Model from {dem_source}',
                    'source': dem_source,
                    'bounds': final_bounds,
                    'resolution_m': resolution
                },
                processing_step={
                    'step_name': 'dem_download',
                    'timestamp': datetime.now(),
                    'parameters': {
                        'bounds': final_bounds,
                        'resolution': resolution,
                        'source': dem_source
                    },
                    'input_files': [],
                    'output_files': [str(dem_file)],
                    'processing_time_seconds': 0.0,
                    'software_version': 'DEMClippingStep'
                },
                coordinate_system='EPSG:4326',
                spatial_extent={
                    'min_x': final_bounds[0],
                    'min_y': final_bounds[1],
                    'max_x': final_bounds[2],
                    'max_y': final_bounds[3]
                }
            )
            
            print(f"SUCCESS: DEM: {dem_file}")
            
        except Exception as e:
            return {
                'success': False,
                'error': f"DEM download failed: {str(e)} - no fallback provided"
            }
        
        # Download Landcover with metadata tracking
        print("Downloading landcover data...")
        try:
            landcover_result = self.landcover_step.execute(
                bounds=final_bounds,
                output_filename="landcover.tif"
            )
            if not landcover_result['success']:
                return {
                    'success': False,
                    'error': f"Landcover download failed: {landcover_result.get('error', 'Unknown error')} - no fallback provided"
                }
            
            # Convert to absolute path and track metadata
            landcover_file = self.path_manager.resolve_path(landcover_result['landcover_file'])
            results['files']['landcover'] = str(landcover_file)
            
            # Track landcover metadata
            self.file_ops.track_output(
                landcover_file,
                source_info={
                    'description': f'Land cover data from {landcover_source}',
                    'source': landcover_source,
                    'bounds': final_bounds
                },
                processing_step={
                    'step_name': 'landcover_download',
                    'timestamp': datetime.now(),
                    'parameters': {
                        'bounds': final_bounds,
                        'source': landcover_source
                    },
                    'input_files': [],
                    'output_files': [str(landcover_file)],
                    'processing_time_seconds': 0.0,
                    'software_version': 'LandcoverExtractionStep'
                },
                coordinate_system='EPSG:4326',
                spatial_extent={
                    'min_x': final_bounds[0],
                    'min_y': final_bounds[1],
                    'max_x': final_bounds[2],
                    'max_y': final_bounds[3]
                }
            )
            
            print(f"SUCCESS: Landcover: {landcover_file}")
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Landcover download failed: {str(e)} - no fallback provided"
            }
        
        # Download Soil with metadata tracking
        print("Downloading soil data...")
        try:
            soil_result = self.soil_step.execute(
                bounds=final_bounds,
                output_filename="soil.tif"
            )
            if not soil_result['success']:
                return {
                    'success': False,
                    'error': f"Soil download failed: {soil_result.get('error', 'Unknown error')} - no fallback provided"
                }
            
            # Convert to absolute path and track metadata
            soil_file = self.path_manager.resolve_path(soil_result['soil_file'])
            results['files']['soil'] = str(soil_file)
            
            # Track soil metadata
            self.file_ops.track_output(
                soil_file,
                source_info={
                    'description': f'Soil data from {soil_source}',
                    'source': soil_source,
                    'bounds': final_bounds
                },
                processing_step={
                    'step_name': 'soil_download',
                    'timestamp': datetime.now(),
                    'parameters': {
                        'bounds': final_bounds,
                        'source': soil_source
                    },
                    'input_files': [],
                    'output_files': [str(soil_file)],
                    'processing_time_seconds': 0.0,
                    'software_version': 'SoilExtractionStep'
                },
                coordinate_system='EPSG:4326',
                spatial_extent={
                    'min_x': final_bounds[0],
                    'min_y': final_bounds[1],
                    'max_x': final_bounds[2],
                    'max_y': final_bounds[3]
                }
            )
            
            print(f"SUCCESS: Soil: {soil_file}")
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Soil download failed: {str(e)} - no fallback provided"
            }
        
        # COORDINATE SYSTEM STANDARDIZATION (BasinMaker-style)
        print("Standardizing coordinate systems...")
        try:
            crs_standardization = self.crs_processor.standardize_coordinate_systems_for_step1(
                dem_path=Path(results['files']['dem']),
                landcover_path=Path(results['files']['landcover']) if 'landcover' in results['files'] else None,
                soil_path=Path(results['files']['soil']) if 'soil' in results['files'] else None,
                outlet_coords=(longitude, latitude)  # For optimal CRS selection
            )
            
            if crs_standardization['success']:
                # Update results with standardized files
                results['coordinate_system_standardization'] = crs_standardization
                results['processing_crs'] = crs_standardization['processing_crs']
                
                # Replace original files with standardized versions
                standardized_files = crs_standardization['reprojected_files']
                for data_type, standardized_path in standardized_files.items():
                    if data_type in results['files']:
                        results['files'][f'{data_type}_original'] = results['files'][data_type]
                        results['files'][data_type] = standardized_path
                        print(f"SUCCESS: {data_type.upper()} standardized to {crs_standardization['processing_crs']}")
                
                # Validate coordinate system standardization
                validation = self.crs_processor.validate_coordinate_system_standardization(crs_standardization)
                results['crs_validation'] = validation
                
                if not validation['success']:
                    print(f"WARNING: CRS standardization validation failed: {validation['errors']}")
                else:
                    print(f"SUCCESS: Coordinate systems standardized to {crs_standardization['processing_crs']}")
                    
            else:
                print(f"WARNING: Coordinate system standardization failed: {crs_standardization.get('error', 'Unknown error')}")
                results['coordinate_system_standardization'] = crs_standardization
                
        except Exception as e:
            print(f"WARNING: Coordinate system standardization failed: {str(e)}")
            results['coordinate_system_standardization'] = {
                'success': False,
                'error': str(e)
            }
        
        # Update results with standardized file structure
        # ULTRA-SIMPLE: Just save direct results - no file manager complexity
        self.save_results(results, "step1_results.json")
        
        print(f"STEP 1 COMPLETE: All files saved to data/ folder")
        print(f"Data directory: {self.data_dir}")
        print(f"Files created: {len(results['geospatial_files_created'])}")
        if 'processing_crs' in results:
            print(f"Processing CRS: {results['processing_crs']}")
        return results


if __name__ == "__main__":
    import math
    
    parser = argparse.ArgumentParser(description='Step 1: Data Preparation')
    parser.add_argument('latitude', type=float, help='Outlet latitude')
    parser.add_argument('longitude', type=float, help='Outlet longitude')
    parser.add_argument('--buffer-km', type=float, default=2.0, help='Buffer distance in km')
    parser.add_argument('--workspace-dir', type=str, help='Workspace directory')
    
    args = parser.parse_args()
    
    step1 = Step1DataPreparation(workspace_dir=args.workspace_dir)
    results = step1.execute(latitude=args.latitude, longitude=args.longitude, buffer_km=args.buffer_km)
    
    if results['success']:
        print("SUCCESS: Step 1 data preparation completed")
        print(f"Files: {len(results['files'])} datasets downloaded")
    else:
        print(f"FAILED: {results['error']}")
        sys.exit(1)