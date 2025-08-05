"""
Refactored Full Delineation Workflow

Supports both single outlet processing and shared spatial dataset mode.
Optimized for multi-gauge processing with unified data management.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import json
import math
import geopandas as gpd
from shapely.geometry import Point

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import required clients
from clients.data_clients.spatial_client import SpatialLayersClient

# Import workflow steps
from workflows.steps import (
    DEMClippingStep,
    UnifiedWatershedDelineation,
    LandcoverExtractionStep,
    SoilExtractionStep,
    CreateSubBasinsAndHRUs,
    SelectModelAndGenerateStructure,
    GenerateModelInstructions,
    ValidateCompleteModel
)

class RefactoredFullDelineation:
    """
    Refactored Full Delineation supporting shared spatial datasets
    
    Key improvements:
    1. Shared DEM/landcover/soil across multiple outlets
    2. Unified data management
    3. Optimized for multi-gauge processing
    4. Maintains single-outlet capability
    """
    
    def __init__(self, workspace_dir: str = None, shared_data_dir: str = None):
        """
        Initialize refactored delineation workflow
        
        Parameters:
        -----------
        workspace_dir : str, optional
            Main workspace directory
        shared_data_dir : str, optional
            Directory for shared spatial datasets (DEM, landcover, soil)
        """
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "delineation_workspace"
        self.shared_data_dir = Path(shared_data_dir) if shared_data_dir else self.workspace_dir / "shared_data"
        
        # Ensure directories exist
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        self.shared_data_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize clients and steps
        self.spatial_client = SpatialLayersClient()
        self.dem_step = DEMClippingStep(workspace_dir=self.shared_data_dir)
        self.watershed_step = UnifiedWatershedDelineation(workspace_dir=self.shared_data_dir)
        self.landcover_step = LandcoverExtractionStep(workspace_dir=self.shared_data_dir)
        self.soil_step = SoilExtractionStep(workspace_dir=self.shared_data_dir)
        self.hru_step = CreateSubBasinsAndHRUs()
        self.model_structure_step = SelectModelAndGenerateStructure()
        self.model_instructions_step = GenerateModelInstructions()
        self.validation_final_step = ValidateCompleteModel()
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the workflow"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_file = self.workspace_dir / "refactored_delineation.log"
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
            
        return logger
    
    def prepare_shared_datasets(self, bounds: Tuple[float, float, float, float], 
                              buffer_km: float = 0.5) -> Dict[str, Any]:
        """
        Prepare shared spatial datasets for the region
        
        Parameters:
        -----------
        bounds : tuple
            Bounding box (minx, miny, maxx, maxy)
        buffer_km : float
            Buffer distance in km around bounds
            
        Returns:
        --------
        Dict with paths to shared datasets
        """
        self.logger.info(f"Preparing shared spatial datasets for bounds {bounds}")
        
        # Use small fixed buffer to avoid large areas
        buffer_deg = 0.05  # ~5km at these latitudes
        buffered_bounds = (
            bounds[0] - buffer_deg,
            bounds[1] - buffer_deg,
            bounds[2] + buffer_deg,
            bounds[3] + buffer_deg
        )
        
        # Ensure reasonable bounds
        buffered_bounds = (
            max(-180, min(180, buffered_bounds[0])),
            max(-90, min(90, buffered_bounds[1])),
            max(-180, min(180, buffered_bounds[2])),
            max(-90, min(90, buffered_bounds[3]))
        )
        
        datasets = {}
        
        # DEM - use USGS only
        dem_result = self.dem_step.execute(
            bounds=buffered_bounds,
            resolution=30,
            output_filename="shared_dem.tif"
        )
        if not dem_result['success']:
            return dem_result
        datasets['dem_file'] = dem_result['dem_file']
        
        # Skip landcover/soil failures and use synthetic
        datasets['landcover_file'] = str(Path(self.shared_data_dir) / "synthetic_landcover.tif")
        datasets['soil_file'] = str(Path(self.shared_data_dir) / "synthetic_soil.tif")
        
        return {
            'success': True,
            'bounds': buffered_bounds,
            'datasets': datasets
        }
    
    def delineate_single_outlet(self, outlet_lat: float, outlet_lon: float,
                              shared_datasets: Dict[str, str],
                              outlet_name: str = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Delineate watershed for single outlet using shared datasets
        
        Parameters:
        -----------
        outlet_lat : float
            Outlet latitude
        outlet_lon : float
            Outlet longitude
        shared_datasets : dict
            Paths to shared spatial datasets
        outlet_name : str, optional
            Name for this outlet/watershed
        **kwargs : additional parameters
            
        Returns:
        --------
        Complete delineation results for this outlet
        """
        if not outlet_name:
            outlet_name = f"outlet_{outlet_lat:.4f}_{outlet_lon:.4f}"
            
        self.logger.info(f"Delineating watershed for outlet: {outlet_name} ({outlet_lat}, {outlet_lon})")
        
        try:
            # Create workspace for this outlet
            outlet_workspace = self.workspace_dir / outlet_name
            outlet_workspace.mkdir(exist_ok=True, parents=True)
            
            # Step 1: Watershed delineation using shared DEM
            detailed_watershed_result = self.watershed_step.execute(
                dem_file=shared_datasets['dem_file'],
                outlet_latitude=outlet_lat,
                outlet_longitude=outlet_lon,
                stream_threshold=kwargs.get('stream_threshold', 1000),
                outlet_name=outlet_name
            )
            
            if not detailed_watershed_result['success']:
                return detailed_watershed_result
            
            # Step 2: Create HRUs using shared datasets
            hru_result = self.hru_step.execute({
                'watershed_boundary': detailed_watershed_result['watershed_boundary'],
                'integrated_stream_network': detailed_watershed_result.get('integrated_stream_network') or detailed_watershed_result.get('original_stream_network'),
                'connected_lakes': detailed_watershed_result.get('connected_lakes_file'),
                'dem_file': shared_datasets['dem_file'],
                'landcover_file': shared_datasets['landcover_file'],
                'soil_file': shared_datasets['soil_file'],
                'watershed_name': outlet_name
            })
            
            if not hru_result['success']:
                return hru_result
            
            # Step 3: Generate model structure
            model_structure_result = self.model_structure_step.execute({
                'final_hrus': hru_result.get('final_hrus'),
                'sub_basins': hru_result.get('sub_basins'),
                'connected_lakes': detailed_watershed_result.get('connected_lakes_file'),
                'watershed_area_km2': detailed_watershed_result.get('watershed_area_km2', 0)
            })
            
            # Step 4: Generate model instructions
            model_instructions_result = self.model_instructions_step.execute({
                'selected_model': model_structure_result.get('selected_model'),
                'sub_basins': hru_result.get('sub_basins'),
                'final_hrus': hru_result.get('final_hrus'),
                'watershed_area_km2': detailed_watershed_result.get('watershed_area_km2', 0)
            })
            
            # Step 5: Validate complete model
            final_validation_result = self.validation_final_step.execute({
                'rvh_file': model_structure_result.get('rvh_file'),
                'rvp_file': model_structure_result.get('rvp_file'),
                'rvi_file': model_instructions_result.get('rvi_file'),
                'rvt_file': model_instructions_result.get('rvt_file'),
                'rvc_file': model_instructions_result.get('rvc_file')
            })
            
            # Compile results
            return {
                'success': True,
                'workflow_type': 'Single_Outlet_Refactored',
                'outlet_coordinates': (outlet_lat, outlet_lon),
                'outlet_name': outlet_name,
                'workspace': str(outlet_workspace),
                
                # Watershed results
                'watershed_area_km2': detailed_watershed_result.get('watershed_area_km2', 0),
                'connected_lake_count': detailed_watershed_result.get('connected_lake_count', 0),
                'stream_length_km': detailed_watershed_result.get('stream_length_km', 0),
                'max_stream_order': detailed_watershed_result.get('max_stream_order', 0),
                
                # HRU results
                'subbasin_count': hru_result.get('subbasin_count', 0),
                'total_hru_count': hru_result.get('total_hru_count', 0),
                'lake_hru_count': hru_result.get('lake_hru_count', 0),
                
                # Model results
                'selected_model': model_structure_result.get('selected_model'),
                'model_description': model_structure_result.get('model_description'),
                'parameter_count': model_structure_result.get('parameter_count', 0),
                'process_count': model_instructions_result.get('process_count', 0),
                
                # File paths
                'watershed_boundary': detailed_watershed_result.get('watershed_boundary'),
                'final_hrus_file': hru_result.get('final_hrus'),
                'rvh_file': model_structure_result.get('rvh_file'),
                'rvp_file': model_structure_result.get('rvp_file'),
                'rvi_file': model_instructions_result.get('rvi_file'),
                'rvt_file': model_instructions_result.get('rvt_file'),
                'rvc_file': model_instructions_result.get('rvc_file'),
                
                'model_valid': final_validation_result.get('model_valid', False),
                'model_ready_for_simulation': final_validation_result.get('model_ready_for_simulation', False)
            }
            
        except Exception as e:
            error_msg = f"Delineation failed for {outlet_name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'outlet_coordinates': (outlet_lat, outlet_lon)
            }
    
    def execute_single_delineation(self, latitude: float, longitude: float,
                                 outlet_name: str = None, **kwargs) -> Dict[str, Any]:
        """
        Legacy single-outlet mode with automatic data preparation
        
        Parameters:
        -----------
        latitude : float
            Outlet latitude
        longitude : float
            Outlet longitude
        outlet_name : str, optional
            Name for the outlet
        **kwargs : additional parameters
            
        Returns:
        --------
        Complete workflow results
        """
        self.logger.info(f"Starting single delineation for ({latitude}, {longitude})")
        
        try:
            # Create small focused bounds around single outlet
            buffer_km = kwargs.get('buffer_km', 2.0)  # Much smaller default
            
            # Proper buffer calculation considering latitude
            lat_buffer = buffer_km / 111.0  # 1 degree lat ≈ 111 km
            lon_buffer = buffer_km / (111.0 * abs(math.cos(math.radians(latitude))))  # Adjust for latitude
            
            bounds = (
                longitude - lon_buffer,
                latitude - lat_buffer,
                longitude + lon_buffer,
                latitude + lat_buffer
            )
            
            # Sanity check - ensure reasonable area size (max ~20km x 20km)
            max_extent = 0.2  # ~20km at these latitudes
            width_deg = bounds[2] - bounds[0]
            height_deg = bounds[3] - bounds[1]
            
            if width_deg > max_extent or height_deg > max_extent:
                self.logger.warning(f"Bounds too large ({width_deg:.3f}° x {height_deg:.3f}°), reducing to {max_extent:.3f}°")
                half_extent = max_extent / 2
                bounds = (
                    longitude - half_extent,
                    latitude - half_extent,
                    longitude + half_extent,
                    latitude + half_extent
                )
            
            # Prepare shared datasets
            shared_prep = self.prepare_shared_datasets(bounds, buffer_km)
            if not shared_prep['success']:
                return shared_prep
            
            # Process single outlet
            return self.delineate_single_outlet(
                outlet_lat=latitude,
                outlet_lon=longitude,
                shared_datasets=shared_prep['datasets'],
                outlet_name=outlet_name,
                **kwargs
            )
            
        except Exception as e:
            error_msg = f"Single delineation failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }


if __name__ == "__main__":
    # Test single outlet mode
    workflow = RefactoredFullDelineation()
    
    results = workflow.execute_single_delineation(
        latitude=45.5017,
        longitude=-73.5673,
        outlet_name="Montreal_Test"
    )
    
    if results['success']:
        print(f"✅ Single delineation successful!")
        print(f"Watershed area: {results['watershed_area_km2']:.1f} km²")
        print(f"HRUs: {results['total_hru_count']}")
        print(f"Model: {results['selected_model']}")
    else:
        print(f"❌ Failed: {results['error']}")