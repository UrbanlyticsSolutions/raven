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
from workflows.steps.watershed_mapping_step import WatershedMappingStep

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
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "delineation"
        self.shared_data_dir = Path(shared_data_dir) if shared_data_dir else self.workspace_dir / "processing_files"
        
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
        self.mapping_step = WatershedMappingStep(workspace_dir=self.shared_data_dir)
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the workflow"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_file = self.workspace_dir / "delineation.log"
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
            
        return logger
    
    def get_hydrosheds_extent(self, lat: float, lon: float, buffer_km: float = 2.0) -> Dict[str, Any]:
        """
        Get HydroSheds watershed extent for a point using MGHydro API
        
        Parameters:
        -----------
        lat : float
            Latitude of the point
        lon : float
            Longitude of the point
        buffer_km : float
            Buffer to add around watershed boundary
            
        Returns:
        --------
        Dict with watershed extent information
        """
        try:
            # Get watershed from MGHydro API (HydroSheds-based)
            watershed_result = self.spatial_client.get_watershed_from_mghydro(
                lat=lat, 
                lng=lon,
                precision="high"  # Use high precision for accurate boundaries
            )
            
            if watershed_result['success']:
                # Extract bbox from result
                watershed_bbox = watershed_result.get('bbox', None)
                if watershed_bbox:
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
                        'watershed_boundary': watershed_result.get('watershed_geometry'),
                        'original_bbox': watershed_bbox,
                        'buffered_bbox': buffered_bbox,
                        'source': 'mghydro_hydrosheds'
                    }
        except Exception as e:
            self.logger.warning(f"MGHydro watershed fetch failed: {e}")
        
        # Fallback to point-based bbox
        lat_buffer = buffer_km / 111.0
        lon_buffer = buffer_km / (111.0 * abs(math.cos(math.radians(lat))))
        
        fallback_bbox = (
            lon - lon_buffer,
            lat - lat_buffer,
            lon + lon_buffer,
            lat + lat_buffer
        )
        
        return {
            'success': True,
            'buffered_bbox': fallback_bbox,
            'source': 'point_buffer'
        }
    
    def calculate_unified_extent(self, locations: List[Tuple[float, float]], 
                               buffer_km: float = 2.0) -> Dict[str, Any]:
        """
        Calculate unified extent from multiple locations using HydroSheds watersheds
        
        Parameters:
        -----------
        locations : list
            List of (lat, lon) tuples
        buffer_km : float
            Buffer to add around combined watersheds
            
        Returns:
        --------
        Dict with unified extent information
        """
        all_watersheds = []
        all_bboxes = []
        
        # Get HydroSheds extent for each location with 500ms delay between API calls
        import time
        for i, (lat, lon) in enumerate(locations):
            if i > 0:  # Add delay before each API call (except the first)
                time.sleep(0.5)  # 500ms delay to prevent API overwhelming
            
            extent = self.get_hydrosheds_extent(lat, lon, buffer_km=0.5)  # Small individual buffer
            if extent['success']:
                all_watersheds.append(extent)
                all_bboxes.append(extent['buffered_bbox'])
        
        if not all_bboxes:
            # No watersheds found, use point-based extent
            lats = [loc[0] for loc in locations]
            lons = [loc[1] for loc in locations]
            
            avg_lat = sum(lats) / len(lats)
            lat_buffer = buffer_km / 111.0
            lon_buffer = buffer_km / (111.0 * abs(math.cos(math.radians(avg_lat))))
            
            unified_bbox = (
                min(lons) - lon_buffer,
                min(lats) - lat_buffer,
                max(lons) + lon_buffer,
                max(lats) + lat_buffer
            )
        else:
            # Calculate union of all watersheds
            minx = min(bbox[0] for bbox in all_bboxes)
            miny = min(bbox[1] for bbox in all_bboxes)
            maxx = max(bbox[2] for bbox in all_bboxes)
            maxy = max(bbox[3] for bbox in all_bboxes)
            
            # Apply final buffer to combined extent
            avg_lat = (miny + maxy) / 2
            lat_buffer = buffer_km / 111.0
            lon_buffer = buffer_km / (111.0 * abs(math.cos(math.radians(avg_lat))))
            
            unified_bbox = (
                minx - lon_buffer,
                miny - lat_buffer,
                maxx + lon_buffer,
                maxy + lat_buffer
            )
        
        # Ensure reasonable bounds
        unified_bbox = (
            max(-180, min(180, unified_bbox[0])),
            max(-90, min(90, unified_bbox[1])),
            max(-180, min(180, unified_bbox[2])),
            max(-90, min(90, unified_bbox[3]))
        )
        
        return {
            'success': True,
            'unified_bbox': unified_bbox,
            'individual_watersheds': all_watersheds,
            'watershed_count': len(all_watersheds)
        }
    
    def prepare_shared_datasets(self, bounds: Tuple[float, float, float, float] = None,
                              locations: List[Tuple[float, float]] = None,
                              buffer_km: float = 2.0) -> Dict[str, Any]:
        """
        Prepare shared spatial datasets using HydroSheds extents
        
        Parameters:
        -----------
        bounds : tuple, optional
            Direct bounding box (minx, miny, maxx, maxy)
        locations : list, optional
            List of (lat, lon) tuples to get HydroSheds extents for
        buffer_km : float
            Buffer distance in km around watersheds
            
        Returns:
        --------
        Dict with paths to shared datasets and extent information
        """
        # Determine extent based on inputs
        if locations:
            self.logger.info(f"Preparing shared datasets for {len(locations)} locations using HydroSheds")
            extent_result = self.calculate_unified_extent(locations, buffer_km)
            final_bounds = extent_result['unified_bbox']
            extent_info = extent_result
        elif bounds:
            self.logger.info(f"Preparing shared datasets for provided bounds {bounds}")
            # Apply small buffer to provided bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            lat_buffer = buffer_km / 111.0
            lon_buffer = buffer_km / (111.0 * abs(math.cos(math.radians(center_lat))))
            
            final_bounds = (
                bounds[0] - lon_buffer,
                bounds[1] - lat_buffer,
                bounds[2] + lon_buffer,
                bounds[3] + lat_buffer
            )
            extent_info = {'source': 'provided_bounds'}
        else:
            return {
                'success': False,
                'error': 'Either bounds or locations must be provided'
            }
        
        self.logger.info(f"Final processing extent: {final_bounds}")
        
        datasets = {}
        
        # DEM - use USGS with HydroSheds extent
        dem_result = self.dem_step.execute(
            bounds=final_bounds,
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
            'bounds': final_bounds,
            'datasets': datasets,
            'extent_info': extent_info
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
            # Use shared workspace - no individual outlet folders for flat structure
            outlet_workspace = self.shared_data_dir
            
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
            # Use 'watershed_file' which is the actual key returned by the watershed step
            watershed_boundary = detailed_watershed_result.get('watershed_file')
            if not watershed_boundary:
                error_msg = f"Watershed step missing watershed file. Available keys: {list(detailed_watershed_result.keys())}"
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg
                }
            
            hru_result = self.hru_step.execute({
                'watershed_boundary': watershed_boundary,
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
            
            # Step 6: Create watershed maps and visualizations
            mapping_result = self.mapping_step.execute(
                watershed_results=detailed_watershed_result,
                outlet_name=outlet_name,
                dem_file=shared_datasets['dem_file'],
                subbasins_file=hru_result.get('sub_basins'),
                create_summary=True
            )
            
            if not mapping_result['success']:
                self.logger.warning(f"Mapping step failed: {mapping_result.get('error')}")
                # Continue workflow even if mapping fails
            
            # Generate actual output files
            output_files = {}
            
            # Shapefiles
            if detailed_watershed_result.get('watershed_file'):
                watershed_shp = detailed_watershed_result['watershed_file']
                output_files['watershed'] = watershed_shp
                
            if hru_result.get('sub_basins'):
                subbasins_shp = str(outlet_workspace / 'subbasins.shp')
                if Path(hru_result['sub_basins']).exists():
                    import shutil
                    shutil.copy2(hru_result['sub_basins'], subbasins_shp)
                    shutil.copy2(hru_result['sub_basins'].replace('.shp', '.shx'), 
                               subbasins_shp.replace('.shp', '.shx'))
                    shutil.copy2(hru_result['sub_basins'].replace('.shp', '.dbf'), 
                               subbasins_shp.replace('.shp', '.dbf'))
                output_files['subbasins'] = subbasins_shp
                
            if hru_result.get('final_hrus'):
                hrus_shp = str(outlet_workspace / 'hrus.shp')
                if Path(hru_result['final_hrus']).exists():
                    import shutil
                    shutil.copy2(hru_result['final_hrus'], hrus_shp)
                    shutil.copy2(hru_result['final_hrus'].replace('.shp', '.shx'), 
                               hrus_shp.replace('.shp', '.shx'))
                    shutil.copy2(hru_result['final_hrus'].replace('.shp', '.dbf'), 
                               hrus_shp.replace('.shp', '.dbf'))
                output_files['hrus'] = hrus_shp
            
            # Raster files
            dem_clip = str(outlet_workspace / 'dem.tif')
            import shutil
            shutil.copy2(shared_datasets['dem_file'], dem_clip)
            output_files['dem'] = dem_clip
            
            # RAVEN files
            raven_files = {}
            for key in ['rvh', 'rvp', 'rvi', 'rvt', 'rvc']:
                file_path = locals().get(f'{key}_file')
                if file_path and Path(file_path).exists():
                    raven_files[key] = file_path
                    output_files[key] = file_path
            
            # Add mapping files to output_files if mapping was successful
            if mapping_result and mapping_result.get('success'):
                if mapping_result.get('main_map_file'):
                    output_files['watershed_map'] = mapping_result['main_map_file']
                if mapping_result.get('summary_plot_file'):
                    output_files['summary_plot'] = mapping_result['summary_plot_file']
            
            # Save summary
            summary = {
                'outlet': {'lat': outlet_lat, 'lon': outlet_lon},
                'watershed_area_km2': detailed_watershed_result.get('watershed_area_km2', 0),
                'subbasins': hru_result.get('subbasin_count', 0),
                'hrus': hru_result.get('total_hru_count', 0),
                'lakes': detailed_watershed_result.get('connected_lake_count', 0),
                'model': model_structure_result.get('selected_model'),
                'mapping': {
                    'success': mapping_result.get('success', False) if mapping_result else False,
                    'components_mapped': mapping_result.get('components_mapped', []) if mapping_result else [],
                    'files_created': len(mapping_result.get('files_created', [])) if mapping_result else 0
                }
            }
            
            summary_file = str(outlet_workspace / 'summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            output_files['summary'] = summary_file
            
            return {
                'success': True,
                'outlet': (outlet_lat, outlet_lon),
                'workspace': str(outlet_workspace),
                'files': output_files,
                'summary': summary
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
        Single-outlet delineation using HydroSheds extent
        
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
        self.logger.info(f"Starting single delineation for ({latitude}, {longitude}) using HydroSheds extent")
        
        try:
            # Get HydroSheds extent for this outlet
            buffer_km = kwargs.get('buffer_km', 2.0)
            
            # Prepare shared datasets using HydroSheds extent
            shared_prep = self.prepare_shared_datasets(
                locations=[(latitude, longitude)],
                buffer_km=buffer_km
            )
            
            if not shared_prep['success']:
                return shared_prep
            
            # Log extent source
            if 'extent_info' in shared_prep:
                self.logger.info(f"Using extent from: {shared_prep['extent_info'].get('source', 'unknown')}")
            
            # Process single outlet
            result = self.delineate_single_outlet(
                outlet_lat=latitude,
                outlet_lon=longitude,
                shared_datasets=shared_prep['datasets'],
                outlet_name=outlet_name,
                **kwargs
            )
            
            # Add extent info to result
            if result['success'] and 'extent_info' in shared_prep:
                result['extent_info'] = shared_prep['extent_info']
            
            return result
            
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