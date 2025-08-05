"""
Unified Watershed Delineation Step

Complete watershed delineation including:
- Watershed boundary delineation from outlet
- Stream network extraction and ordering
- Lake detection and classification (connected vs non-connected)
- Lake-stream network integration
- Overlay and filtering operations

This step combines Steps 3-4 from the Full Delineation Workflow and uses
processors for all actual processing work.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import logging
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from clients.watershed_clients.watershed import ProfessionalWatershedAnalyzer
from processors.lake_detection import ComprehensiveLakeDetector
from processors.lake_classifier import LakeClassifier  
from processors.lake_integrator import LakeIntegrator
from processors.basic_attributes import BasicAttributesCalculator
from processors.improved_outlet_snapping import ImprovedOutletSnapper

class UnifiedWatershedDelineation:
    """
    Unified watershed delineation step that includes complete watershed analysis
    with integrated lake detection, classification, and stream network integration.
    
    Implements the complete watershed delineation workflow from the Full Delineation
    approach including all BasinMaker lake processing logic.
    """
    
    def __init__(self, workspace_dir: Path = None):
        """
        Initialize unified watershed delineation step
        
        Parameters:
        -----------
        workspace_dir : Path, optional
            Working directory for processing
        """
        self.workspace_dir = workspace_dir or Path.cwd() / "watershed_delineation"
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize watershed analyzer (for basic delineation)
        self.watershed_analyzer = ProfessionalWatershedAnalyzer(work_dir=self.workspace_dir / "analysis")
        
        # Initialize processors
        self.lake_detector = ComprehensiveLakeDetector(workspace_dir=self.workspace_dir / "lake_detection")
        self.lake_classifier = LakeClassifier(workspace_dir=self.workspace_dir / "lake_classification")
        self.lake_integrator = LakeIntegrator(workspace_dir=self.workspace_dir / "lake_integration")
        self.basic_attributes = BasicAttributesCalculator(workspace_dir=self.workspace_dir / "attributes")
        self.outlet_snapper = ImprovedOutletSnapper(workspace_dir=self.workspace_dir / "outlet_snapping")
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the step"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(self.workspace_dir / "watershed_delineation.log")
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def execute(self, dem_file: str, outlet_latitude: float, outlet_longitude: float,
                stream_threshold: int = 1000, mghydro_watershed_file: str = None, **kwargs) -> Dict[str, Any]:
        """
        Execute complete unified watershed delineation
        
        Parameters:
        -----------
        dem_file : str
            Path to DEM file for analysis
        outlet_latitude : float
            Outlet latitude coordinate
        outlet_longitude : float
            Outlet longitude coordinate  
        stream_threshold : int, optional
            Flow accumulation threshold for stream extraction
        **kwargs : additional parameters
            Additional processing parameters
            
        Returns:
        --------
        Dict[str, Any]
            Complete watershed delineation results with integrated lakes
        """
        
        self.logger.info(f"Starting unified watershed delineation for outlet ({outlet_latitude}, {outlet_longitude})")
        
        try:
            # Step 1: Basic watershed and stream delineation with improved snapping
            self.logger.info("Step 1: Delineating watershed boundary and stream network with improved snapping")
            basic_results = self._delineate_basic_watershed_improved(
                dem_file, outlet_latitude, outlet_longitude, stream_threshold, mghydro_watershed_file
            )
            if not basic_results['success']:
                return basic_results
            
            # Step 2: Detect lakes within watershed using processor
            self.logger.info("Step 2: Detecting lakes within watershed")
            
            # Get watershed bounds for lake detection
            import geopandas as gpd
            watershed_gdf = gpd.read_file(basic_results['watershed_file'])
            watershed_bounds = watershed_gdf.total_bounds.tolist()  # [minx, miny, maxx, maxy]
            
            lake_detection_results = self.lake_detector.detect_and_classify_lakes(
                bbox=watershed_bounds,
                min_lake_area_m2=500,
                connected_threshold_km2=0.5,
                non_connected_threshold_km2=1.0,
                streams_shapefile=Path(basic_results['stream_network_file'])
            )
            if not lake_detection_results['success']:
                return lake_detection_results
            
            # Step 3: Use already classified lakes from detection step
            self.logger.info("Step 3: Using classified lakes from detection step")
            lake_classification_results = {
                'success': True,
                'connected_lakes_file': lake_detection_results.get('connected_lakes_file'),
                'non_connected_lakes_file': lake_detection_results.get('non_connected_lakes_file'),
                'all_lakes_file': lake_detection_results.get('all_lakes_file'),
                'connected_count': lake_detection_results.get('connected_count', 0),
                'non_connected_count': lake_detection_results.get('non_connected_count', 0),
                'total_lake_area_km2': lake_detection_results.get('raw_total_area_ha', 0) / 100,  # Convert ha to km²
                'files_created': []
            }
            
            # Step 4: Integrate lakes with watershed using processor
            self.logger.info("Step 4: Integrating lakes into watershed routing")
            integration_results = self.lake_integrator.integrate_lakes_into_watershed(
                watershed_results=basic_results,
                lake_classification_results=lake_classification_results,
                basic_attributes=None  # Will be calculated if needed
            )
            if not integration_results['success']:
                return integration_results
            
            # Step 5: Calculate basic attributes using processor
            self.logger.info("Step 5: Calculating watershed attributes")
            attributes_results = self.basic_attributes.calculate_basic_attributes_from_watershed_results(
                watershed_results=integration_results,
                dem_file=dem_file
            )
            if not attributes_results['success']:
                return attributes_results
            
            # Compile complete results using processor outputs
            complete_results = {
                'success': True,
                'step_type': 'unified_watershed_delineation',
                'outlet_coordinates': (outlet_latitude, outlet_longitude),
                'stream_threshold': stream_threshold,
                'workspace': str(self.workspace_dir),
                
                # Basic watershed results
                'watershed_boundary': basic_results['watershed_file'],
                'original_stream_network': basic_results['stream_network_file'],
                'watershed_area_km2': basic_results.get('watershed_area_km2', 0),
                'stream_length_km': basic_results.get('stream_length_km', 0),
                'max_stream_order': basic_results.get('max_stream_order', 0),
                
                # Lake detection results
                'lakes_detected_file': lake_detection_results.get('lake_shapefile'),
                'lakes_detected_count': lake_detection_results.get('raw_lake_count', 0),
                
                # Lake classification results
                'connected_lakes_file': lake_classification_results.get('connected_lakes_file'),
                'non_connected_lakes_file': lake_classification_results.get('non_connected_lakes_file'),
                'all_lakes_file': lake_classification_results.get('all_lakes_file'),
                'connected_lake_count': lake_classification_results.get('connected_count', 0),
                'non_connected_lake_count': lake_classification_results.get('non_connected_count', 0),
                'total_lake_area_km2': lake_classification_results.get('total_lake_area_km2', 0),
                
                # Integration results
                'integrated_catchments_file': integration_results.get('integrated_catchments_file'),
                'modified_routing_table': integration_results.get('modified_routing_table'),
                'lake_routing_file': integration_results.get('lake_routing_file'),
                'lakes_integrated': integration_results.get('lakes_integrated', 0),
                'subbasins_modified': integration_results.get('subbasins_modified', 0),
                
                # Attributes results
                'attributes_file': attributes_results.get('attributes_file'),
                'summary_statistics': attributes_results.get('summary_statistics', {}),
                
                # All created files
                'files_created': []
            }
            
            # Collect all created files from all processors
            all_files = []
            for result_dict in [basic_results, lake_detection_results, 
                              lake_classification_results, integration_results, attributes_results]:
                if 'files_created' in result_dict:
                    all_files.extend(result_dict['files_created'])
                # Also collect individual file paths
                for key, value in result_dict.items():
                    if key.endswith('_file') and isinstance(value, str) and Path(value).exists():
                        all_files.append(value)
            
            complete_results['files_created'] = list(set(all_files))  # Remove duplicates
            complete_results['files_count'] = len(complete_results['files_created'])
            
            self.logger.info(f"Unified watershed delineation completed successfully")
            self.logger.info(f"Watershed area: {complete_results['watershed_area_km2']:.1f} km²")
            self.logger.info(f"Connected lakes: {complete_results['connected_lake_count']}")
            self.logger.info(f"Subbasins modified: {complete_results['subbasins_modified']}")
            
            return complete_results
            
        except Exception as e:
            error_msg = f"Unified watershed delineation failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'step_type': 'unified_watershed_delineation'
            }
    
    def _delineate_basic_watershed(self, dem_file: str, latitude: float, longitude: float,
                                 stream_threshold: int) -> Dict[str, Any]:
        """Delineate basic watershed boundary and stream network"""
        
        try:
            output_dir = self.workspace_dir / "basic_delineation"
            output_dir.mkdir(exist_ok=True)
            
            # Use professional watershed analyzer for basic delineation
            results = self.watershed_analyzer.analyze_watershed_complete(
                dem_path=Path(dem_file),
                outlet_coords=(longitude, latitude),
                output_dir=output_dir,
                stream_threshold=stream_threshold,
                flow_algorithm='d8',
                output_formats=['geojson', 'shapefile'],
                coordinate_system='EPSG:4326'
            )
            
            if not results['success']:
                return {
                    'success': False,
                    'error': f"Basic watershed delineation failed: {results.get('error', 'Unknown error')}"
                }
            
            # Find watershed and stream files
            watershed_file = None
            stream_file = None
            
            for file_path in results['files_created']:
                path_obj = Path(file_path)
                if 'watershed' in path_obj.stem and file_path.endswith(('.geojson', '.shp')):
                    watershed_file = file_path
                elif 'stream' in path_obj.stem and file_path.endswith(('.geojson', '.shp')):
                    stream_file = file_path
            
            if not (watershed_file and stream_file):
                return {
                    'success': False,
                    'error': 'Required watershed or stream files not found in delineation results'  
                }
            
            # Extract statistics
            metadata = results.get('metadata', {})
            statistics = metadata.get('statistics', {})
            
            return {
                'success': True,
                'watershed_file': watershed_file,
                'stream_network_file': stream_file,
                'files_created': results['files_created'],
                'watershed_area_km2': statistics.get('watershed_area_km2', 0),
                'stream_length_km': statistics.get('total_stream_length_km', 0),
                'max_stream_order': statistics.get('max_stream_order', 0),
                'outlet_snapped': True,
                'processing_metadata': metadata
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Basic watershed delineation failed: {str(e)}"
            }
    
    def _delineate_basic_watershed_improved(self, dem_file: str, latitude: float, longitude: float,
                                          stream_threshold: int, mghydro_watershed_file: str = None) -> Dict[str, Any]:
        """Improved watershed delineation with downstream snapping and dual watersheds"""
        
        try:
            output_dir = self.workspace_dir / "basic_delineation"
            output_dir.mkdir(exist_ok=True)
            
            # First, extract streams using standard method
            self.logger.info("Extracting stream network...")
            stream_results = self.watershed_analyzer.analyze_watershed_complete(
                dem_path=Path(dem_file),
                outlet_coords=(longitude, latitude),
                output_dir=output_dir,
                stream_threshold=stream_threshold,
                flow_algorithm='d8',
                output_formats=['geojson', 'shapefile'],
                coordinate_system='EPSG:4326'
            )
            
            if not stream_results['success']:
                return {
                    'success': False,
                    'error': f"Stream extraction failed: {stream_results.get('error', 'Unknown error')}"
                }
            
            # Find stream and flow files
            streams_file = None
            flow_accum_file = None
            flow_dir_file = None
            
            for file_path in stream_results['files_created']:
                path_obj = Path(file_path)
                if 'stream' in path_obj.stem and file_path.endswith(('.geojson', '.shp')):
                    streams_file = file_path
                elif 'flow_accumulation' in path_obj.name:
                    flow_accum_file = file_path
                elif 'flow_direction' in path_obj.name:
                    flow_dir_file = file_path
            
            if not (streams_file and flow_accum_file):
                return {
                    'success': False,
                    'error': 'Required stream or flow accumulation files not found'
                }
            
            # Apply improved downstream snapping
            self.logger.info("Applying improved downstream snapping...")
            snap_results = self.outlet_snapper.snap_outlet_downstream(
                outlet_coords=(longitude, latitude),
                streams_file=streams_file,
                flow_accum_file=flow_accum_file,
                max_search_distance=1000
            )
            
            if not snap_results['success']:
                self.logger.warning(f"Downstream snapping failed: {snap_results['error']}")
                # Fall back to original coordinates
                snapped_coords = (longitude, latitude)
                snap_distance = 0.0
            else:
                snapped_coords = snap_results['snapped_coords']
                snap_distance = snap_results['snap_distance_m']
                self.logger.info(f"Snapped outlet {snap_distance:.1f}m downstream")
            
            # Try watershed delineation with snapped outlet
            if flow_dir_file and snap_distance > 0:
                self.logger.info("Attempting watershed delineation from snapped outlet...")
                delineated_watershed_file = self._try_watershed_delineation(
                    flow_dir_file, snapped_coords, output_dir
                )
            else:
                delineated_watershed_file = None
                self.logger.info("Using original outlet coordinates")
            
            # Prepare results with both MGHydro and delineated watersheds
            results = {
                'success': True,
                'method': 'improved_downstream_snapping',
                'original_outlet': (longitude, latitude),
                'snapped_outlet': snapped_coords,
                'snap_distance_m': snap_distance,
                
                # Stream network (always available)
                'stream_network_file': streams_file,
                'flow_accumulation_file': flow_accum_file,
                'flow_direction_file': flow_dir_file,
                
                # Watershed files
                'mghydro_watershed_file': mghydro_watershed_file,
                'delineated_watershed_file': delineated_watershed_file,
                
                # Files created
                'files_created': stream_results['files_created'],
                
                # Outlet files
                'original_outlet_file': snap_results.get('original_outlet_file'),
                'snapped_outlet_file': snap_results.get('snapped_outlet_file'),
            }
            
            # Add snapped outlet files to created files
            if snap_results.get('files_created'):
                results['files_created'].extend(snap_results['files_created'])
            
            # Extract statistics
            metadata = stream_results.get('metadata', {})
            statistics = metadata.get('statistics', {})
            
            results.update({
                'watershed_area_km2': statistics.get('watershed_area_km2', 0),
                'stream_length_km': statistics.get('total_stream_length_km', 0),
                'max_stream_order': statistics.get('max_stream_order', 0),
                'processing_metadata': metadata
            })
            
            # If we have MGHydro watershed, load and compare
            if mghydro_watershed_file and Path(mghydro_watershed_file).exists():
                try:
                    import geopandas as gpd
                    mghydro_gdf = gpd.read_file(mghydro_watershed_file)
                    mghydro_area = mghydro_gdf.to_crs('EPSG:3857').area.sum() / 1e6
                    results['mghydro_area_km2'] = mghydro_area
                    self.logger.info(f"MGHydro watershed area: {mghydro_area:.1f} km²")
                except Exception as e:
                    self.logger.warning(f"Could not load MGHydro watershed: {e}")
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Improved watershed delineation failed: {str(e)}"
            }
    
    def _try_watershed_delineation(self, flow_dir_file: str, snapped_coords: Tuple[float, float], 
                                 output_dir: Path) -> Optional[str]:
        """Try to delineate watershed from snapped outlet using WhiteboxTools"""
        
        try:
            import whitebox
            import geopandas as gpd
            from shapely.geometry import Point
            
            wbt = whitebox.WhiteboxTools()
            wbt.set_working_dir(str(output_dir.absolute()))
            wbt.set_verbose_mode(False)
            
            # Create snapped outlet shapefile
            snapped_point = Point(snapped_coords)
            snapped_gdf = gpd.GeoDataFrame(
                [{'id': 1, 'type': 'snapped'}],
                geometry=[snapped_point],
                crs='EPSG:4326'
            )
            
            outlet_file = output_dir / "snapped_outlet_for_delineation.shp"
            snapped_gdf.to_file(outlet_file)
            
            # Copy flow direction to working directory
            import shutil
            local_flow_dir = output_dir / "flow_direction_for_delineation.tif"
            shutil.copy2(flow_dir_file, local_flow_dir)
            
            # Try watershed delineation
            watershed_raster = output_dir / "delineated_watershed.tif"
            
            wbt.watershed(
                str(local_flow_dir.name),
                str(outlet_file.name),
                str(watershed_raster.name)
            )
            
            if watershed_raster.exists():
                # Convert to vector
                watershed_vector = output_dir / "delineated_watershed.shp"
                wbt.raster_to_vector_polygons(
                    str(watershed_raster.name),
                    str(watershed_vector.name)
                )
                
                if watershed_vector.exists():
                    self.logger.info(f"Successfully delineated watershed: {watershed_vector}")
                    return str(watershed_vector)
                else:
                    self.logger.warning("Watershed raster created but vector conversion failed")
            else:
                self.logger.warning("Watershed delineation still failed with snapped outlet")
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Watershed delineation attempt failed: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    step = UnifiedWatershedDelineation()
    
    # Test with a sample DEM file and coordinates
    result = step.execute(
        dem_file="/path/to/dem.tif",
        outlet_latitude=45.5017,
        outlet_longitude=-73.5673,
        stream_threshold=1000
    )
    
    if result['success']:
        print(f"Watershed delineation completed successfully!")
        print(f"Watershed area: {result['watershed_area_km2']:.1f} km²")
        print(f"Connected lakes: {result['connected_lake_count']}")
        print(f"Files created: {result['files_count']}")
    else:
        print(f"Watershed delineation failed: {result['error']}")