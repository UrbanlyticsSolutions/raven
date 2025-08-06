"""
Multi-Gauge Delineation Workflow

Refactored to build on RefactoredFullDelineation for shared spatial datasets.
Uses live ECCC gauge data as outlets with unified data management.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime
import json
import math

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from clients.data_clients.hydrometric_client import HydrometricDataClient
from clients.data_clients.spatial_client import SpatialLayersClient

# Import workflow steps directly
from workflows.steps import (
    DEMClippingStep,
    DelineateWatershedAndStreams,
    LandcoverExtractionStep,
    SoilExtractionStep,
    CreateSubBasinsAndHRUs,
    SelectModelAndGenerateStructure,
    GenerateModelInstructions,
    ValidateCompleteModel,
    ProjectManagementStep
)

class MultiGaugeDelineation:
    """
    Multi-Gauge Delineation using refactored workflow
    
    Key features:
    1. Uses RefactoredFullDelineation for shared dataset efficiency
    2. Single DEM/landcover/soil download for entire region
    3. Processes multiple gauge outlets efficiently
    4. Live ECCC hydrometric station integration
    """
    
    def __init__(self):
        """
        Initialize multi-gauge delineation workflow
        
        Project structure will be created by ProjectManagementStep
        based on project_name argument passed to run() method.
        """
        # Initialize clients
        self.hydro_client = HydrometricDataClient()
        self.spatial_client = SpatialLayersClient()
        
        # Initialize project management step
        self.project_step = ProjectManagementStep()
        
        # Initialize workflow steps directly - will be configured per project
        self.dem_step = None
        self.watershed_step = None
        self.landcover_step = None
        self.soil_step = None
        self.hru_step = None
        self.model_step = SelectModelAndGenerateStructure()
        self.instruction_step = GenerateModelInstructions()
        self.validation_step = ValidateCompleteModel()
        
        # Project structure will be set after project creation
        self.project_structure = None
        self.project_name = None
        self.logger = None
    
    def _ensure_directory(self, file_path):
        """Create directory for file path if it doesn't exist"""
        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)
        
    def _setup_logging(self):
        """Setup logging for the workflow"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_file = self.logs_subdir / f"{self.project_name}.log"
            self._ensure_directory(log_file)
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
            
        return logger
    
    def discover_gauges_in_region(self, bbox: Tuple[float, float, float, float], 
                                 buffer_km: float = 1.0, min_years: int = 10) -> List[Dict[str, Any]]:
        """
        Discover ECCC hydrometric gauges within region
        
        Parameters:
        -----------
        bbox : tuple
            Bounding box (minx, miny, maxx, maxy)
        buffer_km : float
            Buffer distance in km around bbox
            
        Returns:
        --------
        List of validated gauge information dictionaries
        """
        self.logger.info(f"Discovering gauges in bbox {bbox} with {buffer_km}km buffer")
        
        # Create buffered bbox - bbox format is (minlon, minlat, maxlon, maxlat)
        minlon, minlat, maxlon, maxlat = bbox
        buffer_deg = buffer_km / 111.0
        
        buffered_bbox = (
            minlon - buffer_deg,
            minlat - buffer_deg,
            maxlon + buffer_deg,
            maxlat + buffer_deg
        )
        
        # Get hydrometric stations
        stations_data = self.hydro_client.get_hydrometric_stations_for_watershed(
            bbox=buffered_bbox,
            output_path=self.data_dir / "discovered_gauges.geojson"
        )
        
        # Also save to metadata folder with more detailed info
        metadata_file = self.metadata_subdir / "station_discovery_metadata.json"
        self._ensure_directory(metadata_file)
        with open(metadata_file, 'w') as f:
            json.dump({
                'search_bbox': bbox,
                'buffered_bbox': buffered_bbox,
                'buffer_km': buffer_km,
                'discovery_time': datetime.now().isoformat(),
                'total_stations_found': len(stations_data.get('features', [])) if 'features' in stations_data else 0
            }, f, indent=2)
        
        if "error" in stations_data:
            self.logger.error(f"Gauge discovery failed: {stations_data['error']}")
            return []
        
        # Process discovered gauges
        gauges = []
        for feature in stations_data.get("features", []):
            props = feature["properties"]
            coords = feature["geometry"]["coordinates"]
            
            gauge_info = {
                'station_id': props.get("STATION_NUMBER"),
                'station_name': props.get("STATION_NAME"),
                'latitude': coords[1],  # GeoJSON: [lon, lat]
                'longitude': coords[0],
                'drainage_area_km2': props.get("DRAINAGE_AREA_GROSS"),
                'province': props.get("PROV_TERR_STATE_LOC"),
                'status': props.get("STATUS_EN", props.get("STATION_STATUS", "ACTIVE")),
                'first_year': props.get("FIRST_YEAR", 0),
                'last_year': props.get("LAST_YEAR", 9999)
            }
            
            # Calculate years of data
            first_year = gauge_info['first_year'] or 0
            last_year = gauge_info['last_year'] or 9999
            current_year = datetime.now().year
            status = gauge_info.get('status', 'ACTIVE').upper()
            
            # Handle special cases for years calculation
            if first_year > 0 and last_year > 0:
                if last_year == 9999 or last_year > current_year:
                    last_year = current_year  # Station is still active
                years_of_data = last_year - first_year
            else:
                # If year data is missing, check status
                # Discontinued stations might still be useful
                if 'DISCONTIN' in status:
                    years_of_data = -1  # Flag for discontinued with unknown years
                    print(f"    Note: Station {gauge_info['station_id']} is discontinued (assuming sufficient historical data)")
                else:
                    years_of_data = 0  # Unknown years for active station
            
            gauge_info['years_of_data'] = years_of_data
            
            # Filter for stations with valid drainage area
            drainage_ok = gauge_info['drainage_area_km2'] and gauge_info['drainage_area_km2'] > 0
            
            # For year filtering: 
            # - If min_years > 0 and we have year data, apply filter
            # - If station is discontinued (-1), assume it has sufficient historical data
            # - If min_years == 0, skip year filtering
            if min_years > 0 and years_of_data >= 0:
                years_ok = years_of_data >= min_years
            elif years_of_data == -1:  # Discontinued station
                years_ok = True  # Assume discontinued stations have sufficient data
            else:
                years_ok = (min_years == 0)  # Only pass if no year requirement
            
            print(f"    Station {gauge_info['station_id']}: drainage={gauge_info['drainage_area_km2']} km², years={years_of_data if years_of_data >= 0 else 'discontinued'}, status={status}, passes_filter={drainage_ok and years_ok}")
            
            if drainage_ok and years_ok:
                gauges.append(gauge_info)
        
        print(f"Found {len(gauges)} qualified gauges (with drainage area >= 0 km² and >= {min_years} years of data)")
        return gauges
    
    def prepare_shared_datasets(self, locations: List[Tuple[float, float]], buffer_km: float = 2.0) -> Dict[str, Any]:
        """
        Prepare shared spatial datasets for locations
        
        Parameters:
        -----------
        locations : list
            List of (lat, lon) tuples
        buffer_km : float
            Buffer distance in km around locations
            
        Returns:
        --------
        Dict with shared dataset paths and extent information
        """
        try:
            # Simple bounding box approach for shared datasets
            if not locations:
                return {'success': False, 'error': 'No locations provided'}
            
            # Calculate unified bounding box with buffer
            lats = [loc[0] for loc in locations]
            lons = [loc[1] for loc in locations]
            
            buffer_deg = buffer_km / 111.0  # Rough conversion
            bounds = (
                min(lons) - buffer_deg,  # minx
                min(lats) - buffer_deg,  # miny  
                max(lons) + buffer_deg,  # maxx
                max(lats) + buffer_deg   # maxy
            )
            
            # Download shared DEM
            shared_dem_path = self.processing_dir / "shared_dem.tif"
            dem_result = self.spatial_client.get_dem_for_watershed(bounds, shared_dem_path)
            
            if not dem_result['success']:
                return {'success': False, 'error': f"Failed to download shared DEM: {dem_result.get('error')}"}
            
            return {
                'success': True,
                'bounds': bounds,
                'datasets': {
                    'dem_path': str(shared_dem_path),
                    'processing_dir': str(self.processing_dir)
                },
                'extent_info': {
                    'locations_count': len(locations),
                    'buffer_km': buffer_km
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def delineate_watershed_direct(self, outlet_lat: float, outlet_lon: float, 
                                 shared_datasets: Dict[str, str], outlet_name: str = None) -> Dict[str, Any]:
        """
        Delineate watershed using direct step calls
        
        Parameters:
        -----------
        outlet_lat : float
            Outlet latitude
        outlet_lon : float  
            Outlet longitude
        shared_datasets : dict
            Paths to shared datasets
        outlet_name : str, optional
            Name for outlet
            
        Returns:
        --------
        Dict with delineation results
        """
        try:
            # Initialize watershed step with project workspace if not already done
            if self.watershed_step is None:
                # Get project workspace from shared datasets
                processing_dir = Path(shared_datasets.get('processing_dir', Path.cwd()))
                self.watershed_step = DelineateWatershedAndStreams(workspace_dir=processing_dir)
            
            # Execute watershed delineation step directly
            result = self.watershed_step.execute(
                dem_file=shared_datasets.get('dem_path'),
                outlet_latitude=outlet_lat,
                outlet_longitude=outlet_lon,
                outlet_name=outlet_name or f"outlet_{outlet_lat}_{outlet_lon}"
            )
            
            # Check for critical errors that should stop workflow
            if not result.get('success', False):
                error_msg = result.get('error', 'Unknown error')
                print(f"[CRITICAL ERROR] Watershed delineation failed: {error_msg}")
                raise RuntimeError(f"Watershed delineation failed: {error_msg}")
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def prepare_unified_datasets_with_hydrosheds(self, gauges: List[Dict[str, Any]], 
                                               buffer_km: float = 2.0) -> Dict[str, Any]:
        """
        Prepare unified datasets using HydroSheds watershed extents
        
        Parameters:
        -----------
        gauges : list
            List of gauge dictionaries
        buffer_km : float
            Buffer distance in km around watersheds
            
        Returns:
        --------
        Dict with unified extent and prepared datasets
        """
        if not gauges:
            return {
                'success': False,
                'error': 'No gauges provided'
            }
        
        # Extract gauge locations
        locations = [(g['latitude'], g['longitude']) for g in gauges]
        
        print(f"Processing {len(gauges)} gauges with unified datasets...")
        
        # Use direct prepare_shared_datasets method
        shared_prep = self.prepare_shared_datasets(
            locations=locations,
            buffer_km=buffer_km
        )
        
        if not shared_prep['success']:
            self.logger.error(f"Failed to prepare shared datasets: {shared_prep.get('error')}")
            return shared_prep
        
        # Log extent information
        if 'extent_info' in shared_prep:
            extent_info = shared_prep['extent_info']
            self.logger.info(f"Unified extent from {extent_info.get('watershed_count', 0)} HydroSheds watersheds")
            self.logger.info(f"Final bounds: {shared_prep['bounds']}")
        
        return {
            'success': True,
            'bounds': shared_prep['bounds'],
            'datasets': shared_prep['datasets'],
            'extent_info': shared_prep.get('extent_info', {})
        }
    
    def execute_multi_gauge_workflow(self, project_name: str,
                                   bbox: Tuple[float, float, float, float], 
                                   buffer_km: float = 1.0,
                                   min_drainage_area_km2: float = 10.0,
                                   individual_buffer_km: float = 2.0,
                                   min_years: int = 10) -> Dict[str, Any]:
        """
        Execute complete multi-gauge delineation workflow with individual processing
        
        Parameters:
        -----------
        project_name : str
            Name of the project for folder structure creation
        bbox : tuple
            Initial bounding box (minx, miny, maxx, maxy)
        buffer_km : float
            Buffer for gauge discovery
        min_drainage_area_km2 : float
            Minimum drainage area for gauge selection
        individual_buffer_km : float
            Buffer for individual gauge processing (much smaller)
            
        Returns:
        --------
        Complete multi-gauge workflow results
        """
        start_time = datetime.now()
        
        # Step 1: Create project structure
        project_result = self.project_step.execute(project_name)
        if not project_result['success']:
            return {'success': False, 'error': f"Failed to create project structure: {project_result['error']}"}
        
        # Set up project structure for workflow
        self.project_name = project_name
        self.project_structure = project_result['folder_structure']
        
        # Set up directory paths from project structure
        self.data_dir = Path(project_result['folder_structure']['input_data'])
        self.processing_dir = Path(project_result['folder_structure']['processing_files'])
        self.results_dir = Path(project_result['folder_structure']['analysis_results'])
        
        # Set up subdirectory paths
        self.logs_subdir = self.data_dir / "logs"
        self.metadata_subdir = self.data_dir / "metadata" 
        self.reports_subdir = self.results_dir / "reports"
        self.maps_subdir = self.results_dir / "maps"
        self.watersheds_subdir = self.results_dir / "watersheds"
        
        # Set workspace for each workflow step (skip None steps)
        for step in [self.model_step, self.instruction_step, self.validation_step]:
            if step is not None:
                step.workspace_dir = self.processing_dir
        
        # Setup logging after project structure is created
        self.logger = self._setup_logging()
        
        self.logger.info(f"Starting Multi-Gauge Workflow for bbox {bbox}")
        self.logger.info(f"Project '{project_name}' created at {project_result['project_root']}")
        
        try:
            # Step 1: Discover gauges
            all_gauges = self.discover_gauges_in_region(bbox, buffer_km, min_years)
            
            # Filter by minimum drainage area
            valid_gauges = [g for g in all_gauges if g['drainage_area_km2'] >= min_drainage_area_km2]
            
            self.logger.info(f"Processing {len(valid_gauges)} gauges with area >= {min_drainage_area_km2} km2")
            
            if not valid_gauges:
                return {
                    'success': False,
                    'error': 'No valid gauges found in region',
                    'gauges_discovered': len(all_gauges),
                    'gauges_filtered': len(valid_gauges)
                }
            
            # Step 2: Prepare unified datasets using HydroSheds extents
            self.logger.info("Preparing unified datasets using HydroSheds watershed extents")
            
            unified_prep = self.prepare_unified_datasets_with_hydrosheds(
                gauges=valid_gauges,
                buffer_km=individual_buffer_km
            )
            
            if not unified_prep['success']:
                return {
                    'success': False,
                    'error': f"Failed to prepare unified datasets: {unified_prep.get('error')}",
                    'workflow_type': 'Multi_Gauge_HydroSheds'
                }
            
            # Step 3: Initialize results structure
            results = {
                'success': True,
                'workflow_type': 'Multi_Gauge_HydroSheds',
                'bbox': bbox,
                'buffer_km': buffer_km,
                'individual_buffer_km': individual_buffer_km,
                'unified_bounds': unified_prep['bounds'],
                'extent_info': unified_prep.get('extent_info', {}),
                'gauges_discovered': len(all_gauges),
                'gauges_processed': len(valid_gauges),
                'gauge_results': {},
                'summary': {
                    'total_watersheds': 0,
                    'total_area_km2': 0,
                    'completed': 0,
                    'failed': 0,
                    'area_validation': {
                        'within_10_percent': 0,
                        'within_25_percent': 0,
                        'over_25_percent': 0
                    }
                }
            }
            
            # Step 4: Process each gauge using shared datasets
            print(f"Delineating {len(valid_gauges)} watersheds...")
            
            for i, gauge in enumerate(valid_gauges):
                station_id = gauge['station_id']
                outlet_name = f"g{station_id.replace('_', '')}"[:6]
                
                print(f"  [{i+1}/{len(valid_gauges)}] {station_id}")
                
                try:
                    # Use direct watershed delineation
                    delineation_result = self.delineate_watershed_direct(
                        outlet_lat=gauge['latitude'],
                        outlet_lon=gauge['longitude'],
                        shared_datasets=unified_prep['datasets'],
                        outlet_name=outlet_name
                    )
                    
                    # Check for critical failure that should stop entire workflow
                    if not delineation_result.get('success', False):
                        error_msg = delineation_result.get('error', 'Unknown error')
                        print(f"[WORKFLOW STOPPED] Critical error processing {station_id}: {error_msg}")
                        print("Stopping entire workflow due to critical error.")
                        return {
                            'success': False,
                            'error': f"Critical error processing {station_id}: {error_msg}",
                            'gauges_processed': i,
                            'partial_results': summary_info
                        }
                    
                    if delineation_result['success']:
                        # Extract results from nested structure
                        if 'summary' in delineation_result:
                            # New format from delineate_single_outlet
                            watershed_area = delineation_result['summary'].get('watershed_area_km2', 0)
                            hru_count = delineation_result['summary'].get('hrus', 0)
                            model_type = delineation_result['summary'].get('model', 'Unknown')
                        else:
                            # Legacy format
                            watershed_area = delineation_result.get('watershed_area_km2', 0)
                            hru_count = delineation_result.get('total_hru_count', 0)
                            model_type = delineation_result.get('selected_model', 'Unknown')
                        
                        # Add gauge-specific metadata
                        delineation_result['gauge_info'] = gauge
                        
                        # Area validation
                        calculated_area = watershed_area
                        reported_area = float(gauge['drainage_area_km2'])
                        area_diff_percent = abs(calculated_area - reported_area) / reported_area * 100 if reported_area > 0 else 999
                        
                        delineation_result['area_validation'] = {
                            'calculated_area_km2': calculated_area,
                            'reported_area_km2': reported_area,
                            'difference_percent': area_diff_percent,
                            'validation_status': 'GOOD' if area_diff_percent < 10 else 'ACCEPTABLE' if area_diff_percent < 25 else 'REVIEW'
                        }
                        
                        # Update summary
                        results['gauge_results'][station_id] = delineation_result
                        results['summary']['completed'] += 1
                        results['summary']['total_watersheds'] += 1
                        results['summary']['total_area_km2'] += calculated_area
                        
                        # Update validation counts
                        if area_diff_percent < 10:
                            results['summary']['area_validation']['within_10_percent'] += 1
                        elif area_diff_percent < 25:
                            results['summary']['area_validation']['within_25_percent'] += 1
                        else:
                            results['summary']['area_validation']['over_25_percent'] += 1
                        
                        # Organize station outputs into proper folders
                        self._organize_station_outputs(station_id, delineation_result)
                            
                        print(f"    [OK] {calculated_area:.1f} km²")
                            
                    else:
                        results['gauge_results'][station_id] = {
                            'success': False,
                            'gauge': gauge,
                            'error': delineation_result.get('error', 'Processing failed')
                        }
                        results['summary']['failed'] += 1
                        print(f"    [FAIL] {delineation_result.get('error', 'Processing failed')}")
                        
                except Exception as e:
                    results['gauge_results'][station_id] = {
                        'success': False,
                        'gauge': gauge,
                        'error': str(e)
                    }
                    results['summary']['failed'] += 1
                    print(f"    [FAIL] Exception: {str(e)}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() / 60
            results['execution_time_minutes'] = execution_time
            
            # Generate summary report
            print(f"Completed: {results['summary']['completed']}/{len(valid_gauges)} successful in {execution_time:.1f} minutes")
            
            # Save detailed results to reports subfolder
            summary_file = self.reports_subdir / f"{self.project_name}_summary.json"
            self._ensure_directory(summary_file)
            with open(summary_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save human-readable summary report
            text_report = self.generate_summary_report(results)
            report_file = self.reports_subdir / f"{self.project_name}_report.txt"
            self._ensure_directory(report_file)
            with open(report_file, 'w') as f:
                f.write(text_report)
            
            return results
            
        except Exception as e:
            error_msg = f"Multi-Gauge Workflow failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'workflow_type': 'Multi_Gauge_Individual'
            }
    
    def _organize_station_outputs(self, station_id: str, delineation_result: Dict[str, Any]):
        """
        Organize station-specific outputs into proper folders with station ID naming
        
        Parameters:
        -----------
        station_id : str
            Station identifier (e.g., '02AA001')
        delineation_result : dict
            Results from delineate_single_outlet
        """
        import shutil
        
        try:
            # Define source files in processing directory with improved organization
            processing_files = {
                'watershed_boundary.shp': self.watersheds_subdir / f"{station_id}_watershed.shp",
                'outlet_snapped.shp': self.outlets_subdir / f"{station_id}_outlet.shp",
                'lakes.shp': self.lakes_subdir / f"{station_id}_lakes.shp",
                'streams.geojson': self.streams_subdir / f"{station_id}_streams.geojson"
            }
            
            # Move and rename files
            for source_name, target_path in processing_files.items():
                source_path = self.processing_dir / source_name
                
                if source_path.exists():
                    # Ensure target directory exists before copying
                    self._ensure_directory(target_path)
                    
                    # Copy all shapefile components (.shp, .shx, .dbf, .prj, etc.)
                    base_source = source_path.with_suffix('')
                    base_target = target_path.with_suffix('')
                    
                    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                        src_file = base_source.with_suffix(ext)
                        tgt_file = base_target.with_suffix(ext)
                        
                        if src_file.exists():
                            shutil.copy2(src_file, tgt_file)
                            self.logger.info(f"Organized: {src_file.name} -> {tgt_file.name}")
                            
        except Exception as e:
            self.logger.warning(f"File organization failed for {station_id}: {str(e)}")

    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate simplified summary report"""
        if not results['success']:
            return f"[FAILED]: {results['error']}"
        
        summary = results.get('summary', {})
        completed = summary.get('completed', 0)
        failed = summary.get('failed', 0)
        total_area = summary.get('total_area_km2', 0)
        exec_time = results.get('execution_time_minutes', 0)
        
        report = f"""
[SUCCESS] Multi-Gauge Workflow Complete
================================
Processed: {completed} successful, {failed} failed
Total Area: {total_area:.1f} km²
Time: {exec_time:.1f} minutes

Results:"""
        
        for station_id, result in results.get('gauge_results', {}).items():
            if result.get('success'):
                area = result.get('summary', {}).get('watershed_area_km2', 0) or result.get('watershed_area_km2', 0)
                status = "[OK]"
            else:
                area = 0
                status = "[FAIL]"
                
            report += f"\n  {status} {station_id}: {area:.1f} km²"
        
        return report
    
    def generate_project_map(self, results: Dict[str, Any], project_name: str = None) -> str:
        """Generate project map and save as PNG"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            if not results.get('success'):
                return None
            
            # Use instance project name if not provided
            if project_name is None:
                project_name = self.project_name
                
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Get bounds from results
            bounds = results.get('unified_bounds')
            if bounds:
                minx, miny, maxx, maxy = bounds
                
                # Set map extent with buffer
                buffer = 0.05
                ax.set_xlim(minx - buffer, maxx + buffer)
                ax.set_ylim(miny - buffer, maxy + buffer)
                
                # Draw extent rectangle
                rect = patches.Rectangle((minx, miny), maxx-minx, maxy-miny, 
                                       linewidth=2, edgecolor='blue', 
                                       facecolor='lightblue', alpha=0.3)
                ax.add_patch(rect)
            
            # Plot gauge locations
            gauge_results = results.get('gauge_results', {})
            for station_id, result in gauge_results.items():
                if 'gauge_info' in result:
                    gauge = result['gauge_info']
                    lat, lon = gauge['latitude'], gauge['longitude']
                    
                    if result.get('success'):
                        ax.plot(lon, lat, 'go', markersize=8, label=f"[OK] {station_id}")
                    else:
                        ax.plot(lon, lat, 'ro', markersize=8, label=f"[FAIL] {station_id}")
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'{project_name.title()} Multi-Gauge Watershed Analysis')
            ax.grid(True, alpha=0.3)
            
            # Save map
            map_file = self.maps_subdir / f"{self.project_name}_overview_map.png"
            self._ensure_directory(map_file)
            plt.savefig(map_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(map_file)
            
        except Exception as e:
            print(f"Warning: Could not generate map - {e}")
            return None
    
    def test_workflow(self, project_name: str = "TestWorkflow", 
                     bbox: Tuple[float, float, float, float] = (-73.8, 45.4, -73.4, 45.6)) -> Dict[str, Any]:
        """
        Test workflow with small Montreal region
        
        Parameters:
        -----------
        project_name : str
            Name for the test project
        bbox : tuple, optional
            Test region bounding box (much smaller now)
            
        Returns:
        --------
        Test results
        """        
        return self.execute_multi_gauge_workflow(
            project_name=project_name,
            bbox=bbox,
            buffer_km=0.5,  # Small discovery buffer
            min_drainage_area_km2=25.0,
            individual_buffer_km=1.5  # Small individual processing buffer
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Gauge Watershed Delineation Workflow')
    parser.add_argument('--bbox', type=str, help='Bounding box as "minlon,minlat,maxlon,maxlat"')
    parser.add_argument('--buffer', type=float, default=22.0, help='Station discovery buffer in km (default: 22.0)')
    parser.add_argument('--min-drainage', type=float, default=25.0, help='Minimum drainage area in km² (default: 25.0)')
    parser.add_argument('--individual-buffer', type=float, default=2.0, help='Individual processing buffer in km (default: 2.0)')
    parser.add_argument('--min-years', type=int, default=10, help='Minimum years of data required (default: 10)')
    parser.add_argument('--project-name', type=str, default='MultiGauge', help='Project name (default: MultiGauge)')
    
    args = parser.parse_args()
    
    workflow = MultiGaugeDelineation()
    
    if args.bbox:
        # Parse bbox string
        bbox_parts = args.bbox.split(',')
        if len(bbox_parts) != 4:
            print("Error: bbox must be 4 comma-separated values: minlon,minlat,maxlon,maxlat")
            exit(1)
        
        try:
            bbox = tuple(float(x) for x in bbox_parts)
        except ValueError:
            print("Error: bbox values must be numbers")
            exit(1)
        
        print(f"Running Multi-Gauge Workflow: {args.project_name}")
        print(f"Bbox: {bbox}")
        print(f"Station buffer: {args.buffer} km")
        print(f"Min drainage: {args.min_drainage} km²")
        print(f"Min years: {args.min_years}")
        print(f"Individual buffer: {args.individual_buffer} km")
        
        results = workflow.execute_multi_gauge_workflow(
            project_name=args.project_name,
            bbox=bbox,
            buffer_km=args.buffer,
            min_drainage_area_km2=args.min_drainage,
            individual_buffer_km=args.individual_buffer,
            min_years=args.min_years
        )
        
        if results['success']:
            print("\n" + "="*60)
            print("SUCCESS! Multi-Gauge Workflow Complete")
            print("="*60)
            print(workflow.generate_summary_report(results))
        else:
            print(f"\n[FAILED] Workflow failed: {results['error']}")
    else:
        # Run test workflow
        print("Running test workflow (use --bbox to specify custom region)")
        results = workflow.test_workflow(project_name="TestWorkflow")
        
        if results['success']:
            print(workflow.generate_summary_report(results))
        else:
            print(f"Test failed: {results['error']}")