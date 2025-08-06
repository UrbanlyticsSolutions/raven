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
from workflows.refactored_full_delineation import RefactoredFullDelineation

class MultiGaugeDelineation:
    """
    Multi-Gauge Delineation using refactored workflow
    
    Key features:
    1. Uses RefactoredFullDelineation for shared dataset efficiency
    2. Single DEM/landcover/soil download for entire region
    3. Processes multiple gauge outlets efficiently
    4. Live ECCC hydrometric station integration
    """
    
    def __init__(self, project_name: str = None, workspace_dir: str = None):
        """
        Initialize multi-gauge delineation
        
        Parameters:
        -----------
        project_name : str, optional
            Project name for folder structure
        workspace_dir : str, optional
            Main workspace directory
        """
        self.project_name = project_name if project_name else "MultiGauge"
        
        if project_name:
            self.workspace_dir = Path.cwd().parent / self.project_name  # Move outside workflows folder
        else:
            self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd().parent / self.project_name
        
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Create flat 3-folder structure with descriptive names
        self.data_dir = self.workspace_dir / "input_data"        # ECCC stations, USGS DEMs, logs
        self.watersheds_dir = self.workspace_dir / "spatial_outputs" # shapefiles, GeoJSON, TIFF
        self.results_dir = self.workspace_dir / "analysis_results"   # JSON outputs, reports, maps
        # Note: RefactoredFullDelineation will create "processing_files" folder = 4 folders total
        
        # Consolidate maps and logs into other directories
        self.maps_dir = self.results_dir  # Maps go in results
        self.logs_dir = self.data_dir     # Logs go in data
        
        for dir_path in [self.data_dir, self.watersheds_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize clients
        self.hydro_client = HydrometricDataClient()
        self.delineation_engine = RefactoredFullDelineation(
            workspace_dir=str(self.workspace_dir)  # Use main workspace to control structure
        )
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the workflow"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_file = self.logs_dir / f"{self.project_name.lower()}.log"
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
                'status': props.get("STATION_STATUS", "ACTIVE"),
                'first_year': props.get("FIRST_YEAR", 0),
                'last_year': props.get("LAST_YEAR", 9999)
            }
            
            # Filter for stations with valid drainage area (years data not available from ECCC API)
            drainage_ok = gauge_info['drainage_area_km2'] and gauge_info['drainage_area_km2'] > 0
            
            print(f"    Station {gauge_info['station_id']}: drainage={gauge_info['drainage_area_km2']}, passes_filter={drainage_ok}")
            
            if drainage_ok:
                gauge_info['years_of_data'] = "Unknown"  # ECCC API doesn't provide year range
                gauges.append(gauge_info)
        
        print(f"Found {len(gauges)} qualified gauges (with drainage area data)")
        return gauges
    
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
        
        # Use RefactoredFullDelineation's HydroSheds-aware prepare method
        shared_prep = self.delineation_engine.prepare_shared_datasets(
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
    
    def execute_multi_gauge_workflow(self, bbox: Tuple[float, float, float, float], 
                                   buffer_km: float = 1.0,
                                   min_drainage_area_km2: float = 10.0,
                                   individual_buffer_km: float = 2.0,
                                   min_years: int = 10) -> Dict[str, Any]:
        """
        Execute complete multi-gauge delineation workflow with individual processing
        
        Parameters:
        -----------
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
        self.logger.info(f"Starting Multi-Gauge Workflow for bbox {bbox}")
        start_time = datetime.now()
        
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
                    # Use delineate_single_outlet with shared datasets
                    delineation_result = self.delineation_engine.delineate_single_outlet(
                        outlet_lat=gauge['latitude'],
                        outlet_lon=gauge['longitude'],
                        shared_datasets=unified_prep['datasets'],
                        outlet_name=outlet_name
                    )
                    
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
                            
                        print(f"    ✅ {calculated_area:.1f} km²")
                            
                    else:
                        results['gauge_results'][station_id] = {
                            'success': False,
                            'gauge': gauge,
                            'error': delineation_result.get('error', 'Processing failed')
                        }
                        results['summary']['failed'] += 1
                        print(f"    ❌ {delineation_result.get('error', 'Processing failed')}")
                        
                except Exception as e:
                    results['gauge_results'][station_id] = {
                        'success': False,
                        'gauge': gauge,
                        'error': str(e)
                    }
                    results['summary']['failed'] += 1
                    print(f"    ❌ Exception: {str(e)}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() / 60
            results['execution_time_minutes'] = execution_time
            
            # Generate summary report
            print(f"Completed: {results['summary']['completed']}/{len(valid_gauges)} successful in {execution_time:.1f} minutes")
            
            # Save detailed results
            summary_file = self.results_dir / "multi_gauge_results.json"
            with open(summary_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            return results
            
        except Exception as e:
            error_msg = f"Multi-Gauge Workflow failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'workflow_type': 'Multi_Gauge_Individual'
            }
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate simplified summary report"""
        if not results['success']:
            return f"❌ FAILED: {results['error']}"
        
        summary = results.get('summary', {})
        completed = summary.get('completed', 0)
        failed = summary.get('failed', 0)
        total_area = summary.get('total_area_km2', 0)
        exec_time = results.get('execution_time_minutes', 0)
        
        report = f"""
✅ Multi-Gauge Workflow Complete
================================
Processed: {completed} successful, {failed} failed
Total Area: {total_area:.1f} km²
Time: {exec_time:.1f} minutes

Results:"""
        
        for station_id, result in results.get('gauge_results', {}).items():
            if result.get('success'):
                area = result.get('summary', {}).get('watershed_area_km2', 0) or result.get('watershed_area_km2', 0)
                status = "✅"
            else:
                area = 0
                status = "❌"
                
            report += f"\n  {status} {station_id}: {area:.1f} km²"
        
        return report
    
    def generate_project_map(self, results: Dict[str, Any], project_name: str = "multi_gauge") -> str:
        """Generate project map and save as PNG"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            if not results.get('success'):
                return None
                
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
                        ax.plot(lon, lat, 'go', markersize=8, label=f"✅ {station_id}")
                    else:
                        ax.plot(lon, lat, 'ro', markersize=8, label=f"❌ {station_id}")
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'{project_name.title()} Multi-Gauge Watershed Analysis')
            ax.grid(True, alpha=0.3)
            
            # Save map
            map_file = self.maps_dir / f"{project_name}_overview_map.png"
            plt.savefig(map_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(map_file)
            
        except Exception as e:
            print(f"Warning: Could not generate map - {e}")
            return None
    
    def test_workflow(self, bbox: Tuple[float, float, float, float] = (-73.8, 45.4, -73.4, 45.6)) -> Dict[str, Any]:
        """
        Test workflow with small Montreal region
        
        Parameters:
        -----------
        bbox : tuple, optional
            Test region bounding box (much smaller now)
            
        Returns:
        --------
        Test results
        """
        self.logger.info("Running test workflow with small focused area")
        
        return self.execute_multi_gauge_workflow(
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
        
        print(f"🚀 Running Multi-Gauge Workflow: {args.project_name}")
        print(f"📍 Bbox: {bbox}")
        print(f"📡 Station buffer: {args.buffer} km")
        print(f"💧 Min drainage: {args.min_drainage} km²")
        print(f"📅 Min years: {args.min_years}")
        print(f"🎯 Individual buffer: {args.individual_buffer} km")
        
        results = workflow.execute_multi_gauge_workflow(
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
            print(f"\n❌ Workflow failed: {results['error']}")
    else:
        # Run test workflow
        print("🧪 Running test workflow (use --bbox to specify custom region)")
        results = workflow.test_workflow()
        
        if results['success']:
            print(workflow.generate_summary_report(results))
        else:
            print(f"Test failed: {results['error']}")