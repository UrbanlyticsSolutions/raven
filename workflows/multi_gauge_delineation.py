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
    
    def __init__(self, workspace_dir: str = None):
        """
        Initialize multi-gauge delineation
        
        Parameters:
        -----------
        workspace_dir : str, optional
            Main workspace directory
        """
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "multi_gauge_delineation"
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize clients
        self.hydro_client = HydrometricDataClient()
        self.delineation_engine = RefactoredFullDelineation(workspace_dir=str(self.workspace_dir))
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the workflow"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_file = self.workspace_dir / "multi_gauge_delineation.log"
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
            
        return logger
    
    def discover_gauges_in_region(self, bbox: Tuple[float, float, float, float], 
                                 buffer_km: float = 1.0) -> List[Dict[str, Any]]:
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
        
        # Create buffered bbox
        minx, miny, maxx, maxy = bbox
        buffer_deg = buffer_km / 111.0
        
        buffered_bbox = (
            minx - buffer_deg,
            miny - buffer_deg,
            maxx + buffer_deg,
            maxy + buffer_deg
        )
        
        # Get hydrometric stations
        stations_data = self.hydro_client.get_hydrometric_stations_for_watershed(
            bbox=buffered_bbox,
            output_path=self.workspace_dir / "discovered_gauges.geojson"
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
            
            # Filter for active stations with drainage area
            if (gauge_info['status'] == "ACTIVE" and 
                gauge_info['drainage_area_km2'] and 
                gauge_info['drainage_area_km2'] > 0):
                
                gauges.append(gauge_info)
        
        self.logger.info(f"Discovered {len(gauges)} active gauges")
        return gauges
    
    def calculate_unified_bounds(self, gauges: List[Dict[str, Any]], 
                                buffer_km: float = 2.0) -> Tuple[float, float, float, float]:
        """
        Calculate unified bounds for all gauges with proper buffer calculation
        
        Parameters:
        -----------
        gauges : list
            List of gauge dictionaries
        buffer_km : float
            Buffer distance in km around all gauges (much smaller now)
            
        Returns:
        --------
        Unified bounding box (minx, miny, maxx, maxy)
        """
        if not gauges:
            return (-73.6, 45.4, -73.5, 45.6)  # Small Montreal area fallback
            
        lats = [g['latitude'] for g in gauges]
        lons = [g['longitude'] for g in gauges]
        
        # Proper buffer calculation considering latitude
        avg_lat = sum(lats) / len(lats)
        lat_buffer = buffer_km / 111.0  # 1 degree lat ≈ 111 km
        lon_buffer = buffer_km / (111.0 * abs(math.cos(math.radians(avg_lat))))  # Adjust for latitude
        
        bounds = (
            min(lons) - lon_buffer,
            min(lats) - lat_buffer,
            max(lons) + lon_buffer,
            max(lats) + lat_buffer
        )
        
        # Sanity check - don't allow areas larger than 50km x 50km
        max_extent = 0.5  # ~50km at these latitudes
        if (bounds[2] - bounds[0]) > max_extent or (bounds[3] - bounds[1]) > max_extent:
            # Fall back to individual processing
            center_lon = sum(lons) / len(lons)
            center_lat = sum(lats) / len(lats)
            half_extent = max_extent / 2
            bounds = (
                center_lon - half_extent,
                center_lat - half_extent,
                center_lon + half_extent,
                center_lat + half_extent
            )
        
        return bounds
    
    def execute_multi_gauge_workflow(self, bbox: Tuple[float, float, float, float], 
                                   buffer_km: float = 1.0,
                                   min_drainage_area_km2: float = 10.0,
                                   individual_buffer_km: float = 2.0) -> Dict[str, Any]:
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
            all_gauges = self.discover_gauges_in_region(bbox, buffer_km)
            
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
            
            # Step 2: Process each gauge individually with focused areas
            self.logger.info("Processing each gauge individually with small focused areas")
            
            results = {
                'success': True,
                'workflow_type': 'Multi_Gauge_Individual',
                'bbox': bbox,
                'buffer_km': buffer_km,
                'individual_buffer_km': individual_buffer_km,
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
            
            # Step 3: Process each gauge individually
            for i, gauge in enumerate(valid_gauges):
                station_id = gauge['station_id']
                outlet_name = f"gauge_{station_id}"
                
                self.logger.info(f"Processing gauge {i+1}/{len(valid_gauges)}: {station_id} ({gauge['latitude']:.4f}, {gauge['longitude']:.4f})")
                
                try:
                    # Use individual delineation with small focused area
                    delineation_result = self.delineation_engine.execute_single_delineation(
                        latitude=gauge['latitude'],
                        longitude=gauge['longitude'],
                        outlet_name=outlet_name,
                        buffer_km=individual_buffer_km  # Small focused area per gauge
                    )
                    
                    if delineation_result['success']:
                        # Add gauge-specific metadata
                        delineation_result['gauge_info'] = gauge
                        
                        # Area validation
                        calculated_area = delineation_result.get('watershed_area_km2', 0)
                        reported_area = float(gauge['drainage_area_km2'])
                        area_diff_percent = abs(calculated_area - reported_area) / reported_area * 100
                        
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
                            
                        self.logger.info(f"  ✅ {station_id}: {calculated_area:.1f} km² (vs {reported_area:.1f} km² reported)")
                            
                    else:
                        results['gauge_results'][station_id] = {
                            'success': False,
                            'gauge': gauge,
                            'error': delineation_result.get('error', 'Processing failed')
                        }
                        results['summary']['failed'] += 1
                        self.logger.error(f"  ❌ {station_id}: {delineation_result.get('error', 'Processing failed')}")
                        
                except Exception as e:
                    results['gauge_results'][station_id] = {
                        'success': False,
                        'gauge': gauge,
                        'error': str(e)
                    }
                    results['summary']['failed'] += 1
                    self.logger.error(f"  ❌ {station_id}: Exception - {str(e)}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() / 60
            results['execution_time_minutes'] = execution_time
            
            # Generate summary report
            self.logger.info(f"Multi-Gauge Workflow completed: {results['summary']['completed']}/{len(valid_gauges)} successful in {execution_time:.1f} minutes")
            
            # Save detailed results
            summary_file = self.workspace_dir / "multi_gauge_results.json"
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
        """Generate human-readable summary report"""
        if not results['success']:
            return f"Workflow failed: {results['error']}"
        
        report = f"""
Multi-Gauge Delineation Summary
===============================

Region: {results['bbox']}
Individual Buffer: {results.get('individual_buffer_km', 'N/A')} km
Gauges Discovered: {results['gauges_discovered']}
Gauges Processed: {results['gauges_processed']}
Successful Delineations: {results['summary']['completed']}
Failed Delineations: {results['summary']['failed']}

Total Watershed Area: {results['summary']['total_area_km2']:.1f} km²
Execution Time: {results.get('execution_time_minutes', 0):.1f} minutes

Area Validation:
- Within 10%: {results['summary']['area_validation']['within_10_percent']}
- Within 25%: {results['summary']['area_validation']['within_25_percent']}
- Over 25%: {results['summary']['area_validation']['over_25_percent']}

Individual Gauge Results:
"""
        
        for station_id, result in results['gauge_results'].items():
            if result['success']:
                report += f"\n  {station_id}: {result['gauge_info']['station_name']}"
                report += f"\n    Area: {result.get('watershed_area_km2', 0):.1f} km²"
                report += f"\n    HRUs: {result.get('total_hru_count', 0)}"
                report += f"\n    Model: {result.get('selected_model', 'Unknown')}"
                report += f"\n    Area Match: {result.get('area_validation', {}).get('validation_status', 'Unknown')}"
            else:
                report += f"\n  {station_id}: FAILED - {result.get('error', 'Unknown error')}"
        
        return report
    
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
    # Test the integrated workflow
    workflow = MultiGaugeDelineation()
    
    # Test with Montreal region
    results = workflow.test_workflow()
    
    if results['success']:
        print(workflow.generate_summary_report(results))
    else:
        print(f"Test failed: {results['error']}")