#!/usr/bin/env python3
"""
Step 1: Data Preparation for RAVEN Single Outlet Delineation
Downloads DEM, Landcover, and Soil data for watershed analysis
"""

import sys
from pathlib import Path
import argparse
import json
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from workflows.steps import DEMClippingStep, LandcoverExtractionStep, SoilExtractionStep
from clients.data_clients.spatial_client import SpatialLayersClient


class Step1DataPreparation:
    """Step 1: Download spatial data (DEM, Landcover, Soil) for watershed"""
    
    def __init__(self, workspace_dir: str = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "data"
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize clients and steps
        self.spatial_client = SpatialLayersClient()
        self.dem_step = DEMClippingStep(workspace_dir=self.workspace_dir)
        self.landcover_step = LandcoverExtractionStep(workspace_dir=self.workspace_dir)
        self.soil_step = SoilExtractionStep(workspace_dir=self.workspace_dir)
    
    def get_watershed_extent(self, lat: float, lon: float, buffer_km: float = 2.0) -> Dict[str, Any]:
        """Get watershed extent using MGHydro API"""
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
        except Exception as e:
            print(f"Warning: MGHydro watershed fetch failed: {e}")
        
        # MGHydro failed - return error
        return {
            'success': False,
            'error': 'MGHydro watershed fetch failed - no synthetic fallback provided'
        }
    
    def execute(self, latitude: float, longitude: float, buffer_km: float = 2.0) -> Dict[str, Any]:
        """Execute Step 1: Download all spatial data"""
        print(f"STEP 1: Preparing spatial data for outlet ({latitude}, {longitude})")
        
        # Get watershed extent
        print("Getting watershed extent from MGHydro...")
        extent_result = self.get_watershed_extent(latitude, longitude, buffer_km)
        
        if not extent_result['success']:
            return extent_result
            
        final_bounds = extent_result['buffered_bbox']
        print(f"Processing extent: {final_bounds}")
        
        results = {
            'success': True,
            'extent_info': extent_result,
            'bounds': final_bounds,
            'files': {}
        }
        
        # Download DEM
        print("Downloading DEM data...")
        dem_result = self.dem_step.execute(
            bounds=final_bounds,
            resolution=30,
            output_filename="dem.tif"
        )
        if not dem_result['success']:
            return dem_result
        results['files']['dem'] = dem_result['dem_file']
        print(f"✅ DEM: {dem_result['dem_file']}")
        
        # Download Landcover
        print("Downloading landcover data...")
        landcover_result = self.landcover_step.execute(
            bounds=final_bounds,
            output_filename="landcover.tif"
        )
        if not landcover_result['success']:
            return landcover_result
        results['files']['landcover'] = landcover_result['landcover_file']
        print(f"✅ Landcover: {landcover_result['landcover_file']}")
        
        # Download Soil
        print("Downloading soil data...")
        soil_result = self.soil_step.execute(
            bounds=final_bounds,
            output_filename="soil.tif"
        )
        if not soil_result['success']:
            return soil_result
        results['files']['soil'] = soil_result['soil_file']
        print(f"✅ Soil: {soil_result['soil_file']}")
        
        # Save results to JSON
        results_file = self.workspace_dir / "step1_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'success': results['success'],
                'extent_info': results['extent_info'],
                'bounds': results['bounds'],
                'files': results['files'],
                'outlet_coordinates': [latitude, longitude]
            }, f, indent=2)
        
        print(f"STEP 1 COMPLETE: Results saved to {results_file}")
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
    results = step1.execute(args.latitude, args.longitude, args.buffer_km)
    
    if results['success']:
        print("SUCCESS: Step 1 data preparation completed")
        print(f"Files: {len(results['files'])} datasets downloaded")
    else:
        print(f"FAILED: {results['error']}")
        sys.exit(1)