#!/usr/bin/env python3
"""
Routing Product Processor for RAVEN-BasinMaker Integration

This processor bridges MGHydro watershed delineation with BasinMaker's
Generate_Raven_Model_Inputs function, handling the complete workflow from
outlet coordinates to RAVEN-ready input files.

Key Functions:
1. Download routing products (DEM, flow direction, streams) for watershed
2. Process MGHydro watershed data into BasinMaker format
3. Generate HRU shapefile with required attributes
4. Create all RAVEN input files using BasinMaker logic
5. Handle lake integration and routing topology

Based on BasinMaker's actual Generate_Raven_Model_Inputs function
"""

import sys
import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import shutil

# Import required components
sys.path.append(str(Path(__file__).parent.parent))

from clients.data_clients.spatial_client import SpatialLayersClient
from processors.raven_generator import RAVENGenerator
from processors.lake_detection import ComprehensiveLakeDetector


class RoutingProductProcessor:
    """
    Complete routing product processing for RAVEN model generation
    
    This class provides the missing link between:
    - MGHydro watershed delineation (API-based)
    - BasinMaker routing product processing
    - RAVEN model file generation
    """
    
    def __init__(self, workspace_dir: Path = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.spatial_client = SpatialLayersClient()
        self.lake_detector = ComprehensiveLakeDetector(self.workspace_dir)
        self.raven_generator = RAVENGenerator(self.workspace_dir)
        
        # Default parameters
        self.default_dem_resolution = 30  # meters
        self.default_stream_threshold = 1000  # cells
        self.default_lake_area_threshold = 0.01  # km²
        self.default_connected_lake_threshold = 0.01  # km²
        self.default_non_connected_threshold = 0.1  # km²
        
    def process_outlet_to_raven(self, 
                              outlet_lat: float, 
                              outlet_lng: float,
                              model_name: str = "raven_model",
                              dem_resolution: int = None,
                              include_lakes: bool = True,
                              lake_area_threshold_km2: float = None,
                              **config_options) -> Dict:
        """
        Complete workflow from outlet coordinates to RAVEN-ready files
        
        Parameters:
        -----------
        outlet_lat : float
            Outlet latitude
        outlet_lng : float
            Outlet longitude  
        model_name : str
            Name for RAVEN model
        dem_resolution : int, optional
            DEM resolution in meters (default: 30m)
        include_lakes : bool
            Whether to include lake processing
        lake_area_threshold_km2 : float, optional
            Minimum lake area threshold
        **config_options : dict
            Additional configuration for RAVEN generation
            
        Returns:
        --------
        Dict with complete workflow results
        """
        
        print("=" * 70)
        print("ROUTING PRODUCT PROCESSOR: OUTLET → RAVEN")
        print("=" * 70)
        print(f"Outlet: {outlet_lat:.4f}, {outlet_lng:.4f}")
        print(f"Model: {model_name}")
        print(f"Workspace: {self.workspace_dir}")
        
        results = {
            'success': False,
            'outlet': {'lat': outlet_lat, 'lng': outlet_lng},
            'model_name': model_name,
            'files_created': [],
            'steps_completed': []
        }
        
        try:
            # Step 1: Get watershed from MGHydro
            print("\n--- STEP 1: Watershed Delineation ---")
            watershed_result = self._get_watershed_boundary(outlet_lat, outlet_lng)
            if not watershed_result['success']:
                results['error'] = watershed_result['error']
                return results
            results['steps_completed'].append('watershed_delineation')
            
            # Step 2: Download routing data
            print("\n--- STEP 2: Routing Data Acquisition ---")
            routing_data = self._acquire_routing_data(
                watershed_result['bbox'],
                dem_resolution or self.default_dem_resolution
            )
            if not routing_data['success']:
                results['error'] = routing_data['error']
                return results
            results['steps_completed'].append('routing_data_acquisition')
            
            # Step 3: Lake detection and classification
            if include_lakes:
                print("\n--- STEP 3: Lake Processing ---")
                lake_results = self._process_lakes(
                    watershed_result['bbox'],
                    lake_area_threshold_km2 or self.default_lake_area_threshold
                )
                if lake_results['success']:
                    results['steps_completed'].append('lake_processing')
                    results['lakes'] = lake_results
            
            # Step 4: Create BasinMaker-compatible HRU data
            print("\n--- STEP 4: HRU Generation ---")
            hru_result = self._create_hru_shapefile(
                watershed_result,
                routing_data,
                results.get('lakes', {})
            )
            if not hru_result['success']:
                results['error'] = hru_result['error']
                return results
            results['steps_completed'].append('hru_generation')
            
            # Step 5: Generate RAVEN input files
            print("\n--- STEP 5: RAVEN File Generation ---")
            raven_result = self._generate_raven_files(
                hru_result['hru_shapefile'],
                model_name,
                **config_options
            )
            if not raven_result['success']:
                results['error'] = raven_result['error']
                return results
            results['steps_completed'].append('raven_generation')
            
            # Step 6: Create workflow summary
            summary_path = self._create_workflow_summary(results)
            results['summary_file'] = summary_path
            
            results['success'] = True
            results['files_created'] = [
                raven_result['raven_files'][f] 
                for f in raven_result['raven_files'] 
                if Path(raven_result['raven_files'][f]).exists()
            ]
            
            print("\n" + "=" * 70)
            print("ROUTING PRODUCT PROCESSING COMPLETE")
            print("=" * 70)
            print(f"RAVEN model ready: {model_name}")
            print(f"Watershed area: {watershed_result.get('area_km2', 0):.1f} km²")
            print(f"Files created: {len(results['files_created'])}")
            print(f"Summary: {summary_path}")
            
        except Exception as e:
            results['error'] = str(e)
            print(f"\nError: {e}")
            
        return results
    
    def _get_watershed_boundary(self, lat: float, lng: float) -> Dict:
        """Get watershed boundary from MGHydro API"""
        
        try:
            # Use the spatial client to get watershed
            watershed_result = self.spatial_client.get_watershed_from_mghydro(lat, lng)
            
            if not watershed_result.get('success'):
                return {'success': False, 'error': watershed_result.get('error', 'Unknown')}
            
            # Save watershed boundary
            watershed_file = self.workspace_dir / "watershed_boundary.geojson"
            with open(watershed_file, 'w') as f:
                json.dump(watershed_result['watershed_geojson'], f, indent=2)
            
            # Calculate bounding box
            from shapely.geometry import shape
            watershed_geom = shape(watershed_result['watershed_geojson']['features'][0]['geometry'])
            bbox = list(watershed_geom.bounds)
            
            return {
                'success': True,
                'watershed_geojson': watershed_result['watershed_geojson'],
                'watershed_shapefile': str(watershed_file),
                'area_km2': watershed_result['area_km2'],
                'bbox': bbox
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _acquire_routing_data(self, bbox: List[float], dem_resolution: int) -> Dict:
        """Acquire all required routing data"""
        
        try:
            # Create routing data directory
            routing_dir = self.workspace_dir / "routing_data"
            routing_dir.mkdir(exist_ok=True)
            
            # Download DEM
            dem_file = routing_dir / "dem.tif"
            dem_result = self.spatial_client.get_dem_for_watershed(
                bbox=bbox,
                output_path=dem_file,
                resolution=dem_resolution,
                source='usgs'
            )
            
            if not dem_result.get('success'):
                return {'success': False, 'error': f"DEM download failed: {dem_result.get('error', 'Unknown')}"}
            
            # Get upstream rivers
            rivers_result = self.spatial_client.get_upstream_rivers_from_mghydro(
                bbox[0] + (bbox[2]-bbox[0])/2,  # Center point
                bbox[1] + (bbox[3]-bbox[1])/2,
                precision="high"
            )
            
            # Save rivers
            rivers_file = routing_dir / "rivers.geojson"
            if rivers_result.get('success'):
                with open(rivers_file, 'w') as f:
                    json.dump(rivers_result['rivers_geojson'], f, indent=2)
            
            return {
                'success': True,
                'dem_file': str(dem_file),
                'rivers_file': str(rivers_file) if rivers_result.get('success') else None,
                'bbox': bbox
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_lakes(self, bbox: List[float], lake_threshold: float) -> Dict:
        """Process lakes for routing integration"""
        
        try:
            lake_results = self.lake_detector.detect_and_classify_lakes(
                bbox=bbox,
                min_lake_area_m2=lake_threshold * 1e6,  # Convert km² to m²
                connected_threshold_km2=self.default_connected_lake_threshold,
                non_connected_threshold_km2=self.default_non_connected_threshold
            )
            
            return lake_results
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_hru_shapefile(self, 
                            watershed_result: Dict,
                            routing_data: Dict,
                            lake_results: Dict) -> Dict:
        """Create BasinMaker-compatible HRU shapefile"""
        
        try:
            # Load watershed boundary
            watershed_gdf = gpd.read_file(watershed_result['watershed_shapefile'])
            
            # Create basic HRU structure
            hru_data = []
            
            # Simple discretization - one HRU per watershed
            watershed = watershed_gdf.iloc[0]
            centroid = watershed.geometry.centroid
            
            # Create HRU record with required BasinMaker attributes
            hru_record = {
                'SubId': 1,
                'DowSubId': -1,  # Outlet
                'HRU_ID': 1,
                'HRU_Area': watershed.geometry.area,
                'HRU_S_mean': 2.0,  # Default slope
                'HRU_A_mean': 180.0,  # Default aspect
                'HRU_E_mean': 250.0,  # Default elevation
                'HRU_CenX': centroid.x,
                'HRU_CenY': centroid.y,
                'LAND_USE_C': 'FOREST',
                'SOIL_PROF': 'LOAM',
                'VEG_C': 'CONIFEROUS',
                'RivLength': 1000.0,  # Default river length
                'RivSlope': 0.01,
                'BkfWidth': 5.0,
                'BkfDepth': 1.0,
                'IsLake': 0,
                'IsObs': 1,  # Outlet is observation point
                'geometry': watershed.geometry
            }
            
            hru_data.append(hru_record)
            
            # Add lake HRUs if lakes are present
            if lake_results.get('success') and lake_results.get('lake_count', 0) > 0:
                lakes_gdf = gpd.read_file(lake_results['lake_shapefile'])
                
                for idx, lake in lakes_gdf.iterrows():
                    lake_record = {
                        'SubId': idx + 2,
                        'DowSubId': 1,  # Flow to watershed outlet
                        'HRU_ID': idx + 2,
                        'HRU_Area': lake.geometry.area,
                        'HRU_S_mean': 0.0,
                        'HRU_A_mean': 0.0,
                        'HRU_E_mean': 250.0,
                        'HRU_CenX': lake.geometry.centroid.x,
                        'HRU_CenY': lake.geometry.centroid.y,
                        'LAND_USE_C': 'WATER',
                        'SOIL_PROF': 'WATER',
                        'VEG_C': 'WATER',
                        'RivLength': 0.0,
                        'RivSlope': 0.0,
                        'BkfWidth': 0.0,
                        'BkfDepth': 0.0,
                        'IsLake': 1,
                        'IsObs': 0,
                        'LakeArea': lake.geometry.area / 1e6,  # km²
                        'LakeDepth': lake.get('depth_m', 5.0),
                        'geometry': lake.geometry
                    }
                    hru_data.append(lake_record)
            
            # Create GeoDataFrame
            hru_gdf = gpd.GeoDataFrame(hru_data, crs=watershed_gdf.crs)
            
            # Save HRU shapefile
            hru_file = self.workspace_dir / "finalcat_info.shp"
            hru_gdf.to_file(hru_file)
            
            return {
                'success': True,
                'hru_shapefile': str(hru_file),
                'hru_count': len(hru_gdf),
                'lake_count': len(hru_gdf[hru_gdf['IsLake'] == 1])
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_raven_files(self, 
                            hru_shapefile: str,
                            model_name: str,
                            **config_options) -> Dict:
        """Generate all RAVEN input files"""
        
        try:
            # Use RAVENGenerator to create all files
            raven_result = self.raven_generator.generate_raven_input_files(
                hru_shapefile_path=Path(hru_shapefile),
                model_name=model_name,
                output_folder=self.workspace_dir / "raven_inputs",
                **config_options
            )
            
            return raven_result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_workflow_summary(self, results: Dict) -> str:
        """Create comprehensive workflow summary"""
        
        summary_file = self.workspace_dir / "ROUTING_WORKFLOW_SUMMARY.md"
        
        content = f"""# Routing Product Processing Summary

## Workflow Overview
Complete processing from outlet coordinates to RAVEN-ready input files.

## Input Parameters
- **Outlet Coordinates**: {results['outlet']['lat']:.4f}, {results['outlet']['lng']:.4f}
- **Model Name**: {results['model_name']}
- **Workspace**: {self.workspace_dir}

## Processing Steps Completed
{chr(10).join(f"- {step.replace('_', ' ').title()}" for step in results['steps_completed'])}

## Generated Files

### RAVEN Input Files
"""
        
        if 'raven_generation' in results['steps_completed']:
            raven_files = results.get('raven_files', {})
            for file_type, file_path in raven_files.items():
                if Path(file_path).exists():
                    content += f"- **{file_type.upper()}**: {Path(file_path).name}\n"
        
        content += f"""
### Supporting Data
- **Watershed Boundary**: watershed_boundary.geojson
- **DEM**: routing_data/dem.tif
"""
        
        if 'lake_processing' in results['steps_completed']:
            lakes = results.get('lakes', {})
            content += f"""
### Lake Data
- **Total Lakes**: {lakes.get('raw_lake_count', 0)}
- **Connected Lakes**: {lakes.get('connected_count', 0)}
- **Non-connected Lakes**: {lakes.get('non_connected_count', 0)}
- **Total Lake Area**: {lakes.get('raw_total_area_ha', 0):.1f} hectares
"""
        
        content += f"""
## Next Steps
1. **Review RAVEN files** in `{self.workspace_dir}/raven_inputs/`
2. **Add forcing data** to .rvt file
3. **Calibrate parameters** in .rvp file
4. **Run RAVEN simulation** using generated files

## Usage
```python
from processors.routing_product_processor import RoutingProductProcessor

# Initialize processor
processor = RoutingProductProcessor("my_workspace")

# Process outlet to RAVEN
results = processor.process_outlet_to_raven(
    outlet_lat=45.1234,
    outlet_lng=-75.5678,
    model_name="my_watershed"
)

# Check results
if results['success']:
    print("RAVEN model ready!")
```

---
*Generated by Routing Product Processor*
*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(summary_file, 'w') as f:
            f.write(content)
            
        return str(summary_file)


def test_routing_processor():
    """Test the complete routing product processor"""
    
    print("Testing Routing Product Processor...")
    print("=" * 50)
    
    # Initialize processor
    processor = RoutingProductProcessor("test_routing")
    
    # Test with Toronto area outlet
    test_result = processor.process_outlet_to_raven(
        outlet_lat=43.6532,
        outlet_lng=-79.3832,
        model_name="toronto_test",
        dem_resolution=90,
        include_lakes=True
    )
    
    if test_result['success']:
        print("\n[SUCCESS] Routing processor test successful!")
        print(f"   Files created: {len(test_result['files_created'])}")
        print(f"   Workspace: {processor.workspace_dir}")
    else:
        print(f"\n[FAILED] Test failed: {test_result.get('error', 'Unknown error')}")
    
    return test_result


if __name__ == "__main__":
    test_routing_processor()