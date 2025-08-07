#!/usr/bin/env python3
"""
Step 3: Lake Processing for RAVEN Single Outlet Delineation
Detects and classifies lakes within the watershed
"""

import sys
from pathlib import Path
import argparse
import json
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from processors.lake_detection import ComprehensiveLakeDetector


class Step3LakeProcessing:
    """Step 3: Detect and classify lakes within watershed"""
    
    def __init__(self, workspace_dir: str = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "data"
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize lake detector
        self.lake_detector = ComprehensiveLakeDetector(workspace_dir=str(self.workspace_dir))
    
    def load_previous_results(self) -> Dict[str, Any]:
        """Load results from previous steps"""
        step1_file = self.workspace_dir / "step1_results.json"
        step2_file = self.workspace_dir / "step2_results.json"
        
        if not step1_file.exists():
            return {
                'success': False,
                'error': 'Step 1 results not found. Run step1_data_preparation.py first.'
            }
        
        if not step2_file.exists():
            return {
                'success': False,
                'error': 'Step 2 results not found. Run step2_watershed_delineation.py first.'
            }
        
        with open(step1_file, 'r') as f:
            step1_results = json.load(f)
        
        with open(step2_file, 'r') as f:
            step2_results = json.load(f)
        
        return {
            'success': True,
            'step1': step1_results,
            'step2': step2_results
        }
    
    def execute(self, latitude: float, longitude: float, min_lake_area_m2: float = 10000) -> Dict[str, Any]:
        """Execute Step 3: Lake detection and classification"""
        print(f"STEP 3: Processing lakes for outlet ({latitude}, {longitude})")
        
        # Load previous results
        previous_results = self.load_previous_results()
        if not previous_results.get('success'):
            return previous_results
        
        step1_data = previous_results['step1']
        step2_data = previous_results['step2']
        
        # Get required files
        dem_file = step1_data['files']['dem']
        watershed_boundary = step2_data['files']['watershed_boundary']
        stream_network = step2_data['files']['stream_network']
        
        print(f"Using DEM: {dem_file}")
        print(f"Using watershed boundary: {watershed_boundary}")
        print(f"Using stream network: {stream_network}")
        
        # Verify files exist
        required_files = {
            'DEM': dem_file,
            'Watershed': watershed_boundary,
            'Streams': stream_network
        }
        
        for name, file_path in required_files.items():
            if not file_path or not Path(file_path).exists():
                return {
                    'success': False,
                    'error': f'{name} file not found: {file_path}'
                }
        
        # Get watershed bounds for lake detection
        import geopandas as gpd
        try:
            watershed_gdf = gpd.read_file(watershed_boundary)
            watershed_bounds = watershed_gdf.total_bounds.tolist()  # [minx, miny, maxx, maxy]
            print(f"Watershed bounds: {watershed_bounds}")
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to read watershed boundary: {str(e)}'
            }
        
        # Execute lake detection and classification
        print(f"Detecting lakes (minimum area: {min_lake_area_m2} m²)...")
        lake_result = self.lake_detector.detect_and_classify_lakes(
            bbox=watershed_bounds,
            min_lake_area_m2=min_lake_area_m2
        )
        
        if not lake_result.get('success'):
            print("Warning: Lake detection failed, continuing without lakes")
            lake_result = {
                'success': True,
                'lakes_detected': 0,
                'connected_lakes': 0,
                'non_connected_lakes': 0,
                'files': {
                    'all_lakes': None,
                    'connected_lakes': None,
                    'non_connected_lakes': None
                }
            }
        
        # Prepare results
        results = {
            'success': True,
            'outlet_coordinates': [latitude, longitude],
            'lake_statistics': {
                'total_lakes': lake_result.get('lakes_detected', 0),
                'connected_lakes': lake_result.get('connected_lakes', 0),
                'non_connected_lakes': lake_result.get('non_connected_lakes', 0),
                'min_area_m2': min_lake_area_m2
            },
            'files': {
                'all_lakes': lake_result.get('files', {}).get('all_lakes'),
                'connected_lakes': lake_result.get('files', {}).get('connected_lakes'),
                'non_connected_lakes': lake_result.get('files', {}).get('non_connected_lakes')
            }
        }
        
        # Verify output files
        for lake_type, file_path in results['files'].items():
            if file_path and Path(file_path).exists():
                print(f"✅ {lake_type}: {file_path}")
            else:
                print(f"ℹ️  {lake_type}: No lakes found or file not created")
        
        # Save results to JSON
        results_file = self.workspace_dir / "step3_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"STEP 3 COMPLETE: Results saved to {results_file}")
        print(f"Lakes found: {results['lake_statistics']['total_lakes']} total")
        print(f"  - Connected: {results['lake_statistics']['connected_lakes']}")
        print(f"  - Non-connected: {results['lake_statistics']['non_connected_lakes']}")
        
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 3: Lake Processing')
    parser.add_argument('latitude', type=float, help='Outlet latitude')
    parser.add_argument('longitude', type=float, help='Outlet longitude')
    parser.add_argument('--min-lake-area', type=float, default=10000, 
                       help='Minimum lake area in m² (default: 10000 = 1 hectare)')
    parser.add_argument('--workspace-dir', type=str, help='Workspace directory')
    
    args = parser.parse_args()
    
    step3 = Step3LakeProcessing(workspace_dir=args.workspace_dir)
    results = step3.execute(args.latitude, args.longitude, args.min_lake_area)
    
    if results['success']:
        print("SUCCESS: Step 3 lake processing completed")
        print(f"Total lakes: {results['lake_statistics']['total_lakes']}")
    else:
        print(f"FAILED: {results['error']}")
        sys.exit(1)