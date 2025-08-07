#!/usr/bin/env python3
"""
Step 4: HRU Generation for RAVEN Single Outlet Delineation
Generates Hydrologic Response Units (HRUs) and subbasins
"""

import sys
from pathlib import Path
import argparse
import json
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from workflows.steps import CreateSubBasinsAndHRUs


class Step4HRUGeneration:
    """Step 4: Generate HRUs (Hydrologic Response Units) and subbasins"""
    
    def __init__(self, workspace_dir: str = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "data"
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize HRU generation step
        self.hru_step = CreateSubBasinsAndHRUs()
    
    def load_previous_results(self) -> Dict[str, Any]:
        """Load results from previous steps"""
        step1_file = self.workspace_dir / "step1_results.json"
        step2_file = self.workspace_dir / "step2_results.json"
        step3_file = self.workspace_dir / "step3_results.json"
        
        required_files = {
            'step1': step1_file,
            'step2': step2_file,
            'step3': step3_file
        }
        
        for step, file_path in required_files.items():
            if not file_path.exists():
                return {
                    'success': False,
                    'error': f'{step.title()} results not found. Run {step}_*.py first.'
                }
        
        results = {}
        for step, file_path in required_files.items():
            with open(file_path, 'r') as f:
                results[step] = json.load(f)
        
        return {
            'success': True,
            **results
        }
    
    def execute(self, latitude: float, longitude: float, outlet_name: str = None) -> Dict[str, Any]:
        """Execute Step 4: HRU and subbasin generation"""
        print(f"STEP 4: Generating HRUs for outlet ({latitude}, {longitude})")
        
        # Load previous results
        previous_results = self.load_previous_results()
        if not previous_results.get('success'):
            return previous_results
        
        step1_data = previous_results['step1']
        step2_data = previous_results['step2']
        step3_data = previous_results['step3']
        
        # Get required files
        dem_file = step1_data['files']['dem']
        landcover_file = step1_data['files']['landcover']
        soil_file = step1_data['files']['soil']
        watershed_boundary = step2_data['files']['watershed_boundary']
        stream_network = step2_data['files']['stream_network']
        connected_lakes = step3_data['files']['connected_lakes']
        
        if not outlet_name:
            outlet_name = f"outlet_{latitude:.4f}_{longitude:.4f}"
        
        print(f"Using files:")
        print(f"  - DEM: {dem_file}")
        print(f"  - Landcover: {landcover_file}")
        print(f"  - Soil: {soil_file}")
        print(f"  - Watershed: {watershed_boundary}")
        print(f"  - Streams: {stream_network}")
        print(f"  - Connected Lakes: {connected_lakes if connected_lakes else 'None'}")
        
        # Verify required files exist
        required_files = {
            'DEM': dem_file,
            'Landcover': landcover_file,
            'Soil': soil_file,
            'Watershed': watershed_boundary,
            'Streams': stream_network
        }
        
        for name, file_path in required_files.items():
            if not file_path or not Path(file_path).exists():
                return {
                    'success': False,
                    'error': f'{name} file not found: {file_path}'
                }
        
        # Execute HRU generation
        print("Generating HRUs and subbasins...")
        hru_result = self.hru_step.execute({
            'watershed_boundary': watershed_boundary,
            'integrated_stream_network': stream_network,
            'connected_lakes': connected_lakes,
            'dem_file': dem_file,
            'landcover_file': landcover_file,
            'soil_file': soil_file,
            'watershed_name': outlet_name
        })
        
        if not hru_result.get('success'):
            return hru_result
        
        # Prepare results
        results = {
            'success': True,
            'outlet_coordinates': [latitude, longitude],
            'outlet_name': outlet_name,
            'files': {
                'subbasins': hru_result.get('sub_basins'),
                'final_hrus': hru_result.get('final_hrus'),
                'hydraulic_parameters': hru_result.get('hydraulic_parameters')
            },
            'statistics': {
                'total_hru_count': hru_result.get('total_hru_count', 0),
                'subbasin_count': hru_result.get('subbasin_count', 0),
                'land_hru_count': hru_result.get('land_hru_count', 0),
                'lake_hru_count': hru_result.get('lake_hru_count', 0)
            },
            'routing': {
                'connectivity': hru_result.get('routing_connectivity', {})
            }
        }
        
        # Verify output files
        for file_type, file_path in results['files'].items():
            if file_path and Path(file_path).exists():
                print(f"✅ {file_type}: {file_path}")
            else:
                print(f"⚠️  {file_type}: Missing or not created")
        
        # Save results to JSON
        results_file = self.workspace_dir / "step4_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"STEP 4 COMPLETE: Results saved to {results_file}")
        print(f"HRU Summary:")
        print(f"  - Total HRUs: {results['statistics']['total_hru_count']}")
        print(f"  - Subbasins: {results['statistics']['subbasin_count']}")
        print(f"  - Land HRUs: {results['statistics']['land_hru_count']}")
        print(f"  - Lake HRUs: {results['statistics']['lake_hru_count']}")
        
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 4: HRU Generation')
    parser.add_argument('latitude', type=float, help='Outlet latitude')
    parser.add_argument('longitude', type=float, help='Outlet longitude')
    parser.add_argument('--outlet-name', type=str, help='Name for the outlet')
    parser.add_argument('--workspace-dir', type=str, help='Workspace directory')
    
    args = parser.parse_args()
    
    step4 = Step4HRUGeneration(workspace_dir=args.workspace_dir)
    results = step4.execute(args.latitude, args.longitude, args.outlet_name)
    
    if results['success']:
        print("SUCCESS: Step 4 HRU generation completed")
        print(f"Total HRUs: {results['statistics']['total_hru_count']}")
        print(f"Subbasins: {results['statistics']['subbasin_count']}")
    else:
        print(f"FAILED: {results['error']}")
        sys.exit(1)