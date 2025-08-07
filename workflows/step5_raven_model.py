#!/usr/bin/env python3
"""
Step 5: RAVEN Model Generation for Single Outlet Delineation
Generates complete RAVEN model files (.rvh, .rvp, .rvi, .rvt, .rvc)
"""

import sys
from pathlib import Path
import argparse
import json
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from workflows.steps import SelectModelAndGenerateStructure, GenerateModelInstructions, ValidateCompleteModel


class Step5RAVENModel:
    """Step 5: Generate complete RAVEN model files"""
    
    def __init__(self, workspace_dir: str = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "data"
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize model generation steps
        self.model_structure_step = SelectModelAndGenerateStructure()
        self.model_instructions_step = GenerateModelInstructions()
        self.validation_step = ValidateCompleteModel()
    
    def load_previous_results(self) -> Dict[str, Any]:
        """Load results from all previous steps"""
        required_steps = ['step1', 'step2', 'step3', 'step4']
        results = {}
        
        for step in required_steps:
            step_file = self.workspace_dir / f"{step}_results.json"
            if not step_file.exists():
                return {
                    'success': False,
                    'error': f'{step.title()} results not found. Run {step}_*.py first.'
                }
            
            with open(step_file, 'r') as f:
                results[step] = json.load(f)
        
        return {
            'success': True,
            **results
        }
    
    def execute(self, latitude: float, longitude: float, outlet_name: str = None, 
                model_type: str = 'GR4JCN') -> Dict[str, Any]:
        """Execute Step 5: RAVEN model generation"""
        print(f"STEP 5: Generating RAVEN model for outlet ({latitude}, {longitude})")
        print(f"Model type: {model_type}")
        
        # Load previous results
        previous_results = self.load_previous_results()
        if not previous_results.get('success'):
            return previous_results
        
        step2_data = previous_results['step2']\n        step3_data = previous_results['step3']\n        step4_data = previous_results['step4']\n        \n        if not outlet_name:\n            outlet_name = f\"outlet_{latitude:.4f}_{longitude:.4f}\"\n        \n        # Get required data\n        final_hrus = step4_data['files']['final_hrus']\n        subbasins = step4_data['files']['subbasins']\n        connected_lakes = step3_data['files']['connected_lakes']\n        watershed_area_km2 = step2_data['characteristics']['watershed_area_km2']\n        \n        print(f\"Using:\")\n        print(f\"  - HRUs: {final_hrus}\")\n        print(f\"  - Subbasins: {subbasins}\")\n        print(f\"  - Connected Lakes: {connected_lakes if connected_lakes else 'None'}\")\n        print(f\"  - Watershed Area: {watershed_area_km2} km²\")\n        \n        # Verify required files\n        required_files = {\n            'HRUs': final_hrus,\n            'Subbasins': subbasins\n        }\n        \n        for name, file_path in required_files.items():\n            if not file_path or not Path(file_path).exists():\n                return {\n                    'success': False,\n                    'error': f'{name} file not found: {file_path}'\n                }\n        \n        # Step 5.1: Generate model structure (.rvh, .rvp)\n        print(\"Generating model structure files (.rvh, .rvp)...\")\n        model_structure_result = self.model_structure_step.execute({\n            'final_hrus': final_hrus,\n            'sub_basins': subbasins,\n            'connected_lakes': connected_lakes,\n            'watershed_area_km2': watershed_area_km2,\n            'model_type': model_type\n        })\n        \n        if not model_structure_result.get('success'):\n            return model_structure_result\n        \n        # Step 5.2: Generate model instructions (.rvi, .rvt, .rvc)\n        print(\"Generating model instructions (.rvi, .rvt, .rvc)...\")\n        model_instructions_result = self.model_instructions_step.execute({\n            'selected_model': model_structure_result.get('selected_model', model_type),\n            'sub_basins': subbasins,\n            'final_hrus': final_hrus,\n            'watershed_area_km2': watershed_area_km2,\n            'outlet_name': outlet_name\n        })\n        \n        if not model_instructions_result.get('success'):\n            return model_instructions_result\n        \n        # Step 5.3: Validate complete model\n        print(\"Validating complete RAVEN model...\")\n        model_files = {\n            'rvh_file': model_structure_result.get('rvh_file'),\n            'rvp_file': model_structure_result.get('rvp_file'),\n            'rvi_file': model_instructions_result.get('rvi_file'),\n            'rvt_file': model_instructions_result.get('rvt_file'),\n            'rvc_file': model_instructions_result.get('rvc_file')\n        }\n        \n        validation_result = self.validation_step.execute(model_files)\n        \n        # Prepare results\n        results = {\n            'success': True,\n            'outlet_coordinates': [latitude, longitude],\n            'outlet_name': outlet_name,\n            'model_info': {\n                'selected_model': model_structure_result.get('selected_model', model_type),\n                'model_description': f\"{model_type} model for {outlet_name}\",\n                'parameter_count': model_structure_result.get('parameter_count', 0),\n                'process_count': model_structure_result.get('process_count', 0)\n            },\n            'files': {\n                'rvh': model_structure_result.get('rvh_file'),\n                'rvp': model_structure_result.get('rvp_file'),\n                'rvi': model_instructions_result.get('rvi_file'),\n                'rvt': model_instructions_result.get('rvt_file'),\n                'rvc': model_instructions_result.get('rvc_file')\n            },\n            'validation': {\n                'is_valid': validation_result.get('success', False),\n                'validation_summary': validation_result.get('validation_summary', {})\n            },\n            'statistics': {\n                'total_hru_count': step4_data['statistics']['total_hru_count'],\n                'subbasin_count': step4_data['statistics']['subbasin_count'],\n                'lake_hru_count': step4_data['statistics']['lake_hru_count'],\n                'watershed_area_km2': watershed_area_km2,\n                'connected_lake_count': step3_data['lake_statistics']['connected_lakes']\n            }\n        }\n        \n        # Verify output files\n        raven_extensions = ['rvh', 'rvp', 'rvi', 'rvt', 'rvc']\n        all_files_exist = True\n        \n        for ext in raven_extensions:\n            file_path = results['files'][ext]\n            if file_path and Path(file_path).exists():\n                print(f\"✅ {ext.upper()}: {file_path}\")\n            else:\n                print(f\"❌ {ext.upper()}: Missing or not created\")\n                all_files_exist = False\n        \n        if not all_files_exist:\n            results['success'] = False\n            results['error'] = 'Some RAVEN model files were not created'\n            return results\n        \n        # Save results to JSON\n        results_file = self.workspace_dir / \"step5_results.json\"\n        with open(results_file, 'w') as f:\n            json.dump(results, f, indent=2)\n        \n        print(f\"STEP 5 COMPLETE: Results saved to {results_file}\")\n        print(f\"Model Summary:\")\n        print(f\"  - Model Type: {results['model_info']['selected_model']}\")\n        print(f\"  - HRUs: {results['statistics']['total_hru_count']}\")\n        print(f\"  - Subbasins: {results['statistics']['subbasin_count']}\")\n        print(f\"  - Watershed Area: {results['statistics']['watershed_area_km2']:.2f} km²\")\n        print(f\"  - Model Valid: {results['validation']['is_valid']}\")\n        \n        return results\n\n\nif __name__ == \"__main__\":\n    parser = argparse.ArgumentParser(description='Step 5: RAVEN Model Generation')\n    parser.add_argument('latitude', type=float, help='Outlet latitude')\n    parser.add_argument('longitude', type=float, help='Outlet longitude')\n    parser.add_argument('--outlet-name', type=str, help='Name for the outlet')\n    parser.add_argument('--model-type', type=str, default='GR4JCN',\n                       choices=['GR4JCN', 'HMETS', 'HBVEC', 'UBCWM'],\n                       help='RAVEN model type (default: GR4JCN)')\n    parser.add_argument('--workspace-dir', type=str, help='Workspace directory')\n    \n    args = parser.parse_args()\n    \n    step5 = Step5RAVENModel(workspace_dir=args.workspace_dir)\n    results = step5.execute(args.latitude, args.longitude, args.outlet_name, args.model_type)\n    \n    if results['success']:\n        print(\"SUCCESS: Step 5 RAVEN model generation completed\")\n        print(f\"Model: {results['model_info']['selected_model']}\")\n        print(f\"Valid: {results['validation']['is_valid']}\")\n    else:\n        print(f\"FAILED: {results['error']}\")\n        sys.exit(1)