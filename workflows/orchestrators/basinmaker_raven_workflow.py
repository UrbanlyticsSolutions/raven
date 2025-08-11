#!/usr/bin/env python3
"""
Streamlined RAVEN Workflow with BasinMaker Integration
Replaces multiple scattered workflow files with a single, clean orchestrator
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreamlinedRAVENWorkflow:
    """
    Clean, BasinMaker-powered RAVEN workflow orchestrator
    """
    
    def __init__(self, workspace_root: Path, config: Optional[Dict[str, Any]] = None):
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        logger.info(f"Initialized RAVEN workflow: {self.workspace_root}")
    
    def execute_complete_workflow(self, 
                                latitude: float, 
                                longitude: float,
                                watershed_name: str = None) -> Dict[str, Any]:
        """
        Execute complete RAVEN workflow using BasinMaker where possible
        """
        
        if not watershed_name:
            watershed_name = f"watershed_{latitude:.4f}_{longitude:.4f}"
        
        logger.info(f"STARTING STREAMLINED RAVEN WORKFLOW")
        logger.info(f"Location: ({latitude}, {longitude})")
        logger.info(f"Watershed: {watershed_name}")
        
        # Create project workspace
        project_workspace = self.workspace_root / watershed_name
        project_workspace.mkdir(parents=True, exist_ok=True)
        
        workflow_results = {
            'workflow_type': 'streamlined_basinmaker_raven',
            'watershed_name': watershed_name,
            'coordinates': [latitude, longitude],
            'start_time': datetime.now().isoformat(),
            'steps': {},
            'success': False
        }
        
        try:
            # STEP 1: Data Preparation
            step1_result = self._execute_step1_data_preparation(
                latitude, longitude, project_workspace
            )
            workflow_results['steps']['step1'] = step1_result
            
            if not step1_result.get('success'):
                return workflow_results
            
            # STEP 2: BasinMaker Watershed Delineation
            step2_result = self._execute_step2_basinmaker_delineation(
                latitude, longitude, project_workspace, step1_result
            )
            workflow_results['steps']['step2'] = step2_result
            
            if not step2_result.get('success'):
                return workflow_results
            
            # STEP 3: BasinMaker Lake Processing  
            step3_result = self._execute_step3_basinmaker_lakes(
                latitude, longitude, project_workspace, step2_result
            )
            workflow_results['steps']['step3'] = step3_result
            
            if not step3_result.get('success'):
                return workflow_results
            
            # STEP 4: BasinMaker HRU Generation
            step4_result = self._execute_step4_basinmaker_hrus(
                latitude, longitude, project_workspace, step3_result
            )
            workflow_results['steps']['step4'] = step4_result
            
            if not step4_result.get('success'):
                return workflow_results
            
            # STEP 5: RAVEN Model Generation
            step5_result = self._execute_step5_raven_generation(
                latitude, longitude, project_workspace, step4_result
            )
            workflow_results['steps']['step5'] = step5_result
            
            workflow_results['success'] = step5_result.get('success', False)
            workflow_results['end_time'] = datetime.now().isoformat()
            
            # Save workflow results
            results_file = project_workspace / "workflow_results.json"
            with open(results_file, 'w') as f:
                json.dump(workflow_results, f, indent=2, default=str)
            
            logger.info(f"WORKFLOW COMPLETED: {'SUCCESS' if workflow_results['success'] else 'FAILED'}")
            
            return workflow_results
            
        except Exception as e:
            logger.error(f"WORKFLOW FAILED: {e}")
            import traceback
            traceback.print_exc()
            
            workflow_results['error'] = str(e)
            workflow_results['end_time'] = datetime.now().isoformat()
            
            return workflow_results
    
    def _execute_step1_data_preparation(self, lat: float, lon: float, 
                                      workspace: Path) -> Dict[str, Any]:
        """Step 1: Data preparation using existing functionality"""
        
        logger.info("STEP 1: Data Preparation")
        
        try:
            from workflows.project_steps.step1_data_preparation.step1_data_preparation import Step1DataPreparation
            
            step1_workspace = workspace / "step1_data"
            # Create workspace directory before initializing step
            step1_workspace.mkdir(parents=True, exist_ok=True)
            step1 = Step1DataPreparation(workspace_dir=step1_workspace)
            
            result = step1.execute(
                latitude=lat,
                longitude=lon,
                buffer_km=10,
                dem_source='USGS_3DEP'
            )
            
            logger.info(f"Step 1: {'SUCCESS' if result.get('success') else 'FAILED'}")
            return result
            
        except Exception as e:
            logger.error(f"Step 1 failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_step2_basinmaker_delineation(self, lat: float, lon: float, 
                                            workspace: Path, step1_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Watershed delineation using BasinMaker integration"""
        
        logger.info("STEP 2: BasinMaker Watershed Delineation")
        
        try:
            # Use the updated Step2 with BasinMaker integration
            from workflows.project_steps.step2_watershed_delineation.step2_watershed_delineation import Step2WatershedDelineation
            
            step2_workspace = workspace / "step2_watershed"  
            # Create workspace directory before initializing step
            step2_workspace.mkdir(parents=True, exist_ok=True)
            step2 = Step2WatershedDelineation(workspace_dir=step2_workspace)
            
            # Pass Step 1 results file as absolute path
            step1_results_file = Path(step1_result.get('workspace', workspace / 'step1_data')) / 'step1_results.json'
            if not step1_results_file.exists():
                # Fallback to direct file path from results
                step1_results_file = workspace / 'step1_data' / 'step1_results.json'
            
            result = step2.execute(
                latitude=lat,
                longitude=lon,
                minimum_drainage_area_km2=5.0,  # BasinMaker merging threshold
                max_snap_distance_m=1000,
                step1_results_file=step1_results_file.resolve()  # Pass absolute path to step1 results
            )
            
            logger.info(f"Step 2: {'SUCCESS' if result.get('success') else 'FAILED'}")
            
            # Log subbasin merging results
            if result.get('success') and 'characteristics' in result:
                merged_count = result['characteristics'].get('merged_subbasin_count', 0)
                logger.info(f"BasinMaker merging result: {merged_count} subbasins")
            
            return result
            
        except Exception as e:
            logger.error(f"Step 2 failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_step3_basinmaker_lakes(self, lat: float, lon: float,
                                      workspace: Path, step2_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Lake processing (minimal, as BasinMaker handles this in Step 2)"""
        
        logger.info("STEP 3: Lake Processing (BasinMaker integrated)")
        
        try:
            # BasinMaker already handles lake processing in the delineation step
            # So this step mainly prepares data for HRU generation
            
            from workflows.project_steps.step3_lake_processing.step3_lake_processing import Step3LakeProcessing
            
            step3_workspace = workspace / "step3_lakes"
            # Create workspace directory before initializing step
            step3_workspace.mkdir(parents=True, exist_ok=True)
            step3 = Step3LakeProcessing(workspace_dir=step3_workspace)
            
            # Pass Step 2 results with absolute paths
            step2_files_absolute = {}
            for key, file_path in step2_result.get('files', {}).items():
                step2_files_absolute[key] = Path(file_path).resolve()
            
            result = step3.execute(
                lat=lat,
                lon=lon,
                minimum_lake_area_m2=50000,
                buffer_distance_m=100,
                use_satellite_detection=True,
                step2_results=step2_result  # Pass complete Step 2 results
            )
            
            logger.info(f"Step 3: {'SUCCESS' if result.get('success') else 'FAILED'}")
            return result
            
        except Exception as e:
            logger.error(f"Step 3 failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_step4_basinmaker_hrus(self, lat: float, lon: float,
                                     workspace: Path, step3_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: HRU generation using BasinMaker"""
        
        logger.info("STEP 4: BasinMaker HRU Generation") 
        
        try:
            # Use consolidated BasinMaker HRU generation
            from workflows.project_steps.step4_hru_generation.step4_hru_generation import Step4HRUGeneration
            from infrastructure.configuration_manager import WorkflowConfiguration, StepConfiguration
            
            step4_workspace = workspace / "step4_hrus"
            # Create workspace directory before initializing step
            step4_workspace.mkdir(parents=True, exist_ok=True)
            
            # Configure BasinMaker HRU generation
            step4_config = StepConfiguration(
                enabled=True,
                parameters={
                    'hru_discretization_method': 'basinmaker',
                    'min_hru_area_km2': 1.0,
                    'landcover_classes': ['forest', 'agriculture', 'urban', 'water'],
                    'soil_texture_classes': ['clay', 'loam', 'sand'],
                    'elevation_bands': 3,
                    'aspect_classes': 4,
                    'merge_small_hrus': True
                }
            )
            
            workflow_config = WorkflowConfiguration(
                workspace_root=str(step4_workspace.absolute()),
                steps={'step4_hru_generation': step4_config}
            )
            
            step4 = Step4HRUGeneration(
                workspace_dir=step4_workspace,
                config=workflow_config
            )
            
            # Pass Step 3 results with absolute paths
            step3_files_absolute = {}
            for key, file_path in step3_result.get('files', {}).items():
                step3_files_absolute[key] = Path(file_path).resolve()
            
            result = step4.execute(
                latitude=lat, 
                longitude=lon,
                step3_results=step3_result  # Pass complete Step 3 results
            )
            
            logger.info(f"Step 4: {'SUCCESS' if result.get('success') else 'FAILED'}")
            
            if result.get('success'):
                hru_count = result.get('metrics', {}).get('total_hrus', 0)
                logger.info(f"HRUs generated: {hru_count}")
            
            return result
            
        except Exception as e:
            logger.error(f"Step 4 failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_step5_raven_generation(self, lat: float, lon: float,
                                      workspace: Path, step4_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: RAVEN model generation"""
        
        logger.info("STEP 5: RAVEN Model Generation")
        
        try:
            from processors.raven_generator import RAVENGenerator
            
            raven_workspace = workspace / "step5_raven_model"
            # Create workspace directory before initializing step
            raven_workspace.mkdir(parents=True, exist_ok=True)
            
            # Get HRU shapefile from Step 4 with absolute path
            hru_files = step4_result.get('files', {})
            hru_shapefile_path = hru_files.get('hru_shapefile')
            
            if not hru_shapefile_path:
                return {'success': False, 'error': 'HRU shapefile path not found in Step 4 results'}
            
            # Convert to absolute Path object
            hru_shapefile = Path(hru_shapefile_path).resolve()
            if not hru_shapefile.exists():
                return {'success': False, 'error': f'HRU shapefile not found at: {hru_shapefile}'}
            
            raven_generator = RAVENGenerator(raven_workspace)
            model_name = f"basinmaker_raven_model"
            
            # Generate RAVEN model files with absolute paths
            result = raven_generator.generate_raven_input_files(
                hru_shapefile_path=hru_shapefile,  # Already converted to absolute Path
                model_name=model_name,
                output_folder=raven_workspace,
                hydrological_processes=['SOIL_BALANCE', 'CANOPY_EVAPORATION', 'INFILTRATION'],
                routing_method='ROUTE_DUMP',
                time_step='1.0',
                simulation_start='2020-01-01',
                simulation_end='2020-12-31'
            )
            
            logger.info(f"Step 5: {'SUCCESS' if result.get('success') else 'FAILED'}")
            return result
            
        except Exception as e:
            logger.error(f"Step 5 failed: {e}")
            return {'success': False, 'error': str(e)}


def main():
    """Main execution function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Streamlined BasinMaker-RAVEN Workflow')
    parser.add_argument('latitude', type=float, help='Outlet latitude (decimal degrees)')
    parser.add_argument('longitude', type=float, help='Outlet longitude (decimal degrees)')
    parser.add_argument('--watershed-name', type=str, help='Optional watershed name')
    parser.add_argument('--workspace', type=str, default="E:/python/Raven/projects/streamlined", 
                       help='Workspace root directory')
    
    args = parser.parse_args()
    
    workspace_root = Path(args.workspace)
    workflow = StreamlinedRAVENWorkflow(workspace_root)
    
    # Execute workflow with provided coordinates
    result = workflow.execute_complete_workflow(
        latitude=args.latitude,
        longitude=args.longitude,
        watershed_name=args.watershed_name
    )
    
    if result['success']:
        print("STREAMLINED WORKFLOW COMPLETED SUCCESSFULLY")
        print(f"Results saved in: {workspace_root}")
    else:
        print(f"WORKFLOW FAILED: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()