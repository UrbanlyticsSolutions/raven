"""
DEM Processing Steps for RAVEN Workflows

This module contains steps for downloading and processing Digital Elevation Models.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from workflows.steps.base_step import WorkflowStep

class DownloadAndPrepareDEM(WorkflowStep):
    """
    Step 2B: Download and prepare DEM
    Used in Approach B (Full Delineation Workflow)
    """
    
    def __init__(self):
        super().__init__(
            step_name="download_prepare_dem",
            step_category="dem_processing",
            description="Download USGS 3DEP DEM and hydrologically condition it"
        )
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._log_step_start()
        
        try:
            # Validate required inputs
            required_inputs = ['dem_bounds', 'dem_resolution']
            self.validate_inputs(inputs, required_inputs)
            
            dem_bounds = inputs['dem_bounds']
            dem_resolution = inputs['dem_resolution']
            
            # Create workspace directory
            workspace_dir = inputs.get('workspace_dir', Path.cwd() / 'dem_processing')
            workspace = Path(workspace_dir)
            workspace.mkdir(exist_ok=True, parents=True)
            
            # Step 1: Download DEM
            self.logger.info(f"Downloading DEM for bounds: {dem_bounds}")
            original_dem = self._download_dem(dem_bounds, dem_resolution, workspace)
            
            # Step 2: Prepare DEM using WhiteboxTools
            self.logger.info("Conditioning DEM for hydrological analysis")
            conditioned_files = self._condition_dem(original_dem, workspace)
            
            outputs = {
                'original_dem': str(original_dem),
                'filled_dem': conditioned_files['filled_dem'],
                'flow_direction': conditioned_files['flow_direction'],
                'flow_accumulation': conditioned_files['flow_accumulation'],
                'dem_bounds': dem_bounds,
                'dem_resolution': dem_resolution,
                'dem_resolution_m': self._get_resolution_meters(dem_resolution),
                'depressions_filled': conditioned_files.get('depressions_filled', 0),
                'processing_time_seconds': conditioned_files.get('processing_time', 0),
                'success': True
            }
            
            created_files = [
                str(original_dem),
                conditioned_files['filled_dem'],
                conditioned_files['flow_direction'],
                conditioned_files['flow_accumulation']
            ]
            
            self._log_step_complete(created_files)
            return outputs
            
        except Exception as e:
            error_msg = f"DEM download and preparation failed: {str(e)}"
            self._log_step_failed(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _download_dem(self, bounds: list, resolution: str, workspace: Path) -> Path:
        """Download DEM from USGS 3DEP or other sources"""
        
        try:
            # Try to use SpatialLayersClient for DEM download
            from clients.data_clients import SpatialLayersClient
            
            spatial_client = SpatialLayersClient()
            
            # Download DEM - ULTRA-SIMPLE: Save to data/ folder with simple name
            data_dir = workspace / "data"
            data_dir.mkdir(exist_ok=True)
            dem_file = data_dir / "dem.tif"
            
            self.logger.info("Attempting USGS 3DEP download...")
            
            # Use the correct method from SpatialLayersClient
            result = spatial_client.get_dem_for_watershed(
                bbox=tuple(bounds),  # Convert to tuple
                output_path=dem_file,
                resolution=self._resolution_to_meters(resolution),
                source='auto'  # Try USGS first, then Canadian sources
            )
            
            if result['success'] and dem_file.exists():
                return dem_file
            else:
                raise RuntimeError(f"DEM download failed: {result.get('error', 'Unknown error')}")
                
        except ImportError as e:
            raise RuntimeError(f"SpatialLayersClient not available: {e}")
        
        except Exception as e:
            raise RuntimeError(f"DEM download failed: {str(e)}")
    
    def _condition_dem(self, dem_file: Path, workspace: Path) -> Dict[str, Any]:
        """Condition DEM using WhiteboxTools"""
        
        start_time = time.time()
        
        try:
            # Try to use WhiteboxTools
            import whitebox
            wbt = whitebox.WhiteboxTools()
            wbt.set_working_dir(str(workspace))
            wbt.set_verbose_mode(False)
            
            # Define output files
            filled_dem = workspace / "dem_filled.tif"
            flow_dir = workspace / "flow_direction.tif"
            flow_accum = workspace / "flow_accumulation.tif"
            
            # Step 1: Fill depressions (use just filenames since we set working dir)
            self.logger.info("Filling depressions...")
            wbt.fill_depressions_wang_and_liu(dem_file.name, filled_dem.name)
            
            # Step 2: Calculate flow direction
            self.logger.info("Calculating flow direction...")
            wbt.d8_pointer(filled_dem.name, flow_dir.name)
            
            # Step 3: Calculate flow accumulation
            self.logger.info("Calculating flow accumulation...")
            wbt.d8_flow_accumulation(flow_dir.name, flow_accum.name)
            
            processing_time = time.time() - start_time
            
            return {
                'filled_dem': str(filled_dem),
                'flow_direction': str(flow_dir),
                'flow_accumulation': str(flow_accum),
                'depressions_filled': 1000,  # Placeholder - would need actual count
                'processing_time': processing_time
            }
            
        except ImportError as e:
            raise RuntimeError(f"WhiteboxTools not available: {e}")
        
        except Exception as e:
            raise RuntimeError(f"WhiteboxTools processing failed: {str(e)}")
    
    def _get_resolution_meters(self, resolution: str) -> float:
        """Convert resolution string to meters"""
        
        if resolution == '10m':
            return 10.0
        elif resolution == '30m':
            return 30.0
        elif resolution == '90m':
            return 90.0
        else:
            return 30.0  # Default
    
    def _resolution_to_meters(self, resolution: str) -> int:
        """Convert resolution string to integer meters for SpatialLayersClient"""
        
        if resolution == '10m':
            return 10
        elif resolution == '30m':
            return 30
        elif resolution == '90m':
            return 90
        else:
            return 30  # Default