"""
DEM Clipping Step

Downloads and clips DEM data for a specified extent.
Uses data clients for actual DEM acquisition.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from clients.data_clients.spatial_client import SpatialLayersClient

class DEMClippingStep:
    """
    DEM clipping step that downloads and clips DEM data for a specified extent.
    Uses SpatialLayersClient for actual DEM acquisition.
    """
    
    def __init__(self, workspace_dir: Path = None):
        """
        Initialize DEM clipping step
        
        Parameters:
        -----------
        workspace_dir : Path, optional
            Working directory for processing
        """
        self.workspace_dir = workspace_dir or Path.cwd() / "dem_clipping"
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize client
        self.spatial_client = SpatialLayersClient()
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the step"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(self.workspace_dir / "dem_clipping.log")
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def execute(self, bounds: Tuple[float, float, float, float], 
                resolution: int = 30, output_filename: str = "clipped_dem.tif") -> Dict[str, Any]:
        """
        Execute DEM clipping for specified bounds
        
        Parameters:
        -----------
        bounds : Tuple[float, float, float, float]
            Bounding box (minx, miny, maxx, maxy) in WGS84
        resolution : int, optional
            DEM resolution in meters (default: 30)
        output_filename : str, optional
            Output filename for clipped DEM
            
        Returns:
        --------
        Dict[str, Any]
            DEM clipping results
        """
        
        self.logger.info(f"Starting DEM clipping for bounds: {bounds}")
        self.logger.info(f"Resolution: {resolution}m")
        
        try:
            # Prepare output path
            dem_file = self.workspace_dir / output_filename
            
            # Download and clip DEM using spatial client
            self.logger.info("Downloading DEM data")
            dem_result = self.spatial_client.get_dem_for_watershed(
                bbox=bounds,
                output_path=dem_file,
                resolution=resolution
            )
            
            if not dem_result['success']:
                return {
                    'success': False,
                    'error': f"DEM download failed: {dem_result.get('error', 'Unknown error')}",
                    'step_type': 'dem_clipping'
                }
            
            # Verify file exists
            if not Path(dem_result['file_path']).exists():
                return {
                    'success': False,
                    'error': f"DEM file not found after download: {dem_result['file_path']}",
                    'step_type': 'dem_clipping'
                }
            
            # Calculate file size
            file_size_mb = Path(dem_result['file_path']).stat().st_size / (1024 * 1024)
            
            results = {
                'success': True,
                'step_type': 'dem_clipping',
                'dem_file': dem_result['file_path'],
                'bounds': bounds,
                'resolution_m': resolution,
                'file_size_mb': round(file_size_mb, 2),
                'source': dem_result.get('source', 'USGS 3DEP'),
                'workspace': str(self.workspace_dir),
                'files_created': [dem_result['file_path']]
            }
            
            self.logger.info(f"DEM clipping completed successfully")
            self.logger.info(f"Output file: {dem_result['file_path']}")
            self.logger.info(f"File size: {file_size_mb:.1f} MB")
            
            return results
            
        except Exception as e:
            error_msg = f"DEM clipping failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'step_type': 'dem_clipping'
            }


if __name__ == "__main__":
    # Example usage
    step = DEMClippingStep()
    
    # Test with sample bounds
    result = step.execute(
        bounds=(-74.0, 45.0, -73.0, 46.0),
        resolution=30,
        output_filename="test_dem.tif"
    )
    
    if result['success']:
        print(f"DEM clipping completed successfully!")
        print(f"DEM file: {result['dem_file']}")
        print(f"File size: {result['file_size_mb']} MB")
    else:
        print(f"DEM clipping failed: {result['error']}")