"""
Soil Extraction Step

Extracts soil data for a specified extent.
Uses data clients and processors for actual soil data acquisition and processing.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import numpy as np
import rasterio
from rasterio.transform import from_bounds

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from clients.data_clients.spatial_client import SpatialLayersClient

class SoilExtractionStep:
    """
    Soil extraction step that gets soil data for a specified extent.
    Uses SpatialLayersClient and creates synthetic data as fallback.
    """
    
    def __init__(self, workspace_dir: Path = None):
        """
        Initialize soil extraction step
        
        Parameters:
        -----------
        workspace_dir : Path, optional
            Working directory for processing
        """
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "soil_extraction"
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize client
        self.spatial_client = SpatialLayersClient()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # RAVEN-compatible soil classes with hydraulic properties
        self.raven_soil_classes = {
            'SAND': {
                'code': 1,
                'hydraulic_conductivity': 120.0,  # mm/hr
                'porosity': 0.45,
                'field_capacity': 0.12,
                'wilting_point': 0.05,
                'bulk_density': 1.65  # g/cm³
            },
            'LOAM': {
                'code': 2,
                'hydraulic_conductivity': 25.0,   # mm/hr
                'porosity': 0.50,
                'field_capacity': 0.27,
                'wilting_point': 0.13,
                'bulk_density': 1.40  # g/cm³
            },
            'CLAY': {
                'code': 3,
                'hydraulic_conductivity': 5.0,    # mm/hr
                'porosity': 0.55,
                'field_capacity': 0.39,
                'wilting_point': 0.25,
                'bulk_density': 1.20  # g/cm³
            },
            'SILT': {
                'code': 4,
                'hydraulic_conductivity': 15.0,   # mm/hr
                'porosity': 0.48,
                'field_capacity': 0.33,
                'wilting_point': 0.15,
                'bulk_density': 1.35  # g/cm³
            },
            'SANDY_LOAM': {
                'code': 5,
                'hydraulic_conductivity': 45.0,   # mm/hr
                'porosity': 0.47,
                'field_capacity': 0.20,
                'wilting_point': 0.09,
                'bulk_density': 1.50  # g/cm³
            },
            'CLAY_LOAM': {
                'code': 6,
                'hydraulic_conductivity': 12.0,   # mm/hr
                'porosity': 0.52,
                'field_capacity': 0.35,
                'wilting_point': 0.20,
                'bulk_density': 1.30  # g/cm³
            }
        }
        
    def _setup_logging(self):
        """Setup logging for the step"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(self.workspace_dir / "soil_extraction.log")
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def execute(self, bounds: Tuple[float, float, float, float], 
                output_filename: str = "soil.tif") -> Dict[str, Any]:
        """
        Execute soil extraction for specified bounds using SoilGrids API
        
        Parameters:
        -----------
        bounds : Tuple[float, float, float, float]
            Bounding box (minx, miny, maxx, maxy) in WGS84
        output_filename : str, optional
            Output filename for soil raster
            
        Returns:
        --------
        Dict[str, Any]
            Soil extraction results from SoilGrids API
        """
        
        self.logger.info(f"Starting SoilGrids soil extraction for bounds: {bounds}")
        
        try:
            # ULTRA-SIMPLE: Save to data/ folder with simple name
            data_dir = self.workspace_dir / "data"
            data_dir.mkdir(exist_ok=True)
            soil_file = data_dir / "soil.tif"
            
            # Get real soil data from SoilGrids API
            self.logger.info("Getting soil data from SoilGrids API")
            soil_result = self._get_soilgrids_data(bounds, soil_file)
            
            if soil_result['success']:
                self.logger.info("Successfully obtained SoilGrids soil data")
                return soil_result
            else:
                error_msg = f"SoilGrids API failed: {soil_result.get('error', 'Unknown error')}"
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'step_type': 'soil_extraction'
                }
            
        except Exception as e:
            error_msg = f"Soil extraction failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'step_type': 'soil_extraction'
            }
    
    def _get_soilgrids_data(self, bounds: Tuple[float, float, float, float], 
                           output_file: Path) -> Dict[str, Any]:
        """Get real soil data from SoilGrids WCS service"""
        
        try:
            from clients.data_clients.soil_client import SoilDataClient
            
            # Initialize soil client with output directory
            soil_client = SoilDataClient(output_dir=str(output_file.parent))
            
            # Get all soil texture rasters (clay, sand, silt) for the bounding box
            print(f"Downloading soil texture data for bounds: {bounds}")
            raster_results = soil_client.get_soil_texture_rasters_for_bbox(
                bbox=bounds,
                depth='0-5cm',  # Top soil layer
                width=256,      # Good resolution for watershed modeling
                height=256
            )
            
            # Check if we got at least one successful raster download
            successful_downloads = [path for path in raster_results.values() if path is not None]
            
            if successful_downloads:
                # Keep all soil texture files separate, create a simple soil.tif copy from clay
                if raster_results.get('clay'):
                    primary_file = Path(raster_results['clay'])
                    # Copy clay file to soil.tif instead of renaming
                    import shutil
                    shutil.copy2(primary_file, output_file)
                else:
                    # Copy the first available raster to soil.tif
                    primary_file = Path(successful_downloads[0])
                    import shutil
                    shutil.copy2(primary_file, output_file)
                
                return {
                    'success': True,
                    'soil_file': str(output_file),
                    'source': 'SoilGrids v2.0 WCS Service',
                    'texture_files': {
                        prop: path for prop, path in raster_results.items() 
                        if path is not None
                    },
                    'successful_downloads': len(successful_downloads),
                    'total_requested': len(raster_results),
                    'step_type': 'soil_extraction'
                }
            else:
                return {
                    'success': False,
                    'error': 'No soil texture rasters could be downloaded from SoilGrids WCS',
                    'step_type': 'soil_extraction'
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"SoilGrids WCS data acquisition failed: {str(e)}",
                'step_type': 'soil_extraction'
            }
    


if __name__ == "__main__":
    # Example usage
    step = SoilExtractionStep()
    
    # Test with sample bounds
    result = step.execute(
        bounds=(-74.0, 45.0, -73.0, 46.0),
        output_filename="test_soil.tif"
    )
    
    if result['success']:
        print(f"Soil extraction completed successfully!")
        print(f"Soil file: {result['soil_file']}")
        print(f"Source: {result['source']}")
        print(f"Class distribution: {result['class_distribution']}")
        print(f"Average properties: {result['average_properties']}")
    else:
        print(f"Soil extraction failed: {result['error']}")