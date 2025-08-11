"""
Landcover Extraction Step

Extracts landcover data for a specified extent.
Uses data clients and processors for actual landcover acquisition and processing.
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

class LandcoverExtractionStep:
    """
    Landcover extraction step that gets landcover data for a specified extent.
    Uses SpatialLayersClient with NO FALLBACKS - real data required.
    """
    
    def __init__(self, workspace_dir: Path = None):
        """
        Initialize landcover extraction step
        
        Parameters:
        -----------
        workspace_dir : Path, optional
            Working directory for processing
        """
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "landcover_extraction"
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize client
        self.spatial_client = SpatialLayersClient()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # RAVEN-compatible landcover classes
        self.raven_landcover_classes = {
            'FOREST': {'code': 42, 'manning_n': 0.35, 'canopy_cover': 0.8},
            'GRASSLAND': {'code': 71, 'manning_n': 0.25, 'canopy_cover': 0.3},
            'CROPLAND': {'code': 82, 'manning_n': 0.20, 'canopy_cover': 0.4},
            'URBAN': {'code': 24, 'manning_n': 0.15, 'canopy_cover': 0.1},
            'WATER': {'code': 11, 'manning_n': 0.03, 'canopy_cover': 0.0},
            'WETLAND': {'code': 95, 'manning_n': 0.40, 'canopy_cover': 0.5},
            'BARREN': {'code': 31, 'manning_n': 0.10, 'canopy_cover': 0.0}
        }
        
    def _setup_logging(self):
        """Setup logging for the step"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(self.workspace_dir / "landcover_extraction.log")
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def execute(self, bounds: Tuple[float, float, float, float], 
                output_filename: str = "landcover.tif") -> Dict[str, Any]:
        """
        Execute landcover extraction for specified bounds
        
        Parameters:
        -----------
        bounds : Tuple[float, float, float, float]
            Bounding box (minx, miny, maxx, maxy) in WGS84
        output_filename : str, optional
            Output filename for landcover raster
            
        Returns:
        --------
        Dict[str, Any]
            Landcover extraction results
        """
        
        self.logger.info(f"Starting landcover extraction for bounds: {bounds}")
        
        try:
            # ULTRA-SIMPLE: Save to data/ folder with simple name
            data_dir = self.workspace_dir / "data"
            data_dir.mkdir(exist_ok=True)
            landcover_file = data_dir / "landcover.tif"
            
            # Try to get real landcover data first
            self.logger.info("Attempting to get real landcover data")
            real_data_result = self._try_get_real_landcover(bounds, landcover_file)
            
            if real_data_result['success']:
                self.logger.info("Successfully obtained real landcover data")
                return real_data_result
            else:
                error_msg = f"Real landcover data unavailable: {real_data_result.get('error', 'Unknown error')}"
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'step_type': 'landcover_extraction'
                }
            
        except Exception as e:
            error_msg = f"Landcover extraction failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'step_type': 'landcover_extraction'
            }
    
    def _try_get_real_landcover(self, bounds: Tuple[float, float, float, float], 
                               output_file: Path) -> Dict[str, Any]:
        """Try to get real landcover data from spatial client"""
        
        try:
            from clients.data_clients.spatial_client import SpatialLayersClient
            
            # Initialize spatial client
            spatial_client = SpatialLayersClient()
            
            # Download real landcover data using NRCan
            result = spatial_client.get_landcover_for_watershed(
                bbox=bounds,
                output_path=output_file,
                year=2020  # Use most recent available
            )
            
            if result.get('success', False) and output_file.exists():
                return {
                    'success': True,
                    'landcover_file': str(output_file),
                    'source': result.get('source', 'nrcan'),
                    'year': 2020
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Landcover download failed')
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Real landcover data acquisition failed: {str(e)}"
            }
    
    # REMOVED: _create_synthetic_landcover - NO FALLBACKS ALLOWED
    # All landcover data must come from real sources via SpatialLayersClient
    
    def _generate_realistic_landcover_pattern(self, width: int, height: int, 
                                            bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """Generate realistic landcover pattern based on geographic context"""
        
        # Initialize with forest as default
        landcover = np.full((height, width), self.raven_landcover_classes['FOREST']['code'], dtype=np.uint8)
        
        # Create random seed based on bounds for reproducible patterns
        seed = int(abs(bounds[0] * 1000) + abs(bounds[1] * 1000)) % 2**32
        np.random.seed(seed)
        
        # Add geographic realism based on latitude
        latitude_center = (bounds[1] + bounds[3]) / 2
        
        # Northern regions - more forest and wetland
        if latitude_center > 50:
            # More wetlands in northern areas
            wetland_mask = np.random.random((height, width)) < 0.15
            landcover[wetland_mask] = self.raven_landcover_classes['WETLAND']['code']
            
            # Some barren areas in far north
            if latitude_center > 60:
                barren_mask = np.random.random((height, width)) < 0.10
                landcover[barren_mask] = self.raven_landcover_classes['BARREN']['code']
        
        # Southern regions - more agriculture and urban
        elif latitude_center < 50:
            # More cropland in agricultural regions
            crop_mask = np.random.random((height, width)) < 0.25
            landcover[crop_mask] = self.raven_landcover_classes['CROPLAND']['code']
            
            # More grassland
            grass_mask = np.random.random((height, width)) < 0.20
            landcover[grass_mask] = self.raven_landcover_classes['GRASSLAND']['code']
        
        # Add urban areas (usually in center or along features)
        urban_center_y, urban_center_x = height // 2, width // 2
        urban_radius = min(width, height) // 8
        
        y_coords, x_coords = np.ogrid[:height, :width]
        urban_distance = np.sqrt((x_coords - urban_center_x)**2 + (y_coords - urban_center_y)**2)
        urban_mask = (urban_distance < urban_radius) & (np.random.random((height, width)) < 0.7)
        landcover[urban_mask] = self.raven_landcover_classes['URBAN']['code']
        
        # Add water bodies (rivers, lakes)
        # Create meandering water features
        for i in range(3):  # 3 water features
            start_y = np.random.randint(0, height)
            start_x = np.random.randint(0, width)
            
            # Create meandering line
            water_points = [(start_y, start_x)]
            current_y, current_x = start_y, start_x
            
            for step in range(100):
                # Random walk with bias
                dy = np.random.randint(-3, 4)
                dx = np.random.randint(-3, 4)
                current_y = max(0, min(height - 1, current_y + dy))
                current_x = max(0, min(width - 1, current_x + dx))
                water_points.append((current_y, current_x))
            
            # Draw water line with width
            for y, x in water_points:
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            landcover[ny, nx] = self.raven_landcover_classes['WATER']['code']
        
        return landcover


if __name__ == "__main__":
    # Example usage
    step = LandcoverExtractionStep()
    
    # Test with sample bounds
    result = step.execute(
        bounds=(-74.0, 45.0, -73.0, 46.0),
        output_filename="test_landcover.tif"
    )
    
    if result['success']:
        print(f"Landcover extraction completed successfully!")
        print(f"Landcover file: {result['landcover_file']}")
        print(f"Source: {result['source']}")
        print(f"Class distribution: {result['class_distribution']}")
    else:
        print(f"Landcover extraction failed: {result['error']}")