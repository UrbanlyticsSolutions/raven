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
    Uses SpatialLayersClient and creates synthetic data as fallback.
    """
    
    def __init__(self, workspace_dir: Path = None):
        """
        Initialize landcover extraction step
        
        Parameters:
        -----------
        workspace_dir : Path, optional
            Working directory for processing
        """
        self.workspace_dir = workspace_dir or Path.cwd() / "landcover_extraction"
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
            # Prepare output path
            landcover_file = self.workspace_dir / output_filename
            
            # Try to get real landcover data first
            self.logger.info("Attempting to get real landcover data")
            real_data_result = self._try_get_real_landcover(bounds, landcover_file)
            
            if real_data_result['success']:
                self.logger.info("Successfully obtained real landcover data")
                return real_data_result
            else:
                error_msg = f"Real landcover data unavailable: {real_data_result.get('error', 'Unknown error')} - No synthetic fallback provided"
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
            # This would be implemented when real landcover client methods are available
            # For now, return failure to trigger synthetic data creation
            return {
                'success': False,
                'error': 'Real landcover data client not yet implemented'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Real landcover data acquisition failed: {str(e)}"
            }
    
    def _create_synthetic_landcover(self, bounds: Tuple[float, float, float, float], 
                                   output_file: Path) -> Dict[str, Any]:
        """Create synthetic landcover data as fallback"""
        
        try:
            # Create landcover raster
            width, height = 1000, 1000  # High resolution synthetic data
            transform = from_bounds(*bounds, width, height)
            
            # Create realistic landcover pattern
            landcover_data = self._generate_realistic_landcover_pattern(width, height, bounds)
            
            # Save landcover raster
            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,
                'dtype': rasterio.uint8,
                'crs': 'EPSG:4326',
                'transform': transform,
                'nodata': 0,
                'compress': 'lzw'
            }
            
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(landcover_data, 1)
            
            # Calculate class statistics
            unique_classes, counts = np.unique(landcover_data[landcover_data > 0], return_counts=True)
            total_pixels = counts.sum()
            
            class_percentages = {}
            for class_code, count in zip(unique_classes, counts):
                percentage = (count / total_pixels) * 100
                # Find class name
                class_name = 'UNKNOWN'
                for name, info in self.raven_landcover_classes.items():
                    if info['code'] == class_code:
                        class_name = name
                        break
                class_percentages[class_name] = round(percentage, 1)
            
            # Calculate file size
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            
            results = {
                'success': True,
                'step_type': 'landcover_extraction',
                'landcover_file': str(output_file),
                'bounds': bounds,
                'source': 'Synthetic',
                'data_type': 'synthetic_realistic',
                'resolution_pixels': f"{width}x{height}",
                'file_size_mb': round(file_size_mb, 2),
                'raven_classes': self.raven_landcover_classes,
                'class_distribution': class_percentages,
                'workspace': str(self.workspace_dir),
                'files_created': [str(output_file)]
            }
            
            self.logger.info(f"Synthetic landcover data created successfully")
            self.logger.info(f"Output file: {output_file}")
            self.logger.info(f"Class distribution: {class_percentages}")
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Synthetic landcover creation failed: {str(e)}",
                'step_type': 'landcover_extraction'
            }
    
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