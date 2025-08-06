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
        self.workspace_dir = workspace_dir or Path.cwd() / "soil_extraction"
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
        Execute soil extraction for specified bounds
        
        Parameters:
        -----------
        bounds : Tuple[float, float, float, float]
            Bounding box (minx, miny, maxx, maxy) in WGS84
        output_filename : str, optional
            Output filename for soil raster
            
        Returns:
        --------
        Dict[str, Any]
            Soil extraction results
        """
        
        self.logger.info(f"Starting soil extraction for bounds: {bounds}")
        
        try:
            # Prepare output path
            soil_file = self.workspace_dir / output_filename
            
            # Try to get real soil data first
            self.logger.info("Attempting to get real soil data")
            real_data_result = self._try_get_real_soil_data(bounds, soil_file)
            
            if real_data_result['success']:
                self.logger.info("Successfully obtained real soil data")
                return real_data_result
            else:
                error_msg = f"Real soil data unavailable: {real_data_result.get('error', 'Unknown error')} - No synthetic fallback provided"
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
    
    def _try_get_real_soil_data(self, bounds: Tuple[float, float, float, float], 
                               output_file: Path) -> Dict[str, Any]:
        """Try to get real soil data from spatial client"""
        
        try:
            # This would be implemented when real soil data client methods are available
            # For now, return failure to trigger synthetic data creation
            return {
                'success': False,
                'error': 'Real soil data client not yet implemented'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Real soil data acquisition failed: {str(e)}"
            }
    
    def _create_synthetic_soil_data(self, bounds: Tuple[float, float, float, float], 
                                   output_file: Path) -> Dict[str, Any]:
        """Create synthetic soil data as fallback"""
        
        try:
            # Create soil raster
            width, height = 1000, 1000  # High resolution synthetic data
            transform = from_bounds(*bounds, width, height)
            
            # Create realistic soil pattern
            soil_data = self._generate_realistic_soil_pattern(width, height, bounds)
            
            # Save soil raster
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
                dst.write(soil_data, 1)
            
            # Calculate class statistics
            unique_classes, counts = np.unique(soil_data[soil_data > 0], return_counts=True)
            total_pixels = counts.sum()
            
            class_percentages = {}
            average_properties = {
                'hydraulic_conductivity': 0.0,
                'porosity': 0.0,
                'field_capacity': 0.0,
                'wilting_point': 0.0,
                'bulk_density': 0.0
            }
            
            for class_code, count in zip(unique_classes, counts):
                percentage = (count / total_pixels) * 100
                weight = percentage / 100.0
                
                # Find class name and properties
                class_name = 'UNKNOWN'
                for name, info in self.raven_soil_classes.items():
                    if info['code'] == class_code:
                        class_name = name
                        # Add weighted contribution to average properties
                        for prop, value in info.items():
                            if prop != 'code' and prop in average_properties:
                                average_properties[prop] += value * weight
                        break
                
                class_percentages[class_name] = round(percentage, 1)
            
            # Round average properties
            for prop in average_properties:
                average_properties[prop] = round(average_properties[prop], 3)
            
            # Calculate file size
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            
            results = {
                'success': True,
                'step_type': 'soil_extraction',
                'soil_file': str(output_file),
                'bounds': bounds,
                'source': 'Synthetic',
                'data_type': 'synthetic_realistic',
                'resolution_pixels': f"{width}x{height}",
                'file_size_mb': round(file_size_mb, 2),
                'raven_soil_classes': self.raven_soil_classes,
                'class_distribution': class_percentages,
                'average_properties': average_properties,
                'workspace': str(self.workspace_dir),
                'files_created': [str(output_file)]
            }
            
            self.logger.info(f"Synthetic soil data created successfully")
            self.logger.info(f"Output file: {output_file}")
            self.logger.info(f"Class distribution: {class_percentages}")
            self.logger.info(f"Average hydraulic conductivity: {average_properties['hydraulic_conductivity']:.1f} mm/hr")
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Synthetic soil data creation failed: {str(e)}",
                'step_type': 'soil_extraction'
            }
    
    def _generate_realistic_soil_pattern(self, width: int, height: int, 
                                       bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """Generate realistic soil pattern based on geographic and topographic context"""
        
        # Initialize with loam as default (most common agricultural soil)
        soil = np.full((height, width), self.raven_soil_classes['LOAM']['code'], dtype=np.uint8)
        
        # Create random seed based on bounds for reproducible patterns
        seed = int(abs(bounds[0] * 1000) + abs(bounds[1] * 1000)) % 2**32
        np.random.seed(seed)
        
        # Create elevation-like gradient for soil distribution
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Simulate elevation gradient (higher = sandier, lower = clayier)
        elevation_gradient = (y_coords / height + x_coords / width) / 2
        elevation_noise = np.random.random((height, width)) * 0.3
        elevation_sim = elevation_gradient + elevation_noise
        
        # Higher elevations - more sandy soils (better drainage)
        high_elevation_mask = elevation_sim > 0.7
        sandy_areas = high_elevation_mask & (np.random.random((height, width)) < 0.6)
        soil[sandy_areas] = self.raven_soil_classes['SAND']['code']
        
        # Some sandy loam in moderately high areas
        mid_high_mask = (elevation_sim > 0.5) & (elevation_sim <= 0.7)
        sandy_loam_areas = mid_high_mask & (np.random.random((height, width)) < 0.4)
        soil[sandy_loam_areas] = self.raven_soil_classes['SANDY_LOAM']['code']
        
        # Lower elevations - more clay soils (poor drainage)
        low_elevation_mask = elevation_sim < 0.3
        clay_areas = low_elevation_mask & (np.random.random((height, width)) < 0.5)
        soil[clay_areas] = self.raven_soil_classes['CLAY']['code']
        
        # Some clay loam in moderately low areas
        mid_low_mask = (elevation_sim >= 0.3) & (elevation_sim <= 0.5)
        clay_loam_areas = mid_low_mask & (np.random.random((height, width)) < 0.3)
        soil[clay_loam_areas] = self.raven_soil_classes['CLAY_LOAM']['code']
        
        # Add some silt areas near water-like features
        # Create meandering patterns that might represent old river deposits
        for i in range(2):  # 2 silt deposit areas
            center_y = np.random.randint(height // 4, 3 * height // 4)
            center_x = np.random.randint(width // 4, 3 * width // 4)
            
            # Create elongated silt deposits
            for j in range(50):
                y_offset = int(np.random.normal(0, height // 8))
                x_offset = int(np.random.normal(0, width // 20))  # More elongated
                
                y = np.clip(center_y + y_offset, 0, height - 1)
                x = np.clip(center_x + x_offset, 0, width - 1)
                
                # Create small silt patches
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < height and 0 <= nx < width and 
                            np.random.random() < 0.7):
                            soil[ny, nx] = self.raven_soil_classes['SILT']['code']
        
        # Add some spatial clustering to make patterns more realistic
        # Apply a smoothing filter to create more natural transitions
        from scipy import ndimage
        try:
            # Apply median filter to smooth transitions
            soil_smoothed = ndimage.median_filter(soil, size=3)
            
            # Only apply smoothing where it makes sense (not too aggressive)
            smooth_mask = np.random.random((height, width)) < 0.3
            soil[smooth_mask] = soil_smoothed[smooth_mask]
        except ImportError:
            # If scipy not available, skip smoothing
            pass
        
        return soil


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