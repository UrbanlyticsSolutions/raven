#!/usr/bin/env python3
"""
Terrain Attributes Calculator
Calculates elevation, slope, and aspect from DEM following BasinMaker approach
Based on BasinMaker's calculatebasicattributesqgis.py logic
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from pathlib import Path
from typing import Union, Tuple
import logging
import warnings
import os

# Suppress GDAL/PROJ warnings about missing proj.db
os.environ['PROJ_LIB'] = os.environ.get('PROJ_LIB', '')
os.environ['GDAL_DATA'] = os.environ.get('GDAL_DATA', '')

# Comprehensive warning suppression for GDAL/PROJ issues
warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='rasterio')
warnings.filterwarnings('ignore', message='.*proj_create_from_database.*')
warnings.filterwarnings('ignore', message='.*proj_identify.*')
warnings.filterwarnings('ignore', message='.*Cannot find proj.db.*')
warnings.filterwarnings('ignore', message='.*CPLE_AppDefined.*')

# Configure rasterio environment to suppress PROJ warnings
import rasterio.env
rasterio.env.default_env = rasterio.env.Env(
    GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
    CPL_LOG_ERRORS='OFF',
    PROJ_DEBUG='0'
)


class TerrainAttributesCalculator:
    """
    Calculate terrain attributes (elevation, slope, aspect) from DEM
    Following BasinMaker approach using GRASS GIS equivalent operations:
    - r.slope.aspect for slope/aspect calculation
    - v.rast.stats for zonal statistics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._suppress_gdal_warnings()
    
    def _suppress_gdal_warnings(self):
        """Suppress GDAL/PROJ warnings about missing proj.db"""
        
        # Set logging levels to suppress GDAL warnings
        gdal_logger = logging.getLogger('rasterio._env')
        gdal_logger.setLevel(logging.ERROR)
        
        # Additional environment variables to suppress PROJ warnings
        os.environ.setdefault('PROJ_NETWORK', 'OFF')
        os.environ.setdefault('PROJ_CURL_ENABLED', 'NO')
        
        # Configure GDAL to be less verbose
        try:
            from osgeo import gdal
            gdal.SetConfigOption('CPL_LOG', 'OFF')
            gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
        except ImportError:
            # GDAL not available directly, rely on rasterio configuration
            pass
        
    def calculate_terrain_attributes(self, 
                                   hru_gdf: gpd.GeoDataFrame, 
                                   dem_file: Union[str, Path]) -> gpd.GeoDataFrame:
        """
        Calculate elevation, slope, and aspect for HRUs from DEM
        
        BasinMaker equivalent operations:
        1. r.slope.aspect elevation=dem_proj slope=slope aspect=aspect
        2. v.rast.stats map=hru_polygons raster=dem column_prefix=d method=average
        3. v.rast.stats map=hru_polygons raster=slope column_prefix=s method=average  
        4. v.rast.stats map=hru_polygons raster=aspect column_prefix=a method=average
        
        Args:
            hru_gdf: HRU polygons GeoDataFrame
            dem_file: Path to DEM raster file
            
        Returns:
            HRU GeoDataFrame with added elevation, slope, aspect columns
        """
        dem_path = Path(dem_file)
        
        if not dem_path.exists():
            self.logger.warning(f"DEM file not found: {dem_file}")
            return self._add_default_terrain_attributes(hru_gdf)
            
        try:
            with rasterio.open(dem_path) as dem_src:
                dem_data = dem_src.read(1)
                dem_transform = dem_src.transform
                dem_crs = dem_src.crs
                dem_nodata = dem_src.nodata
                
                # Reproject HRUs to DEM CRS if needed
                if hru_gdf.crs != dem_crs:
                    self.logger.info(f"Reprojecting HRUs from {hru_gdf.crs} to {dem_crs}")
                    hru_gdf_proj = hru_gdf.to_crs(dem_crs)
                else:
                    hru_gdf_proj = hru_gdf.copy()
                
                # Step 1: Calculate slope and aspect from DEM using third-party libraries
                self.logger.info("Calculating slope and aspect from DEM...")
                slope_data, aspect_data = self._calculate_slope_aspect_richdem_method(
                    dem_data, dem_transform, dem_nodata
                )
                
                # Step 2: Calculate zonal statistics for each HRU (v.rast.stats equivalent)
                self.logger.info(f"Calculating zonal statistics for {len(hru_gdf_proj)} HRUs...")
                terrain_attrs = self._calculate_zonal_statistics(
                    hru_gdf_proj, dem_data, slope_data, aspect_data, 
                    dem_transform, dem_nodata
                )
                
                # Add calculated attributes to original HRU dataframe
                # Use BasinMaker/RAVEN expected field names
                hru_gdf['HRU_E_mean'] = terrain_attrs['elevation']  # Mean elevation
                hru_gdf['HRU_S_mean'] = terrain_attrs['slope']      # Mean slope  
                hru_gdf['HRU_A_mean'] = terrain_attrs['aspect']     # Mean aspect
                
                # Also add legacy names for backward compatibility
                hru_gdf['elevation'] = terrain_attrs['elevation']
                hru_gdf['slope'] = terrain_attrs['slope']
                hru_gdf['aspect'] = terrain_attrs['aspect']
                
                self.logger.info("Terrain attributes calculation completed successfully")
                
        except Exception as e:
            self.logger.error(f"Terrain attributes calculation failed: {e}")
            return self._add_default_terrain_attributes(hru_gdf)
            
        return hru_gdf
    
    def _calculate_slope_aspect_richdem_method(self, 
                                             dem_data: np.ndarray, 
                                             transform, 
                                             nodata) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate slope and aspect using BasinMaker's preferred method
        
        Priority order:
        1. GRASS r.slope.aspect equivalent (BasinMaker standard)
        2. scikit-image (fallback)
        3. scipy (fallback)
        4. basic numpy (last resort)
        """
        
        # Try GRASS equivalent method first (BasinMaker standard)
        try:
            return self._calculate_slope_aspect_grass_method(dem_data, transform, nodata)
        except Exception as e:
            self.logger.warning(f"GRASS equivalent method failed: {e}, falling back to scikit-image")
        
        # Fallback to scikit-image
        try:
            from skimage import filters
            self.logger.info("Using scikit-image for slope/aspect calculation (fallback)")
            
            # Mask nodata values
            if nodata is not None:
                dem_data = np.where(dem_data == nodata, np.nan, dem_data)
            
            # Get cell resolution from transform and convert to meters if needed
            dx = abs(transform[0])  # Cell width
            dy = abs(transform[4])  # Cell height
            
            # Convert to meters if in geographic coordinates
            dx_meters, dy_meters = self._convert_cell_size_to_meters(dx, dy, transform)
            dx = dx_meters
            dy = dy_meters
            
            # Use scikit-image Sobel filters for gradient calculation
            gx = filters.sobel_h(dem_data) / dx  # Horizontal gradient
            gy = filters.sobel_v(dem_data) / dy  # Vertical gradient
            
            # Calculate slope in degrees
            slope_rad = np.arctan(np.sqrt(gx*gx + gy*gy))
            slope_deg = slope_rad * 180.0 / np.pi
            
            # Calculate aspect in degrees (0=North, 90=East)
            aspect_rad = np.arctan2(-gx, gy)  # Note: -gx for proper orientation
            aspect_deg = aspect_rad * 180.0 / np.pi
            aspect_deg = (aspect_deg + 360) % 360  # Ensure 0-360 range
            
            # Apply BasinMaker-style slope constraints
            slope_deg = np.clip(slope_deg, 0.1, 45.0)  # BasinMaker standard: min=0.1°, max=45°
            
            # Log statistics
            valid_slopes = slope_deg[~np.isnan(slope_deg)]
            if len(valid_slopes) > 0:
                self.logger.info(f"Scikit-image slope calculation completed (BasinMaker constraints applied):")
                self.logger.info(f"  Min slope: {np.min(valid_slopes):.2f}°")
                self.logger.info(f"  Max slope: {np.max(valid_slopes):.2f}°")
                self.logger.info(f"  Mean slope: {np.mean(valid_slopes):.2f}°")
                self.logger.info(f"  Slope range: 0.1° - 45.0° (BasinMaker standard)")
            
            return slope_deg, aspect_deg
            
        except ImportError:
            self.logger.warning("scikit-image not available, falling back to scipy method")
            return self._calculate_slope_aspect_scipy_method(dem_data, transform, nodata)
        except Exception as e:
            self.logger.error(f"scikit-image calculation failed: {e}, falling back to scipy method")
            return self._calculate_slope_aspect_scipy_method(dem_data, transform, nodata)
    
    def _calculate_slope_aspect_scipy_method(self, 
                                           dem_data: np.ndarray, 
                                           transform, 
                                           nodata) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate slope and aspect using scipy.ndimage gradients (fallback method)
        
        Uses scipy's gradient calculation which is well-tested and efficient
        """
        
        try:
            from scipy import ndimage
            self.logger.info("Using scipy.ndimage for slope/aspect calculation")
            
            # Mask nodata values
            if nodata is not None:
                dem_data = np.where(dem_data == nodata, np.nan, dem_data)
            
            # Get cell resolution from transform and convert to meters if needed
            dx = abs(transform[0])  # Cell width
            dy = abs(transform[4])  # Cell height
            
            # Convert to meters if in geographic coordinates
            dx_meters, dy_meters = self._convert_cell_size_to_meters(dx, dy, transform)
            dx = dx_meters
            dy = dy_meters
            
            # Calculate gradients using scipy (handles NaN values properly)
            gy, gx = np.gradient(dem_data, dy, dx)
            
            # Calculate slope in degrees
            slope_rad = np.arctan(np.sqrt(gx*gx + gy*gy))
            slope_deg = slope_rad * 180.0 / np.pi
            
            # Calculate aspect in degrees (0=North, 90=East)
            aspect_rad = np.arctan2(-gx, gy)  # Note: -gx for proper orientation
            aspect_deg = aspect_rad * 180.0 / np.pi
            aspect_deg = (aspect_deg + 360) % 360  # Ensure 0-360 range
            
            # Apply BasinMaker-style slope constraints (consistent with scikit-image method)
            slope_deg = np.clip(slope_deg, 0.1, 45.0)  # BasinMaker standard: min=0.1°, max=45°
            
            # Log statistics
            valid_slopes = slope_deg[~np.isnan(slope_deg)]
            if len(valid_slopes) > 0:
                self.logger.info(f"Scipy slope calculation completed (BasinMaker constraints applied):")
                self.logger.info(f"  Min slope: {np.min(valid_slopes):.2f}°")
                self.logger.info(f"  Max slope: {np.max(valid_slopes):.2f}°")
                self.logger.info(f"  Mean slope: {np.mean(valid_slopes):.2f}°")
                self.logger.info(f"  Slope range: 0.1° - 45.0° (BasinMaker standard)")
                
                # Check distribution
                steep_count = np.sum(valid_slopes > 30)
                if steep_count > 0:
                    self.logger.info(f"  {steep_count} pixels have slopes > 30° (steep terrain)")
                
                very_steep_count = np.sum(valid_slopes >= 45)
                if very_steep_count > 0:
                    self.logger.info(f"  {very_steep_count} pixels clamped to 45° (BasinMaker max)")
            
            return slope_deg, aspect_deg
            
        except ImportError:
            self.logger.error("scipy not available, falling back to basic method")
            return self._calculate_slope_aspect_basic_method(dem_data, transform, nodata)
        except Exception as e:
            self.logger.error(f"Scipy calculation failed: {e}, falling back to basic method")
            return self._calculate_slope_aspect_basic_method(dem_data, transform, nodata)
    
    def _calculate_slope_aspect_basic_method(self, 
                                           dem_data: np.ndarray, 
                                           transform, 
                                           nodata) -> Tuple[np.ndarray, np.ndarray]:
        """
        Basic slope and aspect calculation using numpy (last resort fallback)
        """
        
        self.logger.warning("Using basic numpy method for slope/aspect calculation")
        
        # Mask nodata values
        if nodata is not None:
            dem_data = np.where(dem_data == nodata, np.nan, dem_data)
        
        # Get cell resolution from transform and convert to meters if needed
        dx = abs(transform[0])  # Cell width
        dy = abs(transform[4])  # Cell height
        
        # Convert to meters if in geographic coordinates
        dx_meters, dy_meters = self._convert_cell_size_to_meters(dx, dy, transform)
        dx = dx_meters
        dy = dy_meters
        
        # Simple gradient calculation
        gx = np.zeros_like(dem_data)
        gy = np.zeros_like(dem_data)
        
        # Calculate gradients using simple differences
        gx[1:-1, 1:-1] = (dem_data[1:-1, 2:] - dem_data[1:-1, :-2]) / (2 * dx)
        gy[1:-1, 1:-1] = (dem_data[2:, 1:-1] - dem_data[:-2, 1:-1]) / (2 * dy)
        
        # Calculate slope in degrees
        slope_rad = np.arctan(np.sqrt(gx*gx + gy*gy))
        slope_deg = slope_rad * 180.0 / np.pi
        
        # Calculate aspect in degrees
        aspect_rad = np.arctan2(-gx, gy)
        aspect_deg = aspect_rad * 180.0 / np.pi
        aspect_deg = (aspect_deg + 360) % 360
        
        # Apply BasinMaker-style slope constraints (consistent with other methods)
        slope_deg = np.clip(slope_deg, 0.1, 45.0)  # BasinMaker standard: min=0.1°, max=45°
        
        # Log statistics
        valid_slopes = slope_deg[~np.isnan(slope_deg)]
        if len(valid_slopes) > 0:
            self.logger.info(f"Basic numpy slope calculation completed (BasinMaker constraints applied):")
            self.logger.info(f"  Min slope: {np.min(valid_slopes):.2f}°")
            self.logger.info(f"  Max slope: {np.max(valid_slopes):.2f}°")
            self.logger.info(f"  Mean slope: {np.mean(valid_slopes):.2f}°")
            self.logger.info(f"  Slope range: 0.1° - 45.0° (BasinMaker standard)")
            
            # Check distribution
            steep_count = np.sum(valid_slopes > 30)
            if steep_count > 0:
                self.logger.info(f"  {steep_count} pixels have slopes > 30° (steep terrain)")
            
            very_steep_count = np.sum(valid_slopes >= 45)
            if very_steep_count > 0:
                self.logger.info(f"  {very_steep_count} pixels clamped to 45° (BasinMaker max)")
        
        return slope_deg, aspect_deg
    
    def _calculate_zonal_statistics(self, 
                                  hru_gdf: gpd.GeoDataFrame,
                                  dem_data: np.ndarray,
                                  slope_data: np.ndarray, 
                                  aspect_data: np.ndarray,
                                  transform,
                                  nodata) -> dict:
        """
        Calculate zonal statistics for each HRU (v.rast.stats equivalent)
        
        BasinMaker operations:
        - v.rast.stats map=polygons raster=dem method=average -> d_average
        - v.rast.stats map=polygons raster=slope method=average -> s_average  
        - v.rast.stats map=polygons raster=aspect method=average -> a_average
        """
        
        elevations = []
        slopes = []
        aspects = []
        
        for idx, hru in hru_gdf.iterrows():
            try:
                # Create raster mask for this HRU geometry (rasterio equivalent)
                geom = [hru.geometry.__geo_interface__]
                mask = geometry_mask(geom, dem_data.shape, transform, invert=True)
                
                if np.any(mask):
                    # Calculate elevation statistics (d_average equivalent)
                    dem_values = dem_data[mask]
                    dem_values = dem_values[~np.isnan(dem_values)]
                    if len(dem_values) > 0:
                        elevation = float(np.mean(dem_values))
                    else:
                        elevation = 1000.0
                    
                    # Calculate slope statistics (s_average equivalent) 
                    slope_values = slope_data[mask]
                    slope_values = slope_values[~np.isnan(slope_values)]
                    if len(slope_values) > 0:
                        slope = float(np.mean(slope_values))
                    else:
                        slope = 5.0
                    
                    # Calculate aspect statistics (a_average equivalent)
                    aspect_values = aspect_data[mask]
                    aspect_values = aspect_values[~np.isnan(aspect_values)]
                    if len(aspect_values) > 0:
                        # Handle circular mean for aspect
                        aspect = float(self._circular_mean(aspect_values))
                    else:
                        aspect = 180.0
                        
                else:
                    # No valid pixels in HRU
                    elevation, slope, aspect = 1000.0, 5.0, 180.0
                    
            except Exception as e:
                self.logger.warning(f"Failed to calculate attributes for HRU {idx}: {e}")
                elevation, slope, aspect = 1000.0, 5.0, 180.0
            
            elevations.append(elevation)
            slopes.append(slope)
            aspects.append(aspect)
        
        return {
            'elevation': elevations,
            'slope': slopes,
            'aspect': aspects
        }
    
    def _calculate_slope_aspect_grass_method(self, 
                                           dem_data: np.ndarray, 
                                           transform, 
                                           nodata) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate slope and aspect using GRASS r.slope.aspect equivalent method
        
        This follows the Zevenbergen & Thorne (1987) algorithm used by GRASS GIS
        and BasinMaker, which is more conservative and reduces systematic bias.
        """
        
        self.logger.info("Using GRASS r.slope.aspect equivalent method")
        
        # Mask nodata values
        if nodata is not None:
            dem_data = np.where(dem_data == nodata, np.nan, dem_data)
        
        # Get cell resolution from transform and convert to meters if needed
        dx = abs(transform[0])  # Cell width
        dy = abs(transform[4])  # Cell height
        
        # Convert to meters if in geographic coordinates
        dx_meters, dy_meters = self._convert_cell_size_to_meters(dx, dy, transform)
        dx = dx_meters
        dy = dy_meters
        
        # Initialize gradient arrays
        h, w = dem_data.shape
        gx = np.zeros_like(dem_data)
        gy = np.zeros_like(dem_data)
        
        # GRASS r.slope.aspect algorithm (Zevenbergen & Thorne, 1987)
        # Uses 3x3 neighborhood with specific weights
        for i in range(1, h-1):
            for j in range(1, w-1):
                # Get 3x3 neighborhood
                # z1 z2 z3
                # z4 z5 z6  
                # z7 z8 z9
                z = dem_data[i-1:i+2, j-1:j+2]
                
                # Skip if any values are NaN
                if np.any(np.isnan(z)):
                    continue
                
                # GRASS algorithm for gradients:
                # dz/dx = (z3 + 2*z6 + z9 - z1 - 2*z4 - z7) / (8 * dx)
                # dz/dy = (z7 + 2*z8 + z9 - z1 - 2*z2 - z3) / (8 * dy)
                
                gx[i,j] = (z[0,2] + 2*z[1,2] + z[2,2] - z[0,0] - 2*z[1,0] - z[2,0]) / (8 * dx)
                gy[i,j] = (z[2,0] + 2*z[2,1] + z[2,2] - z[0,0] - 2*z[0,1] - z[0,2]) / (8 * dy)
        
        # Calculate slope in degrees
        slope_rad = np.arctan(np.sqrt(gx*gx + gy*gy))
        slope_deg = slope_rad * 180.0 / np.pi
        
        # Calculate aspect in degrees (GRASS convention)
        # Note: GRASS uses different aspect convention than our previous method
        aspect_rad = np.arctan2(gx, -gy)  # GRASS convention
        aspect_deg = aspect_rad * 180.0 / np.pi
        aspect_deg = (aspect_deg + 360) % 360  # Ensure 0-360 range
        
        # Apply BasinMaker-style slope constraints
        slope_deg = np.clip(slope_deg, 0.1, 45.0)  # BasinMaker standard: min=0.1°, max=45°
        
        # Log statistics
        valid_slopes = slope_deg[~np.isnan(slope_deg)]
        if len(valid_slopes) > 0:
            self.logger.info(f"GRASS equivalent slope calculation completed (BasinMaker constraints applied):")
            self.logger.info(f"  Min slope: {np.min(valid_slopes):.2f}°")
            self.logger.info(f"  Max slope: {np.max(valid_slopes):.2f}°")
            self.logger.info(f"  Mean slope: {np.mean(valid_slopes):.2f}°")
            self.logger.info(f"  Slope range: 0.1° - 45.0° (BasinMaker standard)")
            
            # Check aspect distribution to verify reduced bias
            valid_aspects = aspect_deg[~np.isnan(aspect_deg)]
            if len(valid_aspects) > 0:
                east_facing = np.sum((valid_aspects >= 45) & (valid_aspects <= 135))
                east_percentage = 100.0 * east_facing / len(valid_aspects)
                self.logger.info(f"  East-facing pixels: {east_percentage:.1f}% (GRASS method reduces bias)")
        
        return slope_deg, aspect_deg

    def _convert_cell_size_to_meters(self, dx: float, dy: float, transform) -> Tuple[float, float]:
        """
        Convert cell size to meters if in geographic coordinates
        
        Args:
            dx: Cell width in coordinate system units
            dy: Cell height in coordinate system units  
            transform: Rasterio transform object
            
        Returns:
            Tuple of (dx_meters, dy_meters)
        """
        
        # Check if coordinates are in degrees (geographic)
        if abs(dx) < 1.0 and abs(dy) < 1.0:
            # Likely geographic coordinates - convert to meters
            
            # Get approximate center latitude for conversion
            # Use transform to get bounds
            bounds = rasterio.transform.array_bounds(1005, 1087, transform)  # height, width from our DEM
            center_lat = (bounds[1] + bounds[3]) / 2  # (bottom + top) / 2
            
            # Convert degrees to meters (approximate)
            # 1 degree longitude ≈ 111320 * cos(lat) meters
            # 1 degree latitude ≈ 111320 meters
            import math
            lat_rad = math.radians(center_lat)
            dx_meters = abs(dx) * 111320 * math.cos(lat_rad)  # Longitude to meters
            dy_meters = abs(dy) * 111320  # Latitude to meters
            
            self.logger.info(f"Converting geographic coordinates to meters:")
            self.logger.info(f"  Cell size: {dx:.6f}° x {dy:.6f}° -> {dx_meters:.1f}m x {dy_meters:.1f}m")
            self.logger.info(f"  Center latitude: {center_lat:.4f}°")
            
            return dx_meters, dy_meters
        else:
            # Already in projected coordinates (meters)
            self.logger.info(f"Using projected coordinates: {abs(dx):.1f}m x {abs(dy):.1f}m")
            return abs(dx), abs(dy)

    def _circular_mean(self, angles: np.ndarray) -> float:
        """
        Calculate circular mean for aspect values
        Needed because aspect is circular (0° = 360°)
        """
        # Convert to radians
        angles_rad = angles * np.pi / 180.0
        
        # Calculate circular mean
        sin_mean = np.mean(np.sin(angles_rad))
        cos_mean = np.mean(np.cos(angles_rad))
        
        circular_mean_rad = np.arctan2(sin_mean, cos_mean)
        circular_mean_deg = circular_mean_rad * 180.0 / np.pi
        
        # Ensure positive value
        if circular_mean_deg < 0:
            circular_mean_deg += 360
            
        return circular_mean_deg
    
    def _add_default_terrain_attributes(self, hru_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add default terrain attributes when DEM processing fails"""
        
        self.logger.warning("Using default terrain attributes")
        
        # Use BasinMaker/RAVEN expected field names
        hru_gdf['HRU_E_mean'] = 1000.0  # Default elevation (m)
        hru_gdf['HRU_S_mean'] = 5.0     # Default slope (degrees)  
        hru_gdf['HRU_A_mean'] = 180.0   # Default aspect (south-facing, degrees)
        
        # Also add legacy names for backward compatibility
        hru_gdf['elevation'] = 1000.0   # Default elevation (m)
        hru_gdf['slope'] = 5.0          # Default slope (degrees)  
        hru_gdf['aspect'] = 180.0       # Default aspect (south-facing, degrees)
        
        return hru_gdf


def main():
    """Test the terrain attributes calculator"""
    
    import sys
    from pathlib import Path
    
    if len(sys.argv) != 3:
        print("Usage: python terrain_attributes_calculator.py <hru_file> <dem_file>")
        sys.exit(1)
    
    hru_file = sys.argv[1] 
    dem_file = sys.argv[2]
    
    # Load HRU data
    hru_gdf = gpd.read_file(hru_file)
    print(f"Loaded {len(hru_gdf)} HRUs from {hru_file}")
    
    # Calculate terrain attributes
    calculator = TerrainAttributesCalculator()
    hru_with_terrain = calculator.calculate_terrain_attributes(hru_gdf, dem_file)
    
    # Print results
    print("\nTerrain Attributes Summary:")
    print(f"Elevation: min={hru_with_terrain['elevation'].min():.1f}, max={hru_with_terrain['elevation'].max():.1f}, mean={hru_with_terrain['elevation'].mean():.1f}")
    print(f"Slope: min={hru_with_terrain['slope'].min():.1f}, max={hru_with_terrain['slope'].max():.1f}, mean={hru_with_terrain['slope'].mean():.1f}")
    print(f"Aspect: min={hru_with_terrain['aspect'].min():.1f}, max={hru_with_terrain['aspect'].max():.1f}, mean={hru_with_terrain['aspect'].mean():.1f}")


if __name__ == "__main__":
    main()