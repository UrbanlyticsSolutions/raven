#!/usr/bin/env python3
"""
HRU Class Mapper
Maps HRU spatial attributes to classification system classes dynamically
Fixes gaps in HRU parameter assignment for Steps 4 and 5
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from shapely.geometry import shape
from .lookup_table_generator import RAVENLookupTableGenerator

class HRUClassMapper:
    """
    Maps HRU spatial data to classification system classes
    Integrates soil texture analysis, landcover classification, and terrain analysis
    """
    
    def __init__(self, json_database_path: str = None):
        """
        Initialize HRU class mapper
        
        Args:
            json_database_path: Path to raven_lookup_database.json
        """
        self.lookup_generator = RAVENLookupTableGenerator(json_database_path)
        self.database = self.lookup_generator.database
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def assign_soil_classes_from_texture_rasters(self, 
                                                hru_gdf: gpd.GeoDataFrame,
                                                sand_raster: Union[str, Path],
                                                silt_raster: Union[str, Path],
                                                clay_raster: Union[str, Path]) -> gpd.GeoDataFrame:
        """
        Assign soil classes to HRUs based on sand/silt/clay percentage rasters
        
        Args:
            hru_gdf: HRU GeoDataFrame
            sand_raster: Path to sand percentage raster
            silt_raster: Path to silt percentage raster  
            clay_raster: Path to clay percentage raster
            
        Returns:
            HRU GeoDataFrame with SOIL_PROF and Soil_ID columns added
        """
        self.logger.info("Assigning soil classes from texture rasters...")
        
        # Ensure CRS compatibility
        hru_gdf_projected = self._ensure_projected_crs(hru_gdf)
        
        # Extract soil texture percentages for each HRU
        with rasterio.open(sand_raster) as sand_src, \
             rasterio.open(silt_raster) as silt_src, \
             rasterio.open(clay_raster) as clay_src:
            
            # Ensure all rasters have same CRS
            if sand_src.crs != silt_src.crs or sand_src.crs != clay_src.crs:
                raise ValueError("All soil texture rasters must have the same CRS")
            
            # Reproject HRU geometries to raster CRS if needed
            if hru_gdf_projected.crs != sand_src.crs:
                hru_gdf_raster_crs = hru_gdf_projected.to_crs(sand_src.crs)
            else:
                hru_gdf_raster_crs = hru_gdf_projected.copy()
            
            soil_classes = []
            soil_ids = []
            sand_means = []
            silt_means = []
            clay_means = []
            
            for idx, hru in hru_gdf_raster_crs.iterrows():
                try:
                    # Extract raster values for this HRU polygon
                    sand_values = self._extract_raster_values_for_polygon(sand_src, hru.geometry)
                    silt_values = self._extract_raster_values_for_polygon(silt_src, hru.geometry)  
                    clay_values = self._extract_raster_values_for_polygon(clay_src, hru.geometry)
                    
                    # Calculate mean percentages (ignoring nodata)
                    sand_mean = np.nanmean(sand_values) if len(sand_values) > 0 else 40.0
                    silt_mean = np.nanmean(silt_values) if len(silt_values) > 0 else 40.0
                    clay_mean = np.nanmean(clay_values) if len(clay_values) > 0 else 20.0
                    
                    # Store raw percentages
                    sand_means.append(sand_mean)
                    silt_means.append(silt_mean) 
                    clay_means.append(clay_mean)
                    
                    # Classify soil texture
                    soil_class = self.lookup_generator.classify_soil_texture_from_percentages(
                        sand_mean, silt_mean, clay_mean
                    )
                    
                    # Get soil ID
                    soil_id = self.lookup_generator.get_soil_texture_id(soil_class)
                    
                    soil_classes.append(soil_class)
                    soil_ids.append(soil_id)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract soil data for HRU {idx}: {e}")
                    # Default values
                    soil_classes.append('LOAM')
                    soil_ids.append(7)
                    sand_means.append(40.0)
                    silt_means.append(40.0)
                    clay_means.append(20.0)
        
        # Add soil classification columns to original HRU GeoDataFrame
        result_gdf = hru_gdf.copy()
        result_gdf['SOIL_PROF'] = soil_classes
        result_gdf['Soil_ID'] = soil_ids
        result_gdf['Sand_Percent'] = sand_means
        result_gdf['Silt_Percent'] = silt_means
        result_gdf['Clay_Percent'] = clay_means
        
        self.logger.info(f"Assigned soil classes to {len(result_gdf)} HRUs")
        unique_soil_classes = pd.Series(soil_classes).value_counts()
        self.logger.info(f"Soil class distribution: {dict(unique_soil_classes)}")
        
        return result_gdf
    
    def assign_landcover_classes_from_raster(self,
                                           hru_gdf: gpd.GeoDataFrame,
                                           landcover_raster: Union[str, Path],
                                           classification_method: str = 'dominant') -> gpd.GeoDataFrame:
        """
        Assign landcover classes to HRUs based on landcover raster
        
        Args:
            hru_gdf: HRU GeoDataFrame
            landcover_raster: Path to landcover classification raster
            classification_method: Method for class assignment ('dominant', 'majority', 'weighted')
            
        Returns:
            HRU GeoDataFrame with LAND_USE_C, Landuse_ID, and VEG_C columns added
        """
        self.logger.info("Assigning landcover classes from raster...")
        
        # Ensure CRS compatibility
        hru_gdf_projected = self._ensure_projected_crs(hru_gdf)
        
        with rasterio.open(landcover_raster) as lc_src:
            # Reproject HRU geometries to raster CRS if needed
            if hru_gdf_projected.crs != lc_src.crs:
                hru_gdf_raster_crs = hru_gdf_projected.to_crs(lc_src.crs)
            else:
                hru_gdf_raster_crs = hru_gdf_projected.copy()
            
            landuse_classes = []
            landuse_ids = []
            veg_classes = []
            
            for idx, hru in hru_gdf_raster_crs.iterrows():
                try:
                    # Extract landcover values for this HRU polygon
                    lc_values = self._extract_raster_values_for_polygon(lc_src, hru.geometry)
                    
                    if len(lc_values) == 0:
                        # Default to grassland if no data
                        dominant_lc_id = 30  # Grassland in ESA WorldCover
                    else:
                        # Find dominant landcover class
                        if classification_method == 'dominant':
                            # Most frequent value
                            unique_values, counts = np.unique(lc_values, return_counts=True)
                            dominant_idx = np.argmax(counts)
                            dominant_lc_id = int(unique_values[dominant_idx])
                        else:
                            # Default to most frequent (handle NaN values like BasinMaker)
                            lc_values_clean = lc_values.copy()
                            lc_values_clean = np.where(np.isnan(lc_values_clean), -1, lc_values_clean)  # Replace NaN with -1
                            dominant_lc_id = int(np.bincount(lc_values_clean.astype(int)).argmax())
                    
                    # Map WorldCover ID to RAVEN classes
                    landcover_result = self.lookup_generator.classify_landcover_from_worldcover(dominant_lc_id)
                    landuse_class = landcover_result['raven_class']
                    landuse_id = landcover_result['raven_id']
                    
                    # Get vegetation class from landcover mapping
                    veg_mapping = self.database["vegetation_classification"]["basinmaker_format"]["landcover_mapping"]
                    veg_class = veg_mapping.get(landuse_class, f"{landuse_class}_VEG")
                    
                    landuse_classes.append(landuse_class)
                    landuse_ids.append(landuse_id)
                    veg_classes.append(veg_class)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract landcover data for HRU {idx}: {e}")
                    # Default values
                    landuse_classes.append('GRASSLAND')
                    landuse_ids.append(60)
                    veg_classes.append('GRASSLAND_VEG')
        
        # Add landcover classification columns to original HRU GeoDataFrame
        result_gdf = hru_gdf.copy()
        result_gdf['LAND_USE_C'] = landuse_classes
        result_gdf['Landuse_ID'] = landuse_ids
        result_gdf['VEG_C'] = veg_classes
        result_gdf['Veg_ID'] = landuse_ids  # Use same ID for simplicity
        
        self.logger.info(f"Assigned landcover classes to {len(result_gdf)} HRUs")
        unique_landcover_classes = pd.Series(landuse_classes).value_counts()
        self.logger.info(f"Landcover class distribution: {dict(unique_landcover_classes)}")
        
        return result_gdf
    
    def assign_terrain_attributes_from_dem(self,
                                         hru_gdf: gpd.GeoDataFrame,
                                         dem_raster: Union[str, Path],
                                         calculate_slope: bool = True,
                                         calculate_aspect: bool = True) -> gpd.GeoDataFrame:
        """
        Assign terrain attributes to HRUs from DEM raster
        
        Args:
            hru_gdf: HRU GeoDataFrame
            dem_raster: Path to DEM raster
            calculate_slope: Whether to calculate slope statistics
            calculate_aspect: Whether to calculate aspect statistics
            
        Returns:
            HRU GeoDataFrame with elevation, slope, and aspect columns added
        """
        self.logger.info("Assigning terrain attributes from DEM...")
        
        # Ensure CRS compatibility
        hru_gdf_projected = self._ensure_projected_crs(hru_gdf)
        
        with rasterio.open(dem_raster) as dem_src:
            # Reproject HRU geometries to raster CRS if needed
            if hru_gdf_projected.crs != dem_src.crs:
                hru_gdf_raster_crs = hru_gdf_projected.to_crs(dem_src.crs)
            else:
                hru_gdf_raster_crs = hru_gdf_projected.copy()
            
            elevations = []
            slopes = [] if calculate_slope else None
            aspects = [] if calculate_aspect else None
            
            for idx, hru in hru_gdf_raster_crs.iterrows():
                try:
                    # Extract elevation values for this HRU polygon
                    elev_values = self._extract_raster_values_for_polygon(dem_src, hru.geometry)
                    
                    if len(elev_values) == 0:
                        elevation_mean = 1000.0  # Default elevation
                    else:
                        elevation_mean = float(np.nanmean(elev_values))
                    
                    elevations.append(elevation_mean)
                    
                    # Calculate slope if requested (simplified)
                    if calculate_slope:
                        if len(elev_values) > 1:
                            slope_mean = float(np.nanstd(elev_values) * 0.1)  # Simplified slope estimate
                        else:
                            slope_mean = 0.05  # Default slope (5%)
                        slopes.append(slope_mean)
                    
                    # Calculate aspect if requested (simplified - would need proper gradient calculation)
                    if calculate_aspect:
                        aspect_mean = 180.0  # Default south-facing
                        aspects.append(aspect_mean)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract terrain data for HRU {idx}: {e}")
                    elevations.append(1000.0)
                    if calculate_slope:
                        slopes.append(0.05)
                    if calculate_aspect:
                        aspects.append(180.0)
        
        # Add terrain attributes to original HRU GeoDataFrame
        result_gdf = hru_gdf.copy()
        result_gdf['ELEVATION'] = elevations
        
        if calculate_slope:
            result_gdf['SLOPE'] = slopes
        
        if calculate_aspect:
            result_gdf['ASPECT'] = aspects
        
        self.logger.info(f"Assigned terrain attributes to {len(result_gdf)} HRUs")
        self.logger.info(f"Elevation range: {min(elevations):.1f} - {max(elevations):.1f} m")
        
        return result_gdf
    
    def assign_geographic_attributes(self, hru_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Assign geographic attributes (area, centroid coordinates) to HRUs
        
        Args:
            hru_gdf: HRU GeoDataFrame
            
        Returns:
            HRU GeoDataFrame with geographic attributes added
        """
        self.logger.info("Assigning geographic attributes...")
        
        # Ensure projected CRS for area calculation
        hru_gdf_projected = self._ensure_projected_crs(hru_gdf)
        
        # Calculate areas in km²
        areas_km2 = hru_gdf_projected.geometry.area / 1e6
        
        # Calculate centroids in WGS84 for lat/lon
        hru_gdf_wgs84 = hru_gdf_projected.to_crs('EPSG:4326')
        centroids = hru_gdf_wgs84.geometry.centroid
        
        # Extract coordinates
        latitudes = centroids.y
        longitudes = centroids.x
        
        # Add attributes to original GeoDataFrame
        result_gdf = hru_gdf.copy()
        result_gdf['AREA'] = areas_km2
        result_gdf['LATITUDE'] = latitudes
        result_gdf['LONGITUDE'] = longitudes
        
        # Add basin/subbasin IDs if not present
        if 'BASIN_ID' not in result_gdf.columns:
            if 'SubId' in result_gdf.columns:
                result_gdf['BASIN_ID'] = result_gdf['SubId']
            else:
                result_gdf['BASIN_ID'] = 1  # Default basin ID
        
        self.logger.info(f"Assigned geographic attributes to {len(result_gdf)} HRUs")
        self.logger.info(f"Total area: {areas_km2.sum():.2f} km²")
        self.logger.info(f"Area range: {areas_km2.min():.3f} - {areas_km2.max():.3f} km²")
        
        return result_gdf
    
    def assign_complete_hru_classes(self,
                                   hru_gdf: gpd.GeoDataFrame,
                                   sand_raster: Union[str, Path] = None,
                                   silt_raster: Union[str, Path] = None,
                                   clay_raster: Union[str, Path] = None,
                                   landcover_raster: Union[str, Path] = None,
                                   dem_raster: Union[str, Path] = None) -> gpd.GeoDataFrame:
        """
        Assign complete class set to HRUs from all available raster inputs
        
        Args:
            hru_gdf: HRU GeoDataFrame
            sand_raster: Path to sand percentage raster (optional)
            silt_raster: Path to silt percentage raster (optional)
            clay_raster: Path to clay percentage raster (optional)
            landcover_raster: Path to landcover raster (optional)
            dem_raster: Path to DEM raster (optional)
            
        Returns:
            HRU GeoDataFrame with all class assignments and attributes
        """
        self.logger.info("Assigning complete HRU class set...")
        
        result_gdf = hru_gdf.copy()
        
        # Assign geographic attributes first
        result_gdf = self.assign_geographic_attributes(result_gdf)
        
        # Assign soil classes if soil texture rasters available
        if sand_raster and silt_raster and clay_raster:
            if all(Path(raster).exists() for raster in [sand_raster, silt_raster, clay_raster]):
                result_gdf = self.assign_soil_classes_from_texture_rasters(
                    result_gdf, sand_raster, silt_raster, clay_raster
                )
            else:
                self.logger.warning("Some soil texture rasters missing, using default soil classes")
                self._assign_default_soil_classes(result_gdf)
        else:
            self.logger.info("No soil texture rasters provided, using default soil classes")
            self._assign_default_soil_classes(result_gdf)
        
        # Assign landcover classes if landcover raster available
        if landcover_raster and Path(landcover_raster).exists():
            result_gdf = self.assign_landcover_classes_from_raster(result_gdf, landcover_raster)
        else:
            self.logger.info("No landcover raster provided, using default landcover classes")
            self._assign_default_landcover_classes(result_gdf)
        
        # Assign terrain attributes if DEM available
        if dem_raster and Path(dem_raster).exists():
            result_gdf = self.assign_terrain_attributes_from_dem(result_gdf, dem_raster)
        else:
            self.logger.info("No DEM raster provided, using default elevation")
            if 'ELEVATION' not in result_gdf.columns:
                result_gdf['ELEVATION'] = 1000.0  # Default elevation
        
        self.logger.info(f"Complete HRU class assignment finished: {len(result_gdf)} HRUs")
        
        # Log summary statistics
        self._log_hru_summary(result_gdf)
        
        return result_gdf
    
    def _extract_raster_values_for_polygon(self, raster_src, polygon_geom) -> np.ndarray:
        """Extract raster values within a polygon geometry"""
        try:
            # Mask the raster with the polygon
            masked_data, masked_transform = mask(raster_src, [polygon_geom], crop=True, nodata=raster_src.nodata)
            
            # Get valid (non-nodata) values
            if raster_src.nodata is not None:
                valid_mask = masked_data[0] != raster_src.nodata
                values = masked_data[0][valid_mask]
            else:
                values = masked_data[0][masked_data[0] != 0]  # Assume 0 is nodata if not specified
            
            return values
            
        except Exception as e:
            self.logger.warning(f"Failed to extract raster values: {e}")
            return np.array([])
    
    def _ensure_projected_crs(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Ensure GeoDataFrame is in a projected CRS for area calculations"""
        if gdf.crs and gdf.crs.is_geographic:
            # Reproject to appropriate UTM zone based on centroid
            bounds = gdf.total_bounds
            lon_center = (bounds[0] + bounds[2]) / 2
            
            # Simple UTM zone calculation
            utm_zone = int((lon_center + 180) / 6) + 1
            if gdf.total_bounds[1] >= 0:  # Northern hemisphere
                utm_crs = f'EPSG:326{utm_zone:02d}'
            else:  # Southern hemisphere
                utm_crs = f'EPSG:327{utm_zone:02d}'
            
            return gdf.to_crs(utm_crs)
        
        return gdf.copy()
    
    def _assign_default_soil_classes(self, hru_gdf: gpd.GeoDataFrame):
        """Assign default soil classes when soil rasters not available"""
        # Simple area-based assignment
        areas = hru_gdf.geometry.area if hasattr(hru_gdf, 'geometry') else [1.0] * len(hru_gdf)
        
        soil_classes = []
        soil_ids = []
        
        for area in areas:
            if area > 5e6:  # Large areas - assume clay
                soil_class = 'CLAY'
                soil_id = 1
            elif area > 1e6:  # Medium areas - assume loam
                soil_class = 'LOAM'
                soil_id = 7
            else:  # Small areas - assume sandy loam
                soil_class = 'SANDY_LOAM'
                soil_id = 9
            
            soil_classes.append(soil_class)
            soil_ids.append(soil_id)
        
        hru_gdf['SOIL_PROF'] = soil_classes
        hru_gdf['Soil_ID'] = soil_ids
    
    def _assign_default_landcover_classes(self, hru_gdf: gpd.GeoDataFrame):
        """Assign default landcover classes when landcover raster not available"""
        # Default to mixed assignment
        n_hrus = len(hru_gdf)
        
        # Distribute classes based on typical landscape
        landuse_classes = []
        landuse_ids = []
        veg_classes = []
        
        for i in range(n_hrus):
            if i % 3 == 0:  # Forest
                landuse_class = 'FOREST_MIXED'
                landuse_id = 30
                veg_class = 'MIXED_FOREST'
            elif i % 3 == 1:  # Agriculture
                landuse_class = 'AGRICULTURE'
                landuse_id = 40
                veg_class = 'AGRICULTURAL_CROPS'
            else:  # Grassland
                landuse_class = 'GRASSLAND'
                landuse_id = 60
                veg_class = 'GRASSLAND_VEG'
            
            landuse_classes.append(landuse_class)
            landuse_ids.append(landuse_id)
            veg_classes.append(veg_class)
        
        hru_gdf['LAND_USE_C'] = landuse_classes
        hru_gdf['Landuse_ID'] = landuse_ids
        hru_gdf['VEG_C'] = veg_classes
        hru_gdf['Veg_ID'] = landuse_ids
    
    def _log_hru_summary(self, hru_gdf: gpd.GeoDataFrame):
        """Log summary statistics for HRU assignments"""
        self.logger.info("=== HRU CLASS ASSIGNMENT SUMMARY ===")
        
        if 'AREA' in hru_gdf.columns:
            total_area = hru_gdf['AREA'].sum()
            self.logger.info(f"Total watershed area: {total_area:.2f} km²")
            self.logger.info(f"Number of HRUs: {len(hru_gdf)}")
            self.logger.info(f"Mean HRU area: {hru_gdf['AREA'].mean():.3f} km²")
        
        if 'SOIL_PROF' in hru_gdf.columns:
            soil_counts = hru_gdf['SOIL_PROF'].value_counts()
            self.logger.info(f"Soil classes ({len(soil_counts)}): {dict(soil_counts)}")
        
        if 'LAND_USE_C' in hru_gdf.columns:
            landuse_counts = hru_gdf['LAND_USE_C'].value_counts()
            self.logger.info(f"Landuse classes ({len(landuse_counts)}): {dict(landuse_counts)}")
        
        if 'ELEVATION' in hru_gdf.columns:
            elev_stats = hru_gdf['ELEVATION'].describe()
            self.logger.info(f"Elevation range: {elev_stats['min']:.0f} - {elev_stats['max']:.0f} m")


def main():
    """Main function for testing HRU class mapper"""
    mapper = HRUClassMapper()
    
    print("HRU Class Mapper initialized successfully!")
    print("Testing soil texture classification...")
    
    # Test soil texture classification
    test_cases = [
        (65, 23, 12),  # Sandy loam
        (30, 35, 35),  # Clay loam  
        (85, 10, 5),   # Sand
        (20, 60, 20)   # Silt loam
    ]
    
    for sand, silt, clay in test_cases:
        soil_class = mapper.lookup_generator.classify_soil_texture_from_percentages(sand, silt, clay)
        soil_id = mapper.lookup_generator.get_soil_texture_id(soil_class)
        print(f"  Sand:{sand}% Silt:{silt}% Clay:{clay}% → {soil_class} (ID: {soil_id})")
    
    print("\nTesting landcover classification...")
    
    # Test landcover classification
    worldcover_ids = [10, 20, 30, 40, 50, 80]
    
    for wc_id in worldcover_ids:
        lc_result = mapper.lookup_generator.classify_landcover_from_worldcover(wc_id)
        print(f"  WorldCover ID:{wc_id} → {lc_result['raven_class']} (ID: {lc_result['raven_id']})")
    
    print("\nHRU Class Mapper test completed successfully!")


if __name__ == "__main__":
    main()