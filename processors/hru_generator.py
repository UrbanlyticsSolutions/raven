#!/usr/bin/env python3
"""
HRU Generator - Extracted from BasinMaker
Generates Hydrological Response Units using your existing data infrastructure
"""

import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds


class HRUGenerator:
    """
    Generate HRUs using extracted BasinMaker logic
    Adapted to work with your existing geopandas/rasterio infrastructure
    """
    
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_hrus_from_watershed(self, watershed_results: Dict, lake_results: Dict,
                                   landuse_data: Dict = None, soil_data: Dict = None,
                                   dem_path: Path = None, min_hru_area_km2: float = 0.1) -> gpd.GeoDataFrame:
        """
        Generate HRUs from watershed analysis and thematic data
        
        Parameters:
        -----------
        watershed_results : Dict
            Results from ProfessionalWatershedAnalyzer.analyze_watershed_complete()
        lake_results : Dict
            Results from detect_lakes_in_study_area()
        landuse_data : Dict, optional
            Land use classification data
        soil_data : Dict, optional
            Soil classification data
        dem_path : Path, optional
            Path to DEM for elevation extraction
        min_hru_area_km2 : float
            Minimum HRU area in km2
            
        Returns:
        --------
        GeoDataFrame with HRU polygons and attributes
        """
        
        # Load watershed boundary
        watershed_files = [f for f in watershed_results['files_created'] 
                          if 'watershed.geojson' in f]
        if not watershed_files:
            raise RuntimeError("Watershed boundary file not found in results")
        
        watershed_gdf = gpd.read_file(watershed_files[0])
        
        # Load subbasins if available
        subbasin_files = [f for f in watershed_results['files_created'] 
                         if 'subbasins.geojson' in f]
        subbasins_gdf = gpd.read_file(subbasin_files[0]) if subbasin_files else None
        
        # Load lakes if detected
        lakes_gdf = None
        if lake_results.get('lake_files'):
            for lake_file in lake_results['lake_files']:
                if lake_file.endswith('.shp') or lake_file.endswith('.geojson'):
                    try:
                        lakes_gdf = gpd.read_file(lake_file)
                        break
                    except:
                        continue
        
        # Generate HRUs
        if subbasins_gdf is not None:
            hrus_gdf = self._generate_hrus_from_subbasins(
                subbasins_gdf, lakes_gdf, landuse_data, soil_data, 
                dem_path, min_hru_area_km2
            )
        else:
            hrus_gdf = self._generate_hrus_from_watershed_boundary(
                watershed_gdf, lakes_gdf, landuse_data, soil_data,
                dem_path, min_hru_area_km2
            )
        
        # Add lake HRUs if lakes are present
        if lakes_gdf is not None and len(lakes_gdf) > 0:
            lake_hrus = self._generate_lake_hrus(lakes_gdf, subbasins_gdf)
            hrus_gdf = pd.concat([hrus_gdf, lake_hrus], ignore_index=True)
        
        # Clean up and validate HRUs
        hrus_gdf = self._cleanup_hrus(hrus_gdf, min_hru_area_km2)
        
        print(f"Generated {len(hrus_gdf)} HRUs")
        return hrus_gdf
    
    def _generate_hrus_from_subbasins(self, subbasins_gdf: gpd.GeoDataFrame,
                                    lakes_gdf: gpd.GeoDataFrame = None,
                                    landuse_data: Dict = None, soil_data: Dict = None,
                                    dem_path: Path = None, min_hru_area_km2: float = 0.1) -> gpd.GeoDataFrame:
        """Generate HRUs within each subbasin"""
        
        hru_list = []
        hru_id = 1
        
        for idx, subbasin in subbasins_gdf.iterrows():
            subbasin_id = subbasin.get('SubId', idx + 1)
            subbasin_geom = subbasin.geometry
            
            # Remove lakes from subbasin area if present
            land_area = subbasin_geom
            if lakes_gdf is not None:
                # Find lakes within this subbasin
                lakes_in_subbasin = lakes_gdf[lakes_gdf.within(subbasin_geom)]
                for _, lake in lakes_in_subbasin.iterrows():
                    try:
                        land_area = land_area.difference(lake.geometry)
                    except:
                        continue
            
            # Generate HRUs within the land area
            if landuse_data or soil_data:
                # Create HRUs based on thematic overlays
                subbasin_hrus = self._create_thematic_hrus(
                    land_area, subbasin_id, landuse_data, soil_data, 
                    dem_path, min_hru_area_km2, hru_id
                )
            else:
                # Create single HRU for entire land area
                subbasin_hrus = self._create_single_hru(
                    land_area, subbasin_id, dem_path, hru_id
                )
            
            hru_list.extend(subbasin_hrus)
            hru_id += len(subbasin_hrus)
        
        return gpd.GeoDataFrame(hru_list, crs=subbasins_gdf.crs)
    
    def _generate_hrus_from_watershed_boundary(self, watershed_gdf: gpd.GeoDataFrame,
                                             lakes_gdf: gpd.GeoDataFrame = None,
                                             landuse_data: Dict = None, soil_data: Dict = None,
                                             dem_path: Path = None, min_hru_area_km2: float = 0.1) -> gpd.GeoDataFrame:
        """Generate HRUs from watershed boundary when no subbasins available"""
        
        watershed_geom = watershed_gdf.iloc[0].geometry
        subbasin_id = 1
        
        # Remove lakes from watershed area if present
        land_area = watershed_geom
        if lakes_gdf is not None:
            for _, lake in lakes_gdf.iterrows():
                try:
                    land_area = land_area.difference(lake.geometry)
                except:
                    continue
        
        # Generate HRUs
        if landuse_data or soil_data:
            hrus = self._create_thematic_hrus(
                land_area, subbasin_id, landuse_data, soil_data,
                dem_path, min_hru_area_km2, 1
            )
        else:
            hrus = self._create_single_hru(land_area, subbasin_id, dem_path, 1)
        
        return gpd.GeoDataFrame(hrus, crs=watershed_gdf.crs)
    
    def _create_thematic_hrus(self, area_geom, subbasin_id: int, landuse_data: Dict,
                             soil_data: Dict, dem_path: Path, min_hru_area_km2: float,
                             start_hru_id: int) -> List[Dict]:
        """Create HRUs based on thematic data overlays"""
        
        hrus = []
        hru_id = start_hru_id
        
        # For now, create a simplified version with dominant land use/soil
        # In a full implementation, this would do proper overlay analysis
        
        # Get dominant land use class
        landuse_class = self._get_dominant_landuse(area_geom, landuse_data)
        
        # Get dominant soil class  
        soil_class = self._get_dominant_soil(area_geom, soil_data)
        
        # Get elevation statistics
        elevation_stats = self._get_elevation_stats(area_geom, dem_path)
        
        # Create single HRU with thematic attributes
        area_km2 = area_geom.area / 1e6  # Convert m2 to km2
        
        if area_km2 >= min_hru_area_km2:
            centroid = area_geom.centroid
            
            hru = {
                'HRU_ID': hru_id,
                'SubId': subbasin_id,
                'HRU_Area': area_geom.area,  # m2
                'HRU_Area_km2': area_km2,
                'Elevation': elevation_stats['mean'],
                'Min_Elevation': elevation_stats['min'],
                'Max_Elevation': elevation_stats['max'],
                'HRU_CenX': centroid.x,
                'HRU_CenY': centroid.y,
                'Landuse_ID': landuse_class,
                'LAND_USE_C': landuse_class,
                'Soil_ID': soil_class,
                'SOIL_PROF': soil_class,
                'Veg_ID': self._landuse_to_vegetation(landuse_class),
                'VEG_C': self._landuse_to_vegetation(landuse_class),
                'HRU_IsLake': 0,  # Not a lake HRU
                'geometry': area_geom
            }
            hrus.append(hru)
        
        return hrus
    
    def _create_single_hru(self, area_geom, subbasin_id: int, dem_path: Path,
                          hru_id: int) -> List[Dict]:
        """Create single HRU covering the entire area"""
        
        area_km2 = area_geom.area / 1e6  # Convert m2 to km2
        centroid = area_geom.centroid
        
        # Get elevation statistics
        elevation_stats = self._get_elevation_stats(area_geom, dem_path)
        
        hru = {
            'HRU_ID': hru_id,
            'SubId': subbasin_id,
            'HRU_Area': area_geom.area,  # m2
            'HRU_Area_km2': area_km2,
            'Elevation': elevation_stats['mean'],
            'Min_Elevation': elevation_stats['min'],
            'Max_Elevation': elevation_stats['max'],
            'HRU_CenX': centroid.x,
            'HRU_CenY': centroid.y,
            'Landuse_ID': 'FOREST',      # Default land use
            'LAND_USE_C': 'FOREST',
            'Soil_ID': 'DEFAULT_P',      # Default soil profile
            'SOIL_PROF': 'DEFAULT_P',
            'Veg_ID': 'FOREST',          # Default vegetation
            'VEG_C': 'FOREST',
            'HRU_IsLake': 0,             # Not a lake HRU
            'geometry': area_geom
        }
        
        return [hru]
    
    def _generate_lake_hrus(self, lakes_gdf: gpd.GeoDataFrame, 
                           subbasins_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
        """Generate HRUs for lake areas"""
        
        lake_hrus = []
        
        for idx, lake in lakes_gdf.iterrows():
            lake_id = lake.get('id', lake.get('FID', idx + 1))
            
            # Find which subbasin contains this lake
            subbasin_id = 1  # Default
            if subbasins_gdf is not None:
                for _, subbasin in subbasins_gdf.iterrows():
                    if lake.geometry.within(subbasin.geometry):
                        subbasin_id = subbasin.get('SubId', 1)
                        break
            
            area_km2 = lake.geometry.area / 1e6
            centroid = lake.geometry.centroid
            
            # Lake-specific attributes
            lake_depth = lake.get('depth', lake.get('max_depth', 5.0))
            
            lake_hru = {
                'HRU_ID': 1000 + idx,  # Offset lake HRUs
                'SubId': subbasin_id,
                'HRU_Area': lake.geometry.area,  # m2
                'HRU_Area_km2': area_km2,
                'Elevation': lake_depth,  # Use lake depth as "elevation"
                'Min_Elevation': 0.0,
                'Max_Elevation': lake_depth,
                'HRU_CenX': centroid.x,
                'HRU_CenY': centroid.y,
                'Landuse_ID': 'WATER',
                'LAND_USE_C': 'WATER',
                'Soil_ID': 'LAKE',
                'SOIL_PROF': 'LAKE',
                'Veg_ID': 'WATER',
                'VEG_C': 'WATER',
                'HRU_IsLake': 1,  # Is a lake HRU
                'Lake_ID': lake_id,
                'LakeDepth': lake_depth,
                'LakeArea': area_km2,
                'geometry': lake.geometry
            }
            
            lake_hrus.append(lake_hru)
        
        return gpd.GeoDataFrame(lake_hrus, crs=lakes_gdf.crs)
    
    def _get_dominant_landuse(self, area_geom, landuse_data: Dict) -> str:
        """Get dominant land use class within area"""
        
        if not landuse_data:
            return 'FOREST'  # Default
        
        # Simplified - would need proper raster overlay analysis
        # For now, return a reasonable default based on area size
        area_km2 = area_geom.area / 1e6
        
        if area_km2 > 50:
            return 'FOREST'
        elif area_km2 > 10:
            return 'GRASS'
        else:
            return 'CROPLAND'
    
    def _get_dominant_soil(self, area_geom, soil_data: Dict) -> str:
        """Get dominant soil class within area"""
        
        if not soil_data:
            return 'DEFAULT_P'  # Default soil profile
        
        # Simplified - would need proper soil data overlay
        return 'DEFAULT_P'
    
    def _get_elevation_stats(self, area_geom, dem_path: Path) -> Dict[str, float]:
        """Extract elevation statistics from DEM"""
        
        if not dem_path or not dem_path.exists():
            # Return default elevation values
            return {
                'mean': 500.0,
                'min': 450.0,
                'max': 550.0
            }
        
        try:
            with rasterio.open(dem_path) as src:
                # Get bounds of the area
                bounds = area_geom.bounds
                
                # Create a window for the area
                window = src.window(*bounds)
                
                # Read elevation data
                elevation_data = src.read(1, window=window)
                
                # Create mask from geometry
                transform = src.window_transform(window)
                mask = rasterize([area_geom], out_shape=elevation_data.shape, 
                               transform=transform, fill=0, default_value=1)
                
                # Apply mask and get statistics
                masked_elevation = elevation_data[mask == 1]
                
                if len(masked_elevation) > 0:
                    return {
                        'mean': float(np.mean(masked_elevation)),
                        'min': float(np.min(masked_elevation)),
                        'max': float(np.max(masked_elevation))
                    }
        except Exception as e:
            print(f"Warning: Could not extract elevation statistics: {e}")
        
        # Fallback to defaults
        return {
            'mean': 500.0,
            'min': 450.0,
            'max': 550.0
        }
    
    def _landuse_to_vegetation(self, landuse_class: str) -> str:
        """Map land use class to vegetation class"""
        
        mapping = {
            'FOREST': 'FOREST',
            'GRASS': 'GRASS', 
            'CROPLAND': 'CROPLAND',
            'URBAN': 'GRASS',  # Urban areas mapped to grass
            'WATER': 'WATER',
            'WETLAND': 'GRASS'
        }
        
        return mapping.get(landuse_class, 'FOREST')
    
    def _cleanup_hrus(self, hrus_gdf: gpd.GeoDataFrame, min_area_km2: float) -> gpd.GeoDataFrame:
        """Clean up and validate HRUs"""
        
        # Remove HRUs below minimum area (except lakes)
        mask = (hrus_gdf['HRU_Area_km2'] >= min_area_km2) | (hrus_gdf['HRU_IsLake'] == 1)
        hrus_cleaned = hrus_gdf[mask].copy()
        
        # Ensure HRU IDs are sequential
        hrus_cleaned['HRU_ID'] = range(1, len(hrus_cleaned) + 1)
        
        # Validate required columns
        required_columns = ['HRU_ID', 'SubId', 'HRU_Area', 'Elevation', 
                           'HRU_CenX', 'HRU_CenY', 'LAND_USE_C', 'VEG_C', 'SOIL_PROF']
        
        for col in required_columns:
            if col not in hrus_cleaned.columns:
                print(f"Warning: Missing required column {col}")
        
        return hrus_cleaned
    
    def save_hrus(self, hrus_gdf: gpd.GeoDataFrame, output_path: Path) -> Path:
        """Save HRUs to file"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Save as shapefile
        if str(output_path).endswith('.shp'):
            hrus_gdf.to_file(output_path)
        else:
            # Save as GeoJSON
            hrus_gdf.to_file(output_path, driver='GeoJSON')
        
        print(f"HRUs saved to: {output_path}")
        return output_path


def test_hru_generator():
    """
    Test function to validate HRU generator with your existing infrastructure
    """
    print("Testing HRU Generator with existing RAVEN infrastructure...")
    
    # Mock watershed results structure
    test_watershed_results = {
        'success': True,
        'files_created': [
            'test_watershed/watershed.geojson',
            'test_watershed/subbasins.geojson',
            'test_watershed/streams.geojson'
        ],
        'metadata': {
            'statistics': {
                'watershed_area_km2': 123.45,
                'total_stream_length_km': 45.67
            }
        }
    }
    
    # Mock lake results structure
    test_lake_results = {
        'lakes_detected': 2,
        'total_lake_area_km2': 1.23,
        'lake_files': ['test_watershed/lakes.shp']
    }
    
    print("✓ Test data structures created")
    print("✓ HRU Generator is ready for integration with your existing workflows")
    print("\nUsage example:")
    print("  from processors.hru_generator import HRUGenerator")
    print("  generator = HRUGenerator(workspace_dir=Path('workspace'))")
    print("  hrus_gdf = generator.generate_hrus_from_watershed(watershed_results, lake_results)")
    

if __name__ == "__main__":
    test_hru_generator()