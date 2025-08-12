#!/usr/bin/env python3
"""
Step 4: HRU Generation using BasinMaker HRUGenerator
Clean implementation that uses the HRU generator from processors directly
ENHANCED: Integrated with dynamic lookup table generators from JSON database
"""

import os
import sys
from pathlib import Path
import argparse
from typing import Dict, Any

# Fix PROJ.db warnings on Windows - dynamic path loading
def setup_gdal_paths():
    """Set up GDAL_DATA and PROJ_LIB paths dynamically to avoid proj.db warnings"""
    try:
        import pyproj
        proj_data_dir = pyproj.datadir.get_data_dir()
        if proj_data_dir and Path(proj_data_dir).exists():
            os.environ['PROJ_LIB'] = str(proj_data_dir)
        
        import rasterio
        import rasterio.env
        with rasterio.env.Env() as env:
            gdal_data = env.options.get('GDAL_DATA')
            if gdal_data and 'GDAL_DATA' not in os.environ:
                os.environ['GDAL_DATA'] = gdal_data
        
        import logging
        logging.getLogger('rasterio._env').setLevel(logging.ERROR)
        
    except Exception:
        pass

# Set up paths dynamically
setup_gdal_paths()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from processors.hru_generator import HRUGenerator
from utilities.lookup_table_generator import RAVENLookupTableGenerator
from utilities.hru_class_mapper import HRUClassMapper
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.validation import make_valid


def prepare_lakes_shapefile(data_dir: Path) -> Path:
    """Use the correct filtered lakes from step 3: lakes_with_routing_ids.shp"""
    lakes_path = data_dir / "lakes_with_routing_ids.shp"
    
    # Check if lakes exist - if not, return None (no lakes case)
    if not lakes_path.exists():
        print("No lakes shapefile found - proceeding without lakes")
        return None
    
    print("Preparing filtered lakes shapefile from step 3...")
    lakes_gdf = gpd.read_file(lakes_path)
    
    # If empty lakes file, return None
    if len(lakes_gdf) == 0:
        print("Lakes shapefile is empty - proceeding without lakes")
        return None
    
    # Rename lake_id to HyLakeId if needed
    if 'lake_id' in lakes_gdf.columns and 'HyLakeId' not in lakes_gdf.columns:
        lakes_gdf = lakes_gdf.rename(columns={'lake_id': 'HyLakeId'})
        
        # Save updated lakes file
        lakes_fixed_path = data_dir / "lakes_fixed.shp"
        lakes_gdf.to_file(lakes_fixed_path)
        print(f"Created {lakes_fixed_path} with HyLakeId column")
        return lakes_fixed_path
    
    print(f"Lakes shapefile already has correct columns: {list(lakes_gdf.columns)}")
    return lakes_path


def create_landuse_shapefile(data_dir: Path) -> Path:
    """Create landuse shapefile from landcover raster"""
    import rasterio
    import rasterio.features
    import numpy as np
    from shapely.geometry import shape
    
    landcover_path = data_dir / "landcover.tif"
    landuse_path = data_dir / "landuse.shp"
    
    print("Creating landuse shapefile from landcover raster...")
    
    with rasterio.open(landcover_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        
        # Landcover mapping
        landcover_map = {
            1: {'LAND_USE_C': 'FOREST', 'VEG_C': 'CONIFEROUS'},
            2: {'LAND_USE_C': 'SHRUBLAND', 'VEG_C': 'MIXED_SHRUBLAND'}, 
            3: {'LAND_USE_C': 'GRASSLAND', 'VEG_C': 'GRASSLAND'}
        }
        
        # Get unique values
        unique_vals = np.unique(data[data != src.nodata])
        print(f"  Landcover values: {list(unique_vals)}")
        
        # Create polygons for each landcover class
        polys = []
        for val in unique_vals:
            if val in landcover_map:
                mask = data == val
                for geom, value in rasterio.features.shapes(mask.astype(np.uint8), mask=mask, transform=transform):
                    if value == 1:  # Only keep positive mask areas
                        # Fix geometry if needed
                        shp = shape(geom)
                        if not shp.is_valid:
                            shp = shp.buffer(0)  # Fix invalid geometries
                        if shp.is_valid and shp.area > 0:
                            polys.append({
                                'Landuse_ID': int(val),
                                'LAND_USE_C': landcover_map[val]['LAND_USE_C'],
                                'VEG_C': landcover_map[val]['VEG_C'],
                                'geometry': shp
                            })
        
        if polys:
            landuse_gdf = gpd.GeoDataFrame(polys, crs=crs)
            
            # Ensure consistent CRS - reproject to EPSG:32611 if needed
            if landuse_gdf.crs != 'EPSG:32611':
                print(f"  Reprojecting landuse from {landuse_gdf.crs} to EPSG:32611...")
                landuse_gdf = landuse_gdf.to_crs('EPSG:32611')
            
            landuse_gdf.to_file(landuse_path)
            print(f"  Created {landuse_path} with {len(polys)} polygons (CRS: {landuse_gdf.crs})")
        else:
            raise RuntimeError("No landuse polygons created")
    
    return landuse_path


def classify_soil_texture(sand_pct: float, silt_pct: float, clay_pct: float) -> str:
    """Classify soil texture using USDA texture triangle"""
    
    # Ensure percentages sum to 100 (normalize if needed)
    total = sand_pct + silt_pct + clay_pct
    if total > 0:
        sand_pct = (sand_pct / total) * 100
        silt_pct = (silt_pct / total) * 100  
        clay_pct = (clay_pct / total) * 100
    else:
        return 'UNKNOWN'
    
    # USDA soil texture classification
    if clay_pct >= 40:
        if sand_pct >= 45:
            return 'SANDY_CLAY'
        elif silt_pct >= 40:
            return 'SILTY_CLAY'
        else:
            return 'CLAY'
    elif clay_pct >= 27:
        if sand_pct >= 45:
            return 'SANDY_CLAY_LOAM'
        elif silt_pct >= 28:
            return 'SILTY_CLAY_LOAM'
        else:
            return 'CLAY_LOAM'
    elif clay_pct >= 7:
        if sand_pct >= 52:
            if silt_pct >= 28:
                return 'SANDY_LOAM'
            else:
                return 'SANDY_LOAM'
        elif silt_pct >= 50:
            return 'SILT_LOAM'
        else:
            return 'LOAM'
    elif silt_pct >= 80:
        return 'SILT'
    elif sand_pct >= 85:
        return 'SAND'
    elif sand_pct >= 70:
        return 'LOAMY_SAND'
    else:
        return 'LOAM'


def create_soil_shapefile(data_dir: Path) -> Path:
    """Create soil shapefile from sand, silt, clay rasters"""
    import rasterio
    import rasterio.features
    import numpy as np
    from shapely.geometry import shape
    
    sand_path = data_dir / "sand_0-5cm_mean_bbox.tif"
    silt_path = data_dir / "silt_0-5cm_mean_bbox.tif" 
    clay_path = data_dir / "clay_0-5cm_mean_bbox.tif"
    soil_path = data_dir / "soil_polygons.shp"
    
    print("Creating soil shapefile from sand/silt/clay rasters...")
    
    # Read all three rasters
    with rasterio.open(sand_path) as sand_src, \
         rasterio.open(silt_path) as silt_src, \
         rasterio.open(clay_path) as clay_src:
        
        sand_data = sand_src.read(1)
        silt_data = silt_src.read(1) 
        clay_data = clay_src.read(1)
        transform = sand_src.transform
        crs = sand_src.crs
        
        # Create soil texture classification array
        height, width = sand_data.shape
        soil_class_data = np.zeros((height, width), dtype=np.uint8)
        soil_names = {}
        soil_id = 1
        
        print("  Classifying soil textures...")
        for i in range(height):
            for j in range(width):
                if (sand_data[i,j] != sand_src.nodata and 
                    silt_data[i,j] != silt_src.nodata and
                    clay_data[i,j] != clay_src.nodata):
                    
                    soil_class = classify_soil_texture(
                        float(sand_data[i,j]),
                        float(silt_data[i,j]), 
                        float(clay_data[i,j])
                    )
                    
                    if soil_class not in soil_names.values():
                        soil_names[soil_id] = soil_class
                        soil_class_data[i,j] = soil_id
                        soil_id += 1
                    else:
                        # Find existing ID for this soil class
                        for sid, sname in soil_names.items():
                            if sname == soil_class:
                                soil_class_data[i,j] = sid
                                break
        
        print(f"  Found {len(soil_names)} soil texture classes: {list(soil_names.values())}")
        
        # Create polygons for each soil class
        polys = []
        for soil_id, soil_name in soil_names.items():
            mask = soil_class_data == soil_id
            if mask.any():
                for geom, value in rasterio.features.shapes(mask.astype(np.uint8), mask=mask, transform=transform):
                    if value == 1:
                        # Fix geometry if needed
                        shp = shape(geom)
                        if not shp.is_valid:
                            shp = shp.buffer(0)  # Fix invalid geometries
                        if shp.is_valid and shp.area > 0:
                            polys.append({
                                'Soil_ID': int(soil_id),
                                'SOIL_PROF': soil_name,
                                'geometry': shp
                            })
        
        if polys:
            soil_gdf = gpd.GeoDataFrame(polys, crs=crs)
            
            # Ensure consistent CRS - reproject to EPSG:32611 if needed
            if soil_gdf.crs != 'EPSG:32611':
                print(f"  Reprojecting soil from {soil_gdf.crs} to EPSG:32611...")
                soil_gdf = soil_gdf.to_crs('EPSG:32611')
            
            soil_gdf.to_file(soil_path)
            print(f"  Created {soil_path} with {len(polys)} polygons (CRS: {soil_gdf.crs})")
        else:
            raise RuntimeError("No soil polygons created")
    
    return soil_path


def polygonize_raster_per_subbasin(raster_path: Path, subbasins_gdf: gpd.GeoDataFrame, 
                                  value_col: str, class_col: str, target_crs: str) -> Path:
    """
    BasinMaker-style polygonization: clip raster to each subbasin and polygonize classes
    This creates multiple HRUs per subbasin based on raster classes within each subbasin
    """
    import rasterio
    import numpy as np
    from rasterio.mask import mask
    from rasterio.features import shapes
    from shapely.geometry import shape
    
    output_path = raster_path.parent / f"{raster_path.stem}_per_subbasin.shp"
    
    print(f"Polygonizing {raster_path.name} per subbasin (BasinMaker method)...")
    
    all_polys = []
    
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        
        # Process each subbasin separately
        for idx, subbasin in subbasins_gdf.iterrows():
            subid = int(subbasin.get('SubId', idx))
            subbasin_geom = subbasin.geometry
            
            # Reproject subbasin to raster CRS if needed
            if subbasins_gdf.crs != raster_crs:
                subbasin_geom = gpd.GeoSeries([subbasin_geom], crs=subbasins_gdf.crs).to_crs(raster_crs).iloc[0]
            
            try:
                # Clip raster to this subbasin
                out_image, out_transform = mask(src, [subbasin_geom], crop=True, all_touched=True)
                mask_arr = out_image[0]  # first band
                
                # Skip if no valid data
                if src.nodata is not None:
                    mask_arr = np.where(mask_arr == src.nodata, 0, mask_arr)
                
                if not mask_arr.any():
                    continue
                
                # Create polygons for each contiguous pixel group with same value
                for geom_dict, value in shapes(mask_arr, mask=mask_arr > 0, transform=out_transform):
                    if value > 0:
                        shp = shape(geom_dict)
                        if shp.is_valid and shp.area > 0:
                            all_polys.append({
                                'SubId': subid,
                                value_col: int(value),
                                class_col: f'CLASS_{int(value)}',  # Will be mapped later
                                'geometry': shp
                            })
                        
            except Exception as e:
                print(f"    Warning: Could not polygonize SubId {subid}: {e}")
                continue
    
    if all_polys:
        # Create GeoDataFrame with all polygons
        polys_gdf = gpd.GeoDataFrame(all_polys, crs=raster_crs)
        
        # Reproject to target CRS
        if polys_gdf.crs != target_crs:
            polys_gdf = polys_gdf.to_crs(target_crs)
        
        print(f"  Created {len(all_polys)} raw polygons, now dissolving by class...")
        
        # CRITICAL FIX: Dissolve by landuse class within each subbasin to reduce polygon count
        from processors.polygon_overlay import PolygonOverlayProcessor
        overlay_processor = PolygonOverlayProcessor()
        
        # Group by SubId and dissolve by landuse class
        dissolved_polys = []
        for subid in polys_gdf['SubId'].unique():
            sub_polys = polys_gdf[polys_gdf['SubId'] == subid].copy()
            
            # Dissolve by landuse class within this subbasin
            dissolved_sub = overlay_processor.dissolve_by_attribute(sub_polys, class_col)
            dissolved_polys.append(dissolved_sub)
        
        # Combine all dissolved polygons
        if dissolved_polys:
            polys_gdf = pd.concat(dissolved_polys, ignore_index=True)
        
        # Apply minimum area filter (BasinMaker approach)
        MIN_AREA_M2 = 1000  # 1000 m² minimum area
        polys_gdf['area_m2'] = polys_gdf.geometry.area
        polys_gdf = polys_gdf[polys_gdf['area_m2'] >= MIN_AREA_M2].copy()
        polys_gdf = polys_gdf.drop(columns=['area_m2'])
        
        # Save results
        polys_gdf.to_file(output_path)
        print(f"  Final result: {len(polys_gdf)} dissolved polygons across {len(subbasins_gdf)} subbasins")
        print(f"  Saved to: {output_path}")
        
        return output_path
    else:
        raise RuntimeError(f"No polygons created from {raster_path}")


def validate_file_overlays(subbasins_path: Path, lakes_path: Path, landuse_path: Path, soil_path: Path, dem_path: Path) -> Dict[str, Any]:
    """Validate CRS consistency and spatial overlays between all input files"""
    
    print("=== VALIDATING INPUT FILES ===")
    validation_results = {'success': True, 'warnings': [], 'errors': []}
    
    # 1. Check file existence
    files_to_check = {
        'Subbasins': subbasins_path,
        'Lakes': lakes_path, 
        'Landuse': landuse_path,
        'Soil': soil_path,
        'DEM': dem_path
    }
    
    for name, path in files_to_check.items():
        if path is None:
            if name == 'Lakes':
                print(f"  {name}: No lakes file (OK - proceeding without lakes)")
                continue
            else:
                validation_results['errors'].append(f"{name} file is None")
                validation_results['success'] = False
        elif not path.exists():
            validation_results['errors'].append(f"{name} file not found: {path}")
            validation_results['success'] = False
    
    if not validation_results['success']:
        return validation_results
    
    # 2. Load and check CRS consistency
    print("Checking CRS consistency...")
    
    subbasins_gdf = gpd.read_file(subbasins_path)
    landuse_gdf = gpd.read_file(landuse_path)
    soil_gdf = gpd.read_file(soil_path)
    
    # Handle lakes - may be None
    if lakes_path is not None:
        lakes_gdf = gpd.read_file(lakes_path)
    else:
        lakes_gdf = gpd.GeoDataFrame(columns=['HyLakeId', 'geometry'], crs=subbasins_gdf.crs)
    
    with rasterio.open(dem_path) as dem_src:
        dem_crs = dem_src.crs
    
    # Check CRS consistency
    crs_info = {
        'Subbasins': subbasins_gdf.crs,
        'Lakes': lakes_gdf.crs,
        'Landuse': landuse_gdf.crs, 
        'Soil': soil_gdf.crs,
        'DEM': dem_crs
    }
    
    print("CRS Information:")
    for name, crs in crs_info.items():
        print(f"  {name}: {crs}")
    
    # Check if all CRS are compatible (can be reprojected)
    target_crs = subbasins_gdf.crs
    for name, crs in crs_info.items():
        if crs is None:
            validation_results['errors'].append(f"{name} has no CRS defined")
            validation_results['success'] = False
    
    # 3. Check geometry validity
    print("Checking geometry validity...")
    
    geometry_stats = {}
    for name, gdf in [('Subbasins', subbasins_gdf), ('Lakes', lakes_gdf), 
                      ('Landuse', landuse_gdf), ('Soil', soil_gdf)]:
        valid_count = gdf.geometry.is_valid.sum()
        total_count = len(gdf)
        invalid_count = total_count - valid_count
        
        geometry_stats[name] = {
            'total': total_count,
            'valid': valid_count,
            'invalid': invalid_count,
            'valid_pct': (valid_count / total_count) * 100 if total_count > 0 else 0
        }
        
        print(f"  {name}: {valid_count}/{total_count} valid ({geometry_stats[name]['valid_pct']:.1f}%)")
        
        if invalid_count > 0:
            validation_results['warnings'].append(f"{name} has {invalid_count} invalid geometries")
    
    # 4. Check spatial overlaps
    print("Checking spatial overlaps...")
    
    # Reproject all to common CRS for overlap testing
    if target_crs != lakes_gdf.crs:
        lakes_gdf = lakes_gdf.to_crs(target_crs)
    if target_crs != landuse_gdf.crs:
        landuse_gdf = landuse_gdf.to_crs(target_crs)  
    if target_crs != soil_gdf.crs:
        soil_gdf = soil_gdf.to_crs(target_crs)
    
    # Get bounding boxes
    subbasins_bounds = subbasins_gdf.total_bounds
    lakes_bounds = lakes_gdf.total_bounds
    landuse_bounds = landuse_gdf.total_bounds
    soil_bounds = soil_gdf.total_bounds
    
    print(f"Bounding boxes:")
    print(f"  Subbasins: [{subbasins_bounds[0]:.3f}, {subbasins_bounds[1]:.3f}, {subbasins_bounds[2]:.3f}, {subbasins_bounds[3]:.3f}]")
    print(f"  Lakes:     [{lakes_bounds[0]:.3f}, {lakes_bounds[1]:.3f}, {lakes_bounds[2]:.3f}, {lakes_bounds[3]:.3f}]")  
    print(f"  Landuse:   [{landuse_bounds[0]:.3f}, {landuse_bounds[1]:.3f}, {landuse_bounds[2]:.3f}, {landuse_bounds[3]:.3f}]")
    print(f"  Soil:      [{soil_bounds[0]:.3f}, {soil_bounds[1]:.3f}, {soil_bounds[2]:.3f}, {soil_bounds[3]:.3f}]")
    
    # Check actual intersections
    def check_intersection(gdf1, gdf2, name1, name2):
        try:
            # Use spatial index for efficiency
            intersects = gdf1.geometry.bounds.apply(
                lambda x: any(gdf2.geometry.bounds.apply(
                    lambda y: not (x[2] < y[0] or x[0] > y[2] or x[3] < y[1] or x[1] > y[3]), axis=1)), axis=1
            )
            overlap_count = intersects.sum()
            overlap_pct = (overlap_count / len(gdf1)) * 100 if len(gdf1) > 0 else 0
            print(f"  {name1} ↔ {name2}: {overlap_count}/{len(gdf1)} overlaps ({overlap_pct:.1f}%)")
            
            if overlap_count == 0:
                validation_results['errors'].append(f"No spatial overlap between {name1} and {name2}")
                validation_results['success'] = False
            elif overlap_pct < 50:
                validation_results['warnings'].append(f"Low overlap between {name1} and {name2}: {overlap_pct:.1f}%")
                
        except Exception as e:
            validation_results['warnings'].append(f"Could not check intersection between {name1} and {name2}: {e}")
    
    check_intersection(subbasins_gdf, landuse_gdf, "Subbasins", "Landuse")
    check_intersection(subbasins_gdf, soil_gdf, "Subbasins", "Soil")
    check_intersection(subbasins_gdf, lakes_gdf, "Subbasins", "Lakes")
    
    # 5. Check required columns
    print("Checking required columns...")
    
    required_columns = {
        'Subbasins': ['SubId'],
        'Lakes': ['HyLakeId'],  # After renaming
        'Landuse': ['Landuse_ID', 'LAND_USE_C'],
        'Soil': ['Soil_ID', 'SOIL_PROF']
    }
    
    for name, gdf in [('Subbasins', subbasins_gdf), ('Lakes', lakes_gdf), 
                      ('Landuse', landuse_gdf), ('Soil', soil_gdf)]:
        required = required_columns[name]
        missing = [col for col in required if col not in gdf.columns]
        
        if missing:
            validation_results['errors'].append(f"{name} missing columns: {missing}")
            validation_results['success'] = False
        else:
            print(f"  {name}: All required columns present {required}")
    
    # Summary
    print("\n=== VALIDATION SUMMARY ===")
    if validation_results['success']:
        print("SUCCESS: All files valid and ready for HRU generation")
    else:
        print("✗ Validation failed - fix issues before proceeding")
    
    if validation_results['warnings']:
        print(f"Warnings ({len(validation_results['warnings'])}):")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
            
    if validation_results['errors']:
        print(f"Errors ({len(validation_results['errors'])}):")
        for error in validation_results['errors']:
            print(f"  - {error}")
    
    return validation_results


def create_landuse_shapefile_enhanced(data_dir: Path, lookup_generator: RAVENLookupTableGenerator) -> Path:
    """Create enhanced landuse shapefile using comprehensive lookup database"""
    import rasterio
    import rasterio.features
    import numpy as np
    from shapely.geometry import shape
    
    landcover_path = data_dir / "landcover.tif"
    landuse_path = data_dir / "landuse_enhanced.shp"
    
    print("Creating enhanced landuse shapefile from comprehensive lookup database...")
    
    with rasterio.open(landcover_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        
        # Get unique landcover values
        unique_vals = np.unique(data[data != src.nodata])
        print(f"  Found landcover values: {list(unique_vals)}")
        
        # Create polygons for each landcover class using dynamic classification
        polys = []
        for val in unique_vals:
            print(f"  Processing landcover value: {val}")
            
            # Get enhanced classification from lookup database
            landcover_result = lookup_generator.classify_landcover_from_worldcover(int(val))
            raven_class = landcover_result['raven_class']
            raven_id = landcover_result['raven_id']
            
            # Get vegetation class mapping
            veg_mapping = lookup_generator.database["vegetation_classification"]["basinmaker_format"]["landcover_mapping"]
            veg_class = veg_mapping.get(raven_class, f"{raven_class}_VEG")
            
            mask = data == val
            for geom, value in rasterio.features.shapes(mask.astype(np.uint8), mask=mask, transform=transform):
                if value == 1:  # Only keep positive mask areas
                    # Fix geometry if needed
                    shp = shape(geom)
                    if not shp.is_valid:
                        shp = shp.buffer(0)  # Fix invalid geometries
                    if shp.is_valid and shp.area > 0:
                        polys.append({
                            'Landuse_ID': int(raven_id),
                            'LAND_USE_C': raven_class,
                            'VEG_C': veg_class,
                            'geometry': shp
                        })
        
        if polys:
            landuse_gdf = gpd.GeoDataFrame(polys, crs=crs)
            
            # Ensure consistent CRS
            if landuse_gdf.crs != 'EPSG:32611':
                print(f"  Reprojecting landuse from {landuse_gdf.crs} to EPSG:32611...")
                landuse_gdf = landuse_gdf.to_crs('EPSG:32611')
            
            landuse_gdf.to_file(landuse_path)
            
            # Log enhanced classification results
            unique_classes = landuse_gdf['LAND_USE_C'].value_counts()
            print(f"  Created enhanced landuse: {len(polys)} polygons (CRS: {landuse_gdf.crs})")
            print(f"  Enhanced class distribution: {dict(unique_classes)}")
        else:
            raise RuntimeError("No enhanced landuse polygons created")
    
    return landuse_path


def create_soil_shapefile_enhanced(data_dir: Path, lookup_generator: RAVENLookupTableGenerator) -> Path:
    """Create enhanced soil shapefile using JSON database USDA texture classification"""
    import rasterio
    import rasterio.features
    import numpy as np
    from shapely.geometry import shape
    
    sand_path = data_dir / "sand_0-5cm_mean_bbox.tif"
    silt_path = data_dir / "silt_0-5cm_mean_bbox.tif" 
    clay_path = data_dir / "clay_0-5cm_mean_bbox.tif"
    soil_path = data_dir / "soil_polygons_enhanced.shp"
    
    print("Creating enhanced soil shapefile using JSON database classification...")
    
    # Read all three rasters
    with rasterio.open(sand_path) as sand_src, \
         rasterio.open(silt_path) as silt_src, \
         rasterio.open(clay_path) as clay_src:
        
        sand_data = sand_src.read(1)
        silt_data = silt_src.read(1) 
        clay_data = clay_src.read(1)
        transform = sand_src.transform
        crs = sand_src.crs
        
        # Create soil texture classification array using enhanced classification
        height, width = sand_data.shape
        soil_class_data = np.zeros((height, width), dtype=np.uint8)
        soil_names = {}
        
        print("  Applying enhanced USDA soil texture classification from JSON database...")
        for i in range(height):
            for j in range(width):
                if (sand_data[i,j] != sand_src.nodata and 
                    silt_data[i,j] != silt_src.nodata and
                    clay_data[i,j] != clay_src.nodata):
                    
                    # Use enhanced soil texture classification
                    soil_class = lookup_generator.classify_soil_texture_from_percentages(
                        float(sand_data[i,j]),
                        float(silt_data[i,j]), 
                        float(clay_data[i,j])
                    )
                    
                    # Get soil ID from lookup database
                    soil_id = lookup_generator.get_soil_texture_id(soil_class)
                    
                    if soil_id not in soil_names:
                        soil_names[soil_id] = soil_class
                    
                    soil_class_data[i,j] = soil_id
        
        unique_soil_classes = {v: k for k, v in soil_names.items()}
        print(f"  Enhanced soil texture classes found: {list(unique_soil_classes.keys())}")
        
        # Create polygons for each enhanced soil class
        polys = []
        for soil_id, soil_name in soil_names.items():
            mask = soil_class_data == soil_id
            if mask.any():
                for geom, value in rasterio.features.shapes(mask.astype(np.uint8), mask=mask, transform=transform):
                    if value == 1:
                        # Fix geometry if needed
                        shp = shape(geom)
                        if not shp.is_valid:
                            shp = shp.buffer(0)  # Fix invalid geometries
                        if shp.is_valid and shp.area > 0:
                            polys.append({
                                'Soil_ID': int(soil_id),
                                'SOIL_PROF': soil_name,
                                'geometry': shp
                            })
        
        if polys:
            soil_gdf = gpd.GeoDataFrame(polys, crs=crs)
            
            # Ensure consistent CRS
            if soil_gdf.crs != 'EPSG:32611':
                print(f"  Reprojecting soil from {soil_gdf.crs} to EPSG:32611...")
                soil_gdf = soil_gdf.to_crs('EPSG:32611')
            
            soil_gdf.to_file(soil_path)
            
            # Log enhanced results
            soil_class_counts = soil_gdf['SOIL_PROF'].value_counts()
            print(f"  Created enhanced soil: {len(polys)} polygons (CRS: {soil_gdf.crs})")
            print(f"  Enhanced soil distribution: {dict(soil_class_counts)}")
        else:
            raise RuntimeError("No enhanced soil polygons created")
    
    return soil_path


def generate_hrus_clean(workspace_dir: Path, latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Generate HRUs using enhanced approach with dynamic lookup table generators
    ENHANCED: Integrates JSON database for comprehensive soil/landcover classification
    """
    
    data_dir = workspace_dir / "data"
    
    print(f"ENHANCED STEP 4: Generating HRUs with dynamic lookup tables for outlet ({latitude}, {longitude})")
    
    # Initialize dynamic generators
    print("Initializing dynamic lookup table generators...")
    lookup_generator = RAVENLookupTableGenerator(output_dir=data_dir / "lookup_tables")
    hru_mapper = HRUClassMapper()
    
    # Generate all BasinMaker-compatible lookup tables
    print("Generating BasinMaker-compatible lookup tables...")
    lookup_files = lookup_generator.generate_all_lookup_tables()
    
    for table_name, file_path in lookup_files.items():
        print(f"  Generated {table_name}: {file_path}")
    
    # Check required input files
    subbasins_path = data_dir / "subbasins_with_lakes.shp"
    dem_path = data_dir / "dem.tif"
    
    if not subbasins_path.exists():
        raise FileNotFoundError(f"Subbasins file not found: {subbasins_path}")
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")
    
    # Prepare input files for HRU generator
    print("Preparing input files with dynamic classification...")
    
    # Step 1: Fix lakes shapefile
    lakes_path = prepare_lakes_shapefile(data_dir)
    
    # Step 2: Create landuse shapefile using dynamic classification
    print("Creating landuse polygons with dynamic classification...")
    subbasins_gdf = gpd.read_file(subbasins_path)
    landcover_raster = data_dir / "landcover.tif"
    
    if landcover_raster.exists():
        landuse_path = polygonize_raster_per_subbasin(
            landcover_raster, subbasins_gdf, 
            'Landuse_ID', 'LAND_USE_C', 'EPSG:32611'
        )
        
        # Load and enhance landuse data with dynamic classification
        landuse_gdf = gpd.read_file(landuse_path)
        
        # Apply dynamic landcover classification using lookup database
        print("  Applying dynamic landcover classification from JSON database...")
        enhanced_classes = []
        veg_classes = []
        
        for idx, row in landuse_gdf.iterrows():
            landcover_id = row.get('Landuse_ID', 30)  # Default to grassland
            
            # Get classification from lookup database
            landcover_result = lookup_generator.classify_landcover_from_worldcover(landcover_id)
            raven_class = landcover_result['raven_class']
            
            # Get vegetation class mapping
            veg_mapping = lookup_generator.database["vegetation_classification"]["basinmaker_format"]["landcover_mapping"]
            veg_class = veg_mapping.get(raven_class, f"{raven_class}_VEG")
            
            enhanced_classes.append(raven_class)
            veg_classes.append(veg_class)
        
        landuse_gdf['LAND_USE_C'] = enhanced_classes
        landuse_gdf['VEG_C'] = veg_classes
        landuse_gdf.to_file(landuse_path)
        
        unique_classes = pd.Series(enhanced_classes).value_counts()
        print(f"  Enhanced landuse classification: {len(landuse_gdf)} polygons")
        print(f"  Class distribution: {dict(unique_classes)}")
    else:
        landuse_path = create_landuse_shapefile_enhanced(data_dir, lookup_generator)
    
    # Step 3: Create soil shapefile using enhanced USDA classification from JSON database
    print("Creating soil polygons with enhanced USDA texture classification...")
    soil_path = create_soil_shapefile_enhanced(data_dir, lookup_generator)
    
    # --- MAKE MULTI-HRU POLYGONS (landuse × soil) PER SUBBASIN ---
    print("Creating multi-HRU polygons (landuse × soil per subbasin)...")
    landuse_gdf = gpd.read_file(landuse_path)        # has SubId, Landuse_ID, LAND_USE_C, VEG_C
    soil_gdf    = gpd.read_file(soil_path)           # has Soil_ID, SOIL_PROF
    subs_gdf    = gpd.read_file(subbasins_path)      # has SubId
    
    # Handle lakes - may be None if no lakes exist
    if lakes_path is not None:
        lakes_gdf = gpd.read_file(lakes_path)          # has HyLakeId
    else:
        # Create empty lakes GeoDataFrame with correct CRS
        lakes_gdf = gpd.GeoDataFrame(columns=['HyLakeId', 'geometry'], crs=subs_gdf.crs)

    # 1) Ensure common CRS
    target_crs = subs_gdf.crs
    if landuse_gdf.crs != target_crs:
        landuse_gdf = landuse_gdf.to_crs(target_crs)
    if soil_gdf.crs != target_crs:
        soil_gdf = soil_gdf.to_crs(target_crs)
    if lakes_gdf.crs != target_crs:
        lakes_gdf = lakes_gdf.to_crs(target_crs)

    # 2) Clip soil to subbasins and tag SubId
    print("  Clipping soil to subbasins...")
    soil_in_subs = gpd.overlay(soil_gdf, subs_gdf[['SubId','geometry']], how='intersection')
    
    # 3) Landuse is already per-subbasin (from polygonize_raster_per_subbasin)
    landuse_in_subs = landuse_gdf

    # 4) Build HRU candidates using BasinMaker overlay logic
    print("  Creating landuse × soil intersections using BasinMaker logic...")
    from processors.polygon_overlay import PolygonOverlayProcessor
    overlay_processor = PolygonOverlayProcessor(data_dir)
    
    # Use intersection for HRU generation (landuse AND soil, not landuse OR soil)
    hru_candidates = gpd.overlay(landuse_in_subs, soil_in_subs, how='intersection')
    # Apply BasinMaker geometry fixes to intersection result
    hru_candidates = overlay_processor._fix_geometries(hru_candidates)
    
    # Fix duplicate SubId columns from overlay (SubId_1 and SubId_2 -> SubId)
    if 'SubId_1' in hru_candidates.columns:
        hru_candidates['SubId'] = hru_candidates['SubId_1']
        hru_candidates = hru_candidates.drop(columns=['SubId_1', 'SubId_2'], errors='ignore')

    # 5) Carve out lakes using BasinMaker approach
    if len(lakes_gdf) > 0:
        print("  Carving out lakes using BasinMaker logic...")
        # Remove lake areas from HRU candidates using difference operation
        lakes_union = lakes_gdf.union_all()
        hru_candidates['geometry'] = hru_candidates.geometry.difference(lakes_union)
        hru_candidates = hru_candidates[~hru_candidates.geometry.is_empty]

    # 6) Apply BasinMaker geometry fixes and area filtering
    print("  Applying BasinMaker geometry fixes...")
    hru_candidates = overlay_processor._fix_geometries(hru_candidates)
    
    MIN_HRU_KM2 = 0.01
    hru_candidates['HRU_Area'] = hru_candidates.geometry.area / 1e6
    hru_candidates = hru_candidates[hru_candidates['HRU_Area'] >= MIN_HRU_KM2].copy()

    # 7) Save multi-HRU candidates  
    import time
    multi_hru_path = data_dir / f"hrus_land_soil_candidates_{int(time.time())}.shp"
    hru_candidates.to_file(multi_hru_path)
    print(f"  Multi-HRU candidates: {len(hru_candidates)} polygons across {hru_candidates['SubId'].nunique()} subbasins")
    
    # Keep the multi-HRU path separate - don't overwrite landuse_path
    # landuse_path = multi_hru_path  # REMOVED - this was causing 349 HRUs instead of 25
    
    # Step 4: Validate all files before HRU generation
    validation_results = validate_file_overlays(subbasins_path, lakes_path, landuse_path, soil_path, dem_path)
    
    if not validation_results['success']:
        raise RuntimeError(f"File validation failed: {validation_results['errors']}")
    
    if validation_results['warnings']:
        print("PROCEEDING WITH WARNINGS - this may cause overlay issues")
    
    # Step 5: Create final HRU dataset with enhanced class mapping
    print("Finalizing HRU dataset with enhanced class mapping...")
    
    # Load the multi-HRU candidates (already has landuse×soil intersections)
    final_hrus = gpd.read_file(multi_hru_path)
    
    # Apply enhanced HRU class mapping using the HRU mapper
    print("  Applying enhanced HRU class assignments...")
    
    # Check if we have soil raster files for enhanced mapping
    sand_raster = data_dir / "sand_0-5cm_mean_bbox.tif"
    silt_raster = data_dir / "silt_0-5cm_mean_bbox.tif"
    clay_raster = data_dir / "clay_0-5cm_mean_bbox.tif"
    landcover_raster = data_dir / "landcover.tif"
    dem_raster = data_dir / "dem.tif"
    
    # Apply complete enhanced HRU class assignment
    final_hrus = hru_mapper.assign_complete_hru_classes(
        final_hrus,
        sand_raster=sand_raster if sand_raster.exists() else None,
        silt_raster=silt_raster if silt_raster.exists() else None,
        clay_raster=clay_raster if clay_raster.exists() else None,
        landcover_raster=landcover_raster if landcover_raster.exists() else None,
        dem_raster=dem_raster if dem_raster.exists() else None
    )
    
    # Add HRU IDs and additional required attributes
    final_hrus['HRU_ID'] = range(1, len(final_hrus) + 1)
    final_hrus['HRULake_ID'] = final_hrus['SubId'] * 1000 + final_hrus['HRU_ID']  # Unique HRU identifier
    final_hrus['HRU_IsLake'] = 0  # All are land HRUs (lakes added separately)
    final_hrus['HyLakeId'] = -1   # No lake association for land HRUs
    
    # Calculate centroids for HRU_CenX, HRU_CenY - MUST be in geographic degrees (EPSG:4326)
    print("  Converting HRU centroids to geographic coordinates (lat/lon degrees)...")
    
    # Check current CRS and convert to EPSG:4326 if needed
    if final_hrus.crs != 'EPSG:4326':
        print(f"    Converting from {final_hrus.crs} to EPSG:4326")
        # Convert to geographic coordinates
        centroids_geo = final_hrus.geometry.centroid.to_crs('EPSG:4326')
        final_hrus['HRU_CenX'] = centroids_geo.x  # Longitude in degrees
        final_hrus['HRU_CenY'] = centroids_geo.y  # Latitude in degrees
    else:
        centroids = final_hrus.geometry.centroid
        final_hrus['HRU_CenX'] = centroids.x
        final_hrus['HRU_CenY'] = centroids.y
    
    # Validate coordinate ranges for geographic coordinates
    lon_valid = (final_hrus['HRU_CenX'] >= -180) & (final_hrus['HRU_CenX'] <= 180)
    lat_valid = (final_hrus['HRU_CenY'] >= -90) & (final_hrus['HRU_CenY'] <= 90)
    
    if not lon_valid.all():
        invalid_lon = final_hrus[~lon_valid]
        print(f"    WARNING: {len(invalid_lon)} HRUs have invalid longitude values")
        print(f"    Longitude range: {final_hrus['HRU_CenX'].min():.6f} to {final_hrus['HRU_CenX'].max():.6f}")
    
    if not lat_valid.all():
        invalid_lat = final_hrus[~lat_valid]
        print(f"    WARNING: {len(invalid_lat)} HRUs have invalid latitude values")
        print(f"    Latitude range: {final_hrus['HRU_CenY'].min():.6f} to {final_hrus['HRU_CenY'].max():.6f}")
    
    if lon_valid.all() and lat_valid.all():
        print(f"    SUCCESS: All HRU coordinates are valid geographic degrees")
        print(f"    Longitude range: {final_hrus['HRU_CenX'].min():.6f} to {final_hrus['HRU_CenX'].max():.6f}")
        print(f"    Latitude range: {final_hrus['HRU_CenY'].min():.6f} to {final_hrus['HRU_CenY'].max():.6f}")
    
    # Apply BasinMaker HRU consolidation
    print("  Applying BasinMaker HRU consolidation...")
    from utilities.hru_consolidator import BasinMakerHRUConsolidator
    
    # Set consolidation parameters (BasinMaker defaults)
    min_hru_pct_sub_area = 0.05  # 5% of subbasin area minimum (more aggressive)
    importance_order = ['Landuse_ID', 'Soil_ID', 'Veg_ID']
    
    consolidator = BasinMakerHRUConsolidator(
        min_hru_pct_sub_area=min_hru_pct_sub_area,
        importance_order=importance_order
    )
    
    # Apply consolidation to land HRUs
    final_hrus = consolidator.consolidate_hrus(final_hrus)
    print(f"    HRU consolidation complete: {len(final_hrus)} final HRUs")

    # Add lake HRUs if any
    if len(lakes_gdf) > 0:
        print("  Adding lake HRUs...")
        lakes_as_hrus = lakes_gdf.copy()
        lakes_as_hrus['HRU_ID'] = range(len(final_hrus) + 1, len(final_hrus) + len(lakes_as_hrus) + 1)
        lakes_as_hrus['HRULake_ID'] = lakes_as_hrus['HyLakeId'] + 10000  # Unique lake HRU IDs
        lakes_as_hrus['HRU_IsLake'] = 1
        
        # Fix: Assign proper SubIds to lakes based on spatial containment
        print("  Assigning SubIds to lake HRUs based on spatial location...")
        lakes_as_hrus['SubId'] = -1  # Initialize with -1
        
        # Spatial assignment using centroids
        for idx, lake_hru in lakes_as_hrus.iterrows():
            lake_centroid = lake_hru.geometry.centroid
            
            # Find which subbasin contains this lake centroid
            containing_subbasin = subbasins_gdf[subbasins_gdf.geometry.contains(lake_centroid)]
            
            if len(containing_subbasin) > 0:
                # Use the SubId of the containing subbasin
                subid = containing_subbasin.iloc[0]['SubId']
                lakes_as_hrus.loc[idx, 'SubId'] = subid
                print(f"    Lake HRU {idx} (HyLakeId: {lake_hru['HyLakeId']}) -> SubId: {subid}")
            else:
                # Fallback: find the closest subbasin
                min_distance = float('inf')
                closest_subid = None
                
                for _, subbasin in subbasins_gdf.iterrows():
                    distance = lake_centroid.distance(subbasin.geometry.centroid)
                    if distance < min_distance:
                        min_distance = distance
                        closest_subid = subbasin['SubId']
                
                if closest_subid is not None:
                    lakes_as_hrus.loc[idx, 'SubId'] = closest_subid
                    print(f"    Lake HRU {idx} (HyLakeId: {lake_hru['HyLakeId']}) -> SubId: {closest_subid} (closest)")
                else:
                    # Ultimate fallback
                    lakes_as_hrus.loc[idx, 'SubId'] = subbasins_gdf['SubId'].iloc[0]
                    print(f"    Lake HRU {idx} (HyLakeId: {lake_hru['HyLakeId']}) -> SubId: {subbasins_gdf['SubId'].iloc[0]} (fallback)")
        lakes_as_hrus['Landuse_ID'] = 99
        lakes_as_hrus['LAND_USE_C'] = 'LAKE'
        lakes_as_hrus['Soil_ID'] = 99
        lakes_as_hrus['SOIL_PROF'] = 'WATER'
        lakes_as_hrus['Veg_ID'] = 99
        lakes_as_hrus['VEG_C'] = 'WATER'
        lakes_as_hrus['HRU_Area'] = lakes_as_hrus.geometry.area / 1e6
        
        # Ensure lake HRU coordinates are also in geographic degrees
        if lakes_as_hrus.crs != 'EPSG:4326':
            print(f"    Converting lake HRU centroids from {lakes_as_hrus.crs} to EPSG:4326")
            centroids_geo = lakes_as_hrus.geometry.centroid.to_crs('EPSG:4326')
            lakes_as_hrus['HRU_CenX'] = centroids_geo.x  # Longitude in degrees
            lakes_as_hrus['HRU_CenY'] = centroids_geo.y  # Latitude in degrees
        else:
            centroids = lakes_as_hrus.geometry.centroid
            lakes_as_hrus['HRU_CenX'] = centroids.x
            lakes_as_hrus['HRU_CenY'] = centroids.y
        
        # Combine land and lake HRUs
        final_hrus = pd.concat([final_hrus, lakes_as_hrus], ignore_index=True)
    
    print(f"  Created {len(final_hrus)} total HRUs ({len(final_hrus[final_hrus['HRU_IsLake']==0])} land, {len(final_hrus[final_hrus['HRU_IsLake']==1])} lake)")
    
    # Save the final HRU results
    hru_output_path = data_dir / "hrus.geojson"  # Use standard name expected by step5
    final_hrus.to_file(hru_output_path, driver='GeoJSON')
    
    # Also save as hru_output.geojson for backward compatibility
    hru_output_path_alt = data_dir / "hru_output.geojson"
    final_hrus.to_file(hru_output_path_alt, driver='GeoJSON')
    
    # Create results dictionary
    hru_results = {
        'hru_gdf': final_hrus,
        'hru_output_file': str(hru_output_path),
        'n_hrus': len(final_hrus),
        'n_land_hrus': len(final_hrus[final_hrus['HRU_IsLake']==0]),
        'n_lake_hrus': len(final_hrus[final_hrus['HRU_IsLake']==1]),
        'subbasins_processed': final_hrus[final_hrus['HRU_IsLake']==0]['SubId'].nunique()
    }
    
    print(f"SUCCESS: Generated {hru_results['n_hrus']} HRUs")
    print(f"  Lake HRUs: {hru_results['n_lake_hrus']}")
    print(f"  Land HRUs: {hru_results['n_land_hrus']}")
    print(f"  Subbasins processed: {hru_results['subbasins_processed']}")
    print(f"  Average HRUs per subbasin: {hru_results['n_land_hrus'] / hru_results['subbasins_processed']:.1f}")
    print(f"  Output: {hru_results['hru_output_file']}")
    
    # Save summary
    import json
    with open(data_dir / "step4_results.json", "w") as f:
        json.dump({k: v for k, v in hru_results.items() if k != 'hru_gdf'}, f, indent=2)
    
    return hru_results


def main():
    parser = argparse.ArgumentParser(description="Clean Step 4: HRU Generation using HRUGenerator")
    parser.add_argument("latitude", type=float, help="Outlet latitude")
    parser.add_argument("longitude", type=float, help="Outlet longitude") 
    parser.add_argument("--workspace-dir", required=True, type=Path, help="Workspace directory")
    
    args = parser.parse_args()
    
    try:
        results = generate_hrus_clean(args.workspace_dir, args.latitude, args.longitude)
        print(f"HRU generation completed successfully!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()