#!/usr/bin/env python3
"""
Comprehensive Lake Detection Processor
Merged functionality from lake_detection_final.py with lake_classifier and lake_filter
Provides complete lake detection, classification, and filtering capabilities
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Tuple, Union

# Import infrastructure
sys.path.append(str(Path(__file__).parent.parent))

from clients.data_clients.spatial_client import SpatialLayersClient
from processors.lake_classifier import LakeClassifier
from processors.lake_filter import LakeFilter
from infrastructure.path_manager import AbsolutePathManager, PathResolutionError, FileAccessError
from infrastructure.file_operations import SecureFileOperations, FileValidationError


class ComprehensiveLakeDetector:
    """
    Comprehensive lake detection processor combining:
    1. DEM-based lake detection (from lake_detection_final)
    2. Lake classification (connected vs non-connected)
    3. Lake filtering by area thresholds
    4. Integration with watershed analysis
    """
    
    def __init__(self, workspace_dir: Path = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        
        # Initialize secure file operations with path validation
        try:
            self.path_manager = AbsolutePathManager(self.workspace_dir)
            self.file_ops = SecureFileOperations(self.path_manager)
            # Validate workspace permissions
            self.path_manager.validate_workspace_permissions()
        except (PathResolutionError, FileAccessError) as e:
            raise RuntimeError(f"Failed to initialize lake detector workspace: {e}")
        
        # Initialize sub-components
        self.spatial_client = SpatialLayersClient()
        self.classifier = LakeClassifier(self.workspace_dir)
        self.filter = LakeFilter(self.workspace_dir)
        
        # Note: All thresholds should now be passed as parameters to functions
        # No default hardcoded values - they must be specified by the caller
        
    def detect_and_classify_lakes(self, 
                                bbox: List[float],
                                min_lake_area_m2: float = None,
                                connected_threshold_km2: float = None,
                                non_connected_threshold_km2: float = None,
                                lake_depth_threshold: float = None,
                                streams_shapefile: Path = None,
                                watershed_boundary: Path = None) -> Dict:
        """
        Complete lake detection and classification workflow
        
        Parameters:
        -----------
        bbox : List[float]
            Study area bounding box [minx, miny, maxx, maxy]
        min_lake_area_m2 : float, optional
            Minimum lake area in square meters (default: 500)
        connected_threshold_km2 : float, optional
            Minimum area for connected lakes in km² (default: 0.01)
        non_connected_threshold_km2 : float, optional
            Minimum area for non-connected lakes in km² (default: 0.1)
        lake_depth_threshold : float, optional
            Minimum depression depth for lake detection in meters (default: 0.3)
        streams_shapefile : Path, optional
            Path to streams shapefile for connectivity analysis
        watershed_boundary : Path, optional
            Path to watershed boundary shapefile for filtering lakes to watershed only
            
        Returns:
        --------
        Dict with complete lake detection and classification results
        """
        
        print("=" * 60)
        print("COMPREHENSIVE LAKE DETECTION & CLASSIFICATION")
        print("=" * 60)
        print(f"Area: {bbox}")
        print(f"Working directory: {self.workspace_dir}")
        
        # All parameters must be provided - no defaults
        if min_lake_area_m2 is None:
            raise ValueError("min_lake_area_m2 must be specified")
        if connected_threshold_km2 is None:
            raise ValueError("connected_threshold_km2 must be specified")
        if non_connected_threshold_km2 is None:
            raise ValueError("non_connected_threshold_km2 must be specified")
        if lake_depth_threshold is None:
            raise ValueError("lake_depth_threshold must be specified")
        
        results = {
            'success': False,
            'bbox': bbox,
            'parameters': {
                'min_lake_area_m2': min_lake_area_m2,
                'connected_threshold_km2': connected_threshold_km2,
                'non_connected_threshold_km2': non_connected_threshold_km2,
                'lake_depth_threshold': lake_depth_threshold
            },
            'files_created': [],
            'statistics': {}
        }
        
        try:
            # Step 1: DEM-based lake detection
            print("\n--- Step 1: DEM-based Lake Detection ---")
            detection_results = self._detect_lakes_from_dem(
                bbox, min_lake_area_m2, lake_depth_threshold
            )
            
            if not detection_results['success']:
                results['error'] = detection_results['error']
                return results
                
            results.update({
                'dem_file': detection_results['dem_file'],
                'lake_shapefile': detection_results['lake_shapefile'],
                'raw_lake_count': detection_results['lake_count'],
                'raw_total_area_ha': detection_results['total_area_ha']
            })
            
            # Step 1.5: Filter lakes to watershed boundary (if provided)
            if watershed_boundary and Path(watershed_boundary).exists():
                print("\n--- Step 1.5: Filtering Lakes to Watershed ---")
                watershed_filtering_results = self._filter_lakes_to_watershed(
                    detection_results['lake_shapefile'],
                    watershed_boundary
                )
                
                if watershed_filtering_results['success']:
                    # Update to use filtered lakes for classification
                    results.update({
                        'lake_shapefile': watershed_filtering_results['filtered_lakes_file'],
                        'filtered_lake_count': watershed_filtering_results['filtered_count'],
                        'lakes_removed_count': watershed_filtering_results['removed_count'],
                        'watershed_filter_applied': True
                    })
                    print(f"   Filtered: {results['raw_lake_count']} → {results['filtered_lake_count']} lakes (removed {results['lakes_removed_count']} outside watershed)")
                else:
                    print(f"   WARNING: Watershed filtering failed: {watershed_filtering_results['error']}")
                    results['watershed_filter_applied'] = False
            else:
                results['watershed_filter_applied'] = False
                results['filtered_lake_count'] = results['raw_lake_count']
            
            # Step 2: Lake classification
            print("\n--- Step 2: Lake Classification ---")
            classification_results = self._classify_detected_lakes(
                detection_results['lake_shapefile'],
                streams_shapefile,
                connected_threshold_km2,
                non_connected_threshold_km2
            )
            
            if not classification_results['success']:
                results['error'] = classification_results['error']
                return results
                
            results.update(classification_results)
            
            # Step 3: Create comprehensive summary
            print("\n--- Step 3: Creating Summary ---")
            summary = self._create_comprehensive_summary(results)
            results['summary_file'] = summary
            
            results['success'] = True
            
            print("\n" + "=" * 60)
            print("LAKE DETECTION & CLASSIFICATION COMPLETE")
            print("=" * 60)
            print(f"Total lakes detected: {results['raw_lake_count']}")
            print(f"Total lake area: {results['raw_total_area_ha']:.1f} hectares")
            print(f"Connected lakes: {results['connected_count']}")
            print(f"Non-connected lakes: {results['non_connected_count']}")
            print(f"All files in: {self.workspace_dir}")
            
        except Exception as e:
            results['error'] = str(e)
            print(f"\nError: {e}")
            
        return results
    
    def _detect_lakes_from_dem(self, bbox: List[float], min_area_m2: float, depth_threshold: float) -> Dict:
        """DEM-based lake detection using WhiteboxTools"""
        
        try:
            dem_file = self.workspace_dir / "study_area_dem.tif"
            
            # Download DEM
            dem_result = self.spatial_client.get_dem_for_watershed(
                bbox=bbox,
                output_path=dem_file,
                resolution=30,
                source='usgs'
            )
            
            if not dem_result.get('success', False):
                return {'success': False, 'error': f"DEM download failed: {dem_result.get('error', 'Unknown')}"}
            
            # Validate DEM
            if not self._validate_dem_quality(dem_file):
                return {'success': False, 'error': "DEM validation failed"}
            
            # Process to lakes
            lakes_result = self._process_dem_to_lakes(dem_file, min_area_m2, depth_threshold)
            
            return lakes_result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_dem_quality(self, dem_file: Path) -> bool:
        """Validate DEM has reasonable elevation data"""
        
        try:
            import rasterio
            
            with rasterio.open(dem_file) as src:
                data = src.read(1, window=((0, min(500, src.height)), (0, min(500, src.width))))
                
                if src.nodata is not None:
                    valid_data = data[data != src.nodata]
                else:
                    valid_data = data[(data > -1000) & (data < 10000)]
                
                if valid_data.size == 0:
                    return False
                
                min_elev = float(valid_data.min())
                max_elev = float(valid_data.max())
                range_elev = max_elev - min_elev
                
                print(f"  DEM elevation: {min_elev:.1f} to {max_elev:.1f} m (range: {range_elev:.1f} m)")
                
                return range_elev >= 2
                
        except Exception as e:
            print(f"  DEM validation failed: {e}")
            return False
    
    def _process_dem_to_lakes(self, dem_file: Union[str, Path], min_area_m2: float, depth_threshold: float) -> Dict:
        """Process DEM to create lake shapefile"""
        
        try:
            import whitebox
            import rasterio
            from rasterio.features import shapes
            from shapely.geometry import shape
            import shutil
            
            # CRITICAL FIX: Ensure dem_file is a Path object
            dem_file = Path(dem_file)
            
            print(f"=== LAKE DETECTION DEBUG ===")
            print(f"Input DEM file: {dem_file}")
            print(f"DEM exists: {dem_file.exists()}")
            print(f"Min area threshold: {min_area_m2} m²")
            print(f"Depth threshold: {depth_threshold} m")
            print(f"Workspace: {self.workspace_dir}")
            
            # CRITICAL: Validate input DEM file
            if not dem_file.exists():
                error_msg = f"CRITICAL ERROR: Input DEM file does not exist: {dem_file}"
                print(error_msg)
                return {'success': False, 'error': error_msg}
            
            # Validate DEM file can be opened
            try:
                with rasterio.open(dem_file) as src:
                    print(f"DEM validation:")
                    print(f"  - Size: {src.width} x {src.height} pixels")
                    print(f"  - CRS: {src.crs}")
                    print(f"  - Bounds: {src.bounds}")
                    print(f"  - NoData: {src.nodata}")
                    
                    # Read a small sample to check data quality
                    sample = src.read(1, window=((0, min(100, src.height)), (0, min(100, src.width))))
                    valid_pixels = np.sum(sample != src.nodata) if src.nodata is not None else sample.size
                    print(f"  - Valid pixels in sample: {valid_pixels}/{sample.size}")
                    
                    if valid_pixels == 0:
                        error_msg = f"CRITICAL ERROR: DEM contains no valid data!"
                        print(error_msg)
                        return {'success': False, 'error': error_msg}
                        
            except Exception as e:
                error_msg = f"CRITICAL ERROR: Cannot read DEM file: {str(e)}"
                print(error_msg)
                return {'success': False, 'error': error_msg}
            
            # Initialize Whitebox
            wbt = whitebox.WhiteboxTools()
            wbt.set_working_dir(str(self.workspace_dir.absolute()))
            wbt.set_verbose_mode(True)  # Enable verbose for debugging
            
            print(f"WhiteboxTools working directory: {wbt.work_dir}")
            
            # File names
            filled_name = "filled_dem.tif"
            depression_name = "depressions.tif"
            
            print(f"Processing DEM for lake detection...")
            
            # CRITICAL: Handle DEM file location correctly
            # For WhiteboxTools compatibility, always copy DEM to workspace root
            workspace_dem = self.workspace_dir / dem_file.name
            print(f"Target workspace DEM location: {workspace_dem}")
            
            # Always copy to ensure WhiteboxTools can access it directly
            if not workspace_dem.exists() or workspace_dem.stat().st_size == 0:
                print(f"Copying DEM to workspace root for WhiteboxTools...")
                print(f"  From: {dem_file}")
                print(f"  To: {workspace_dem}")
                try:
                    import shutil
                    shutil.copy2(dem_file, workspace_dem)
                    print(f"Successfully copied DEM to workspace root: {workspace_dem}")
                except Exception as e:
                    error_msg = f"CRITICAL ERROR: Failed to copy DEM to workspace: {e}"
                    print(error_msg)
                    return {'success': False, 'error': error_msg}
            else:
                print(f"DEM already exists in workspace root: {workspace_dem}")
                
            # Validate the workspace DEM
            if not workspace_dem.exists() or workspace_dem.stat().st_size == 0:
                error_msg = f"CRITICAL ERROR: Workspace DEM is missing or empty: {workspace_dem}"
                print(error_msg)
                return {'success': False, 'error': error_msg}
                
            print(f"DEM ready for WhiteboxTools: {workspace_dem}")
            
            # STEP 1: Fill depressions
            print(f"Step 1: Filling depressions...")
            try:
                # Use just the filename since DEM is now in workspace root
                dem_filename = workspace_dem.name
                print(f"Using DEM filename for WhiteboxTools: {dem_filename}")
                
                wbt.fill_depressions_wang_and_liu(
                    dem=dem_filename,
                    output=filled_name,
                    fix_flats=True
                )
                
                filled_path = self.workspace_dir / filled_name
                if not filled_path.exists():
                    error_msg = f"CRITICAL ERROR: Depression filling failed - output file not created: {filled_path}"
                    print(error_msg)
                    return {'success': False, 'error': error_msg}
                    
                print(f"Depression filling completed: {filled_path}")
                
            except Exception as e:
                error_msg = f"CRITICAL ERROR: Depression filling failed: {str(e)}"
                print(error_msg)
                return {'success': False, 'error': error_msg}
            
            # STEP 2: Calculate depression depths
            print(f"Step 2: Calculating depression depths...")
            try:
                wbt.subtract(
                    input1=filled_name,
                    input2=dem_filename,
                    output=depression_name
                )
                
                depression_file = self.workspace_dir / depression_name
                if not depression_file.exists():
                    error_msg = f"CRITICAL ERROR: Depression calculation failed - output file not created: {depression_file}"
                    print(error_msg)
                    return {'success': False, 'error': error_msg}
                    
                print(f"Depression calculation completed: {depression_file}")
                
                # Validate depression file has data
                with rasterio.open(depression_file) as src:
                    sample = src.read(1, window=((0, min(100, src.height)), (0, min(100, src.width))))
                    max_depression = np.max(sample)
                    print(f"  - Maximum depression depth in sample: {max_depression:.3f} m")
                    print(f"  - Depression file size: {src.width} x {src.height} pixels")
                    
                    if max_depression <= 0:
                        print(f"WARNING: No positive depression depths found in sample!")
                        print(f"This may indicate:")
                        print(f"  1. DEM already has depressions filled")
                        print(f"  2. Very flat terrain with no natural depressions")
                        print(f"  3. DEM processing issues")
                        
            except Exception as e:
                error_msg = f"CRITICAL ERROR: Depression calculation failed: {str(e)}"
                print(error_msg)
                return {'success': False, 'error': error_msg}
            
            # Convert to lake polygons
            with rasterio.open(depression_file) as src:
                depression_data = src.read(1)
                transform = src.transform
                crs = src.crs
                
                # Apply depth threshold
                lake_mask = (depression_data >= depth_threshold).astype(np.uint8)
                
                depression_pixels = np.sum(lake_mask)
                print(f"    Found {depression_pixels} pixels >= {depth_threshold}m depth")
                
                if depression_pixels == 0:
                    # Create empty shapefile
                    empty_gdf = gpd.GeoDataFrame(
                        columns=['lake_id', 'depth_m', 'area_m2', 'lake_type', 'geometry'], 
                        crs=crs
                    )
                    shapefile_path = self.workspace_dir / "lakes.shp"
                    try:
                        # For empty GeoDataFrame, write directly (secure operations expect non-empty data)
                        empty_gdf.to_file(shapefile_path)
                    except Exception as e:
                        return {'success': False, 'error': f'Failed to write empty lakes shapefile: {e}'}
                    return {
                        'success': True,
                        'dem_file': str(dem_file),
                        'lake_shapefile': str(shapefile_path),
                        'lake_count': 0,
                        'total_area_ha': 0
                    }
                
                # Vectorize lake areas
                lake_polygons = []
                for geom, value in shapes(lake_mask, mask=lake_mask, transform=transform):
                    if value == 1:
                        lake_polygons.append(shape(geom))
                
                if not lake_polygons:
                    empty_gdf = gpd.GeoDataFrame(
                        columns=['lake_id', 'depth_m', 'area_m2', 'lake_type', 'geometry'], 
                        crs=crs
                    )
                    shapefile_path = self.workspace_dir / "lakes.shp"
                    empty_gdf.to_file(shapefile_path)
                    return {
                        'success': True,
                        'dem_file': str(dem_file),
                        'lake_shapefile': str(shapefile_path),
                        'lake_count': 0,
                        'total_area_ha': 0
                    }
                
                # Create GeoDataFrame with actual depth calculations
                lake_depths = []
                for polygon in lake_polygons:
                    # Calculate actual depth for each lake polygon
                    # Create a mask for this specific polygon
                    from rasterio.mask import mask as rasterio_mask
                    
                    try:
                        # Extract depression values within this polygon
                        with rasterio.open(depression_file) as depth_src:
                            polygon_mask, polygon_transform = rasterio_mask(depth_src, [polygon], crop=True)
                            polygon_depths = polygon_mask[0]
                            
                            # Calculate actual depth statistics for this polygon
                            valid_depths = polygon_depths[polygon_depths > 0]
                            if len(valid_depths) > 0:
                                # Use maximum depth within the polygon
                                actual_depth = float(np.max(valid_depths))
                                # Ensure it meets minimum threshold
                                actual_depth = max(actual_depth, depth_threshold)
                            else:
                                # Fallback to threshold if no valid depths
                                actual_depth = depth_threshold
                                
                    except Exception as e:
                        print(f"Warning: Could not calculate depth for polygon, using threshold: {e}")
                        actual_depth = depth_threshold
                    
                    lake_depths.append(actual_depth)
                
                lakes_gdf = gpd.GeoDataFrame({
                    'geometry': lake_polygons,
                    'depth_m': lake_depths
                }, crs=crs)
                
                print(f"    Calculated actual depths - Range: {min(lake_depths):.3f} to {max(lake_depths):.3f} m")
                
                # Calculate areas
                if crs.is_geographic:
                    utm_crs = lakes_gdf.estimate_utm_crs()
                    lakes_projected = lakes_gdf.to_crs(utm_crs)
                    areas_m2 = lakes_projected.geometry.area
                    lakes_gdf['area_m2'] = areas_m2
                else:
                    lakes_gdf['area_m2'] = lakes_gdf.geometry.area
                
                # Filter by minimum area
                lakes_filtered = lakes_gdf[lakes_gdf['area_m2'] >= min_area_m2].copy()
                
                print(f"    Total polygons: {len(lakes_gdf)}")
                print(f"    After {min_area_m2}m² filter: {len(lakes_filtered)}")
                
                if len(lakes_filtered) == 0:
                    empty_gdf = gpd.GeoDataFrame(
                        columns=['lake_id', 'depth_m', 'area_m2', 'lake_type', 'geometry'], 
                        crs=crs
                    )
                    shapefile_path = self.workspace_dir / "lakes.shp"
                    empty_gdf.to_file(shapefile_path)
                    return {
                        'success': True,
                        'dem_file': str(dem_file),
                        'lake_shapefile': str(shapefile_path),
                        'lake_count': 0,
                        'total_area_ha': 0
                    }
                
                # Add lake IDs
                lakes_filtered['lake_id'] = range(1, len(lakes_filtered) + 1)
                lakes_filtered['lake_type'] = 'undetermined'
                
                # Save shapefile using secure operations
                shapefile_path = self.workspace_dir / "lakes.shp"
                lakes_filtered = lakes_filtered[['lake_id', 'depth_m', 'area_m2', 'lake_type', 'geometry']]
                try:
                    shapefile_path = self.file_ops.safe_write_shapefile(lakes_filtered, shapefile_path)
                except (PathResolutionError, FileAccessError, FileValidationError) as e:
                    return {'success': False, 'error': f'Failed to write lakes shapefile: {e}'}
                
                total_area_ha = lakes_filtered['area_m2'].sum() / 10000
                
                return {
                    'success': True,
                    'dem_file': str(dem_file),
                    'lake_shapefile': str(shapefile_path),
                    'lake_count': len(lakes_filtered),
                    'total_area_ha': total_area_ha
                }
                
        except ImportError as e:
            return {
                'success': False, 
                'error': f'Missing library: {e}. Install: pip install whitebox rasterio geopandas'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _classify_detected_lakes(self, 
                               lakes_shapefile: str,
                               streams_shapefile: Path = None,
                               connected_threshold: float = None,
                               non_connected_threshold: float = None) -> Dict:
        """Classify detected lakes using the LakeClassifier"""
        
        try:
            # Create watershed results structure for compatibility
            watershed_results = {'files_created': []}
            if streams_shapefile and streams_shapefile.exists():
                watershed_results['files_created'].append(str(streams_shapefile))
            
            # Use LakeClassifier
            classification_results = self.classifier.classify_lakes_from_watershed_results(
                watershed_results,
                Path(lakes_shapefile),
                connected_threshold,
                non_connected_threshold
            )
            
            return classification_results
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_comprehensive_summary(self, results: Dict) -> str:
        """Create comprehensive summary report"""
        
        summary_file = self.workspace_dir / "LAKE_DETECTION_SUMMARY.md"
        
        content = f"""# Lake Detection & Classification Summary

## Workflow Parameters
- **Study Area**: {results['bbox']}
- **Minimum Lake Area**: {results['parameters']['min_lake_area_m2']} m²
- **Connected Lake Threshold**: {results['parameters']['connected_threshold_km2']} km²
- **Non-connected Lake Threshold**: {results['parameters']['non_connected_threshold_km2']} km²
- **Lake Depth Threshold**: {results['parameters']['lake_depth_threshold']} m

## Detection Results
- **Total Lakes Detected**: {results['raw_lake_count']}
- **Total Lake Area**: {results['raw_total_area_ha']:.1f} hectares
- **Connected Lakes**: {results['connected_count']}
- **Non-connected Lakes**: {results['non_connected_count']}

## Files Generated
- **DEM**: {Path(results['dem_file']).name}
- **Lakes Shapefile**: {Path(results['lake_shapefile']).name}
- **Connected Lakes**: {Path(results['connected_lakes_file']).name if results['connected_lakes_file'] else 'None'}
- **Non-connected Lakes**: {Path(results['non_connected_lakes_file']).name if results['non_connected_lakes_file'] else 'None'}
- **All Lakes**: {Path(results['all_lakes_file']).name if results['all_lakes_file'] else 'None'}

## File Locations
All files are located in: `{self.workspace_dir}`

## Next Steps
1. Review lake shapefiles in GIS software
2. Validate lake classification results
3. Use classified lakes in watershed modeling
4. Apply lake filtering if needed for specific applications

---
*Generated by Comprehensive Lake Detector*
"""
        
        with open(summary_file, 'w') as f:
            f.write(content)
            
        return str(summary_file)
    
    def _filter_lakes_to_watershed(self, lakes_shapefile: Path, watershed_boundary: Path) -> Dict:
        """
        Filter lakes to only include those within the watershed boundary
        
        Parameters:
        -----------
        lakes_shapefile : Path
            Path to lakes shapefile
        watershed_boundary : Path
            Path to watershed boundary shapefile
            
        Returns:
        --------
        Dict with filtering results
        """
        
        try:
            # Load data
            lakes_gdf = gpd.read_file(lakes_shapefile)
            watershed_gdf = gpd.read_file(watershed_boundary)
            
            print(f"   Total lakes before filtering: {len(lakes_gdf)}")
            
            # Ensure consistent CRS
            if lakes_gdf.crs != watershed_gdf.crs:
                lakes_gdf = lakes_gdf.to_crs(watershed_gdf.crs)
                print(f"   Reprojected lakes to watershed CRS: {watershed_gdf.crs}")
            
            # Get watershed geometry
            watershed_geom = watershed_gdf.geometry.iloc[0]
            
            # Filter lakes to those within or intersecting watershed
            # Use intersects() to include lakes that cross the boundary
            lakes_within_watershed = lakes_gdf[lakes_gdf.geometry.intersects(watershed_geom)]
            
            print(f"   Lakes after watershed filtering: {len(lakes_within_watershed)}")
            print(f"   Lakes removed (outside watershed): {len(lakes_gdf) - len(lakes_within_watershed)}")
            
            # Save filtered lakes
            filtered_lakes_file = self.workspace_dir / "lakes_watershed_filtered.shp"
            lakes_within_watershed.to_file(filtered_lakes_file)
            
            return {
                'success': True,
                'filtered_lakes_file': filtered_lakes_file,
                'filtered_count': len(lakes_within_watershed),
                'removed_count': len(lakes_gdf) - len(lakes_within_watershed),
                'original_count': len(lakes_gdf)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Watershed filtering failed: {str(e)}"
            }
    
    def filter_existing_lakes(self, 
                            lakes_shapefile: str,
                            connected_threshold_km2: float = None,
                            non_connected_threshold_km2: float = None) -> Dict:
        """
        Filter existing lakes using LakeFilter
        
        Parameters:
        -----------
        lakes_shapefile : str
            Path to lakes shapefile to filter
        connected_threshold_km2 : float, optional
            Minimum area for connected lakes
        non_connected_threshold_km2 : float, optional
            Minimum area for non-connected lakes
            
        Returns:
        --------
        Dict with filtering results
        """
        
        # All thresholds must be provided - no defaults
        if connected_threshold_km2 is None:
            raise ValueError("connected_threshold_km2 must be specified")
        if non_connected_threshold_km2 is None:
            raise ValueError("non_connected_threshold_km2 must be specified")
            
        # Use LakeClassifier to classify first, then apply thresholds
        return self.classifier.classify_from_existing_lakes(
            Path(lakes_shapefile),
            connected_threshold=connected_threshold_km2,
            non_connected_threshold=non_connected_threshold_km2
        )


def test_comprehensive_lake_detection():
    """Test the comprehensive lake detection processor"""
    
    print("Testing Comprehensive Lake Detection Processor...")
    print("=" * 50)
    
    # Initialize detector
    detector = ComprehensiveLakeDetector("test_lakes")
    
    # Test parameters
    test_bbox = [-79.8, 43.5, -79.2, 44.0]  # Toronto area
    
    # Run complete workflow
    results = detector.detect_and_classify_lakes(
        bbox=test_bbox,
        min_lake_area_m2=1000,  # 1000 m² for testing
        connected_threshold_km2=0.005,
        non_connected_threshold_km2=0.01
    )
    
    if results['success']:
        print("\nComprehensive lake detection test successful!")
        print(f"   Lakes detected: {results['raw_lake_count']}")
        print(f"   Total area: {results['raw_total_area_ha']:.1f} hectares")
        print(f"   Files created: {len(results['files_created'])}")
    else:
        print(f"\n[FAILED] Test failed: {results.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    test_comprehensive_lake_detection()