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
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize sub-components
        self.spatial_client = SpatialLayersClient()
        self.classifier = LakeClassifier(self.workspace_dir)
        self.filter = LakeFilter(self.workspace_dir)
        
        # Default parameters
        self.default_min_lake_area_m2 = 500  # 500 m² minimum
        self.default_connected_threshold_km2 = 0.01  # 1 hectare
        self.default_non_connected_threshold_km2 = 0.1  # 10 hectares
        self.default_lake_depth_threshold = 0.3  # 0.3m minimum depth
        
    def detect_and_classify_lakes(self, 
                                bbox: List[float],
                                min_lake_area_m2: float = None,
                                connected_threshold_km2: float = None,
                                non_connected_threshold_km2: float = None,
                                lake_depth_threshold: float = None,
                                streams_shapefile: Path = None) -> Dict:
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
            
        Returns:
        --------
        Dict with complete lake detection and classification results
        """
        
        print("=" * 60)
        print("COMPREHENSIVE LAKE DETECTION & CLASSIFICATION")
        print("=" * 60)
        print(f"Area: {bbox}")
        print(f"Working directory: {self.workspace_dir}")
        
        # Use defaults if not provided
        min_lake_area_m2 = min_lake_area_m2 or self.default_min_lake_area_m2
        connected_threshold_km2 = connected_threshold_km2 or self.default_connected_threshold_km2
        non_connected_threshold_km2 = non_connected_threshold_km2 or self.default_non_connected_threshold_km2
        lake_depth_threshold = lake_depth_threshold or self.default_lake_depth_threshold
        
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
    
    def _process_dem_to_lakes(self, dem_file: Path, min_area_m2: float, depth_threshold: float) -> Dict:
        """Process DEM to create lake shapefile"""
        
        try:
            import whitebox
            import rasterio
            from rasterio.features import shapes
            from shapely.geometry import shape
            
            # Initialize Whitebox
            wbt = whitebox.WhiteboxTools()
            wbt.set_working_dir(str(self.workspace_dir.absolute()))
            wbt.set_verbose_mode(False)
            
            # File names
            filled_name = "filled_dem.tif"
            depression_name = "depressions.tif"
            
            print(f"  Processing DEM...")
            
            # Fill depressions
            wbt.fill_depressions_wang_and_liu(
                dem=str(dem_file.name),
                output=filled_name,
                fix_flats=True
            )
            
            if not (self.workspace_dir / filled_name).exists():
                return {'success': False, 'error': 'Depression filling failed'}
            
            # Calculate depression depths
            wbt.subtract(
                input1=filled_name,
                input2=str(dem_file.name),
                output=depression_name
            )
            
            depression_file = self.workspace_dir / depression_name
            if not depression_file.exists():
                return {'success': False, 'error': 'Depression calculation failed'}
            
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
                    empty_gdf.to_file(shapefile_path)
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
                
                # Create GeoDataFrame
                lakes_gdf = gpd.GeoDataFrame({
                    'geometry': lake_polygons,
                    'depth_m': [depth_threshold] * len(lake_polygons)
                }, crs=crs)
                
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
                
                # Save shapefile
                shapefile_path = self.workspace_dir / "lakes.shp"
                lakes_filtered = lakes_filtered[['lake_id', 'depth_m', 'area_m2', 'lake_type', 'geometry']]
                lakes_filtered.to_file(shapefile_path)
                
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
            # Create mock watershed results for compatibility
            mock_watershed_results = {'files_created': []}
            if streams_shapefile and streams_shapefile.exists():
                mock_watershed_results['files_created'].append(str(streams_shapefile))
            
            # Use LakeClassifier
            classification_results = self.classifier.classify_lakes_from_watershed_results(
                mock_watershed_results,
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
        
        # Use LakeClassifier to classify first, then apply thresholds
        return self.classifier.classify_from_existing_lakes(
            Path(lakes_shapefile),
            connected_threshold=connected_threshold_km2 or self.default_connected_threshold_km2,
            non_connected_threshold=non_connected_threshold_km2 or self.default_non_connected_threshold_km2
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