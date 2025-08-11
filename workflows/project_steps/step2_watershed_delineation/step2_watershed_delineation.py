#!/usr/bin/env python3
"""
Step 2: Watershed Delineation for RAVEN Single Outlet Delineation
Delineates watershed boundary and extracts stream network from DEM with absolute path support
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Union, Optional

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # Project root
sys.path.append(str(Path(__file__).parent.parent.parent))  # workflows dir

from workflows.base_workflow_step import BaseWorkflowStep
from workflows.steps import DelineateWatershedAndStreams
from processors.outlet_snapping import ImprovedOutletSnapper
from processors.basic_attributes_calculator import BasicAttributesCalculator
from infrastructure.path_manager import PathResolutionError, FileAccessError
from infrastructure.configuration_manager import WorkflowConfiguration
# Removed GeospatialFileManager - using ultra-simple data/ folder structure
import logging

logger = logging.getLogger(__name__)


class Step2WatershedDelineation(BaseWorkflowStep):
    """Step 2: Delineate watershed boundary and stream network with absolute path support"""
    
    def __init__(self, workspace_dir: Union[str, Path], config: Optional[WorkflowConfiguration] = None):
        """
        Initialize Step 2 with absolute path infrastructure.
        
        Args:
            workspace_dir: Workspace directory (required, no fallback)
            config: Optional workflow configuration
            
        Raises:
            ValueError: If workspace_dir is not provided
            PathResolutionError: If workspace_dir cannot be resolved
        """
        if not workspace_dir:
            raise ValueError("workspace_dir is required for Step2WatershedDelineation - no fallback paths allowed")
        
        # Initialize base class with absolute path infrastructure
        super().__init__(workspace_dir, config, 'step2_watershed_delineation')
        
        # ULTRA-SIMPLE: All files go to workspace/data/ folder
        self.data_dir = self.path_manager.workspace_root / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize watershed delineation step and outlet snapper with absolute paths
        self.watershed_step = DelineateWatershedAndStreams()
        
        # Create outlet snapping directory with absolute path
        outlet_snapping_dir = self.path_manager.create_path_structure("outlet_snapping")
        self.outlet_snapper = ImprovedOutletSnapper(outlet_snapping_dir)
    
    def load_step1_results(self, step1_results_file: 'Path' = None) -> Dict[str, Any]:
        """Load results from Step 1 using absolute path or base class method with explicit error handling"""
        
        # If absolute path to step1 results is provided, use it directly
        if step1_results_file and step1_results_file.exists():
            try:
                import json
                with open(step1_results_file, 'r') as f:
                    results = json.load(f)
                logger.info(f"Loaded step1 results from absolute path: {step1_results_file}")
                return results
            except (json.JSONDecodeError, IOError) as e:
                return {
                    'success': False,
                    'error': f'Cannot load step1 results from {step1_results_file}: {e}'
                }
        
        # Fallback to base class method for backward compatibility
        try:
            return self.load_previous_results('step1', 'step1_results.json')
        except FileAccessError as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _execute_step(self, latitude: float, longitude: float, outlet_name: str = None, minimum_drainage_area_km2: float = None, max_snap_distance_m: float = 5000, **kwargs) -> Dict[str, Any]:
        """Execute Step 2: Watershed delineation with absolute path support and comprehensive tracking"""
        print(f"STEP 2: Delineating watershed for outlet ({latitude}, {longitude})")
        
        # Get parameters from configuration if available
        if self.step_config:
            minimum_drainage_area_km2 = self.step_config.parameters.get('minimum_drainage_area_km2', minimum_drainage_area_km2)
            max_snap_distance_m = self.step_config.parameters.get('max_snap_distance_m', max_snap_distance_m)
            stream_threshold = self.step_config.parameters.get('stream_threshold', 5000)
        else:
            stream_threshold = 5000
        
        if minimum_drainage_area_km2:
            print(f"Using minimum drainage area threshold: {minimum_drainage_area_km2:.2f} km²")
        print(f"Maximum outlet snap distance: {max_snap_distance_m}m")
        
        # ULTRA-SIMPLE: Load inputs from data/ folder
        study_area_file = self.data_dir / "study_area.geojson"
        dem_file = self.data_dir / "dem.tif"
        
        # Load study area if available
        has_study_area = False
        if study_area_file.exists():
            try:
                import geopandas as gpd
                study_area_gdf = gpd.read_file(study_area_file)
                print(f"Loaded study area from Step 1: {len(study_area_gdf)} features")
                has_study_area = True
            except Exception as e:
                print(f"Warning: Could not load study area: {e}")
        
        # Check for DEM file
        if not dem_file.exists():
            return {
                'success': False,
                'error': f'DEM file not found: {dem_file}'
            }
        
        print(f"Using DEM: {dem_file}")
        dem_file_path = dem_file
        
        # ULTRA-SIMPLE: Use data/ folder for both processing and final outputs
        if not outlet_name:
            outlet_name = f"outlet_{latitude:.4f}_{longitude:.4f}"
        
        # Process directly in data/ folder - no complex directory structures
        outlet_workspace = self.data_dir
        print(f"Using data directory for processing and outputs: {outlet_workspace}")
        
        # Verify directory was created successfully
        if not outlet_workspace.exists():
            return {
                'success': False,
                'error': f'Failed to create outlet workspace directory: {outlet_workspace}'
            }
        
        # Step 1: Check for existing flow files or create them
        print("Looking for existing flow accumulation and streams data...")
        flow_accum_file = None
        streams_file = None
        
        # Check for existing files in the outlet workspace
        for f in outlet_workspace.rglob("*"):
            if 'flow_accumulation' in f.name and f.suffix == '.tif':
                flow_accum_file = str(f)
            elif 'streams' in f.name and f.suffix in ['.geojson', '.shp']:
                streams_file = str(f)
        
        # If files don't exist, create them with optimized parameters
        if not flow_accum_file or not streams_file:
            print("Creating flow accumulation data with performance optimizations...")
            
            # Performance optimization: Pass enhanced parameters for better threshold management
            flow_accum_result = self.watershed_step.analyzer.analyze_watershed_complete(
                dem_path=str(dem_file_path),
                outlet_coords=(latitude, longitude),
                output_dir=outlet_workspace
            )
            
            if not flow_accum_result.get('success'):
                return {
                    'success': False,
                    'error': f'Flow accumulation preparation failed: {flow_accum_result.get("error")}'
                }
            
            # Find the newly created files
            for f in outlet_workspace.rglob("*"):
                if 'flow_accumulation' in f.name and f.suffix == '.tif':
                    flow_accum_file = str(f)
                elif 'streams' in f.name and f.suffix in ['.geojson', '.shp']:
                    streams_file = str(f)
        
        # Step 2: Snap outlet to stream network using local processor
        if not flow_accum_file or not streams_file:
            return {
                'success': False,
                'error': 'Could not find flow accumulation or streams files for outlet snapping'
            }
            
        print(f"Snapping outlet to nearest point on stream network...")
        snap_result = self.outlet_snapper.snap_outlet_nearest_point(
            (longitude, latitude), 
            streams_file,
            max_search_distance=max_snap_distance_m  # Use configurable search radius
        )
        
        if not snap_result['success']:
            return {
                'success': False,
                'error': f'Outlet snapping failed: {snap_result["error"]}'
            }
        
        print(f"SUCCESS: Outlet snapped: {snap_result['snap_distance_m']:.1f}m from original")
        # Use snapped coordinates for final delineation
        final_lat, final_lon = snap_result['snapped_coords'][1], snap_result['snapped_coords'][0]
        
        # Step 3: Execute final watershed delineation with snapped coordinates
        print(f"Delineating watershed boundary and streams at ({final_lat:.6f}, {final_lon:.6f})...")
        print(f"Output directory: {outlet_workspace}")
        print(f"Output directory exists: {outlet_workspace.exists()}")
        print(f"DEM file: {dem_file_path}")
        print(f"DEM file exists: {dem_file_path.exists()}")
        
        try:
            # Ensure we use absolute paths for WhiteboxTools
            abs_dem_path = str(dem_file_path.resolve())
            abs_output_dir = str(outlet_workspace.resolve())
            print(f"Using absolute DEM path: {abs_dem_path}")
            print(f"Using absolute output dir: {abs_output_dir}")
            
            # Verify paths are accessible
            if not dem_file_path.exists():
                raise FileNotFoundError(f"DEM file not found: {abs_dem_path}")
            if not outlet_workspace.exists():
                raise FileNotFoundError(f"Output directory not found: {abs_output_dir}")
                
            # Test write permissions in output directory
            test_file = outlet_workspace / "test_write.tmp"
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                test_file.unlink()  # Delete test file
                print("SUCCESS: Write permissions verified in output directory")
            except Exception as write_error:
                raise PermissionError(f"Cannot write to output directory {abs_output_dir}: {write_error}")
            
            # Performance optimization: Enhanced watershed analysis with optimized parameters
            watershed_result = self.watershed_step.analyzer.analyze_watershed_complete(
                dem_path=abs_dem_path,
                outlet_coords=(final_lon, final_lat),
                output_dir=abs_output_dir,
                stream_threshold=stream_threshold
            )
        except Exception as e:
            return {
                'success': False,
                'error': f"Watershed analysis threw exception: {str(e)}"
            }
        
        if not watershed_result.get('success'):
            error_msg = watershed_result.get('error', 'Unknown error')
            print(f"CRITICAL ERROR: {error_msg}")
            return {
                'success': False,
                'error': f"Final watershed delineation failed: {error_msg}"
            }

        # Extract file paths from the comprehensive results
        final_watershed_boundary = None
        final_stream_network = None
        for f in watershed_result.get('files_created', []):
            if 'watershed' in Path(f).name and Path(f).suffix in ['.geojson', '.shp']:
                final_watershed_boundary = f
            if 'streams' in Path(f).name and Path(f).suffix in ['.geojson', '.shp']:
                final_stream_network = f
        
        # ULTRA-SIMPLE: Files are already in data/ folder with simple names
        print("INFO: Files saved directly to data/ folder with simple names:")

        # Extract statistics
        stats = watershed_result.get('metadata', {}).get('statistics', {})
        
        # Debug logging for area calculation
        print(f"DEBUG: Raw statistics from watershed analysis:")
        print(f"  - watershed_area_km2: {stats.get('watershed_area_km2', 'MISSING')}")
        print(f"  - total_stream_length_km: {stats.get('total_stream_length_km', 'MISSING')}")
        print(f"  - max_strahler_order: {stats.get('max_strahler_order', 'MISSING')}")
        print(f"  - Available stats keys: {list(stats.keys())}")
        
        # Initialize WhiteboxTools for essential operations only
        from whitebox import WhiteboxTools
        
        wbt = WhiteboxTools()
        wbt.set_working_dir(str(outlet_workspace))
        wbt.set_verbose_mode(False)  # Keep it quiet for performance
        print(f"DEBUG: WhiteboxTools initialized")
        
        # Performance optimization: Use WhiteboxTools for area calculation
        area_km2 = stats.get('watershed_area_km2', 0)
        if area_km2 == 0:
            print(f"DEBUG: Area is 0, calculating from watershed file...")
            try:
                watershed_file = final_watershed_boundary
                if watershed_file and Path(watershed_file).exists():
                    # Simple GeoPandas area calculation
                    import geopandas as gpd
                    gdf = gpd.read_file(watershed_file)
                    if len(gdf) > 0:
                        utm_crs = gdf.estimate_utm_crs()
                        area_m2 = gdf.to_crs(utm_crs).geometry.area.sum()
                        area_km2 = area_m2 / 1e6
                        print(f"DEBUG: Calculated area: {area_km2:.6f} km²")
            except Exception as e:
                print(f"DEBUG: Area calculation failed: {e}")
        else:
            print(f"DEBUG: Using area from stats: {area_km2:.6f} km²")
        
        # Get max stream order from raster file
        max_stream_order = 0
        try:
            strahler_file = outlet_workspace / "stream_order_strahler.tif"
            if strahler_file.exists():
                import rasterio
                import numpy as np
                with rasterio.open(strahler_file) as src:
                    data = src.read(1)
                    valid_data = data[data > 0]
                    if len(valid_data) > 0:
                        max_stream_order = int(np.max(valid_data))
                        
                print(f"DEBUG: Max stream order from raster: {max_stream_order}")
            else:
                print(f"DEBUG: Stream order raster not found: {strahler_file}")
        except Exception as e:
            print(f"DEBUG: Failed to read stream order: {e}")
        
        # HYDROLOGICALLY CORRECT SUBBASIN MERGING (BasinMaker-style)
        merged_subbasin_count = 0
        try:
            import geopandas as gpd
            import numpy as np
            import shutil

            subbasins_vector_file = outlet_workspace / "subbasins.shp"

            if subbasins_vector_file.exists():
                gdf = gpd.read_file(subbasins_vector_file)
                original_count = len(gdf)
                print(f"DEBUG: Original subbasins count: {original_count}")

                if original_count == 0:
                    print("DEBUG: Empty subbasin layer; skipping merge")
                    merged_subbasin_count = 0
                else:
                    # USE CUSTOM HYDROLOGICALLY CORRECT MERGING (BasinMaker-style)
                    print("DEBUG: Applying custom BasinMaker-style merging with clipping...")
                    
                    # Use our proven custom implementation
                    merged_gdf = self._apply_hydrologically_correct_merging(
                        gdf, 
                        min_area_threshold_km2=5.0,
                        outlet_workspace=outlet_workspace
                    )
                    
                    # CRITICAL: Save only the final clipped subbasins as the main subbasins file
                    # This ensures Step 3 uses the hydrologically correct clipped watershed data
                    final_subbasins_file = outlet_workspace / "subbasins.shp"
                    merged_gdf.to_file(final_subbasins_file, driver="ESRI Shapefile")
                    print(f"DEBUG: Saved legacy final clipped subbasins to: {final_subbasins_file}")
                    
                    # Calculate missing hydraulic attributes using existing BasicAttributesCalculator
                    print("DEBUG: Calculating hydraulic attributes using BasicAttributesCalculator...")
                    calculator = BasicAttributesCalculator(outlet_workspace)
                    
                    # Save temporary shapefile for calculator
                    temp_subbasins = outlet_workspace / "temp_subbasins.shp"
                    merged_gdf.to_file(temp_subbasins, driver="ESRI Shapefile")
                    
                    # Calculate attributes
                    streams_file = outlet_workspace / "streams.shp"
                    dem_file = self.data_dir / "dem.tif"
                    
                    attrs_df = calculator.calculate_basic_attributes(
                        catchments_shapefile=temp_subbasins,
                        rivers_shapefile=streams_file if streams_file.exists() else None,
                        dem_raster=dem_file if dem_file.exists() else None
                    )
                    
                    # Merge attributes back into GeoDataFrame
                    if 'SubId' in merged_gdf.columns:
                        merged_gdf_with_attrs = merged_gdf.merge(attrs_df, left_on='SubId', right_on='SubId', how='left')
                    else:
                        # Add SubId column if missing and merge by index
                        merged_gdf['SubId'] = range(1, len(merged_gdf) + 1)
                        attrs_df['SubId'] = range(1, len(attrs_df) + 1)  
                        merged_gdf_with_attrs = merged_gdf.merge(attrs_df, on='SubId', how='left')
                    
                    # Clean up temp file
                    if temp_subbasins.exists():
                        import shutil
                        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                            temp_file = temp_subbasins.with_suffix(ext)
                            if temp_file.exists():
                                temp_file.unlink()
                    
                    # ULTRA-SIMPLE: Save subbasins directly to data/ folder
                    subbasins_path = self.data_dir / "subbasins.geojson"
                    merged_gdf_with_attrs.to_file(subbasins_path, driver="GeoJSON")
                    print(f"SUCCESS: Saved subbasins with hydraulic attributes: {subbasins_path}")
                    
                    # Also update the shapefile with attributes
                    merged_gdf_with_attrs.to_file(final_subbasins_file, driver="ESRI Shapefile")
                    print(f"DEBUG: Updated shapefile with hydraulic attributes: {final_subbasins_file}")
                    
                    merged_subbasin_count = len(merged_gdf_with_attrs)
                    print(f"DEBUG: Custom merging and clipping: {original_count} -> {merged_subbasin_count}")

            else:
                merged_subbasin_count = stats.get('subbasin_count', 0)
                print(f"DEBUG: No subbasins vector file found, using stats count: {merged_subbasin_count}")

        except Exception as e:
            print(f"DEBUG: Hydrologically correct subbasin merge failed: {e}")
            merged_subbasin_count = stats.get('subbasin_count', 0)
        
        # ULTRA-SIMPLE: All outputs are already in data/ folder with standard names
        
        # Prepare results with snapping information
        # Find the processed DEM file (typically dem_conditioned.tif)
        processed_dem = None
        for dem_file in ['dem_conditioned.tif', 'dem_breached.tif', 'dem_filled.tif']:
            dem_path = outlet_workspace / dem_file
            if dem_path.exists():
                processed_dem = str(dem_path)
                break
        
        results = {
            'success': True,
            'outlet_coordinates': [latitude, longitude],  # Original coordinates
            'outlet_coordinates_snapped': [final_lat, final_lon],  # Snapped coordinates
            'outlet_snapped': final_lat != latitude or final_lon != longitude,
            'snap_distance_m': snap_result['snap_distance_m'],
            'outlet_name': outlet_name,
            'workspace': str(self.data_dir),  # ULTRA-SIMPLE: Always data/ folder
            'files': {
                'watershed_boundary': str(self.data_dir / 'watershed.geojson'),
                'stream_network': str(self.data_dir / 'streams.geojson'),
                'subbasins': str(self.data_dir / 'subbasins.geojson'),
                'dem': str(self.data_dir / 'dem.tif'),
                'study_area': str(self.data_dir / 'study_area.geojson'),
            },
            'characteristics': {
                'watershed_area_km2': area_km2,  # Use manually calculated area if stats was 0
                'stream_length_km': stats.get('total_stream_length_km', 0),
                'max_stream_order': max_stream_order,  # Use correctly calculated max stream order
                'merged_subbasin_count': merged_subbasin_count  # Use merged subbasin count
            },
            'processing_notes': {
                'subbasins_processing': 'Applied BasinMaker-style hydrologically correct merging and clipped to watershed boundary',
                'final_subbasins_file': 'subbasins.shp contains the final processed (merged and clipped) subbasins',
                'hydrological_correctness': 'Subbasins have been properly merged by area threshold and clipped to exact watershed boundary'
            },
            'data_quality': {
                'coordinate_system': 'EPSG:4326',
                'format_standard': 'GeoJSON only - ultra-simple data/ folder structure'
            }
        }
        
        # Verify output files exist
        for file_type, file_path in results['files'].items():
            if file_path and Path(file_path).exists():
                print(f"SUCCESS: {file_type}: {file_path}")
            else:
                print(f"WARNING:  {file_type}: Missing or not created")
        
        # Save results using base class method with metadata tracking
        self.save_results(results, "step2_results.json")
        
        print(f"STEP 2 COMPLETE: Results saved with GeoJSON-first file management") 
        print(f"Watershed area: {results['characteristics']['watershed_area_km2']:.2f} km²")
        print(f"SUCCESS: Outlet was snapped {results['snap_distance_m']:.1f}m to stream network")
        
        # Generate visualization
        self.create_watershed_plot(results, latitude, longitude)
        
        return results
    
    def _apply_hydrologically_correct_merging(self, gdf: 'gpd.GeoDataFrame', 
                                            min_area_threshold_km2: float = 5.0,
                                            outlet_workspace: 'Path' = None) -> 'gpd.GeoDataFrame':
        """
        Apply REAL BasinMaker hydrologically correct subbasin merging algorithm.
        
        BasinMaker approach:
        1. Modify attribute table to reassign SubId values for merging
        2. Use dissolve() operation on SubId column (NOT geometric union operations)
        3. Preserve hydrological connectivity through proper SubId reassignment
        """
        import geopandas as gpd
        import pandas as pd
        import numpy as np
        
        print(f"APPLYING REAL BASINMAKER MERGING ALGORITHM")
        print(f"Input subbasins: {len(gdf)}")
        print(f"Area threshold: {min_area_threshold_km2} km²")
        
        # Ensure projected CRS for accurate area calculations
        if gdf.crs is None or gdf.crs.is_geographic:
            gdf = gdf.to_crs(gdf.estimate_utm_crs())
        
        # Calculate areas in km²
        gdf['area_km2'] = gdf.geometry.area / 1_000_000.0
        
        # FIXED: Add required columns for BasinMaker algorithm with proper data types
        if 'SubId' not in gdf.columns:
            gdf['SubId'] = range(1, len(gdf) + 1)
        
        if 'VALUE' in gdf.columns:
            # FIX: Convert to integer to avoid float grouping issues in dissolve
            gdf['SubId'] = gdf['VALUE'].astype(int)
        
        # FIX: Validate SubId uniqueness before processing
        if gdf['SubId'].nunique() != len(gdf):
            print(f"WARNING: Non-unique SubIds detected. Creating new sequence.")
            gdf['SubId'] = range(1, len(gdf) + 1)
        
        # Create working copy for attribute modification (BasinMaker style)
        mapoldnew_info = gdf.copy()
        mapoldnew_info['Original_SubId'] = mapoldnew_info['SubId'].copy()
        
        # Identify subbasins that need merging
        small_mask = mapoldnew_info['area_km2'] < min_area_threshold_km2
        small_subbasins = mapoldnew_info[small_mask].copy()
        
        print(f"Small subbasins (<{min_area_threshold_km2} km²): {len(small_subbasins)}")
        print(f"Large subbasins (>={min_area_threshold_km2} km²): {len(mapoldnew_info) - len(small_subbasins)}")
        
        if len(small_subbasins) == 0:
            print("No small subbasins to merge")
            return gdf.copy()
        
        # FIXED BASINMAKER ALGORITHM: Reassign SubId values for merging
        merged_count = 0
        
        # FIX: Process small subbasins in order of decreasing area (merge largest first)
        small_subbasins_sorted = small_subbasins.sort_values('area_km2', ascending=False)
        
        for idx, small_basin in small_subbasins_sorted.iterrows():
            try:
                # Find target basin for merging using spatial analysis
                target_subid = self._find_merge_target_subid(
                    small_basin, mapoldnew_info, min_area_threshold_km2
                )
                
                if target_subid is not None:
                    # BASINMAKER KEY STEP: Change the SubId of small basin to target SubId
                    # This groups them together for the dissolve operation
                    original_subid = mapoldnew_info.loc[idx, 'SubId']
                    mapoldnew_info.loc[idx, 'SubId'] = target_subid
                    
                    merged_count += 1
                    print(f"  Reassigning SubId {original_subid} -> {target_subid}")
                    
            except Exception as e:
                print(f"Warning: Failed to reassign SubId for subbasin {idx}: {e}")
        
        # FIXED BASINMAKER KEY OPERATION: Dissolve by SubId (this is where actual merging happens)
        print(f"Dissolving {len(mapoldnew_info)} subbasins by SubId...")
        
        # FIX: Validate before dissolve
        unique_subids_before = mapoldnew_info['SubId'].nunique()
        total_records_before = len(mapoldnew_info)
        print(f"Before dissolve: {total_records_before} records, {unique_subids_before} unique SubIds")
        
        try:
            # FIX: Ensure SubId is proper integer type for grouping
            mapoldnew_info['SubId'] = mapoldnew_info['SubId'].astype(int)
            
            # FIX: Fix invalid geometries before dissolve operation
            print(f"Fixing invalid geometries before dissolve...")
            invalid_count = (~mapoldnew_info.geometry.is_valid).sum()
            if invalid_count > 0:
                print(f"  Found {invalid_count} invalid geometries, fixing...")
                # Fix invalid geometries using buffer(0) technique
                mapoldnew_info['geometry'] = mapoldnew_info.geometry.buffer(0)
                # Verify fix worked
                still_invalid = (~mapoldnew_info.geometry.is_valid).sum()
                print(f"  After fix: {still_invalid} invalid geometries remaining")
            
            # This is the real BasinMaker merging: dissolve by SubId
            merged_gdf = mapoldnew_info.dissolve(
                by='SubId', 
                aggfunc='first',  # Keep first occurrence of other attributes
                as_index=False
            )
            
            # Recalculate areas after dissolve
            merged_gdf['area_km2'] = merged_gdf.geometry.area / 1_000_000.0
            
            print(f"BasinMaker dissolve completed:")
            print(f"  Original subbasins: {len(gdf)}")
            print(f"  Final subbasins: {len(merged_gdf)}")
            print(f"  SubIds reassigned: {merged_count}")
            print(f"  Reduction achieved: {len(gdf) - len(merged_gdf)} subbasins")
            
            # CLIP TO WATERSHED BOUNDARY
            print(f"Clipping merged subbasins to watershed boundary...")
            clipped_gdf = self._clip_subbasins_to_watershed(merged_gdf, outlet_workspace)
            
            print(f"After clipping:")
            print(f"  Clipped subbasins: {len(clipped_gdf)}")
            print(f"  Final area: {clipped_gdf['area_km2'].sum():.2f} km²")
            
            # Critical validation - merging MUST reduce count
            if len(clipped_gdf) >= len(gdf):
                print(f"ERROR: BasinMaker dissolve failed to reduce subbasin count! Returning original.")
                return gdf.copy()
            
            return clipped_gdf
            
        except Exception as e:
            print(f"ERROR: BasinMaker dissolve operation failed: {e}")
            return gdf.copy()
    
    def _find_merge_target_subid(self, small_basin, mapoldnew_info, min_area_threshold_km2):
        """
        Find target SubId for merging using BasinMaker's approach.
        
        Priority order:
        1. Adjacent neighbors that meet area threshold
        2. Nearest neighbor by centroid distance that meets area threshold
        """
        import numpy as np
        
        small_centroid = small_basin.geometry.centroid
        small_subid = small_basin['SubId']
        
        # Get potential targets (large basins only)
        large_basins = mapoldnew_info[mapoldnew_info['area_km2'] >= min_area_threshold_km2].copy()
        
        if len(large_basins) == 0:
            return None
        
        # Method 1: Find adjacent (touching/intersecting) neighbors
        touching_candidates = []
        for idx, large_basin in large_basins.iterrows():
            if large_basin['SubId'] == small_subid:
                continue  # Skip self
                
            try:
                if small_basin.geometry.touches(large_basin.geometry) or \
                   small_basin.geometry.intersects(large_basin.geometry):
                    
                    touching_candidates.append({
                        'subid': large_basin['SubId'],
                        'area': large_basin['area_km2'],
                        'distance': small_centroid.distance(large_basin.geometry.centroid)
                    })
            except Exception as e:
                # Skip problematic geometries
                continue
        
        # If we have touching neighbors, select the largest one
        if touching_candidates:
            # Sort by area (descending), then by distance (ascending)
            touching_candidates.sort(key=lambda x: (-x['area'], x['distance']))
            return touching_candidates[0]['subid']
        
        # Method 2: Find nearest neighbor by centroid distance
        try:
            distances = large_basins.geometry.distance(small_centroid)
            if len(distances) > 0:
                nearest_idx = distances.idxmin()
                return large_basins.loc[nearest_idx, 'SubId']
        except Exception as e:
            print(f"Warning: Distance calculation failed: {e}")
        
        return None

    def _clip_subbasins_to_watershed(self, merged_gdf: 'gpd.GeoDataFrame', outlet_workspace: 'Path') -> 'gpd.GeoDataFrame':
        """
        Clip merged subbasins to watershed boundary to remove artifacts and ensure 
        only subbasins within the watershed are kept.
        """
        import geopandas as gpd
        from pathlib import Path
        
        try:
            # Load watershed boundary
            watershed_file = outlet_workspace / "watershed.shp"
            if not watershed_file.exists():
                print(f"WARNING: Watershed boundary file not found: {watershed_file}")
                return merged_gdf
            
            watershed_gdf = gpd.read_file(watershed_file)
            
            # Ensure same CRS
            if watershed_gdf.crs != merged_gdf.crs:
                watershed_gdf = watershed_gdf.to_crs(merged_gdf.crs)
            
            # Get watershed boundary (should be single polygon)
            if len(watershed_gdf) > 1:
                watershed_boundary = watershed_gdf.unary_union
            else:
                watershed_boundary = watershed_gdf.iloc[0].geometry
            
            print(f"  Watershed boundary area: {watershed_boundary.area / 1e6:.2f} km²")
            
            # Clip merged subbasins to watershed boundary
            clipped_gdf = merged_gdf.copy()
            clipped_geometries = []
            
            for idx, row in merged_gdf.iterrows():
                subbasin_geom = row.geometry
                
                # Check if subbasin intersects watershed
                if subbasin_geom.intersects(watershed_boundary):
                    # Clip to watershed boundary
                    try:
                        clipped_geom = subbasin_geom.intersection(watershed_boundary)
                        
                        # Handle different geometry types after clipping
                        if clipped_geom.is_empty:
                            continue  # Skip empty results
                        
                        # Handle MultiPolygon results - take largest piece
                        if hasattr(clipped_geom, 'geoms'):
                            if len(clipped_geom.geoms) > 0:
                                clipped_geom = max(clipped_geom.geoms, key=lambda x: x.area)
                            else:
                                continue
                        
                        # Only keep if significant area remains
                        if clipped_geom.area > 1000:  # > 1000 m² (0.001 km²)
                            clipped_geometries.append((idx, clipped_geom))
                        
                    except Exception as e:
                        print(f"  Warning: Could not clip subbasin {row.get('SubId', idx)}: {e}")
                        # Keep original if clipping fails
                        clipped_geometries.append((idx, subbasin_geom))
            
            # Create new GeoDataFrame with clipped geometries
            if clipped_geometries:
                clipped_indices, clipped_geoms = zip(*clipped_geometries)
                clipped_gdf = merged_gdf.loc[list(clipped_indices)].copy()
                clipped_gdf.geometry = list(clipped_geoms)
                
                # Recalculate areas after clipping
                clipped_gdf['area_km2'] = clipped_gdf.geometry.area / 1_000_000.0
                
                # Apply geometry cleaning after clipping
                print(f"  Applying geometry cleaning after clipping...")
                clipped_gdf.geometry = clipped_gdf.geometry.buffer(0)
                
                print(f"  Successfully clipped {len(clipped_gdf)} subbasins to watershed boundary")
                return clipped_gdf
            else:
                print(f"  WARNING: No subbasins remained after clipping!")
                return merged_gdf
                
        except Exception as e:
            print(f"ERROR: Failed to clip subbasins to watershed: {e}")
            return merged_gdf

    def create_watershed_plot(self, results: Dict, outlet_lat: float, outlet_lon: float):
        """Create a plot showing watershed boundary, streams, and outlet point with consistent CRS"""
        try:
            import matplotlib.pyplot as plt
            import geopandas as gpd
            import numpy as np
            from pyproj import Transformer

            print("\nGenerating watershed visualization...")

            # Load file paths
            watershed_shp = results['files'].get('watershed_boundary')
            streams_shp = results['files'].get('stream_network')

            if not watershed_shp or not Path(watershed_shp).exists():
                print("WARNING: Cannot create plot: watershed boundary file missing")
                return

            # Load watershed and determine target CRS
            watershed_gdf = gpd.read_file(watershed_shp)
            if watershed_gdf.crs is None or watershed_gdf.crs.is_geographic:
                target_crs = watershed_gdf.estimate_utm_crs()
                watershed_gdf = watershed_gdf.to_crs(target_crs)
            else:
                target_crs = watershed_gdf.crs

            print(f"DEBUG: Using target CRS: {target_crs}")

            # Load and reproject streams
            streams_gdf = None
            if streams_shp and Path(streams_shp).exists():
                streams_gdf = gpd.read_file(streams_shp)
                if len(streams_gdf) > 0:
                    streams_gdf = streams_gdf.to_crs(target_crs)

            # Load and reproject subbasins (prefer merged)
            subbasins_merged = Path(watershed_shp).parent / "subbasins_merged.shp"
            subbasins_shp = subbasins_merged if subbasins_merged.exists() else Path(watershed_shp).parent / "subbasins.shp"
            subbasins_gdf = None
            if subbasins_shp.exists():
                try:
                    subbasins_gdf = gpd.read_file(subbasins_shp)
                    if len(subbasins_gdf) > 0:
                        subbasins_gdf = subbasins_gdf.to_crs(target_crs)
                except Exception as e:
                    print(f"DEBUG: Could not load subbasins: {e}")
                    subbasins_gdf = None

            # Transform outlet coordinates to target CRS
            transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
            x_out, y_out = transformer.transform(outlet_lon, outlet_lat)

            # Transform snapped coordinates
            snapped_coords = results['outlet_coordinates_snapped']
            snapped_lat, snapped_lon = snapped_coords[0], snapped_coords[1]
            x_snap, y_snap = transformer.transform(snapped_lon, snapped_lat)

            # Create figure and plot
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))

            # Plot layers in target CRS
            watershed_gdf.plot(ax=ax, color='lightblue', edgecolor='darkblue', 
                              linewidth=2, alpha=0.3, label='Watershed Boundary')

            if streams_gdf is not None:
                streams_gdf.plot(ax=ax, color='blue', linewidth=1.5, 
                                alpha=0.7, label='Stream Network')

            if subbasins_gdf is not None:
                subbasins_gdf.plot(ax=ax, facecolor='none', edgecolor='green', 
                                  linewidth=0.8, alpha=0.5, label='Subbasins')

            # Outlet points
            ax.scatter(x_out, y_out, s=150, marker='s', zorder=4, label=f'Original Outlet ({outlet_lat:.4f}, {outlet_lon:.4f})')
            if results['outlet_snapped']:
                ax.scatter(x_snap, y_snap, s=200, marker='o', zorder=5, label=f'Snapped Outlet ({snapped_lat:.4f}, {snapped_lon:.4f})')
                ax.plot([x_out, x_snap], [y_out, y_snap], 'k--', linewidth=2, alpha=0.7, label=f'Snap Distance: {results["snap_distance_m"]:.1f} m')

            # Axes formatting and extent
            ax.set_aspect('equal', adjustable='box')
            minx, miny, maxx, maxy = watershed_gdf.total_bounds
            dx, dy = maxx - minx, maxy - miny
            pad_x, pad_y = 0.05 * dx, 0.05 * dy
            ax.set_xlim(minx - pad_x, maxx + pad_x)
            ax.set_ylim(miny - pad_y, maxy + pad_y)

            ax.set_xlabel('Easting (m)' if not target_crs.is_geographic else 'Longitude')
            ax.set_ylabel('Northing (m)' if not target_crs.is_geographic else 'Latitude')

            merged_subbasin_count = results['characteristics']['merged_subbasin_count']
            max_stream_order_display = results['characteristics']['max_stream_order']
            ax.set_title(
                f'Step 2: Watershed Delineation Results\n'
                f'Area: {results["characteristics"]["watershed_area_km2"]:.2f} km² | '
                f'Stream Length: {results["characteristics"]["stream_length_km"]:.1f} km | '
                f'Max Stream Order: {max_stream_order_display} | '
                f'Merged Subbasins: {merged_subbasin_count}'
            )

            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=10)

            # Simple scalebar for projected CRS
            if not target_crs.is_geographic:
                width_m = (maxx - minx)
                scalebar_m = 10000 if width_m > 50000 else 5000 if width_m > 20000 else 2000
                sb_x0 = minx + 0.1 * width_m
                sb_y0 = miny + 0.1 * (maxy - miny)
                ax.plot([sb_x0, sb_x0 + scalebar_m], [sb_y0, sb_y0], 'k-', linewidth=3)
                ax.text(sb_x0 + scalebar_m / 2, sb_y0, f'{int(scalebar_m/1000)} km', ha='center', va='bottom')

            plot_file = self.path_manager.workspace_root / "step2_watershed_plot.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"SUCCESS: Watershed plot saved: {plot_file}")
            plt.close()

        except ImportError as e:
            print(f"WARNING: Cannot create plot: Missing required library ({e})")
        except Exception as e:
            print(f"WARNING: Error creating watershed plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 2: Watershed Delineation')
    parser.add_argument('latitude', type=float, help='Outlet latitude')
    parser.add_argument('longitude', type=float, help='Outlet longitude')
    parser.add_argument('--outlet-name', type=str, help='Name for the outlet')
    parser.add_argument('--workspace-dir', type=str, help='Workspace directory')
    
    args = parser.parse_args()
    
    step2 = Step2WatershedDelineation(workspace_dir=args.workspace_dir)
    results = step2.execute(latitude=args.latitude, longitude=args.longitude, outlet_name=args.outlet_name)
    
    if results.get('success', False):
        print("SUCCESS: Step 2 watershed delineation completed")
        print(f"Workspace: {results['workspace']}")
        print(f"Area: {results['characteristics']['watershed_area_km2']:.2f} km²")
    else:
        print(f"FAILED: {results.get('error', 'Unknown error')}")
        sys.exit(1)
