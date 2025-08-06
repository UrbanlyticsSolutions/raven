#!/usr/bin/env python3
"""
Improved Outlet Snapping Processor

Provides multiple snapping methods including downstream snapping
that respects the existing processor architecture.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point, LineString
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class ImprovedOutletSnapper:
    """
    Improved outlet snapping processor with multiple methods:
    1. WhiteboxTools standard snapping (flow accumulation based)
    2. Downstream snapping (nearest stream downstream point)
    3. Multi-criteria snapping (flow + distance + stream proximity)
    """
    
    def __init__(self, workspace_dir: Path = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "outlet_snapping"
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Snapping parameters
        self.max_search_distance_m = 1000  # Maximum search distance in meters
        self.default_snap_distance_m = 200  # Default snap distance
        
    def _setup_logging(self):
        """Setup logging for the processor"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(self.workspace_dir / "outlet_snapping.log")
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def snap_outlet_downstream(self, outlet_coords: Tuple[float, float], 
                              streams_file: str, flow_accum_file: str,
                              max_search_distance: float = None) -> Dict[str, Any]:
        """
        Snap outlet to downstream point of nearest stream
        
        Parameters:
        -----------
        outlet_coords : tuple
            Original outlet coordinates (longitude, latitude)
        streams_file : str
            Path to streams shapefile/geojson
        flow_accum_file : str
            Path to flow accumulation raster
        max_search_distance : float, optional
            Maximum search distance in meters
            
        Returns:
        --------
        Dict with snapping results including both original and snapped outlets
        """
        
        self.logger.info(f"Starting downstream outlet snapping for ({outlet_coords[0]:.6f}, {outlet_coords[1]:.6f})")
        
        max_distance = max_search_distance or self.default_snap_distance_m
        original_point = Point(outlet_coords)
        
        try:
            # Load streams
            streams_gdf = gpd.read_file(streams_file)
            self.logger.info(f"Loaded {len(streams_gdf)} stream segments")
            
            # Step 1: Find nearest stream
            distances = streams_gdf.geometry.distance(original_point)
            closest_idx = distances.idxmin()
            closest_distance_deg = distances.min()
            closest_distance_m = closest_distance_deg * 111000  # Approximate conversion
            
            self.logger.info(f"Nearest stream: {closest_distance_m:.1f}m away")
            
            if closest_distance_m > max_distance:
                return {
                    'success': False,
                    'error': f'Nearest stream is {closest_distance_m:.1f}m away (> {max_distance}m limit)',
                    'original_coords': outlet_coords,
                    'method': 'downstream_snapping'
                }
            
            # Step 2: Get closest stream and find downstream point
            closest_stream = streams_gdf.iloc[closest_idx]
            stream_line = closest_stream.geometry
            stream_coords = list(stream_line.coords)
            
            # Step 3: Load flow accumulation and find downstream point
            with rasterio.open(flow_accum_file) as src:
                flow_data = src.read(1)
                
                # Sample flow accumulation at each stream point
                flow_values = []
                valid_coords = []
                
                for coord in stream_coords:
                    try:
                        row, col = src.index(coord[0], coord[1])
                        if 0 <= row < src.height and 0 <= col < src.width:
                            flow_val = flow_data[row, col]
                            if not np.isnan(flow_val) and flow_val >= 0:
                                flow_values.append(flow_val)
                                valid_coords.append(coord)
                    except:
                        continue
                
                if not flow_values:
                    return {
                        'success': False,
                        'error': 'No valid flow accumulation values on nearest stream',
                        'original_coords': outlet_coords,
                        'method': 'downstream_snapping'
                    }
                
                # Step 4: Find downstream point (highest flow accumulation)
                max_flow_idx = np.argmax(flow_values)
                downstream_coords = valid_coords[max_flow_idx]
                max_flow_value = flow_values[max_flow_idx]
                
                # Calculate snap distance
                downstream_point = Point(downstream_coords)
                snap_distance_deg = original_point.distance(downstream_point)
                snap_distance_m = snap_distance_deg * 111000
                
                self.logger.info(f"Downstream point found: ({downstream_coords[0]:.6f}, {downstream_coords[1]:.6f})")
                self.logger.info(f"Flow accumulation: {max_flow_value:.0f}")
                self.logger.info(f"Snap distance: {snap_distance_m:.1f}m")
                
                # Create output shapefiles
                original_gdf = gpd.GeoDataFrame(
                    [{'id': 1, 'type': 'original', 'flow_accum': 0}],
                    geometry=[original_point],
                    crs='EPSG:4326'
                )
                
                snapped_gdf = gpd.GeoDataFrame(
                    [{'id': 1, 'type': 'snapped_downstream', 'flow_accum': max_flow_value}],
                    geometry=[downstream_point],
                    crs='EPSG:4326'
                )
                
                # Save outlets
                original_outlet_file = self.workspace_dir / "original_outlet.shp"
                snapped_outlet_file = self.workspace_dir / "snapped_outlet_downstream.shp"
                
                original_gdf.to_file(original_outlet_file)
                snapped_gdf.to_file(snapped_outlet_file)
                
                self.logger.info(f"Saved original outlet: {original_outlet_file}")
                self.logger.info(f"Saved snapped outlet: {snapped_outlet_file}")
                
                return {
                    'success': True,
                    'method': 'downstream_snapping',
                    'original_coords': outlet_coords,
                    'snapped_coords': downstream_coords,
                    'snap_distance_m': snap_distance_m,
                    'flow_accumulation': max_flow_value,
                    'stream_id': closest_idx,
                    'original_outlet_file': str(original_outlet_file),
                    'snapped_outlet_file': str(snapped_outlet_file),
                    'files_created': [str(original_outlet_file), str(snapped_outlet_file)]
                }
                
        except Exception as e:
            error_msg = f"Downstream snapping failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'original_coords': outlet_coords,
                'method': 'downstream_snapping'
            }
    
    def compare_snapping_methods(self, outlet_coords: Tuple[float, float],
                                streams_file: str, flow_accum_file: str, flow_dir_file: str) -> Dict[str, Any]:
        """
        Compare multiple snapping methods and return results for all
        
        Returns results from:
        1. Downstream snapping (this processor)
        2. WhiteboxTools standard snapping (if available)
        3. Analysis of both methods
        """
        
        self.logger.info("Comparing snapping methods")
        
        results = {
            'success': True,
            'original_coords': outlet_coords,
            'comparison_results': {}
        }
        
        # Method 1: Downstream snapping
        downstream_result = self.snap_outlet_downstream(outlet_coords, streams_file, flow_accum_file)
        results['comparison_results']['downstream'] = downstream_result
        
        # Method 2: WhiteboxTools snapping (if available)
        try:
            import whitebox
            wbt = whitebox.WhiteboxTools()
            wbt.set_working_dir(str(self.workspace_dir.absolute()))
            wbt.set_verbose_mode(False)
            
            # Create original outlet shapefile for WhiteboxTools
            original_point = Point(outlet_coords)
            original_gdf = gpd.GeoDataFrame(
                [{'id': 1, 'type': 'original'}],
                geometry=[original_point],
                crs='EPSG:4326'
            )
            
            wbt_original_file = self.workspace_dir / "wbt_original_outlet.shp"
            wbt_snapped_file = self.workspace_dir / "wbt_snapped_outlet.shp"
            
            original_gdf.to_file(wbt_original_file)
            
            # Copy flow accumulation to workspace
            import shutil
            local_flow_accum = self.workspace_dir / "flow_accumulation.tif"
            shutil.copy2(flow_accum_file, local_flow_accum)
            
            # Run WhiteboxTools snapping
            wbt.snap_pour_points(
                str(wbt_original_file.name),
                str(local_flow_accum.name),
                str(wbt_snapped_file.name),
                50  # 50 meter snap distance
            )
            
            if wbt_snapped_file.exists():
                wbt_snapped_gdf = gpd.read_file(wbt_snapped_file)
                if len(wbt_snapped_gdf) > 0:
                    wbt_snapped_point = wbt_snapped_gdf.geometry.iloc[0]
                    wbt_snap_distance = original_point.distance(wbt_snapped_point) * 111000
                    
                    results['comparison_results']['whitebox'] = {
                        'success': True,
                        'method': 'whitebox_snap_pour_points',
                        'snapped_coords': (wbt_snapped_point.x, wbt_snapped_point.y),
                        'snap_distance_m': wbt_snap_distance,
                        'snapped_outlet_file': str(wbt_snapped_file)
                    }
                    
                    self.logger.info(f"WhiteboxTools snapping: {wbt_snap_distance:.1f}m")
                else:
                    results['comparison_results']['whitebox'] = {
                        'success': False,
                        'error': 'WhiteboxTools produced empty result'
                    }
            else:
                results['comparison_results']['whitebox'] = {
                    'success': False,
                    'error': 'WhiteboxTools snap_pour_points failed'
                }
                
        except Exception as e:
            results['comparison_results']['whitebox'] = {
                'success': False,
                'error': f'WhiteboxTools not available or failed: {str(e)}'
            }
        
        # Create comparison summary
        results['summary'] = self._create_comparison_summary(results['comparison_results'])
        
        return results
    
    def _create_comparison_summary(self, comparison_results: Dict) -> Dict[str, Any]:
        """Create a summary comparing the different snapping methods"""
        
        summary = {
            'methods_tested': len(comparison_results),
            'successful_methods': sum(1 for r in comparison_results.values() if r.get('success', False)),
            'recommendations': []
        }
        
        # Analyze each method
        for method_name, result in comparison_results.items():
            if result.get('success', False):
                snap_distance = result.get('snap_distance_m', 0)
                
                summary[f'{method_name}_distance'] = snap_distance
                
                if snap_distance < 10:
                    summary['recommendations'].append(f"{method_name}: Excellent snap (< 10m)")
                elif snap_distance < 100:
                    summary['recommendations'].append(f"{method_name}: Good snap (< 100m)")
                else:
                    summary['recommendations'].append(f"{method_name}: Long distance snap ({snap_distance:.0f}m)")
        
        return summary


if __name__ == "__main__":
    # Test the improved snapping
    snapper = ImprovedOutletSnapper("test_snapping")
    
    # Vancouver coordinates
    outlet_coords = (-123.1207, 49.2827)
    
    # Use existing data
    streams_file = "fixed_bc_workflow/watershed/basic_delineation/streams.geojson"
    flow_accum_file = "fixed_bc_workflow/watershed/basic_delineation/flow_accumulation_d8.tif"
    flow_dir_file = "fixed_bc_workflow/watershed/basic_delineation/flow_direction_d8.tif"
    
    # Test downstream snapping
    result = snapper.snap_outlet_downstream(outlet_coords, streams_file, flow_accum_file)
    
    if result['success']:
        print(f"Downstream snapping successful!")
        print(f"Original: {result['original_coords']}")
        print(f"Snapped: {result['snapped_coords']}")
        print(f"Distance: {result['snap_distance_m']:.1f}m")
    else:
        print(f"Downstream snapping failed: {result['error']}")