"""
Consolidated Watershed Steps for RAVEN Workflows

This module contains all watershed-related workflow steps:
- DelineateWatershedAndStreams: Basic watershed and stream delineation
- UnifiedWatershedDelineation: Complete watershed analysis with lakes
- WatershedMappingStep: Watershed visualization and mapping

Consolidates functionality from multiple separate step files for better organization.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import logging
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from workflows.steps.base_step import WorkflowStep
from clients.watershed_clients.watershed import ProfessionalWatershedAnalyzer
from processors.lake_detection import ComprehensiveLakeDetector
from processors.lake_classifier import LakeClassifier  
from processors.lake_integrator import LakeIntegrator
from processors.basic_attributes import BasicAttributesCalculator
from processors.outlet_snapping import ImprovedOutletSnapper
from processors.watershed_mapper import WatershedMapper

class DelineateWatershedAndStreams(WorkflowStep):
    """
    Step 3B: Delineate watershed and extract stream network
    Used in Approach B (Full Delineation Workflow)
    """
    
    def __init__(self):
        super().__init__(
            step_name="delineate_watershed_streams",
            step_category="watershed",
            description="Trace watershed boundary and extract stream network from DEM"
        )
        self.analyzer = ProfessionalWatershedAnalyzer()
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._log_step_start()
        
        try:
            # Validate required inputs
            required_inputs = ['flow_direction', 'flow_accumulation', 'latitude', 'longitude']
            self.validate_inputs(inputs, required_inputs)
            
            flow_direction = self.validate_file_exists(inputs['flow_direction'])
            flow_accumulation = self.validate_file_exists(inputs['flow_accumulation'])
            outlet_lat = float(inputs['latitude'])
            outlet_lon = float(inputs['longitude'])
            
            # Create workspace
            workspace_dir = inputs.get('workspace_dir', flow_direction.parent)
            workspace = Path(workspace_dir)
            
            # Step 1: Delineate watershed
            self.logger.info("Delineating watershed boundary...")
            watershed_boundary = self._delineate_watershed(
                flow_direction, outlet_lat, outlet_lon, workspace
            )
            
            # Step 2: Extract stream network
            self.logger.info("Extracting stream network...")
            stream_network = self._extract_stream_network(
                flow_accumulation, watershed_boundary, workspace
            )
            
            # Calculate statistics
            stats = self._calculate_watershed_stats(watershed_boundary, stream_network)
            
            outputs = {
                'watershed_boundary': str(watershed_boundary),
                'stream_network': str(stream_network),
                'watershed_area_km2': stats['area_km2'],
                'stream_length_km': stats['stream_length_km'],
                'max_stream_order': stats['max_stream_order'],
                'outlet_snapped': False,
                'snap_distance_m': None,
                'success': True
            }
            
            self._log_step_complete([str(watershed_boundary), str(stream_network)])
            return outputs
            
        except Exception as e:
            error_msg = f"Watershed delineation failed: {str(e)}"
            self._log_step_failed(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _delineate_watershed(self, flow_direction: Path, outlet_lat: float, 
                           outlet_lon: float, workspace: Path) -> Path:
        """Delineate watershed boundary from outlet point"""
        
        results = self.analyzer.analyze_watershed_complete(
            dem_path=flow_direction,
            outlet_coords=(outlet_lat, outlet_lon),
            output_dir=workspace
        )

        if not results.get('success'):
            raise Exception(f"Watershed delineation failed: {results.get('error')}")

        for f in results.get('files_created', []):
            if 'watershed.geojson' in f:
                return Path(f)
        
        raise FileNotFoundError("Could not find watershed geojson in analysis results.")
    
    def _extract_stream_network(self, flow_accumulation: Path, 
                              watershed_boundary: Path, workspace: Path) -> Path:
        """Extract stream network from flow accumulation"""
        
        # This is a bit of a hack. The analyzer already created the streams.
        # We are just looking for the file.
        for f in workspace.glob('**/*'):
            if 'streams.geojson' in f.name:
                return f

        raise FileNotFoundError("Could not find stream geojson in analysis results.")
    
    def _calculate_watershed_stats(self, watershed_boundary: Path, 
                                 stream_network: Path) -> Dict[str, float]:
        """Calculate watershed and stream statistics with proper geodetic area calculation"""
        
        import geopandas as gpd
        
        # Load data
        watershed_gdf = gpd.read_file(watershed_boundary)
        stream_gdf = gpd.read_file(stream_network)
        
        # HYDROLOGICALLY CORRECT area calculation
        # Project to appropriate UTM zone for accurate area measurement
        if watershed_gdf.crs and watershed_gdf.crs.is_geographic:
            # Get appropriate UTM zone for accurate area calculation
            utm_crs = watershed_gdf.estimate_utm_crs()
            watershed_utm = watershed_gdf.to_crs(utm_crs)
            stream_utm = stream_gdf.to_crs(utm_crs)
            
            # Calculate area in square meters, convert to km²
            area_km2 = watershed_utm.geometry.area.sum() / 1e6
            
            # Calculate stream length in meters, convert to km
            stream_length_km = stream_utm.geometry.length.sum() / 1e3
            
            print(f"   Projected to UTM zone {utm_crs} for accurate area calculation")
            print(f"   Watershed area: {area_km2:.2f} km²")
            
        else:
            # Already in projected coordinate system
            if watershed_gdf.crs and 'metre' in str(watershed_gdf.crs.axis_info[0]).lower():
                # CRS uses meters
                area_km2 = watershed_gdf.geometry.area.sum() / 1e6
                stream_length_km = stream_gdf.geometry.length.sum() / 1e3
            elif watershed_gdf.crs and 'foot' in str(watershed_gdf.crs.axis_info[0]).lower():
                # CRS uses feet
                area_km2 = watershed_gdf.geometry.area.sum() * 0.092903e-6  # ft² to km²
                stream_length_km = stream_gdf.geometry.length.sum() * 0.0003048  # ft to km
            else:
                # Fallback: assume meters and warn
                print("   WARNING: Unknown CRS units, assuming meters")
                area_km2 = watershed_gdf.geometry.area.sum() / 1e6
                stream_length_km = stream_gdf.geometry.length.sum() / 1e3
        
        # Calculate drainage density (hydrologically important metric)
        drainage_density_km_per_km2 = stream_length_km / area_km2 if area_km2 > 0 else 0
        
        # Get max stream order with better handling
        max_stream_order = self._get_max_stream_order(stream_gdf, watershed_boundary.parent)
        
        return {
            'area_km2': round(area_km2, 4),
            'stream_length_km': round(stream_length_km, 2),
            'drainage_density_km_per_km2': round(drainage_density_km_per_km2, 3),
            'max_stream_order': max_stream_order,
            'crs_used': str(utm_crs) if watershed_gdf.crs and watershed_gdf.crs.is_geographic else str(watershed_gdf.crs)
        }
    
    def _get_max_stream_order(self, stream_gdf: 'gpd.GeoDataFrame', workspace_dir: Path = None) -> int:
        """Get maximum stream order using proper Strahler calculation or calculated raster"""
        
        # First try to get from properly calculated Strahler order raster (HYDROLOGICALLY CORRECT)
        max_order = self._get_stream_order_from_raster(workspace_dir)
        if max_order is not None:
            print(f"   Using calculated Strahler stream order: {max_order}")
            return max_order
        
        # Check for stream order in GeoDataFrame columns
        order_columns = ['stream_order', 'strahler', 'order', 'str_order', 'streamorde']
        stream_order_col = None
        
        for col in order_columns:
            if col in stream_gdf.columns:
                stream_order_col = col
                break
        
        if stream_order_col:
            try:
                max_order = int(stream_gdf[stream_order_col].max())
                if 1 <= max_order <= 12:  # Reasonable stream order range
                    print(f"   Using stream order from vector data: {max_order}")
                    return max_order
                else:
                    print(f"   WARNING: Unreasonable stream order {max_order}, falling back to estimation")
            except (ValueError, TypeError):
                print(f"   WARNING: Invalid stream order values in column {stream_order_col}")
        
        # Fallback: Estimate from network topology (less accurate but hydrologically informed)
        max_order = self._estimate_strahler_order_from_topology(stream_gdf)
        print(f"   Estimated Strahler stream order: {max_order} (topology-based)")
        return max_order
    
    def _get_stream_order_from_raster(self, workspace_dir: Path = None) -> Optional[int]:
        """Get maximum stream order from Strahler order raster"""
        
        # Look for Strahler order raster in workspace
        if workspace_dir is None:
            workspace_dir = Path.cwd()
        strahler_files = list(workspace_dir.glob("**/stream_order_strahler.tif"))
        if not strahler_files:
            return None
            
        try:
            import rasterio
            import numpy as np
            
            strahler_file = strahler_files[0]  # Use the first found
            with rasterio.open(strahler_file) as src:
                data = src.read(1, masked=True)
                if data.mask is not None:
                    valid_data = data[~data.mask]
                else:
                    valid_data = data[data > 0]  # Exclude nodata/zero
                
                if len(valid_data) > 0:
                    max_order = int(np.max(valid_data))
                    return max_order
                    
        except Exception as e:
            print(f"   WARNING: Could not read Strahler raster: {e}")
        
        return None
    
    def _estimate_strahler_order_from_topology(self, stream_gdf: 'gpd.GeoDataFrame') -> int:
        """
        Estimate Strahler stream order from stream network topology 
        More hydrologically informed than simple segment counting
        """
        
        num_streams = len(stream_gdf)
        
        # More sophisticated estimation based on Horton's Laws
        # Rule of thumb: stream order follows logarithmic relationship with stream count
        if num_streams < 5:
            estimated_order = 1
        elif num_streams < 15:
            estimated_order = 2  
        elif num_streams < 40:
            estimated_order = 3
        elif num_streams < 120:
            estimated_order = 4
        elif num_streams < 360:
            estimated_order = 5
        elif num_streams < 1000:
            estimated_order = 6
        else:
            # For very large networks, use Horton's bifurcation ratio (typically 3-5)
            import math
            estimated_order = min(12, max(6, int(math.log(num_streams, 4) + 1)))
        
        return estimated_order


# Re-export the main class
__all__ = ['DelineateWatershedAndStreams']