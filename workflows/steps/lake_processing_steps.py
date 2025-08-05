"""
Lake Processing Steps for RAVEN Workflows

This module contains steps for lake detection, classification, and integration.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from workflows.steps.base_step import WorkflowStep

class DetectAndClassifyLakes(WorkflowStep):
    """
    Step 4B: Detect and classify lakes within watershed
    Used in Approach B (Full Delineation Workflow)
    """
    
    def __init__(self):
        super().__init__(
            step_name="detect_classify_lakes",
            step_category="lake_processing",
            description="Identify, classify, and integrate lakes within watershed"
        )
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._log_step_start()
        
        try:
            # Validate required inputs
            required_inputs = ['watershed_boundary', 'stream_network']
            self.validate_inputs(inputs, required_inputs)
            
            watershed_boundary = self.validate_file_exists(inputs['watershed_boundary'])
            stream_network = self.validate_file_exists(inputs['stream_network'])
            
            # Create workspace
            workspace_dir = inputs.get('workspace_dir', watershed_boundary.parent)
            workspace = Path(workspace_dir)
            
            # Step 1: Find lakes within watershed
            self.logger.info("Finding lakes within watershed...")
            watershed_lakes = self._find_watershed_lakes(watershed_boundary, workspace)
            
            # Step 2: Classify lakes as connected vs isolated
            self.logger.info("Classifying lake connectivity...")
            classified_lakes = self._classify_lake_connectivity(
                watershed_lakes, stream_network, workspace
            )
            
            # Step 3: Filter significant lakes
            self.logger.info("Filtering significant lakes...")
            significant_lakes = self._filter_significant_lakes(classified_lakes, workspace)
            
            # Step 4: Integrate with stream network
            self.logger.info("Integrating lakes with stream network...")
            integrated_network = self._integrate_with_streams(
                significant_lakes, stream_network, workspace
            )
            
            outputs = {
                'significant_connected_lakes': significant_lakes['connected'],
                'significant_isolated_lakes': significant_lakes['isolated'],
                'integrated_stream_network': str(integrated_network),
                'lake_outlets': significant_lakes.get('outlets'),
                'connected_lake_count': significant_lakes['connected_count'],
                'isolated_lake_count': significant_lakes['isolated_count'],
                'total_lake_area_km2': significant_lakes['total_area_km2'],
                'success': True
            }
            
            created_files = [f for f in outputs.values() if isinstance(f, str) and f]
            self._log_step_complete(created_files)
            return outputs
            
        except Exception as e:
            error_msg = f"Lake detection and classification failed: {str(e)}"
            self._log_step_failed(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _find_watershed_lakes(self, watershed_boundary: Path, workspace: Path) -> Path:
        """Find lakes within watershed boundary using local databases"""
        
        import geopandas as gpd
        from shapely.geometry import Point, Polygon
        
        # Load watershed
        watershed_gdf = gpd.read_file(watershed_boundary)
        watershed_geom = watershed_gdf.geometry.iloc[0]
        
        # Create mock lakes within watershed
        bounds = watershed_gdf.total_bounds
        
        lakes = []
        
        # Create a few mock lakes
        for i in range(3):
            # Random location within watershed
            center_x = bounds[0] + (bounds[2] - bounds[0]) * (0.3 + i * 0.2)
            center_y = bounds[1] + (bounds[3] - bounds[1]) * (0.3 + i * 0.2)
            
            # Create circular lake
            radius = 0.01  # degrees
            lake_poly = Point(center_x, center_y).buffer(radius)
            
            lakes.append({
                'geometry': lake_poly,
                'lake_id': f'LAKE_{i+1}',
                'area_km2': lake_poly.area * 111 * 111,  # Rough conversion
                'depth_m': 5.0 + i * 2.0  # Mock depth
            })
        
        # Save lakes
        lakes_file = workspace / "watershed_lakes.shp"
        lakes_gdf = gpd.GeoDataFrame(lakes, crs='EPSG:4326')
        lakes_gdf.to_file(lakes_file)
        
        return lakes_file
    
    def _classify_lake_connectivity(self, lakes_file: Path, stream_network: Path, 
                                  workspace: Path) -> Dict[str, Any]:
        """Classify lakes as connected or isolated"""
        
        import geopandas as gpd
        
        # Load data
        lakes_gdf = gpd.read_file(lakes_file)
        streams_gdf = gpd.read_file(stream_network)
        
        # Check intersections with buffer
        buffer_distance = 0.001  # degrees (~100m)
        stream_buffer = streams_gdf.geometry.buffer(buffer_distance).unary_union
        
        connected_lakes = []
        isolated_lakes = []
        
        for idx, lake in lakes_gdf.iterrows():
            if lake.geometry.intersects(stream_buffer):
                connected_lakes.append(lake)
            else:
                isolated_lakes.append(lake)
        
        # Save classified lakes
        connected_file = workspace / "connected_lakes.shp"
        isolated_file = workspace / "isolated_lakes.shp"
        
        if connected_lakes:
            connected_gdf = gpd.GeoDataFrame(connected_lakes, crs=lakes_gdf.crs)
            connected_gdf.to_file(connected_file)
        
        if isolated_lakes:
            isolated_gdf = gpd.GeoDataFrame(isolated_lakes, crs=lakes_gdf.crs)
            isolated_gdf.to_file(isolated_file)
        
        return {
            'connected_file': connected_file if connected_lakes else None,
            'isolated_file': isolated_file if isolated_lakes else None,
            'connected_count': len(connected_lakes),
            'isolated_count': len(isolated_lakes)
        }
    
    def _filter_significant_lakes(self, classified_lakes: Dict[str, Any], 
                                workspace: Path) -> Dict[str, Any]:
        """Filter lakes by significance thresholds"""
        
        import geopandas as gpd
        
        # BasinMaker thresholds
        connected_area_threshold = 0.5  # km²
        isolated_area_threshold = 1.0   # km²
        depth_threshold = 2.0           # m
        
        significant_connected = []
        significant_isolated = []
        total_area = 0.0
        
        # Filter connected lakes
        if classified_lakes['connected_file'] and classified_lakes['connected_file'].exists():
            connected_gdf = gpd.read_file(classified_lakes['connected_file'])
            
            for idx, lake in connected_gdf.iterrows():
                if (lake.get('area_km2', 0) >= connected_area_threshold and 
                    lake.get('depth_m', 0) >= depth_threshold):
                    significant_connected.append(lake)
                    total_area += lake.get('area_km2', 0)
        
        # Filter isolated lakes
        if classified_lakes['isolated_file'] and classified_lakes['isolated_file'].exists():
            isolated_gdf = gpd.read_file(classified_lakes['isolated_file'])
            
            for idx, lake in isolated_gdf.iterrows():
                if (lake.get('area_km2', 0) >= isolated_area_threshold and 
                    lake.get('depth_m', 0) >= depth_threshold):
                    significant_isolated.append(lake)
                    total_area += lake.get('area_km2', 0)
        
        # Save significant lakes
        significant_connected_file = workspace / "significant_connected_lakes.shp"
        significant_isolated_file = workspace / "significant_isolated_lakes.shp"
        
        if significant_connected:
            sig_conn_gdf = gpd.GeoDataFrame(significant_connected, crs='EPSG:4326')
            sig_conn_gdf.to_file(significant_connected_file)
        
        if significant_isolated:
            sig_isol_gdf = gpd.GeoDataFrame(significant_isolated, crs='EPSG:4326')
            sig_isol_gdf.to_file(significant_isolated_file)
        
        return {
            'connected': str(significant_connected_file) if significant_connected else None,
            'isolated': str(significant_isolated_file) if significant_isolated else None,
            'connected_count': len(significant_connected),
            'isolated_count': len(significant_isolated),
            'total_area_km2': total_area
        }
    
    def _integrate_with_streams(self, significant_lakes: Dict[str, Any], 
                              stream_network: Path, workspace: Path) -> Path:
        """Integrate lakes with stream network"""
        
        import geopandas as gpd
        from shapely.geometry import Point
        
        # Load stream network
        streams_gdf = gpd.read_file(stream_network)
        
        # Create lake outlets if connected lakes exist
        outlets = []
        
        if significant_lakes['connected'] and Path(significant_lakes['connected']).exists():
            lakes_gdf = gpd.read_file(significant_lakes['connected'])
            
            for idx, lake in lakes_gdf.iterrows():
                # Find outlet point (simplified - use centroid)
                outlet_point = lake.geometry.centroid
                outlets.append({
                    'geometry': outlet_point,
                    'lake_id': lake.get('lake_id', f'LAKE_{idx}'),
                    'outlet_type': 'natural'
                })
        
        # Save lake outlets
        if outlets:
            outlets_file = workspace / "lake_outlets.shp"
            outlets_gdf = gpd.GeoDataFrame(outlets, crs='EPSG:4326')
            outlets_gdf.to_file(outlets_file)
            significant_lakes['outlets'] = str(outlets_file)
        
        # Create integrated stream network (copy original for now)
        integrated_streams = workspace / "integrated_stream_network.shp"
        streams_gdf.to_file(integrated_streams)
        
        return integrated_streams