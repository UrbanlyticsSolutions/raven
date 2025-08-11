#!/usr/bin/env python3
"""
Lake Processor with River Routing Integration
Based on BasinMaker logic with upstream/downstream routing, 
bankfull characteristics, and connected lake integration
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
import networkx as nx
from scipy import ndimage
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LakeProcessor:
    """
    Lake processing with river routing integration
    Includes:
    1. Upstream/downstream lake connectivity analysis
    2. River flow direction and routing through lakes
    3. Bankfull width/depth calculations for lake outlets
    4. Lake outflow routing and pour point analysis
    5. Connected lake integration with HRU generation
    """
    
    def __init__(self, workspace_dir: Path = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Default hydraulic parameters (from BasinMaker)
        self.default_min_slope = 0.001  # Minimum river slope
        self.default_manning_n_min = 0.025  # Minimum Manning's n
        self.default_manning_n_max = 0.15   # Maximum Manning's n
        self.default_bkf_width = 1.2345     # Default bankfull width
        self.default_bkf_depth = 1.2345     # Default bankfull depth
        self.default_bkf_discharge = 1.2345 # Default bankfull discharge
        
        # Lake processing thresholds
        self.min_lake_outflow_length = 10.0  # meters
        self.lake_routing_buffer = 50.0      # meters
        
    def process_lakes_with_routing(self, 
                                 dem_path: Path,
                                 watershed_boundary: Path,
                                 stream_network: Path,
                                 lakes_shapefile: Path,
                                 subbasins_shapefile: Path = None,
                                 buffer_distance_km: float = 5.0) -> Dict:
        """
        Complete lake processing with river routing integration
        Addresses boundary truncation by expanding analysis area
        
        Parameters:
        -----------
        dem_path : Path
            Digital elevation model for flow analysis
        watershed_boundary : Path
            Watershed boundary shapefile
        stream_network : Path
            Stream network shapefile with flow direction
        lakes_shapefile : Path
            Detected lakes shapefile
        subbasins_shapefile : Path, optional
            Subbasins for HRU integration
        buffer_distance_km : float
            Buffer distance around watershed for extended routing analysis
            
        Returns:
        --------
        Dict with complete lake routing analysis
        """
        
        logger.info("Starting lake processing with river routing...")
        logger.info(f"Using {buffer_distance_km}km buffer for extended routing network")
        
        # Step 0: Create extended analysis area to avoid boundary truncation
        logger.info("Step 0: Creating extended analysis area...")
        extended_area = self._create_extended_analysis_area(
            watershed_boundary, buffer_distance_km
        )
        
        # Load input data with extended area consideration
        input_data = self._load_input_data_extended(
            dem_path, watershed_boundary, stream_network, 
            lakes_shapefile, subbasins_shapefile, extended_area
        )
        
        if not input_data['success']:
            return input_data
        
        dem = input_data['dem']
        watershed = input_data['watershed']
        extended_boundary = input_data['extended_boundary']
        streams = input_data['streams']
        lakes = input_data['lakes']
        subbasins = input_data.get('subbasins')
        
        # Step 1: Detect additional lakes in extended area
        logger.info("Step 1: Detecting additional lakes in extended area...")
        extended_lakes = self._detect_extended_lakes(
            dem, extended_boundary, streams, lakes
        )
        
        # Step 2: Analyze complete stream-lake connectivity
        logger.info("Step 2: Analyzing complete stream-lake connectivity...")
        connectivity_results = self._analyze_lake_stream_connectivity_extended(
            extended_lakes, streams, dem, watershed
        )
        
        # Step 3: Build complete routing network
        logger.info("Step 3: Building complete river routing network...")
        routing_network = self._build_complete_routing_network(
            extended_lakes, streams, connectivity_results, dem, watershed
        )
        
        # Step 4: Calculate flow directions and pour points
        logger.info("Step 4: Calculating flow directions and pour points...")
        flow_analysis = self._analyze_flow_directions(
            extended_lakes, streams, routing_network, dem
        )
        
        # Step 5: Calculate bankfull characteristics
        logger.info("Step 5: Calculating bankfull characteristics...")
        hydraulic_properties = self._calculate_complete_hydraulics(
            extended_lakes, streams, routing_network, flow_analysis, dem
        )
        
        # Step 6: Determine complete upstream/downstream relationships
        logger.info("Step 6: Determining complete lake topology...")
        lake_topology = self._determine_complete_lake_topology(
            extended_lakes, routing_network, hydraulic_properties, watershed
        )
        
        # Step 7: Classify lakes by watershed relationship
        logger.info("Step 7: Classifying lakes by watershed relationship...")
        lake_classification = self._classify_lakes_by_watershed(
            extended_lakes, watershed, lake_topology
        )
        
        # Step 8: Integrate with subbasin routing
        logger.info("Step 8: Integrating with subbasin routing...")
        if subbasins is not None:
            subbasin_integration = self._integrate_complete_lakes_with_subbasins(
                extended_lakes, subbasins, lake_topology, hydraulic_properties, watershed
            )
        else:
            subbasin_integration = {'success': True, 'message': 'No subbasins provided'}
        
        # Step 9: Create enhanced lake attributes for RAVEN
        logger.info("Step 9: Creating enhanced lake attributes...")
        enhanced_lakes = self._create_complete_lake_attributes(
            extended_lakes, connectivity_results, hydraulic_properties, 
            lake_topology, lake_classification, subbasin_integration, watershed
        )
        
        # Step 10: Save all results
        logger.info("Step 10: Saving complete results...")
        output_files = self._save_complete_results(
            enhanced_lakes, routing_network, hydraulic_properties, 
            lake_topology, lake_classification, flow_analysis
        )
        
        return {
            'success': True,
            'files_created': output_files,
            'extended_area_km2': extended_boundary.area / 1e6,
            'watershed_area_km2': watershed.geometry.iloc[0].area / 1e6,
            'connectivity_results': connectivity_results,
            'routing_network': routing_network,
            'flow_analysis': flow_analysis,
            'hydraulic_properties': hydraulic_properties,
            'lake_topology': lake_topology,
            'lake_classification': lake_classification,
            'subbasin_integration': subbasin_integration,
            'enhanced_lakes': enhanced_lakes,
            'statistics': self._generate_complete_statistics(enhanced_lakes, lake_topology, lake_classification)
        }
    
    def _create_extended_analysis_area(self, watershed_boundary, buffer_distance_km) -> 'Polygon':
        """Create extended analysis area to avoid boundary truncation"""
        
        watershed = gpd.read_file(watershed_boundary)
        
        # Convert to projected CRS for accurate buffering (use UTM)
        # Get UTM zone from watershed centroid
        centroid = watershed.geometry.iloc[0].centroid
        utm_crs = f"EPSG:{32610 + int((centroid.x + 180) / 6)}"  # Simplified UTM zone calc
        
        watershed_projected = watershed.to_crs(utm_crs)
        buffer_distance_m = buffer_distance_km * 1000
        
        # Create buffer
        buffered = watershed_projected.buffer(buffer_distance_m)
        
        # Convert back to original CRS
        buffered_original_crs = buffered.to_crs(watershed.crs)
        
        return buffered_original_crs.geometry.iloc[0]
    
    def _load_input_data_extended(self, dem_path, watershed_boundary, stream_network, 
                                lakes_shapefile, subbasins_shapefile, extended_boundary) -> Dict:
        """Load input data considering extended analysis area"""
        
        try:
            # Load DEM
            with rasterio.open(dem_path) as src:
                dem_data = {
                    'array': src.read(1),
                    'transform': src.transform,
                    'crs': src.crs,
                    'nodata': src.nodata
                }
            
            # Load vector data
            watershed = gpd.read_file(watershed_boundary)
            streams = gpd.read_file(stream_network)
            lakes = gpd.read_file(lakes_shapefile)
            
            # Ensure consistent CRS
            target_crs = streams.crs
            if watershed.crs != target_crs:
                watershed = watershed.to_crs(target_crs)
            if lakes.crs != target_crs:
                lakes = lakes.to_crs(target_crs)
            
            # Create extended boundary GeoDataFrame
            extended_boundary_gdf = gpd.GeoDataFrame([1], geometry=[extended_boundary], crs=target_crs)
            
            # Filter streams and lakes to extended area (not just watershed)
            streams_extended = streams[streams.intersects(extended_boundary)]
            lakes_extended = lakes[lakes.intersects(extended_boundary)]
            
            logger.info(f"Extended area analysis:")
            logger.info(f"  Original lakes: {len(lakes)} -> Extended: {len(lakes_extended)}")
            logger.info(f"  Original streams: {len(streams)} -> Extended: {len(streams_extended)}")
            
            # Load subbasins if provided
            subbasins = None
            if subbasins_shapefile and Path(subbasins_shapefile).exists():
                subbasins = gpd.read_file(subbasins_shapefile)
                if subbasins.crs != target_crs:
                    subbasins = subbasins.to_crs(target_crs)
            
            return {
                'success': True,
                'dem': dem_data,
                'watershed': watershed,
                'extended_boundary': extended_boundary,
                'streams': streams_extended,
                'lakes': lakes_extended,
                'subbasins': subbasins
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to load extended input data: {str(e)}"
            }
    
    def _detect_extended_lakes(self, dem, extended_boundary, streams, existing_lakes):
        """Detect additional lakes in the extended area using DEM analysis"""
        
        # For now, use existing lakes but could extend with additional detection
        # This is where you could add more sophisticated lake detection
        logger.info("Using existing lake detection - could be enhanced with DEM-based detection")
        return existing_lakes
    
    def _analyze_lake_stream_connectivity_extended(self, lakes, streams, dem, watershed) -> Dict:
        """
        Analyze stream-lake intersections with extended area context
        Classify lakes by their relationship to the watershed boundary
        """
        
        watershed_geom = watershed.geometry.iloc[0]
        
        connectivity_results = {
            'connected_lakes': [],
            'non_connected_lakes': [],
            'lake_inflows': {},
            'lake_outflows': {},
            'lake_watershed_relationship': {},
            'boundary_crossings': {}
        }
        
        for idx, lake in lakes.iterrows():
            lake_id = lake.get('lake_id', idx)
            lake_geom = lake.geometry
            
            # Classify lake by watershed relationship
            if watershed_geom.contains(lake_geom):
                relationship = 'internal'
            elif watershed_geom.intersects(lake_geom):
                relationship = 'boundary'
            else:
                relationship = 'external'
            
            connectivity_results['lake_watershed_relationship'][lake_id] = relationship
            
            # Find streams that intersect with this lake
            intersecting_streams = streams[streams.intersects(lake_geom)]
            
            if len(intersecting_streams) > 0:
                connectivity_results['connected_lakes'].append(lake_id)
                
                # Determine inflows and outflows with watershed context
                inflows, outflows = self._classify_lake_flows_extended(
                    lake_geom, intersecting_streams, dem, watershed_geom, relationship
                )
                
                connectivity_results['lake_inflows'][lake_id] = inflows
                connectivity_results['lake_outflows'][lake_id] = outflows
                
                # Check for boundary crossings
                connectivity_results['boundary_crossings'][lake_id] = self._check_boundary_crossings(
                    lake_geom, intersecting_streams, watershed_geom
                )
                
            else:
                connectivity_results['non_connected_lakes'].append(lake_id)
                connectivity_results['lake_inflows'][lake_id] = []
                connectivity_results['lake_outflows'][lake_id] = []
                connectivity_results['boundary_crossings'][lake_id] = []
        
        return connectivity_results
    
    def _classify_lake_flows_extended(self, lake_geom, intersecting_streams, dem, 
                                    watershed_geom, relationship) -> Tuple[List, List]:
        """
        Classify streams as inflows or outflows with watershed boundary context
        """
        
        inflows = []
        outflows = []
        
        lake_boundary = lake_geom.boundary
        
        for idx, stream in intersecting_streams.iterrows():
            stream_geom = stream.geometry
            
            # Find intersection points
            intersection = stream_geom.intersection(lake_boundary)
            
            if intersection.is_empty:
                continue
            
            # Extract intersection coordinates
            if hasattr(intersection, 'geoms'):
                int_points = [geom for geom in intersection.geoms if geom.geom_type == 'Point']
            else:
                int_points = [intersection] if intersection.geom_type == 'Point' else []
            
            for point in int_points:
                # Enhanced flow direction considering watershed context
                flow_direction = self._determine_flow_direction_extended(
                    point, stream_geom, dem, watershed_geom, relationship
                )
                
                # Check if stream crosses watershed boundary
                crosses_boundary = watershed_geom.intersects(stream_geom) and not watershed_geom.contains(stream_geom)
                
                flow_info = {
                    'stream_id': idx,
                    'intersection_point': point,
                    'stream_segment': stream_geom,
                    'crosses_watershed_boundary': crosses_boundary,
                    'flow_confidence': flow_direction.get('confidence', 'medium')
                }
                
                if flow_direction['direction'] == 'into_lake':
                    inflows.append(flow_info)
                elif flow_direction['direction'] == 'out_of_lake':
                    outflows.append(flow_info)
        
        return inflows, outflows
    
    def _determine_flow_direction_extended(self, point, stream_geom, dem, 
                                         watershed_geom, relationship) -> Dict:
        """
        Enhanced flow direction determination with watershed context
        """
        
        x, y = point.x, point.y
        
        try:
            dem_transform = dem['transform']
            dem_array = dem['array']
            
            # Get pixel coordinates
            row, col = rasterio.transform.rowcol(dem_transform, x, y)
            
            # Sample larger neighborhood for better gradient analysis
            window_size = 5
            half_window = window_size // 2
            
            row_start = max(0, row - half_window)
            row_end = min(dem_array.shape[0], row + half_window + 1)
            col_start = max(0, col - half_window)
            col_end = min(dem_array.shape[1], col + half_window + 1)
            
            elevation_window = dem_array[row_start:row_end, col_start:col_end]
            
            if elevation_window.size == 0:
                return {'direction': 'unknown', 'confidence': 'low'}
            
            center_elevation = dem_array[row, col] if 0 <= row < dem_array.shape[0] and 0 <= col < dem_array.shape[1] else np.nan
            
            # Calculate gradients along stream direction
            coords = list(stream_geom.coords)
            if len(coords) < 2:
                return {'direction': 'unknown', 'confidence': 'low'}
            
            # Find closest points on stream to current intersection
            point_coords = np.array([x, y])
            stream_coords = np.array(coords)
            distances = np.linalg.norm(stream_coords - point_coords, axis=1)
            closest_idx = np.argmin(distances)
            
            # Determine upstream and downstream directions
            if closest_idx > 0:
                upstream_point = coords[closest_idx - 1]
                upstream_elev = self._sample_dem_elevation(upstream_point, dem)
            else:
                upstream_elev = center_elevation
            
            if closest_idx < len(coords) - 1:
                downstream_point = coords[closest_idx + 1]
                downstream_elev = self._sample_dem_elevation(downstream_point, dem)
            else:
                downstream_elev = center_elevation
            
            # Determine flow direction based on elevation gradient
            if not np.isnan(upstream_elev) and not np.isnan(downstream_elev):
                if upstream_elev > center_elevation > downstream_elev:
                    direction = 'out_of_lake'
                    confidence = 'high'
                elif downstream_elev > center_elevation > upstream_elev:
                    direction = 'into_lake'
                    confidence = 'high'
                else:
                    # Use watershed boundary context for ambiguous cases
                    if relationship == 'external':
                        # External lakes likely feed into or receive from watershed
                        direction = 'into_lake' if upstream_elev > downstream_elev else 'out_of_lake'
                    else:
                        direction = 'out_of_lake' if upstream_elev > downstream_elev else 'into_lake'
                    confidence = 'medium'
            else:
                direction = 'unknown'
                confidence = 'low'
            
            return {
                'direction': direction,
                'confidence': confidence,
                'upstream_elevation': upstream_elev,
                'downstream_elevation': downstream_elev,
                'center_elevation': center_elevation
            }
                
        except Exception as e:
            logger.warning(f"Flow direction analysis failed: {str(e)}")
            return {'direction': 'unknown', 'confidence': 'low'}
    
    def _check_boundary_crossings(self, lake_geom, intersecting_streams, watershed_geom) -> List:
        """Check which streams cross the watershed boundary"""
        
        boundary_crossings = []
        
        for idx, stream in intersecting_streams.iterrows():
            stream_geom = stream.geometry
            
            # Check if stream crosses watershed boundary
            if watershed_geom.intersects(stream_geom) and not watershed_geom.contains(stream_geom):
                # Find boundary crossing points
                boundary_intersection = stream_geom.intersection(watershed_geom.boundary)
                
                boundary_crossings.append({
                    'stream_id': idx,
                    'crossing_type': 'watershed_boundary',
                    'intersection_geometry': boundary_intersection
                })
        
        return boundary_crossings
    
    def _analyze_lake_stream_connectivity(self, lakes, streams, dem) -> Dict:
        """
        Analyze stream-lake intersections and connectivity
        Based on BasinMaker definelaketypeqgis.py logic
        """
        
        connectivity_results = {
            'connected_lakes': [],
            'non_connected_lakes': [],
            'lake_inflows': {},
            'lake_outflows': {},
            'connectivity_matrix': {}
        }
        
        for idx, lake in lakes.iterrows():
            lake_id = lake.get('lake_id', idx)
            lake_geom = lake.geometry
            
            # Find streams that intersect with this lake
            intersecting_streams = streams[streams.intersects(lake_geom)]
            
            if len(intersecting_streams) > 0:
                # Connected lake - analyze inflows and outflows
                connectivity_results['connected_lakes'].append(lake_id)
                
                # Determine inflows and outflows based on flow direction
                inflows, outflows = self._classify_lake_flows(
                    lake_geom, intersecting_streams, dem
                )
                
                connectivity_results['lake_inflows'][lake_id] = inflows
                connectivity_results['lake_outflows'][lake_id] = outflows
                
            else:
                # Non-connected lake
                connectivity_results['non_connected_lakes'].append(lake_id)
                connectivity_results['lake_inflows'][lake_id] = []
                connectivity_results['lake_outflows'][lake_id] = []
        
        return connectivity_results
    
    def _classify_lake_flows(self, lake_geom, intersecting_streams, dem) -> Tuple[List, List]:
        """
        Classify streams as inflows or outflows based on elevation and flow direction
        """
        
        inflows = []
        outflows = []
        
        # Get lake boundary points
        lake_boundary = lake_geom.boundary
        
        for idx, stream in intersecting_streams.iterrows():
            stream_geom = stream.geometry
            
            # Find intersection points
            intersection = stream_geom.intersection(lake_boundary)
            
            if intersection.is_empty:
                continue
            
            # Extract intersection coordinates
            if hasattr(intersection, 'geoms'):
                # Multiple intersections
                int_points = [geom for geom in intersection.geoms if geom.geom_type == 'Point']
            else:
                # Single intersection
                int_points = [intersection] if intersection.geom_type == 'Point' else []
            
            for point in int_points:
                # Determine flow direction at intersection
                flow_direction = self._determine_flow_direction_at_point(
                    point, stream_geom, dem
                )
                
                if flow_direction == 'into_lake':
                    inflows.append({
                        'stream_id': idx,
                        'intersection_point': point,
                        'stream_segment': stream_geom
                    })
                elif flow_direction == 'out_of_lake':
                    outflows.append({
                        'stream_id': idx,
                        'intersection_point': point,
                        'stream_segment': stream_geom
                    })
        
        return inflows, outflows
    
    def _determine_flow_direction_at_point(self, point, stream_geom, dem) -> str:
        """
        Determine flow direction at a specific point using DEM gradients
        """
        
        # Get coordinates
        x, y = point.x, point.y
        
        # Sample DEM elevations in a small neighborhood
        try:
            # Transform to DEM coordinates
            dem_transform = dem['transform']
            dem_array = dem['array']
            
            # Get pixel coordinates
            row, col = rasterio.transform.rowcol(dem_transform, x, y)
            
            # Sample elevations in 3x3 neighborhood
            elevation_window = dem_array[max(0, row-1):row+2, max(0, col-1):col+2]
            
            if elevation_window.size == 0:
                return 'unknown'
            
            center_elevation = dem_array[row, col] if 0 <= row < dem_array.shape[0] and 0 <= col < dem_array.shape[1] else np.nan
            
            # Calculate gradient along stream direction
            # This is simplified - in practice would use more sophisticated flow direction analysis
            upstream_elevation = np.max(elevation_window)
            downstream_elevation = np.min(elevation_window)
            
            if center_elevation > downstream_elevation:
                return 'out_of_lake'
            elif center_elevation < upstream_elevation:
                return 'into_lake'
            else:
                return 'unknown'
                
        except Exception:
            return 'unknown'
    
    def _build_lake_routing_network(self, lakes, streams, connectivity_results, dem) -> Dict:
        """
        Build a routing network that includes lakes and streams
        """
        
        # Create network graph
        G = nx.DiGraph()
        
        # Add lake nodes
        for idx, lake in lakes.iterrows():
            lake_id = lake.get('lake_id', idx)
            G.add_node(f"lake_{lake_id}", 
                      type='lake',
                      geometry=lake.geometry,
                      area_km2=lake.geometry.area / 1e6,
                      lake_id=lake_id)
        
        # Add stream nodes and edges
        for idx, stream in streams.iterrows():
            stream_id = stream.get('stream_id', idx)
            G.add_node(f"stream_{stream_id}",
                      type='stream',
                      geometry=stream.geometry,
                      stream_id=stream_id)
        
        # Add connectivity edges based on inflows/outflows
        for lake_id in connectivity_results['connected_lakes']:
            inflows = connectivity_results['lake_inflows'][lake_id]
            outflows = connectivity_results['lake_outflows'][lake_id]
            
            # Connect inflow streams to lake
            for inflow in inflows:
                stream_id = inflow['stream_id']
                G.add_edge(f"stream_{stream_id}", f"lake_{lake_id}",
                          type='inflow',
                          intersection_point=inflow['intersection_point'])
            
            # Connect lake to outflow streams
            for outflow in outflows:
                stream_id = outflow['stream_id']
                G.add_edge(f"lake_{lake_id}", f"stream_{stream_id}",
                          type='outflow',
                          intersection_point=outflow['intersection_point'])
        
        return {
            'graph': G,
            'lake_nodes': [n for n in G.nodes() if n.startswith('lake_')],
            'stream_nodes': [n for n in G.nodes() if n.startswith('stream_')],
            'connectivity_edges': list(G.edges())
        }
    
    def _calculate_lake_outlet_hydraulics(self, lakes, streams, routing_network, dem) -> Dict:
        """
        Calculate bankfull width, depth, and discharge for lake outlets
        Based on BasinMaker calbkfwidthdepthqgis.py
        """
        
        hydraulic_properties = {}
        G = routing_network['graph']
        
        for idx, lake in lakes.iterrows():
            lake_id = lake.get('lake_id', idx)
            lake_node = f"lake_{lake_id}"
            
            if lake_node not in G:
                continue
            
            # Find outflow streams
            outflow_edges = [(u, v) for u, v in G.edges() if u == lake_node]
            
            lake_props = {
                'lake_id': lake_id,
                'area_km2': lake.geometry.area / 1e6,
                'outflow_count': len(outflow_edges),
                'bankfull_width': self.default_bkf_width,
                'bankfull_depth': self.default_bkf_depth,
                'bankfull_discharge': self.default_bkf_discharge,
                'manning_n': (self.default_manning_n_min + self.default_manning_n_max) / 2,
                'slope': self.default_min_slope,
                'outflow_streams': []
            }
            
            # Calculate hydraulics for each outflow
            for u, v in outflow_edges:
                outflow_stream_id = int(v.split('_')[1])
                
                # Find the corresponding stream geometry
                stream_row = streams[streams.index == outflow_stream_id]
                if len(stream_row) > 0:
                    stream_geom = stream_row.iloc[0].geometry
                    
                    # Calculate hydraulic properties based on drainage area and stream characteristics
                    drainage_area_km2 = lake_props['area_km2']  # Simplified
                    
                    # Empirical relationships (simplified - from literature)
                    # These should be calibrated for your region
                    bankfull_width = max(1.0, 2.3 * (drainage_area_km2 ** 0.5))
                    bankfull_depth = max(0.1, 0.27 * (drainage_area_km2 ** 0.3))
                    
                    # Calculate slope from stream geometry and DEM
                    stream_slope = self._calculate_stream_slope(stream_geom, dem)
                    
                    lake_props['outflow_streams'].append({
                        'stream_id': outflow_stream_id,
                        'bankfull_width': bankfull_width,
                        'bankfull_depth': bankfull_depth,
                        'slope': stream_slope,
                        'drainage_area_km2': drainage_area_km2
                    })
                    
                    # Use the primary outflow for lake-level properties
                    if len(lake_props['outflow_streams']) == 1:
                        lake_props['bankfull_width'] = bankfull_width
                        lake_props['bankfull_depth'] = bankfull_depth
                        lake_props['slope'] = stream_slope
            
            hydraulic_properties[lake_id] = lake_props
        
        return hydraulic_properties
    
    def _calculate_stream_slope(self, stream_geom, dem) -> float:
        """Calculate stream slope from geometry and DEM"""
        
        try:
            # Get stream coordinates
            coords = list(stream_geom.coords)
            if len(coords) < 2:
                return self.default_min_slope
            
            # Sample elevations at start and end
            start_point = coords[0]
            end_point = coords[-1]
            
            # Get elevations from DEM
            start_elev = self._sample_dem_elevation(start_point, dem)
            end_elev = self._sample_dem_elevation(end_point, dem)
            
            # Calculate distance
            start_geom = Point(start_point)
            end_geom = Point(end_point)
            distance = start_geom.distance(end_geom)
            
            if distance > 0 and not np.isnan(start_elev) and not np.isnan(end_elev):
                slope = abs(start_elev - end_elev) / distance
                return max(self.default_min_slope, slope)
            else:
                return self.default_min_slope
                
        except Exception:
            return self.default_min_slope
    
    def _sample_dem_elevation(self, point, dem) -> float:
        """Sample elevation from DEM at a specific point"""
        
        try:
            x, y = point[0], point[1]
            dem_transform = dem['transform']
            dem_array = dem['array']
            
            row, col = rasterio.transform.rowcol(dem_transform, x, y)
            
            if 0 <= row < dem_array.shape[0] and 0 <= col < dem_array.shape[1]:
                elevation = dem_array[row, col]
                if elevation != dem['nodata']:
                    return float(elevation)
            
            return np.nan
            
        except Exception:
            return np.nan
    
    def _determine_lake_topology(self, lakes, routing_network, hydraulic_properties) -> Dict:
        """
        Determine upstream/downstream relationships between lakes
        """
        
        G = routing_network['graph']
        lake_topology = {
            'upstream_lakes': {},
            'downstream_lakes': {},
            'lake_order': {},
            'flow_paths': {}
        }
        
        # Find all lake nodes
        lake_nodes = [n for n in G.nodes() if n.startswith('lake_')]
        
        for lake_node in lake_nodes:
            lake_id = int(lake_node.split('_')[1])
            
            # Find upstream lakes (lakes that flow into this lake)
            upstream = []
            for pred in G.predecessors(lake_node):
                if pred.startswith('lake_'):
                    upstream_id = int(pred.split('_')[1])
                    upstream.append(upstream_id)
            
            # Find downstream lakes (lakes that this lake flows into)
            downstream = []
            for succ in G.successors(lake_node):
                if succ.startswith('lake_'):
                    downstream_id = int(succ.split('_')[1])
                    downstream.append(downstream_id)
            
            lake_topology['upstream_lakes'][lake_id] = upstream
            lake_topology['downstream_lakes'][lake_id] = downstream
        
        # Calculate lake order (similar to stream order)
        lake_topology['lake_order'] = self._calculate_lake_order(lake_topology)
        
        # Find flow paths through lake system
        lake_topology['flow_paths'] = self._find_lake_flow_paths(G, lake_nodes)
        
        return lake_topology
    
    def _calculate_lake_order(self, topology) -> Dict:
        """Calculate lake order similar to stream order"""
        
        lake_order = {}
        
        # Start with lakes that have no upstream lakes (order 1)
        for lake_id, upstream in topology['upstream_lakes'].items():
            if len(upstream) == 0:
                lake_order[lake_id] = 1
        
        # Propagate order downstream
        changed = True
        while changed:
            changed = False
            for lake_id in topology['upstream_lakes'].keys():
                if lake_id not in lake_order:
                    upstream = topology['upstream_lakes'][lake_id]
                    if all(up_id in lake_order for up_id in upstream):
                        if len(upstream) == 0:
                            lake_order[lake_id] = 1
                        else:
                            max_upstream_order = max(lake_order[up_id] for up_id in upstream)
                            lake_order[lake_id] = max_upstream_order + 1
                        changed = True
        
        return lake_order
    
    def _find_lake_flow_paths(self, G, lake_nodes) -> List:
        """Find all flow paths through the lake system"""
        
        flow_paths = []
        
        # Find all source lakes (no upstream lakes)
        source_lakes = [node for node in lake_nodes 
                       if len(list(G.predecessors(node))) == 0]
        
        # Find all sink lakes (no downstream lakes)
        sink_lakes = [node for node in lake_nodes 
                     if len(list(G.successors(node))) == 0]
        
        # Find paths from each source to each sink
        for source in source_lakes:
            for sink in sink_lakes:
                try:
                    if nx.has_path(G, source, sink):
                        path = nx.shortest_path(G, source, sink)
                        # Filter to only include lakes in the path
                        lake_path = [node for node in path if node.startswith('lake_')]
                        if len(lake_path) > 1:
                            flow_paths.append(lake_path)
                except:
                    continue
        
        return flow_paths
    
    def _integrate_lakes_with_subbasins(self, lakes, subbasins, lake_topology, hydraulic_properties) -> Dict:
        """
        Integrate lake routing with subbasin structure for HRU generation
        """
        
        integration_results = {
            'lake_subbasin_mapping': {},
            'subbasin_lake_count': {},
            'routing_modifications': {}
        }
        
        # Map each lake to its containing subbasin(s)
        for idx, lake in lakes.iterrows():
            lake_id = lake.get('lake_id', idx)
            lake_geom = lake.geometry
            
            # Find subbasins that contain or overlap with this lake
            containing_subbasins = []
            for sub_idx, subbasin in subbasins.iterrows():
                subbasin_id = subbasin.get('SubId', sub_idx)
                if subbasin.geometry.intersects(lake_geom):
                    overlap_area = subbasin.geometry.intersection(lake_geom).area
                    overlap_pct = overlap_area / lake_geom.area * 100
                    containing_subbasins.append({
                        'subbasin_id': subbasin_id,
                        'overlap_area_m2': overlap_area,
                        'overlap_percentage': overlap_pct
                    })
            
            integration_results['lake_subbasin_mapping'][lake_id] = containing_subbasins
        
        # Count lakes per subbasin
        for sub_idx, subbasin in subbasins.iterrows():
            subbasin_id = subbasin.get('SubId', sub_idx)
            subbasin_geom = subbasin.geometry
            
            lake_count = 0
            for idx, lake in lakes.iterrows():
                if subbasin_geom.intersects(lake.geometry):
                    lake_count += 1
            
            integration_results['subbasin_lake_count'][subbasin_id] = lake_count
        
        return integration_results
    
    def _create_enhanced_lake_attributes(self, lakes, connectivity_results, 
                                       hydraulic_properties, lake_topology, 
                                       subbasin_integration) -> gpd.GeoDataFrame:
        """
        Create enhanced lake attributes for RAVEN integration
        """
        
        enhanced_lakes = lakes.copy()
        
        # Add connectivity attributes
        enhanced_lakes['is_connected'] = False
        enhanced_lakes['inflow_count'] = 0
        enhanced_lakes['outflow_count'] = 0
        enhanced_lakes['lake_order'] = 0
        enhanced_lakes['upstream_lake_count'] = 0
        enhanced_lakes['downstream_lake_count'] = 0
        
        # Add hydraulic attributes
        enhanced_lakes['bankfull_width'] = self.default_bkf_width
        enhanced_lakes['bankfull_depth'] = self.default_bkf_depth
        enhanced_lakes['bankfull_discharge'] = self.default_bkf_discharge
        enhanced_lakes['outlet_slope'] = self.default_min_slope
        enhanced_lakes['manning_n'] = (self.default_manning_n_min + self.default_manning_n_max) / 2
        
        # Add routing attributes
        enhanced_lakes['routing_type'] = 'non_connected'
        enhanced_lakes['subbasin_id'] = -1
        enhanced_lakes['lake_subbasin_overlap_pct'] = 0.0
        
        # Populate attributes for each lake
        for idx, lake in enhanced_lakes.iterrows():
            lake_id = lake.get('lake_id', idx)
            
            # Connectivity attributes
            if lake_id in connectivity_results['connected_lakes']:
                enhanced_lakes.loc[idx, 'is_connected'] = True
                enhanced_lakes.loc[idx, 'routing_type'] = 'connected'
                enhanced_lakes.loc[idx, 'inflow_count'] = len(connectivity_results['lake_inflows'].get(lake_id, []))
                enhanced_lakes.loc[idx, 'outflow_count'] = len(connectivity_results['lake_outflows'].get(lake_id, []))
            
            # Topology attributes
            if lake_id in lake_topology['lake_order']:
                enhanced_lakes.loc[idx, 'lake_order'] = lake_topology['lake_order'][lake_id]
            
            if lake_id in lake_topology['upstream_lakes']:
                enhanced_lakes.loc[idx, 'upstream_lake_count'] = len(lake_topology['upstream_lakes'][lake_id])
            
            if lake_id in lake_topology['downstream_lakes']:
                enhanced_lakes.loc[idx, 'downstream_lake_count'] = len(lake_topology['downstream_lakes'][lake_id])
            
            # Hydraulic attributes
            if lake_id in hydraulic_properties:
                props = hydraulic_properties[lake_id]
                enhanced_lakes.loc[idx, 'bankfull_width'] = props['bankfull_width']
                enhanced_lakes.loc[idx, 'bankfull_depth'] = props['bankfull_depth']
                enhanced_lakes.loc[idx, 'bankfull_discharge'] = props['bankfull_discharge']
                enhanced_lakes.loc[idx, 'outlet_slope'] = props['slope']
                enhanced_lakes.loc[idx, 'manning_n'] = props['manning_n']
            
            # Subbasin integration attributes
            if lake_id in subbasin_integration['lake_subbasin_mapping']:
                mappings = subbasin_integration['lake_subbasin_mapping'][lake_id]
                if mappings:
                    # Use the subbasin with highest overlap
                    best_mapping = max(mappings, key=lambda x: x['overlap_percentage'])
                    enhanced_lakes.loc[idx, 'subbasin_id'] = best_mapping['subbasin_id']
                    enhanced_lakes.loc[idx, 'lake_subbasin_overlap_pct'] = best_mapping['overlap_percentage']
        
        return enhanced_lakes
    
    def _save_results(self, enhanced_lakes, routing_network, hydraulic_properties, lake_topology) -> List[str]:
        """Save all results to files"""
        
        output_files = []
        
        # Save enhanced lakes
        enhanced_lakes_file = self.workspace_dir / "enhanced_lakes.shp"
        enhanced_lakes.to_file(enhanced_lakes_file)
        output_files.append(str(enhanced_lakes_file))
        
        # Save connected and non-connected lakes separately
        connected_lakes = enhanced_lakes[enhanced_lakes['is_connected'] == True]
        if len(connected_lakes) > 0:
            connected_file = self.workspace_dir / "connected_lakes_enhanced.shp"
            connected_lakes.to_file(connected_file)
            output_files.append(str(connected_file))
        
        non_connected_lakes = enhanced_lakes[enhanced_lakes['is_connected'] == False]
        if len(non_connected_lakes) > 0:
            non_connected_file = self.workspace_dir / "non_connected_lakes_enhanced.shp"
            non_connected_lakes.to_file(non_connected_file)
            output_files.append(str(non_connected_file))
        
        # Save hydraulic properties as CSV
        hydraulic_df = pd.DataFrame.from_dict(hydraulic_properties, orient='index')
        hydraulic_file = self.workspace_dir / "lake_hydraulic_properties.csv"
        hydraulic_df.to_csv(hydraulic_file)
        output_files.append(str(hydraulic_file))
        
        # Save lake topology as JSON
        import json
        topology_file = self.workspace_dir / "lake_topology.json"
        with open(topology_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_topology = {}
            for key, value in lake_topology.items():
                if isinstance(value, dict):
                    json_topology[key] = {str(k): v for k, v in value.items()}
                else:
                    json_topology[key] = value
            json.dump(json_topology, f, indent=2)
        output_files.append(str(topology_file))
        
        return output_files
    
    def _generate_statistics(self, enhanced_lakes, lake_topology) -> Dict:
        """Generate summary statistics"""
        
        total_lakes = len(enhanced_lakes)
        connected_lakes = len(enhanced_lakes[enhanced_lakes['is_connected'] == True])
        non_connected_lakes = total_lakes - connected_lakes
        
        total_area_km2 = enhanced_lakes.geometry.area.sum() / 1e6
        connected_area_km2 = enhanced_lakes[enhanced_lakes['is_connected'] == True].geometry.area.sum() / 1e6
        
        return {
            'total_lakes': total_lakes,
            'connected_lakes': connected_lakes,
            'non_connected_lakes': non_connected_lakes,
            'total_area_km2': float(total_area_km2),
            'connected_area_km2': float(connected_area_km2),
            'non_connected_area_km2': float(total_area_km2 - connected_area_km2),
            'avg_lake_area_km2': float(total_area_km2 / total_lakes) if total_lakes > 0 else 0,
            'max_lake_order': max(lake_topology['lake_order'].values()) if lake_topology['lake_order'] else 0,
            'flow_paths_count': len(lake_topology['flow_paths'])
        }
