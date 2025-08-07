"""
HydroSHEDS to BasinMaker Format Adapter

This module adapts HydroRIVERS/HydroLAKES data to work with BasinMaker routing product workflows.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import geopandas as gpd
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

class HydroShedsAdapter:
    """
    Adapter to convert HydroSHEDS data format to BasinMaker routing product format
    
    MAPPING:
    - HYRIV_ID -> SubId
    - NEXT_DOWN -> DowSubId (0 -> -1 for outlets)
    - LENGTH_KM -> RivLength  
    - ORD_STRA -> Strahler
    - HYBAS_L12 -> Basin identifier
    - Hylak_id -> Lake ID
    - Pour_long/Pour_lat -> Lake outlet coordinates
    """
    
    def __init__(self):
        self.rivers_file = Path("data/canadian/rivers/HydroRIVERS_v10_na_shp/HydroRIVERS_v10_na.shp")
        self.lakes_file = Path("data/canadian/lakes/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp")
        self.canadian_gpkg = Path("data/canadian/canadian_hydro.gpkg")
        
        # Setup logging
        import logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def validate_routing_product_availability(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Check if HydroSHEDS data covers the given coordinates
        """
        
        try:
            # Check if data files exist
            if not self.rivers_file.exists():
                return {
                    'available': False,
                    'error': f'HydroRIVERS file not found: {self.rivers_file}'
                }
            
            if not self.lakes_file.exists():
                return {
                    'available': False,
                    'error': f'HydroLAKES file not found: {self.lakes_file}'
                }
            
            # Check geographic coverage with a bounding box around the point
            buffer = 0.5  # degrees
            bbox = (
                longitude - buffer,
                latitude - buffer, 
                longitude + buffer,
                latitude + buffer
            )
            
            # Check if rivers exist in the area
            rivers_gdf = gpd.read_file(self.rivers_file, bbox=bbox)
            
            if len(rivers_gdf) == 0:
                return {
                    'available': False,
                    'error': f'No river network found for coordinates ({latitude}, {longitude})'
                }
            
            return {
                'available': True,
                'region': 'hydrosheds_north_america',
                'product_path': self.rivers_file.parent,
                'version': 'hydrosheds_v10',
                'river_segments': len(rivers_gdf),
                'max_strahler_order': rivers_gdf['ORD_STRA'].max()
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': f'Error checking HydroSHEDS availability: {str(e)}'
            }
    
    def find_target_subbasin(self, latitude: float, longitude: float) -> int:
        """
        Find the target subbasin (river segment) that contains or is nearest to the outlet point
        """
        
        try:
            from shapely.geometry import Point
            
            point = Point(longitude, latitude)
            buffer = 0.1  # degrees
            bbox = (longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer)
            
            # Load rivers in the area
            rivers_gdf = gpd.read_file(self.rivers_file, bbox=bbox)
            
            if len(rivers_gdf) == 0:
                raise ValueError(f"No rivers found near coordinates ({latitude}, {longitude})")
            
            # Find the nearest river segment
            distances = rivers_gdf.geometry.distance(point)
            nearest_idx = distances.idxmin()
            target_hyriv_id = rivers_gdf.iloc[nearest_idx]['HYRIV_ID']
            
            # Return HYRIV_ID as the target subbasin ID
            return int(target_hyriv_id)
            
        except Exception as e:
            raise RuntimeError(f"Error finding target subbasin: {str(e)}")
    
    def extract_upstream_network(self, target_subbasin_id: int, output_folder: Path) -> Dict[str, Any]:
        """
        Extract upstream river network and lakes using HydroSHEDS connectivity
        """
        
        try:
            # Create output folder
            output_folder.mkdir(exist_ok=True, parents=True)
            
            # Find the target river segment
            target_bbox = self._get_search_bbox_for_subbasin(target_subbasin_id)
            
            # Load river network - try bbox first, then expand search
            try:
                rivers_gdf = gpd.read_file(self.rivers_file, bbox=target_bbox)
                
                if target_subbasin_id not in rivers_gdf['HYRIV_ID'].values:
                    # Expand search progressively if not found
                    self.logger.info(f"Target subbasin {target_subbasin_id} not found in initial bbox, expanding search...")
                    
                    # Try larger samples
                    for sample_size in [50000, 100000, 200000]:
                        rivers_gdf = gpd.read_file(self.rivers_file, rows=sample_size)
                        rivers_subset = rivers_gdf[rivers_gdf['HYRIV_ID'] == target_subbasin_id]
                        
                        if len(rivers_subset) > 0:
                            # Found it! Get the basin for more targeted search
                            target_basin = rivers_subset.iloc[0]['HYBAS_L12']
                            rivers_gdf = rivers_gdf[rivers_gdf['HYBAS_L12'] == target_basin]
                            self.logger.info(f"Found target subbasin in basin {target_basin} with {len(rivers_gdf)} river segments")
                            break
                    else:
                        raise ValueError(f"Target subbasin {target_subbasin_id} not found in HydroRIVERS dataset")
                        
            except Exception as e:
                if "Target subbasin" in str(e):
                    raise e
                else:
                    raise ValueError(f"Error reading HydroRIVERS data: {str(e)}")
            
            # Extract upstream network using NEXT_DOWN connectivity
            upstream_ids = self._trace_upstream_network(rivers_gdf, target_subbasin_id)
            upstream_rivers = rivers_gdf[rivers_gdf['HYRIV_ID'].isin(upstream_ids)]
            
            # Convert to BasinMaker format
            basinmaker_rivers = self._convert_rivers_to_basinmaker_format(upstream_rivers)
            
            # Generate simplified catchments from river network
            basinmaker_catchments = self._generate_catchments_from_rivers(basinmaker_rivers)
            
            # Extract lakes in the watershed
            watershed_bounds = upstream_rivers.total_bounds
            lakes_bbox = (watershed_bounds[0], watershed_bounds[1], watershed_bounds[2], watershed_bounds[3])
            lakes_gdf = gpd.read_file(self.lakes_file, bbox=lakes_bbox)
            
            # Convert lakes to BasinMaker format
            basinmaker_lakes = self._convert_lakes_to_basinmaker_format(lakes_gdf) if len(lakes_gdf) > 0 else None
            
            # Save extracted data
            rivers_file = output_folder / "extracted_rivers.shp"
            catchments_file = output_folder / "extracted_catchments.shp"
            
            basinmaker_rivers.to_file(rivers_file)
            basinmaker_catchments.to_file(catchments_file)
            
            lakes_file = None
            if basinmaker_lakes is not None:
                lakes_file = output_folder / "extracted_lakes.shp"
                basinmaker_lakes.to_file(lakes_file)
            
            return {
                'success': True,
                'extracted_rivers': str(rivers_file),
                'extracted_catchments': str(catchments_file),
                'extracted_lakes': str(lakes_file) if lakes_file else None,
                'extracted_gauges': None,  # No gauge data in HydroSHEDS
                'subbasin_count': len(basinmaker_catchments),
                'total_area_km2': basinmaker_catchments['BasArea'].sum() / 1e6,  # Convert m² to km²
                'total_stream_length_km': basinmaker_rivers['RivLength'].sum(),
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Upstream network extraction failed: {str(e)}'
            }
    
    def _get_search_bbox_for_subbasin(self, subbasin_id: int) -> Tuple[float, float, float, float]:
        """Get a reasonable bounding box to search for a subbasin"""
        # Use a large bounding box covering most of North America
        # This ensures we can find subbasins across different regions
        return (-170.0, 25.0, -50.0, 85.0)
    
    def _trace_upstream_network(self, rivers_gdf: gpd.GeoDataFrame, target_id: int) -> set:
        """Trace all upstream river segments using NEXT_DOWN connectivity"""
        
        # Build connectivity lookup
        downstream_map = dict(zip(rivers_gdf['HYRIV_ID'], rivers_gdf['NEXT_DOWN']))
        upstream_map = {}
        
        for hyriv_id, next_down in downstream_map.items():
            if next_down != 0:  # 0 means outlet
                if next_down not in upstream_map:
                    upstream_map[next_down] = []
                upstream_map[next_down].append(hyriv_id)
        
        # Trace upstream from target
        upstream_ids = {target_id}
        to_process = [target_id]
        
        while to_process:
            current_id = to_process.pop(0)
            if current_id in upstream_map:
                for upstream_id in upstream_map[current_id]:
                    if upstream_id not in upstream_ids:
                        upstream_ids.add(upstream_id)
                        to_process.append(upstream_id)
        
        return upstream_ids
    
    def _convert_rivers_to_basinmaker_format(self, rivers_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convert HydroRIVERS format to BasinMaker river format"""
        
        # Create BasinMaker-compatible columns
        basinmaker_rivers = rivers_gdf.copy()
        
        # Map HydroSHEDS columns to BasinMaker columns
        basinmaker_rivers['SubId'] = basinmaker_rivers['HYRIV_ID']
        basinmaker_rivers['DowSubId'] = basinmaker_rivers['NEXT_DOWN'].replace(0, -1)  # 0 -> -1 for outlets
        basinmaker_rivers['RivLength'] = basinmaker_rivers['LENGTH_KM'] * 1000  # Convert km to m
        
        # PRESERVE HYBAS_L12 for basin polygon linkage
        if 'HYBAS_L12' in basinmaker_rivers.columns:
            basinmaker_rivers['HYBAS_L12'] = basinmaker_rivers['HYBAS_L12']
        basinmaker_rivers['Strahler'] = basinmaker_rivers['ORD_STRA']
        
        # Validate required HydroRIVERS attributes are present
        required_cols = ['LENGTH_KM', 'CATCH_SKM', 'DIS_AV_CMS', 'ORD_STRA']
        missing_cols = [col for col in required_cols if col not in basinmaker_rivers.columns]
        if missing_cols:
            raise ValueError(f"Missing required HydroRIVERS columns: {missing_cols} - cannot proceed without real data")
        
        # Check for missing data in critical columns
        if basinmaker_rivers['LENGTH_KM'].isna().any():
            raise ValueError("Missing LENGTH_KM data in HydroRIVERS - cannot calculate RivLength")
        if basinmaker_rivers['CATCH_SKM'].isna().any():
            raise ValueError("Missing CATCH_SKM data in HydroRIVERS - cannot calculate BasArea")
        if basinmaker_rivers['DIS_AV_CMS'].isna().any():
            raise ValueError("Missing DIS_AV_CMS data in HydroRIVERS - cannot calculate Q_Mean")
            
        # Use real data only - NO DEFAULT VALUES
        # River slope must be calculated from DEM - fail if not available
        if 'RIV_SLOPE' in basinmaker_rivers.columns and not basinmaker_rivers['RIV_SLOPE'].isna().all():
            basinmaker_rivers['RivSlope'] = basinmaker_rivers['RIV_SLOPE']
        else:
            raise ValueError("River slope data not available in HydroRIVERS - requires DEM-based calculation")
            
        basinmaker_rivers['BasArea'] = basinmaker_rivers['CATCH_SKM'] * 1e6  # Convert km² to m²
        
        # Basin slope and aspect must be calculated from DEM - fail if not available  
        if 'BAS_SLOPE' in basinmaker_rivers.columns and not basinmaker_rivers['BAS_SLOPE'].isna().all():
            basinmaker_rivers['BasSlope'] = basinmaker_rivers['BAS_SLOPE']
        else:
            raise ValueError("Basin slope data not available in HydroRIVERS - requires DEM-based calculation")
            
        if 'BAS_ASPECT' in basinmaker_rivers.columns and not basinmaker_rivers['BAS_ASPECT'].isna().all():
            basinmaker_rivers['BasAspect'] = basinmaker_rivers['BAS_ASPECT']
        else:
            raise ValueError("Basin aspect data not available in HydroRIVERS - requires DEM-based calculation")
        
        # Channel geometry must be calculated from hydraulic analysis - fail if not available
        if 'BKF_WIDTH' in basinmaker_rivers.columns and not basinmaker_rivers['BKF_WIDTH'].isna().all():
            basinmaker_rivers['BkfWidth'] = basinmaker_rivers['BKF_WIDTH']
        else:
            raise ValueError("Channel width data not available in HydroRIVERS - requires hydraulic geometry analysis")
            
        if 'BKF_DEPTH' in basinmaker_rivers.columns and not basinmaker_rivers['BKF_DEPTH'].isna().all():
            basinmaker_rivers['BkfDepth'] = basinmaker_rivers['BKF_DEPTH']
        else:
            raise ValueError("Channel depth data not available in HydroRIVERS - requires hydraulic geometry analysis")
        
        # Lake attributes - only set if real lake data exists
        basinmaker_rivers['IsLake'] = -9999  # Will be updated if lakes are linked
        basinmaker_rivers['HyLakeId'] = -9999
        basinmaker_rivers['LakeVol'] = -9999.0
        basinmaker_rivers['LakeDepth'] = -9999.0
        basinmaker_rivers['LakeArea'] = -9999.0
        basinmaker_rivers['Laketype'] = -9999
        basinmaker_rivers['IsObs'] = -9999
        
        # Elevation must be calculated from DEM - fail if not available
        if 'MEAN_ELEV' in basinmaker_rivers.columns and not basinmaker_rivers['MEAN_ELEV'].isna().all():
            basinmaker_rivers['MeanElev'] = basinmaker_rivers['MEAN_ELEV']
        else:
            raise ValueError("Mean elevation data not available in HydroRIVERS - requires DEM-based calculation")
            
        # Manning's n values must be estimated from landcover - fail if not available
        if 'FLOOD_N' in basinmaker_rivers.columns and not basinmaker_rivers['FLOOD_N'].isna().all():
            basinmaker_rivers['FloodP_n'] = basinmaker_rivers['FLOOD_N']
        else:
            raise ValueError("Floodplain Manning's n not available - requires landcover analysis")
            
        # Use actual discharge data from HydroRIVERS
        basinmaker_rivers['Q_Mean'] = basinmaker_rivers['DIS_AV_CMS']
        
        # Channel Manning's n must be estimated from substrate analysis - fail if not available
        if 'CH_N' in basinmaker_rivers.columns and not basinmaker_rivers['CH_N'].isna().all():
            basinmaker_rivers['Ch_n'] = basinmaker_rivers['CH_N']
        else:
            raise ValueError("Channel Manning's n not available - requires substrate/roughness analysis")
            
        basinmaker_rivers['DA'] = basinmaker_rivers['CATCH_SKM'] * 1e6  # Drainage area in m²
        
        # Keep essential columns for BasinMaker compatibility (including HYBAS_L12 for basin linkage)
        essential_cols = [
            'SubId', 'DowSubId', 'RivSlope', 'RivLength', 'BasSlope', 'BasAspect', 
            'BasArea', 'BkfWidth', 'BkfDepth', 'IsLake', 'HyLakeId', 'LakeVol', 
            'LakeDepth', 'LakeArea', 'Laketype', 'IsObs', 'MeanElev', 'FloodP_n', 
            'Q_Mean', 'Ch_n', 'DA', 'Strahler', 'HYBAS_L12', 'geometry'
        ]
        
        # Filter to only columns that actually exist
        existing_cols = [col for col in essential_cols if col in basinmaker_rivers.columns]
        return basinmaker_rivers[existing_cols]
    
    def _generate_catchments_from_rivers(self, rivers_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Extract real catchment polygons from HydroBASINS using HYBAS_L12 codes"""
        
        try:
            # Get HYBAS_L12 codes from the river network
            hybas_codes = rivers_gdf['HYBAS_L12'].unique()
            self.logger.info(f"Looking for {len(hybas_codes)} unique basin polygons")
            
            # Load Canadian HydroBASINS Level 12 from geopackage
            if not self.canadian_gpkg.exists():
                raise FileNotFoundError(f"Canadian geopackage not found: {self.canadian_gpkg}")
            
            # Load all Canadian basins (Level 12 for detailed watersheds)
            canadian_basins = gpd.read_file(self.canadian_gpkg, layer='hydrobasins_level12')
            self.logger.info(f"Loaded {len(canadian_basins)} Canadian basin polygons")
            
            # Filter to basins that match our river network
            matching_basins = canadian_basins[canadian_basins['HYBAS_ID'].isin(hybas_codes)]
            self.logger.info(f"Found {len(matching_basins)} matching basin polygons")
            
            if len(matching_basins) == 0:
                self.logger.error("No matching basins found - no synthetic fallback provided")
                raise ValueError("No matching basins found")
            
            # Convert HydroBASINS format to BasinMaker catchment format
            catchments_data = []
            
            for _, basin in matching_basins.iterrows():
                # Find the corresponding river segment for this basin
                matching_rivers = rivers_gdf[rivers_gdf['HYBAS_L12'] == basin['HYBAS_ID']]
                
                if len(matching_rivers) > 0:
                    # Use the first matching river for attributes
                    river = matching_rivers.iloc[0]
                    
                    # Create catchment with real basin polygon and river attributes
                    # Use the river's attributes (already converted to BasinMaker format)
                    catchment_data = {
                        'SubId': river['SubId'],  # River segment ID (already converted)
                        'DowSubId': river['DowSubId'],  # Downstream connection (already converted)
                        'RivLength': river['RivLength'],  # River length (already converted)
                        'Strahler': river['Strahler'],  # Stream order (already converted)
                        'BasArea': basin['SUB_AREA'] * 1e6,  # Use REAL basin area (km² to m²)
                        'BkfWidth': river['BkfWidth'],  # Use existing estimates
                        'BkfDepth': river['BkfDepth'],  # Use existing estimates
                        'Q_Mean': river['Q_Mean'],  # Use existing discharge
                        'HYBAS_ID': basin['HYBAS_ID'],  # Basin identifier
                        'UP_AREA': basin['UP_AREA'],  # Upstream area
                        'geometry': basin.geometry  # REAL watershed polygon!
                    }
                    
                    catchments_data.append(catchment_data)
            
            catchments_gdf = gpd.GeoDataFrame(catchments_data, crs=canadian_basins.crs)
            self.logger.info(f"Created {len(catchments_gdf)} real watershed catchments")
            
            return catchments_gdf
            
        except Exception as e:
            self.logger.error(f"Error extracting real catchments: {e}")
            raise Exception(f"Real catchment extraction failed: {e} - no synthetic fallback provided")
    
    
    def _convert_lakes_to_basinmaker_format(self, lakes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convert HydroLAKES format to BasinMaker lake format"""
        
        if len(lakes_gdf) == 0:
            return None
        
        # Create BasinMaker-compatible lake data
        basinmaker_lakes = lakes_gdf.copy()
        
        # Map HydroLAKES columns to BasinMaker lake columns
        basinmaker_lakes['SubId'] = basinmaker_lakes['Hylak_id']
        basinmaker_lakes['Lake_Cat'] = 1  # Connected lake type
        basinmaker_lakes['LakeArea'] = basinmaker_lakes['Lake_area']
        basinmaker_lakes['LakeVol'] = basinmaker_lakes['Vol_total'] / 1000  # Convert to km³
        basinmaker_lakes['LakeDepth'] = basinmaker_lakes['Depth_avg']
        basinmaker_lakes['Pour_long'] = basinmaker_lakes['Pour_long']
        basinmaker_lakes['Pour_lat'] = basinmaker_lakes['Pour_lat']
        
        return basinmaker_lakes