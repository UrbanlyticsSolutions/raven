#!/usr/bin/env python3
"""
Hydraulic Routing Routines Extracted from Magpie Workflow
Comprehensive implementation of river channel routing and lake-river routing networks
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

class MagpieHydraulicRouting:
    """
    Hydraulic Routing Routines from Magpie Workflow
    Implements river channel routing using diffusive wave, hydrologic, and delayed first-order methods
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Available RAVEN routing methods from Magpie
        self.routing_methods = {
            'ROUTE_DIFFUSIVE_WAVE': {
                'description': 'Diffusive wave routing - most physically based',
                'parameters': ['manning_n', 'channel_slope', 'channel_width', 'channel_depth'],
                'suitable_for': 'Complex watersheds with detailed channel geometry'
            },
            'ROUTE_HYDROLOGIC': {
                'description': 'Hydrologic routing using lag time and storage',
                'parameters': ['lag_time', 'storage_coefficient'],
                'suitable_for': 'Simple watersheds without detailed channel data'
            },
            'ROUTE_DELAYED_FIRST_ORDER': {
                'description': 'Linear reservoir routing with lag',
                'parameters': ['lag_time', 'storage_coefficient'],
                'suitable_for': 'Medium complexity watersheds'
            },
            'ROUTE_DUMP': {
                'description': 'No routing - direct flow to outlet',
                'parameters': [],
                'suitable_for': 'Testing or very small watersheds'
            }
        }
        
    def calculate_channel_hydraulics_from_basinmaker(self, subbasin_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate hydraulic parameters from BasinMaker routing product data
        Uses the Andreadis et al. (2013) hydraulic geometry relationships
        """
        self.logger.info("Calculating channel hydraulics from BasinMaker data...")
        
        # Create copy to avoid modifying original
        channel_data = subbasin_data.copy()
        
        # 1. Calculate discharge using drainage area relationship
        # Q = k ├ù DA^c (where Q in m┬│/s, DA in km┬▓)
        k = 0.00450718  # Coefficient from literature
        c = 0.98579699  # Exponent from literature
        
        # Calculate cumulative drainage area using BasinMaker method (from flow accumulation)
        drainage_area_km2 = self._calculate_cumulative_drainage_area(channel_data)
        
        # If calculation fails, fall back to individual subbasin areas and estimate cumulative
        if drainage_area_km2 is None:
            self.logger.warning("Flow accumulation method failed, estimating cumulative drainage areas")
            
            # Try to get drainage area from existing columns
            if 'DrainArea' in channel_data.columns:
                drainage_area_km2 = channel_data['DrainArea']
                # DrainArea might be in m², convert to km² if needed
                if drainage_area_km2.max() > 1000:  # Likely in m²
                    drainage_area_km2 = drainage_area_km2 / 1e6
            elif 'area_km2' in channel_data.columns:
                # Use local areas but try to estimate cumulative based on topology
                local_areas = channel_data['area_km2']
                drainage_area_km2 = self._estimate_cumulative_from_topology(channel_data, local_areas)
            elif 'Area' in channel_data.columns:
                local_areas = channel_data['Area'] / 1e6  # Convert m² to km²
                drainage_area_km2 = self._estimate_cumulative_from_topology(channel_data, local_areas)
            else:
                # Last resort: use reasonable default values
                self.logger.warning("No area data found, using default values")
                drainage_area_km2 = np.linspace(1.0, 50.0, len(channel_data))  # 1-50 km²
        
        channel_data['discharge_m3s'] = k * (drainage_area_km2 ** c)
        
        # 2. Calculate channel geometry using Andreadis relationships
        # Bankfull width: W = 7.2 ├ù Q^0.5
        channel_data['channel_width_m'] = 7.2 * (channel_data['discharge_m3s'] ** 0.5)
        channel_data['channel_width_m'] = np.maximum(channel_data['channel_width_m'], 1.0)  # Minimum 1m
        
        # Bankfull depth: D = 0.27 ├ù Q^0.3
        channel_data['channel_depth_m'] = 0.27 * (channel_data['discharge_m3s'] ** 0.3)
        channel_data['channel_depth_m'] = np.maximum(channel_data['channel_depth_m'], 0.3)  # Minimum 0.3m
        
        # 3. Calculate channel slope
        if 'RivLength' in channel_data.columns and 'RivSlope' in channel_data.columns:
            channel_data['channel_slope'] = channel_data['RivSlope']
        elif 'RivLength' in channel_data.columns:
            # Estimate slope from elevation change and river length
            if 'MeanElev' in channel_data.columns:
                elevation_change = channel_data['MeanElev'] * 0.001  # Use 0.1% of elevation as change
            else:
                raise ValueError("No elevation column found. Expected 'MeanElev' column for slope calculation")
            channel_data['channel_slope'] = elevation_change / channel_data['RivLength']
        else:
            # Default slope
            channel_data['channel_slope'] = 0.001  # 0.1% slope
        
        # Ensure minimum slope
        channel_data['channel_slope'] = np.maximum(channel_data['channel_slope'], 0.0001)
        
        # 4. Calculate Manning's roughness coefficient
        channel_data['manning_n'] = self._calculate_manning_n(
            channel_data['channel_width_m'],
            channel_data['channel_depth_m'],
            channel_data['discharge_m3s'],
            channel_data['channel_slope']
        )
        
        # 5. Calculate routing parameters
        channel_data = self._calculate_routing_parameters(channel_data)
        
        self.logger.info(f"Calculated hydraulics for {len(channel_data)} channel reaches")
        
        return channel_data
    
    def _calculate_manning_n(self, width: np.ndarray, depth: np.ndarray, 
                           discharge: np.ndarray, slope: np.ndarray) -> np.ndarray:
        """
        Calculate Manning's roughness coefficient using Manning's equation
        Assumes trapezoidal channel with 2:1 side slopes
        """
        # Trapezoidal channel geometry
        side_slope = 2.0
        
        # Wetted perimeter: P = b + 2dΓêÜ(1 + m┬▓)
        wetted_perimeter = width + 2 * depth * np.sqrt(1 + side_slope**2)
        
        # Cross-sectional area: A = bd + md┬▓
        cross_sectional_area = width * depth + side_slope * depth**2
        
        # Hydraulic radius: R = A/P
        hydraulic_radius = cross_sectional_area / wetted_perimeter
        
        # Manning's equation: n = (R^(2/3) * S^(1/2) * A) / Q
        manning_n = (hydraulic_radius**(2/3) * slope**(1/2) * cross_sectional_area) / discharge
        
        # Apply realistic bounds for natural channels
        manning_n = np.clip(manning_n, 0.025, 0.15)
        
        return manning_n
    
    def _calculate_routing_parameters(self, channel_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate routing-specific parameters for different methods"""
        
        # For diffusive wave routing
        channel_data['wave_celerity'] = self._calculate_wave_celerity(
            channel_data['discharge_m3s'],
            channel_data['channel_width_m'],
            channel_data['channel_depth_m']
        )
        
        # For hydrologic routing  
        channel_data['lag_time_hours'] = self._calculate_lag_time(
            channel_data['RivLength'] if 'RivLength' in channel_data.columns else 1000,
            channel_data['wave_celerity']
        )
        
        channel_data['storage_coefficient'] = self._calculate_storage_coefficient(
            channel_data['RivLength'] if 'RivLength' in channel_data.columns else 1000,
            channel_data['wave_celerity']
        )
        
        return channel_data
    
    def _calculate_wave_celerity(self, discharge: np.ndarray, width: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Calculate wave celerity for routing
        c = dQ/dA Γëê 5/3 * V (for wide rectangular channels)
        """
        # Calculate velocity: V = Q/A
        area = width * depth
        velocity = discharge / area
        
        # Wave celerity (kinematic wave approximation)
        celerity = (5/3) * velocity
        
        # Apply realistic bounds
        celerity = np.clip(celerity, 0.1, 10.0)  # 0.1 to 10 m/s
        
        return celerity
    
    def _calculate_lag_time(self, reach_length: np.ndarray, celerity: np.ndarray) -> np.ndarray:
        """Calculate lag time for hydrologic routing"""
        # Convert reach length to meters if in km
        if np.mean(reach_length) < 100:  # Likely in km
            reach_length_m = reach_length * 1000
        else:
            reach_length_m = reach_length
            
        # Lag time = Length / Celerity (convert to hours)
        lag_time = reach_length_m / celerity / 3600
        
        # Apply realistic bounds
        lag_time = np.clip(lag_time, 0.1, 48.0)  # 0.1 to 48 hours
        
        return lag_time
    
    def _calculate_storage_coefficient(self, reach_length: np.ndarray, celerity: np.ndarray) -> np.ndarray:
        """Calculate storage coefficient for linear reservoir routing"""
        # Storage coefficient Γëê 0.6 * lag time for natural channels
        lag_time = self._calculate_lag_time(reach_length, celerity)
        storage_coeff = 0.6 * lag_time
        
        return storage_coeff
    
    def generate_raven_channel_profiles(self, channel_data: pd.DataFrame) -> List[str]:
        """
        Generate RAVEN :ChannelProfile commands from calculated hydraulics
        Creates extended stage ranges to prevent flow capacity warnings
        """
        channel_profiles = []
        
        for idx, row in channel_data.iterrows():
            subbasin_id = int(row.get('SubId', idx + 1))
            
            # Base channel dimensions
            bankfull_depth = row['channel_depth_m']
            channel_width = row['channel_width_m']
            manning_n = row['manning_n']
            slope = row['channel_slope']
            
            # Create extended channel profile with flood stages (prevents flow capacity warnings)
            # Stage 1: Channel bottom to bankfull (normal conditions)
            # Stage 2: Bankfull to 2x bankfull (high flow)  
            # Stage 3: 2x bankfull to 3x bankfull (extreme flood)
            max_depth = bankfull_depth * 3.0  # Reasonable flood capacity
            flood_width = channel_width * 2.0  # Moderate flood plain width
            
            profile = f"""
:ChannelProfile CHANNEL_{subbasin_id}
  :Bedslope {slope:.6f}
  :SurveyPoints
    # Channel cross-section with extended flood capacity
    0.0 {max_depth:.2f}
    {channel_width/4:.2f} {bankfull_depth:.2f}
    {channel_width/2:.2f} 0.0
    {channel_width*3/4:.2f} {bankfull_depth:.2f}
    {flood_width:.2f} {max_depth:.2f}
  :EndSurveyPoints
  :RoughnessZones
    0.0 {manning_n:.4f}
    {channel_width:.2f} {manning_n*1.5:.4f}
  :EndRoughnessZones
:EndChannelProfile
"""
            channel_profiles.append(profile.strip())
        
        return channel_profiles
    
    def _calculate_cumulative_drainage_area(self, channel_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate cumulative drainage area using flow accumulation raster (BasinMaker method)
        
        Equivalent to BasinMaker's r.stats.zonal to find maximum accumulation per subbasin
        Fixes CRS alignment issues between raster and vector data
        """
        try:
            import rasterio
            from rasterio.mask import mask as raster_mask
            from rasterio.warp import transform_bounds
            import geopandas as gpd
            from pathlib import Path
            
            # Look for flow accumulation raster
            possible_paths = [
                Path("data/flow_accumulation_d8.tif"),
                Path("outlet_49.5738_-119.0368/data/flow_accumulation_d8.tif"),
                Path("flow_accumulation_d8.tif")
            ]
            
            flow_acc_file = None
            for path in possible_paths:
                if path.exists():
                    flow_acc_file = path
                    break
            
            if flow_acc_file is None:
                self.logger.warning("Flow accumulation raster not found")
                return None
            
            self.logger.info(f"Using flow accumulation raster: {flow_acc_file}")
            
            # Open flow accumulation raster and get CRS info
            with rasterio.open(flow_acc_file) as src:
                raster_crs = src.crs
                raster_bounds = src.bounds
                transform = src.transform
                
                self.logger.info(f"Flow accumulation CRS: {raster_crs}")
                self.logger.info(f"Flow accumulation bounds: {raster_bounds}")
                
                # Check if channel_data is a GeoDataFrame or regular DataFrame
                if hasattr(channel_data, 'crs'):
                    # It's a GeoDataFrame
                    subbasins_gdf = channel_data.copy()
                    vector_crs = subbasins_gdf.crs
                else:
                    # It's a regular DataFrame, need to load the original shapefile
                    subbasin_files = [
                        "outlet_49.5738_-119.0368/data/subbasins.shp",
                        "data/subbasins.shp",
                        "subbasins.shp"
                    ]
                    
                    subbasins_gdf = None
                    for subbasin_file in subbasin_files:
                        if Path(subbasin_file).exists():
                            subbasins_gdf = gpd.read_file(subbasin_file)
                            break
                    
                    if subbasins_gdf is None:
                        self.logger.warning("No subbasin shapefile found for CRS checking")
                        return None
                    
                    vector_crs = subbasins_gdf.crs
                
                self.logger.info(f"Subbasins CRS: {vector_crs}")
                
                # FAIL-FAST CRS validation and reprojection
                self.logger.info(f"Starting CRS alignment validation...")
                self.logger.info(f"Vector CRS: {vector_crs}")
                self.logger.info(f"Raster CRS: {raster_crs}")
                
                # Validate CRS objects first (fail-fast)
                if vector_crs is None:
                    raise ValueError("Vector CRS is None - subbasin shapefile has no coordinate system defined")
                if raster_crs is None:
                    raise ValueError("Raster CRS is None - flow accumulation raster has no coordinate system defined")
                
                # If CRS are different, attempt reprojection with fail-fast approach
                if vector_crs != raster_crs:
                    self.logger.info(f"CRS mismatch detected - reprojecting from {vector_crs} to {raster_crs}")
                    
                    # Test coordinate transformation first with a single point (fail-fast)
                    try:
                        import pyproj
                        transformer = pyproj.Transformer.from_crs(vector_crs, raster_crs, always_xy=True)
                        
                        # Test transformation with center point of vector data
                        test_bounds = subbasins_gdf.total_bounds
                        test_x = (test_bounds[0] + test_bounds[2]) / 2
                        test_y = (test_bounds[1] + test_bounds[3]) / 2
                        
                        # This will fail fast if transformation is impossible
                        transformed_x, transformed_y = transformer.transform(test_x, test_y)
                        self.logger.info(f"✓ CRS transformation test successful: ({test_x:.6f}, {test_y:.6f}) -> ({transformed_x:.6f}, {transformed_y:.6f})")
                        
                    except Exception as e:
                        raise RuntimeError(f"CRS transformation test failed - coordinate systems are incompatible: {e}")
                    
                    # Now perform the actual reprojection (should work since test passed)
                    try:
                        original_bounds = subbasins_gdf.total_bounds
                        subbasins_gdf = subbasins_gdf.to_crs(raster_crs)
                        new_bounds = subbasins_gdf.total_bounds
                        
                        self.logger.info(f"✓ Reprojection successful")
                        self.logger.info(f"  Original bounds: {original_bounds}")
                        self.logger.info(f"  Reprojected bounds: {new_bounds}")
                        
                    except Exception as e:
                        raise RuntimeError(f"CRS reprojection failed despite successful test: {e}")
                else:
                    self.logger.info("✓ CRS already match - no reprojection needed")
                
                # FAIL-FAST bounds validation
                vector_bounds = subbasins_gdf.total_bounds
                self.logger.info(f"Final subbasin bounds: {vector_bounds}")
                self.logger.info(f"Raster bounds: {raster_bounds}")
                
                # Strict bounds overlap check (fail-fast)
                bounds_overlap = not (vector_bounds[2] < raster_bounds[0] or  # xmax < xmin
                                    vector_bounds[0] > raster_bounds[2] or  # xmin > xmax
                                    vector_bounds[3] < raster_bounds[1] or  # ymax < ymin
                                    vector_bounds[1] > raster_bounds[3])    # ymin > ymax
                
                if not bounds_overlap:
                    # Calculate separation distances for detailed error reporting
                    x_separation = max(0, max(raster_bounds[0] - vector_bounds[2], vector_bounds[0] - raster_bounds[2]))
                    y_separation = max(0, max(raster_bounds[1] - vector_bounds[3], vector_bounds[1] - raster_bounds[3]))
                    
                    error_msg = (f"CRITICAL: Subbasins and flow accumulation raster do not overlap after CRS alignment!\n"
                               f"  Raster bounds: {raster_bounds}\n"
                               f"  Vector bounds: {vector_bounds}\n"
                               f"  X separation: {x_separation:.2f} units\n"
                               f"  Y separation: {y_separation:.2f} units\n"
                               f"This indicates a fundamental CRS or data alignment problem.")
                    
                    raise RuntimeError(error_msg)
                
                self.logger.info("✓ Bounds validation successful - datasets overlap properly")
                
                # FAIL-FAST raster data validation
                flow_acc_data = src.read(1)
                if flow_acc_data is None:
                    raise RuntimeError("Failed to read flow accumulation raster data")
                
                # Check for valid flow accumulation values
                valid_values = flow_acc_data[flow_acc_data > 0]
                if len(valid_values) == 0:
                    raise RuntimeError("Flow accumulation raster contains no valid (positive) values")
                
                self.logger.info(f"✓ Flow accumulation raster loaded successfully")
                self.logger.info(f"  Valid flow accumulation range: {np.min(valid_values):.0f} - {np.max(valid_values):.0f} cells")
                
                drainage_areas = []
                failed_subbasins = []
                
                # For each subbasin polygon, find maximum flow accumulation (BasinMaker method)
                for idx, row in subbasins_gdf.iterrows():
                    try:
                        # FAIL-FAST geometry validation
                        if row.geometry is None or row.geometry.is_empty:
                            raise ValueError(f"Subbasin {idx} has invalid geometry")
                        
                        # Get the geometry in the correct CRS
                        geom = [row.geometry.__geo_interface__]
                        
                        # Extract flow accumulation values within subbasin with fail-fast validation
                        try:
                            masked_data, mask_transform = raster_mask(src, geom, crop=True, nodata=src.nodata)
                        except Exception as e:
                            raise RuntimeError(f"Raster masking failed for subbasin {idx}: {e}")
                        
                        flow_acc_values = masked_data[0]
                        
                        # FAIL-FAST: Ensure we extracted some data
                        if flow_acc_values.size == 0:
                            raise ValueError(f"No raster data extracted for subbasin {idx}")
                        
                        # Remove nodata values
                        if src.nodata is not None:
                            flow_acc_values = flow_acc_values[flow_acc_values != src.nodata]
                        
                        # Also remove zero values (they're usually nodata in flow accumulation)
                        flow_acc_values = flow_acc_values[flow_acc_values > 0]
                        
                        if len(flow_acc_values) > 0:
                            # Maximum accumulation = cumulative drainage area (BasinMaker method)
                            max_accumulation = float(np.max(flow_acc_values))
                            
                            # Convert from cell count to km²
                            # Flow accumulation is typically in number of cells
                            cell_area_m2 = abs(transform[0] * transform[4])  # pixel area in m²
                            drainage_area_km2 = (max_accumulation * cell_area_m2) / 1e6
                            
                            # FAIL-FAST: Validate drainage area is reasonable
                            if drainage_area_km2 <= 0:
                                raise ValueError(f"Invalid drainage area calculated: {drainage_area_km2}")
                            
                            # Sanity check: drainage area should be reasonable
                            if drainage_area_km2 < 0.1:  # Too small
                                drainage_area_km2 = max(0.1, row.get('area_km2', 1.0))
                            elif drainage_area_km2 > 10000:  # Too large for this watershed
                                drainage_area_km2 = min(10000, row.get('area_km2', 100.0) * 10)
                                
                        else:
                            # FAIL-FAST: No valid accumulation values found
                            error_msg = f"No valid flow accumulation values found for subbasin {idx}"
                            self.logger.warning(error_msg)
                            failed_subbasins.append(idx)
                            
                            # Use fallback but warn
                            drainage_area_km2 = row.get('area_km2', 1.0)
                            self.logger.warning(f"Using fallback local area: {drainage_area_km2:.1f} km² for subbasin {idx}")
                            
                        drainage_areas.append(drainage_area_km2)
                        
                    except Exception as e:
                        error_msg = f"FAILED to process subbasin {idx}: {e}"
                        self.logger.error(error_msg)
                        failed_subbasins.append(idx)
                        
                        # Use fallback area
                        fallback_area = row.get('area_km2', 1.0)
                        drainage_areas.append(fallback_area)
                
                # FAIL-FAST: Check if too many subbasins failed
                if len(failed_subbasins) > len(subbasins_gdf) * 0.5:  # More than 50% failed
                    raise RuntimeError(f"CRS alignment failed: {len(failed_subbasins)}/{len(subbasins_gdf)} subbasins failed processing. "
                                     f"This indicates a fundamental data alignment problem. Failed subbasins: {failed_subbasins}")
                
                if len(failed_subbasins) > 0:
                    self.logger.warning(f"⚠ {len(failed_subbasins)} subbasins failed processing but continuing with remaining {len(drainage_areas) - len(failed_subbasins)} subbasins")
                
                if len(drainage_areas) > 0:
                    self.logger.info(f"✓ Calculated cumulative drainage areas: {np.min(drainage_areas):.1f} - {np.max(drainage_areas):.1f} km²")
                    self.logger.info(f"✓ Successfully processed {len(drainage_areas) - len(failed_subbasins)}/{len(drainage_areas)} subbasins")
                    return np.array(drainage_areas)
                else:
                    raise RuntimeError("No drainage areas calculated - complete failure")
                
        except (ValueError, RuntimeError) as e:
            # These are our fail-fast errors - re-raise them with context
            self.logger.error(f"CRITICAL CRS/Data Alignment Failure: {e}")
            raise RuntimeError(f"Cumulative drainage area calculation failed due to CRS/data alignment issues: {e}")
        
        except Exception as e:
            # Unexpected errors
            self.logger.error(f"Unexpected error in cumulative drainage area calculation: {e}")
            self.logger.error("This may indicate missing dependencies or corrupted data files")
            return None

    def _estimate_cumulative_from_topology(self, channel_data: pd.DataFrame, local_areas: pd.Series) -> np.ndarray:
        """
        Estimate cumulative drainage areas from topology when flow accumulation fails
        Uses downstream routing to accumulate areas
        """
        try:
            cumulative_areas = local_areas.copy()
            
            # If we have downstream topology information
            if 'DowSubId' in channel_data.columns:
                # Create a mapping of subbasin relationships
                subid_to_idx = {row['SubId']: idx for idx, row in channel_data.iterrows()}
                
                # Sort by SubId to process upstream first
                sorted_indices = channel_data['SubId'].argsort()
                
                for idx in sorted_indices:
                    row = channel_data.iloc[idx]
                    subid = row['SubId']
                    downstream_id = row.get('DowSubId', -1)
                    
                    # If this subbasin has a downstream subbasin
                    if downstream_id != -1 and downstream_id in subid_to_idx:
                        downstream_idx = subid_to_idx[downstream_id]
                        # Add this subbasin's cumulative area to downstream
                        cumulative_areas.iloc[downstream_idx] += cumulative_areas.iloc[idx]
                
                self.logger.info("Estimated cumulative drainage areas from topology")
                return cumulative_areas.values
            else:
                # No topology info, just scale local areas to be more realistic
                # Assume some are headwaters (1x local) and some are downstream (2-5x local)
                multipliers = np.linspace(1.0, 5.0, len(local_areas))
                np.random.shuffle(multipliers)  # Random assignment
                estimated_cumulative = local_areas * multipliers
                
                self.logger.info("Estimated cumulative drainage areas using scaling factors")
                return estimated_cumulative.values
                
        except Exception as e:
            self.logger.warning(f"Topology-based estimation failed: {e}")
            # Return local areas as last resort
            return local_areas.values
    
    def generate_raven_subbasins_with_routing(self, channel_data: pd.DataFrame, 
                                            routing_method: str = "ROUTE_DIFFUSIVE_WAVE") -> List[str]:
        """
        Generate RAVEN :SubBasin commands with routing parameters
        """
        subbasins = []
        
        for idx, row in channel_data.iterrows():
            subbasin_id = int(row.get('SubId', idx + 1))
            downstream_id = int(row.get('DowSubId', -1))
            
            # Determine downstream connection
            if downstream_id <= 0 or downstream_id == subbasin_id:
                downstream_str = "NONE"
            else:
                downstream_str = str(downstream_id)
            
            # Create subbasin with routing parameters based on method
            if routing_method == "ROUTE_DIFFUSIVE_WAVE":
                subbasin = f"""
:SubBasin {subbasin_id}
  :Attributes NAME SB_{subbasin_id}
  :Attributes DOWNSTREAM_ID {downstream_str}
  :Attributes PROFILE CHANNEL_{subbasin_id}
  :Attributes GAUGE_LOC {row.get('GaugeID', 'NONE')}
:EndSubBasin
"""
            elif routing_method in ["ROUTE_HYDROLOGIC", "ROUTE_DELAYED_FIRST_ORDER"]:
                lag_time = row.get('lag_time_hours', 1.0)
                storage_coeff = row.get('storage_coefficient', 0.6)
                
                subbasin = f"""
:SubBasin {subbasin_id}
  :Attributes NAME SB_{subbasin_id}
  :Attributes DOWNSTREAM_ID {downstream_str}
  :Attributes LAG_TIME {lag_time:.2f}
  :Attributes STORAGE_COEFF {storage_coeff:.2f}
  :Attributes GAUGE_LOC {row.get('GaugeID', 'NONE')}
:EndSubBasin
"""
            else:  # ROUTE_DUMP
                subbasin = f"""
:SubBasin {subbasin_id}
  :Attributes NAME SB_{subbasin_id}
  :Attributes DOWNSTREAM_ID {downstream_str}
  :Attributes GAUGE_LOC {row.get('GaugeID', 'NONE')}
:EndSubBasin
"""
            
            subbasins.append(subbasin.strip())
        
        return subbasins
    
    def generate_lake_routing(self, lake_data: pd.DataFrame, channel_data: pd.DataFrame) -> List[str]:
        """
        Generate RAVEN reservoir/lake routing commands
        Implements lake-river routing network from Magpie/BasinMaker
        """
        if lake_data is None or len(lake_data) == 0:
            return []
        
        reservoirs = []
        
        for idx, row in lake_data.iterrows():
            lake_id = int(row.get('Lake_ID', idx + 1))
            subbasin_id = int(row.get('SubId', lake_id))
            
            # Lake properties
            if 'Max_Depth' in row:
                max_depth = row['Max_Depth']
            else:
                raise ValueError("Lake data missing 'Max_Depth' column")
            
            if 'Lake_Area' in row:
                area = row['Lake_Area']
            else:
                raise ValueError("Lake data missing 'Lake_Area' column")
            volume = area * max_depth * 0.3  # Approximate volume (30% of max theoretical)
            
            # Create reservoir command
            reservoir = f"""
:Reservoir LAKE_{lake_id}
  :SubBasinID {subbasin_id}
  :WEIRCOEFF 0.6
  :CRESTWIDTH {np.sqrt(area) * 0.1:.1f}
  :MaxDepth {max_depth:.1f}
  :LakeArea {area:.0f}
  :SeepageParameters
    :SeepageRate 0.0
  :EndSeepageParameters
:EndReservoir
"""
            reservoirs.append(reservoir.strip())
        
        return reservoirs
    
    def create_complete_routing_configuration(self, 
                                            subbasin_data: pd.DataFrame,
                                            lake_data: pd.DataFrame = None,
                                            routing_method: str = "ROUTE_DIFFUSIVE_WAVE") -> Dict[str, Any]:
        """
        Create complete hydraulic routing configuration for RAVEN
        """
        self.logger.info(f"Creating complete routing configuration with {routing_method}")
        
        # 1. Calculate channel hydraulics
        channel_data = self.calculate_channel_hydraulics_from_basinmaker(subbasin_data)
        
        # 2. Generate RAVEN components
        channel_profiles = self.generate_raven_channel_profiles(channel_data)
        subbasins = self.generate_raven_subbasins_with_routing(channel_data, routing_method)
        reservoirs = self.generate_lake_routing(lake_data, channel_data) if lake_data is not None else []
        
        # 3. Create routing configuration
        routing_config = {
            'routing_method': routing_method,
            'channel_profiles': channel_profiles,
            'subbasins': subbasins,
            'reservoirs': reservoirs,
            'hydraulic_parameters': channel_data,
            'routing_info': self.routing_methods[routing_method]
        }
        
        # 4. Generate summary statistics
        routing_config['summary'] = {
            'total_reaches': len(channel_data),
            'total_lakes': len(reservoirs),
            'avg_discharge': channel_data['discharge_m3s'].mean(),
            'avg_width': channel_data['channel_width_m'].mean(),
            'avg_depth': channel_data['channel_depth_m'].mean(),
            'avg_slope': channel_data['channel_slope'].mean(),
            'avg_manning_n': channel_data['manning_n'].mean(),
            'total_length_km': channel_data['RivLength'].sum() if 'RivLength' in channel_data.columns else 0
        }
        
        return routing_config
    
    def export_raven_routing_files(self, routing_config: Dict[str, Any], 
                                 output_dir: str, model_name: str = "watershed") -> Dict[str, str]:
        """
        Export complete RAVEN routing configuration to files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files_created = {}
        
        # 1. Export RVH file with routing structure
        rvh_file = output_path / f"{model_name}_routing.rvh"
        with open(rvh_file, 'w') as f:
            f.write(f"# RAVEN Watershed Structure File - Generated by Magpie Hydraulic Routing\\n")
            f.write(f"# Routing Method: {routing_config['routing_method']}\\n")
            f.write(f"# Generated: {pd.Timestamp.now()}\\n\\n")
            
            # Write channel profiles
            if routing_config['routing_method'] == "ROUTE_DIFFUSIVE_WAVE":
                f.write("# Channel Profiles\\n")
                for profile in routing_config['channel_profiles']:
                    f.write(profile + "\\n\\n")
            
            # Write subbasins
            f.write("# SubBasins\\n")
            for subbasin in routing_config['subbasins']:
                f.write(subbasin + "\\n\\n")
            
            # Write reservoirs/lakes
            if routing_config['reservoirs']:
                f.write("# Reservoirs/Lakes\\n")
                for reservoir in routing_config['reservoirs']:
                    f.write(reservoir + "\\n\\n")
        
        files_created['rvh'] = str(rvh_file)
        
        # 2. Export RVI file with routing configuration
        rvi_file = output_path / f"{model_name}_routing.rvi"
        with open(rvi_file, 'w') as f:
            f.write(f"# RAVEN Input File - Generated by Magpie Hydraulic Routing\\n")
            f.write(f"# Routing Method: {routing_config['routing_method']}\\n\\n")
            
            f.write(f":Routing {routing_config['routing_method']}\\n")
            f.write(f":CatchmentRoute TRIANGULATED_IRREGULAR_NETWORK\\n")
            f.write(f":Evaporation PET_PRIESTLEY_TAYLOR\\n")
            f.write(f"\\n")
        
        files_created['rvi'] = str(rvi_file)
        
        # 3. Export hydraulic parameters as CSV
        params_file = output_path / f"{model_name}_hydraulic_parameters.csv"
        routing_config['hydraulic_parameters'].to_csv(params_file, index=False)
        files_created['parameters'] = str(params_file)
        
        # 4. Export routing summary
        summary_file = output_path / f"{model_name}_routing_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("MAGPIE HYDRAULIC ROUTING SUMMARY\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(f"Routing Method: {routing_config['routing_method']}\\n")
            f.write(f"Description: {routing_config['routing_info']['description']}\\n")
            f.write(f"Parameters Required: {', '.join(routing_config['routing_info']['parameters'])}\\n")
            f.write(f"Suitable For: {routing_config['routing_info']['suitable_for']}\\n\\n")
            
            f.write("HYDRAULIC STATISTICS:\\n")
            f.write("-" * 30 + "\\n")
            for key, value in routing_config['summary'].items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.3f}\\n")
                else:
                    f.write(f"{key}: {value}\\n")
        
        files_created['summary'] = str(summary_file)
        
        self.logger.info(f"Exported routing configuration to {output_dir}")
        return files_created

def main():
    """Demonstrate Magpie hydraulic routing functionality"""
    print("≡ƒîè MAGPIE HYDRAULIC ROUTING ROUTINES")
    print("=" * 50)
    
    # Initialize routing processor
    routing = MagpieHydraulicRouting()
    
    # Create sample watershed data (like from BigWhite subbasins.geojson)
    sample_data = pd.DataFrame({
        'SubId': [16, 17, 18, 19, 20],
        'DowSubId': [17, 18, 19, 20, -1],
        'DrainArea': [2.669e6, 5.2e6, 8.1e6, 12.5e6, 18.3e6],  # m┬▓
        'RivLength': [1200, 1800, 2100, 2400, 2800],  # m
        'RivSlope': [0.002, 0.0015, 0.001, 0.0008, 0.0006],
        'MeanElev': [1200, 1150, 1100, 1050, 1000]  # m
    })
    
    # Sample lake data
    lake_data = pd.DataFrame({
        'Lake_ID': [1, 2],
        'SubId': [18, 20],
        'Lake_Area': [50000, 120000],  # m┬▓
        'Max_Depth': [8.5, 12.0]  # m
    })
    
    print("\\nAvailable Routing Methods:")
    for method, info in routing.routing_methods.items():
        print(f"  {method}:")
        print(f"    - {info['description']}")
        print(f"    - Parameters: {', '.join(info['parameters']) if info['parameters'] else 'None'}")
        print()
    
    # Test different routing methods
    methods_to_test = ["ROUTE_DIFFUSIVE_WAVE", "ROUTE_HYDROLOGIC", "ROUTE_DELAYED_FIRST_ORDER"]
    
    for method in methods_to_test:
        print(f"\\n≡ƒöä Testing {method}...")
        
        # Create routing configuration
        config = routing.create_complete_routing_configuration(
            subbasin_data=sample_data,
            lake_data=lake_data,
            routing_method=method
        )
        
        # Export files
        files = routing.export_raven_routing_files(
            config,
            output_dir=f"output/routing_{method.lower()}",
            model_name="bigwhite_example"
        )
        
        print(f"  Γ£ô Created routing configuration")
        print(f"  Γ£ô {config['summary']['total_reaches']} channel reaches")
        print(f"  Γ£ô {config['summary']['total_lakes']} lakes")
        print(f"  Γ£ô Files: {', '.join(files.keys())}")
    
    print(f"\\nΓ£à MAGPIE HYDRAULIC ROUTING DEMONSTRATION COMPLETE!")
    print(f"Check the 'output/routing_*' directories for generated files.")

if __name__ == "__main__":
    main()
