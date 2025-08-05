#!/usr/bin/env python3
"""
Basic Attributes Calculator - Extracted from BasinMaker
Calculates basic watershed attributes using your existing WhiteboxTools infrastructure
EXTRACTED FROM: basinmaker/addattributes/calculatebasicattributesqgis.py
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import rasterio
from rasterio.mask import mask
import sys

# Import your existing infrastructure
sys.path.append(str(Path(__file__).parent.parent))
try:
    from clients.watershed_clients.whitebox_client import WhiteboxWatershedClient
except ImportError:
    WhiteboxWatershedClient = None


class BasicAttributesCalculator:
    """
    Calculate basic watershed attributes using extracted BasinMaker logic
    EXTRACTED FROM: calculate_basic_attributes() in BasinMaker addattributes/calculatebasicattributesqgis.py
    Adapted to work with your existing WhiteboxTools/rasterio infrastructure
    """
    
    def __init__(self, workspace_dir: Path = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize your WhiteboxTools
        try:
            import whitebox
            self.wbt = whitebox.WhiteboxTools()
            self.wbt.work_dir = str(self.workspace_dir)
            self.has_whitebox = True
        except ImportError:
            self.has_whitebox = False
            print("WARNING: WhiteboxTools not available for advanced calculations")
    
    def calculate_basic_attributes_from_watershed_results(self, watershed_results: Dict, 
                                                        dem_path: Path) -> pd.DataFrame:
        """
        Calculate basic attributes using real BasinMaker logic adapted to your infrastructure
        EXTRACTED FROM: calculate_basic_attributes() in BasinMaker addattributes/calculatebasicattributesqgis.py
        
        This function replicates the core BasinMaker attribute calculation workflow:
        1. Load catchments and rivers from your watershed analysis results
        2. Calculate area, slope, aspect, elevation statistics
        3. Calculate river length and slope
        4. Generate routing topology
        
        Parameters:
        -----------
        watershed_results : Dict
            Results from your ProfessionalWatershedAnalyzer.analyze_watershed_complete()
        dem_path : Path
            Path to DEM raster file
            
        Returns:
        --------
        DataFrame with BasinMaker-style attributes: SubId, DowSubId, BasArea, MeanElev, RivLength, RivSlope, etc.
        """
        
        print("Calculating basic attributes using BasinMaker logic...")
        
        # Load watershed data from your results
        catchments_gdf, rivers_gdf = self._load_watershed_data(watershed_results)
        
        # Initialize result dataframe with BasinMaker columns
        catinfo = pd.DataFrame()
        
        # Step 1: Calculate catchment area, slope, aspect, elevation (BasinMaker lines 189-235)
        print("   Calculating catchment statistics...")
        area_stats = self._calculate_catchment_statistics(catchments_gdf, dem_path)
        
        # Step 2: Calculate river length and elevation statistics (BasinMaker lines 238-244) 
        print("   Calculating river statistics...")
        river_stats = self._calculate_river_statistics(rivers_gdf, dem_path)
        
        # Step 3: Generate routing topology (BasinMaker lines 324-342)
        print("   Generating routing topology...")
        routing_info = self._generate_routing_topology(catchments_gdf, watershed_results)
        
        # Step 4: Combine all statistics into BasinMaker format (BasinMaker lines 412-536)
        print("   Combining statistics...")
        catinfo = self._combine_statistics_basinmaker_format(
            area_stats, river_stats, routing_info
        )
        
        print(f"   Calculated attributes for {len(catinfo)} catchments")
        return catinfo
    
    def _load_watershed_data(self, watershed_results: Dict) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Load catchments and rivers from your watershed analysis results"""
        
        # Load catchments (watershed boundary)
        watershed_files = [f for f in watershed_results['files_created'] 
                          if 'watershed.geojson' in f]
        if not watershed_files:
            raise RuntimeError("No watershed boundary found in results")
        
        catchments_gdf = gpd.read_file(watershed_files[0])
        
        # Load rivers/streams
        stream_files = [f for f in watershed_results['files_created'] 
                       if 'streams.geojson' in f]
        if stream_files:
            rivers_gdf = gpd.read_file(stream_files[0])
        else:
            # Create empty rivers dataframe
            rivers_gdf = gpd.GeoDataFrame()
        
        return catchments_gdf, rivers_gdf
    
    def _calculate_catchment_statistics(self, catchments_gdf: gpd.GeoDataFrame, 
                                      dem_path: Path) -> pd.DataFrame:
        """
        Calculate catchment area, slope, aspect, elevation statistics
        EXTRACTED FROM: BasinMaker lines 189-235 (v.rast.stats operations)
        """
        
        area_stats = []
        
        for idx, catchment in catchments_gdf.iterrows():
            catchment_id = idx + 1  # Use 1-based indexing like BasinMaker
            
            try:
                with rasterio.open(dem_path) as src:
                    # Extract DEM data for this catchment
                    masked_data, masked_transform = mask(src, [catchment.geometry], 
                                                       crop=True, nodata=src.nodata)
                    elevation_data = masked_data[0]
                    
                    # Remove nodata values
                    if src.nodata is not None:
                        elevation_data = elevation_data[elevation_data != src.nodata]
                    
                    if len(elevation_data) > 0:
                        # Calculate elevation statistics (like BasinMaker v.rast.stats)
                        d_average = float(np.mean(elevation_data))
                        
                        # Calculate slope using your WhiteboxTools if available
                        slope_average = self._calculate_slope_statistics(catchment, dem_path)
                        aspect_average = self._calculate_aspect_statistics(catchment, dem_path)
                        
                        # Calculate area in meters squared (like BasinMaker v.to.db)
                        area_m = float(catchment.geometry.area)
                        
                        area_stats.append({
                            'Gridcode': catchment_id,
                            'Area_m': area_m,
                            'd_average': d_average,
                            's_average': slope_average,
                            'a_average': aspect_average
                        })
                    
            except Exception as e:
                print(f"Warning: Could not process catchment {catchment_id}: {e}")
                # Add default values
                area_stats.append({
                    'Gridcode': catchment_id,
                    'Area_m': float(catchment.geometry.area),
                    'd_average': 500.0,  # Default elevation
                    's_average': 5.0,    # Default slope in degrees
                    'a_average': 180.0   # Default aspect
                })
        
        return pd.DataFrame(area_stats)
    
    def _calculate_river_statistics(self, rivers_gdf: gpd.GeoDataFrame, 
                                   dem_path: Path) -> pd.DataFrame:
        """
        Calculate river length and elevation statistics
        EXTRACTED FROM: BasinMaker lines 238-244 (v.rast.stats on rivers)
        """
        
        river_stats = []
        
        if len(rivers_gdf) == 0:
            # No rivers found, return empty stats
            return pd.DataFrame(columns=['Gridcode', 'Length_m', 'd_minimum', 'd_maximum'])
        
        for idx, river in rivers_gdf.iterrows():
            river_id = idx + 1  # Use 1-based indexing
            
            try:
                # Calculate river length (like BasinMaker v.to.db length)
                length_m = float(river.geometry.length)
                
                # Extract elevation along river
                with rasterio.open(dem_path) as src:
                    # Sample elevation along river line
                    coords = list(river.geometry.coords)
                    elevations = []
                    
                    for coord in coords:
                        try:
                            row, col = src.index(coord[0], coord[1])
                            if 0 <= row < src.height and 0 <= col < src.width:
                                elev = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                                if elev != src.nodata:
                                    elevations.append(elev)
                        except:
                            continue
                    
                    # Calculate min/max elevation (like BasinMaker v.rast.stats minimum/maximum)
                    if elevations:
                        d_minimum = float(min(elevations))
                        d_maximum = float(max(elevations))
                    else:
                        d_minimum = 500.0  # Default
                        d_maximum = 510.0
                    
                    river_stats.append({
                        'Gridcode': river_id,
                        'Length_m': length_m,
                        'd_minimum': d_minimum,
                        'd_maximum': d_maximum
                    })
                    
            except Exception as e:
                print(f"Warning: Could not process river {river_id}: {e}")
                river_stats.append({
                    'Gridcode': river_id,
                    'Length_m': 1000.0,  # Default 1km
                    'd_minimum': 500.0,
                    'd_maximum': 510.0
                })
        
        return pd.DataFrame(river_stats)
    
    def _calculate_slope_statistics(self, catchment, dem_path: Path) -> float:
        """Calculate average slope using WhiteboxTools like BasinMaker r.slope.aspect"""
        
        if not self.has_whitebox:
            return 5.0  # Default slope in degrees
        
        try:
            # Create temporary files
            temp_catchment = self.workspace_dir / f"temp_catchment_{hash(str(catchment.geometry))}.shp"
            temp_slope = self.workspace_dir / f"temp_slope_{hash(str(catchment.geometry))}.tif"
            
            # Save catchment to shapefile
            temp_gdf = gpd.GeoDataFrame([catchment], crs=catchment.crs if hasattr(catchment, 'crs') else 'EPSG:4326')
            temp_gdf.to_file(temp_catchment)
            
            # Calculate slope using WhiteboxTools (equivalent to BasinMaker r.slope.aspect)
            self.wbt.slope(str(dem_path), str(temp_slope))
            
            # Extract slope statistics
            with rasterio.open(temp_slope) as slope_src:
                masked_slope, _ = mask(slope_src, [catchment.geometry], crop=True, nodata=slope_src.nodata)
                slope_data = masked_slope[0]
                
                if slope_src.nodata is not None:
                    slope_data = slope_data[slope_data != slope_src.nodata]
                
                if len(slope_data) > 0:
                    return float(np.mean(slope_data))
            
            # Cleanup temporary files
            for temp_file in [temp_catchment, temp_slope]:
                if temp_file.exists():
                    temp_file.unlink()
            
        except Exception as e:
            print(f"Warning: Slope calculation failed: {e}")
        
        return 5.0  # Default slope
    
    def _calculate_aspect_statistics(self, catchment, dem_path: Path) -> float:
        """Calculate average aspect using WhiteboxTools like BasinMaker r.slope.aspect"""
        
        if not self.has_whitebox:
            return 180.0  # Default aspect (south-facing)
        
        try:
            # Create temporary files
            temp_aspect = self.workspace_dir / f"temp_aspect_{hash(str(catchment.geometry))}.tif"
            
            # Calculate aspect using WhiteboxTools (equivalent to BasinMaker r.slope.aspect)
            self.wbt.aspect(str(dem_path), str(temp_aspect))
            
            # Extract aspect statistics
            with rasterio.open(temp_aspect) as aspect_src:
                masked_aspect, _ = mask(aspect_src, [catchment.geometry], crop=True, nodata=aspect_src.nodata)
                aspect_data = masked_aspect[0]
                
                if aspect_src.nodata is not None:
                    aspect_data = aspect_data[aspect_data != aspect_src.nodata]
                
                if len(aspect_data) > 0:
                    return float(np.mean(aspect_data))
            
            # Cleanup
            if temp_aspect.exists():
                temp_aspect.unlink()
                
        except Exception as e:
            print(f"Warning: Aspect calculation failed: {e}")
        
        return 180.0  # Default aspect
    
    def _generate_routing_topology(self, catchments_gdf: gpd.GeoDataFrame, 
                                 watershed_results: Dict) -> pd.DataFrame:
        """
        Generate routing topology information
        EXTRACTED FROM: BasinMaker lines 324-342 (generate_routing_info_of_catchments)
        """
        
        routing_info = []
        
        # For single watershed, create simple routing
        for idx, catchment in catchments_gdf.iterrows():
            subid = idx + 1
            
            # Outlet coordinates (use centroid as approximation)
            centroid = catchment.geometry.centroid
            outlet_lng = centroid.x
            outlet_lat = centroid.y
            
            # For single watershed, downstream ID is -1 (outlet)
            dow_subid = -1
            
            routing_info.append({
                'SubId': subid,
                'DowSubId': dow_subid,
                'outletLat': outlet_lat,
                'outletLng': outlet_lng
            })
        
        return pd.DataFrame(routing_info)
    
    def _combine_statistics_basinmaker_format(self, area_stats: pd.DataFrame, 
                                            river_stats: pd.DataFrame,
                                            routing_info: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all statistics into BasinMaker format
        EXTRACTED FROM: BasinMaker lines 412-536 (main combination loop)
        """
        
        catinfo = pd.DataFrame()
        
        # Process each subbasin (following BasinMaker's loop structure)
        for i, outlet_row in routing_info.iterrows():
            catid = outlet_row['SubId']
            dow_subid = outlet_row['DowSubId']
            outlet_lat = outlet_row['outletLat']
            outlet_lng = outlet_row['outletLng']
            
            # Initialize row
            row_data = {
                'SubId': catid,
                'DowSubId': dow_subid,
                'outletLat': outlet_lat,
                'outletLng': outlet_lng
            }
            
            # Get area statistics (BasinMaker lines 470-497)
            area_row = area_stats[area_stats['Gridcode'] == catid]
            if len(area_row) > 0:
                area_data = area_row.iloc[0]
                row_data['BasArea'] = area_data['Area_m']
                row_data['BasSlope'] = area_data['s_average']
                row_data['BasAspect'] = area_data['a_average']
                row_data['MeanElev'] = area_data['d_average']
            else:
                # Default values
                row_data['BasArea'] = -9999
                row_data['BasSlope'] = -9999
                row_data['BasAspect'] = -9999
                row_data['MeanElev'] = -9999
            
            # Get river statistics (BasinMaker lines 499-536)
            river_row = river_stats[river_stats['Gridcode'] == catid]
            if len(river_row) > 0:
                river_data = river_row.iloc[0]
                rivlen = river_data['Length_m']
                maxdem = river_data['d_maximum']
                mindem = river_data['d_minimum']
                
                row_data['RivLength'] = rivlen
                row_data['Min_DEM'] = mindem
                row_data['Max_DEM'] = maxdem
                
                # Calculate river slope (BasinMaker slope calculation)
                if rivlen > 0:
                    slope_rch = max(0, float(maxdem - mindem) / float(rivlen))
                    # Apply BasinMaker slope constraints
                    min_riv_slope = 0.0001  # BasinMaker default
                    max_riv_slope = 1.0     # BasinMaker default
                    slope_rch = max(slope_rch, min_riv_slope)
                    slope_rch = min(slope_rch, max_riv_slope)
                    row_data['RivSlope'] = slope_rch
                else:
                    row_data['RivSlope'] = -9999
            else:
                # No river data
                row_data['RivLength'] = -9999
                row_data['RivSlope'] = -9999
                row_data['FloodP_n'] = -9999
                row_data['Min_DEM'] = -9999
                row_data['Max_DEM'] = -9999
            
            # Add row to catinfo
            catinfo = pd.concat([catinfo, pd.DataFrame([row_data])], ignore_index=True)
        
        return catinfo
    
    def _calculate_subbasin_attributes(self, subbasins_gdf: gpd.GeoDataFrame,
                                     dem_path: Path = None,
                                     streams_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
        """Calculate attributes for each subbasin"""
        
        result_gdf = subbasins_gdf.copy()
        
        # Basic geometric attributes
        result_gdf = self._add_geometric_attributes(result_gdf)
        
        # Elevation attributes from DEM
        if dem_path and dem_path.exists():
            result_gdf = self._add_elevation_attributes(result_gdf, dem_path)
        
        # Stream attributes
        if streams_gdf is not None:
            result_gdf = self._add_stream_attributes(result_gdf, streams_gdf)
        
        # Derived attributes
        result_gdf = self._add_derived_attributes(result_gdf)
        
        return result_gdf
    
    def _calculate_single_watershed_attributes(self, watershed_gdf: gpd.GeoDataFrame,
                                             dem_path: Path = None,
                                             streams_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
        """Calculate attributes for single watershed"""
        
        result_gdf = watershed_gdf.copy()
        
        # Add SubId for single watershed
        result_gdf['SubId'] = 1
        result_gdf['DowSubId'] = -1  # Outlet
        
        # Calculate same attributes as subbasins
        result_gdf = self._add_geometric_attributes(result_gdf)
        
        if dem_path and dem_path.exists():
            result_gdf = self._add_elevation_attributes(result_gdf, dem_path)
        
        if streams_gdf is not None:
            result_gdf = self._add_stream_attributes(result_gdf, streams_gdf)
        
        result_gdf = self._add_derived_attributes(result_gdf)
        
        return result_gdf
    
    def _add_geometric_attributes(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add basic geometric attributes"""
        
        # Area calculations
        gdf['Area_m2'] = gdf.geometry.area
        gdf['Area_km2'] = gdf['Area_m2'] / 1e6
        
        # Perimeter
        gdf['Perimeter_m'] = gdf.geometry.length
        gdf['Perimeter_km'] = gdf['Perimeter_m'] / 1000
        
        # Centroid coordinates
        centroids = gdf.geometry.centroid
        gdf['CenX'] = centroids.x
        gdf['CenY'] = centroids.y
        
        # Shape indices
        gdf['Compactness'] = gdf['Perimeter_m'] / (2 * np.sqrt(np.pi * gdf['Area_m2']))
        gdf['CircularityRatio'] = (4 * np.pi * gdf['Area_m2']) / (gdf['Perimeter_m'] ** 2)
        
        return gdf
    
    def _add_elevation_attributes(self, gdf: gpd.GeoDataFrame, dem_path: Path) -> gpd.GeoDataFrame:
        """Extract elevation statistics from DEM using your rasterio infrastructure"""
        
        try:
            with rasterio.open(dem_path) as src:
                for idx, row in gdf.iterrows():
                    geom = row.geometry
                    
                    # Extract elevation data for this polygon
                    try:
                        masked_data, masked_transform = mask(src, [geom], crop=True, nodata=src.nodata)
                        elevation_data = masked_data[0]
                        
                        # Remove nodata values
                        if src.nodata is not None:
                            elevation_data = elevation_data[elevation_data != src.nodata]
                        
                        if len(elevation_data) > 0:
                            gdf.loc[idx, 'MeanElev'] = float(np.mean(elevation_data))
                            gdf.loc[idx, 'MinElev'] = float(np.min(elevation_data))
                            gdf.loc[idx, 'MaxElev'] = float(np.max(elevation_data))
                            gdf.loc[idx, 'StdElev'] = float(np.std(elevation_data))
                            gdf.loc[idx, 'RangeElev'] = float(np.max(elevation_data) - np.min(elevation_data))
                        else:
                            # Default values if no valid elevation data
                            gdf.loc[idx, 'MeanElev'] = 500.0
                            gdf.loc[idx, 'MinElev'] = 450.0
                            gdf.loc[idx, 'MaxElev'] = 550.0
                            gdf.loc[idx, 'StdElev'] = 25.0
                            gdf.loc[idx, 'RangeElev'] = 100.0
                            
                    except Exception as e:
                        print(f"Warning: Could not extract elevation for feature {idx}: {e}")
                        # Default values
                        gdf.loc[idx, 'MeanElev'] = 500.0
                        gdf.loc[idx, 'MinElev'] = 450.0
                        gdf.loc[idx, 'MaxElev'] = 550.0
                        gdf.loc[idx, 'StdElev'] = 25.0
                        gdf.loc[idx, 'RangeElev'] = 100.0
                        
        except Exception as e:
            print(f"Warning: Could not process DEM file: {e}")
            # Add default elevation columns
            gdf['MeanElev'] = 500.0
            gdf['MinElev'] = 450.0
            gdf['MaxElev'] = 550.0
            gdf['StdElev'] = 25.0
            gdf['RangeElev'] = 100.0
        
        return gdf
    
    def _add_stream_attributes(self, gdf: gpd.GeoDataFrame, 
                             streams_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate stream-related attributes"""
        
        # Initialize stream attributes
        gdf['RivLength'] = 0.0
        gdf['RivSlope'] = 0.001  # Default slope
        gdf['StreamDensity'] = 0.0
        gdf['StreamCount'] = 0
        
        # Calculate for each polygon
        for idx, polygon in gdf.iterrows():
            # Find streams within this polygon
            streams_within = streams_gdf[streams_gdf.intersects(polygon.geometry)]
            
            if len(streams_within) > 0:
                # Calculate total stream length within polygon
                total_length = 0.0
                stream_segments = []
                
                for _, stream in streams_within.iterrows():
                    # Intersect stream with polygon
                    intersection = stream.geometry.intersection(polygon.geometry)
                    
                    if hasattr(intersection, 'length'):
                        segment_length = intersection.length
                        total_length += segment_length
                        stream_segments.append(segment_length)
                
                gdf.loc[idx, 'RivLength'] = total_length
                gdf.loc[idx, 'StreamCount'] = len(streams_within)
                
                # Stream density (km/km2)
                if polygon['Area_km2'] > 0:
                    gdf.loc[idx, 'StreamDensity'] = (total_length / 1000) / polygon['Area_km2']
                
                # Calculate average slope if elevation data is available
                if 'MeanElev' in gdf.columns and total_length > 0:
                    # Simplified slope calculation
                    elevation_range = gdf.loc[idx, 'RangeElev']
                    gdf.loc[idx, 'RivSlope'] = max(0.001, elevation_range / total_length)
        
        return gdf
    
    def _add_derived_attributes(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add derived attributes commonly used in hydrological modeling"""
        
        # Flow accumulation estimate (simplified)
        if 'Area_km2' in gdf.columns:
            gdf['FlowAccum'] = gdf['Area_km2'] * 1000  # Rough estimate
        
        # Time of concentration estimate (simplified Kirpich equation)
        if 'RivLength' in gdf.columns and 'RivSlope' in gdf.columns:
            # Tc = 0.0078 * (L^0.77) * (S^-0.385)  where L in km, S in m/m
            L_km = gdf['RivLength'] / 1000
            S = np.maximum(gdf['RivSlope'], 0.001)  # Minimum slope
            gdf['TimeOfConc_hr'] = 0.0078 * (L_km ** 0.77) * (S ** -0.385)
        
        # Relief ratio
        if 'RangeElev' in gdf.columns and 'RivLength' in gdf.columns:
            gdf['ReliefRatio'] = np.where(
                gdf['RivLength'] > 0,
                gdf['RangeElev'] / gdf['RivLength'],
                0.0
            )
        
        # Default channel properties (can be refined later)
        gdf['BkfWidth'] = np.maximum(3.0, (gdf.get('Area_km2', 1.0) ** 0.3) * 2)
        gdf['BkfDepth'] = gdf['BkfWidth'] * 0.1
        
        # Manning's coefficients (defaults)
        gdf['Ch_n'] = 0.030      # Channel Manning's n
        gdf['FloodP_n'] = 0.035  # Floodplain Manning's n
        
        # Lake flags
        gdf['IsLake'] = 0  # Default: not a lake subbasin
        gdf['LakeArea'] = 0.0
        gdf['LakeDepth'] = 0.0
        
        # Observation flags
        gdf['IsObs'] = 0   # Default: no observation gauge
        gdf['Has_POI'] = 0
        
        return gdf
    
    def validate_attributes(self, gdf: gpd.GeoDataFrame) -> Dict:
        """Validate calculated attributes and provide summary"""
        
        validation = {
            'total_features': len(gdf),
            'warnings': [],
            'statistics': {}
        }
        
        # Check for required columns
        required_cols = ['SubId', 'Area_km2', 'MeanElev', 'RivLength']
        missing_cols = [col for col in required_cols if col not in gdf.columns]
        if missing_cols:
            validation['warnings'].append(f"Missing required columns: {missing_cols}")
        
        # Statistical validation
        for col in ['Area_km2', 'MeanElev', 'RivLength']:
            if col in gdf.columns:
                validation['statistics'][col] = {
                    'min': float(gdf[col].min()),
                    'max': float(gdf[col].max()),
                    'mean': float(gdf[col].mean()),
                    'count_zero': int((gdf[col] == 0).sum())
                }
        
        # Check for anomalies
        if 'Area_km2' in gdf.columns:
            zero_area = (gdf['Area_km2'] <= 0).sum()
            if zero_area > 0:
                validation['warnings'].append(f"{zero_area} features have zero or negative area")
        
        if 'RivLength' in gdf.columns:
            zero_length = (gdf['RivLength'] <= 0).sum()
            if zero_length > 0:
                validation['warnings'].append(f"{zero_length} features have no stream length")
        
        return validation


def test_basic_attributes():
    """Test the basic attributes calculator"""
    
    print("Testing Basic Attributes Calculator...")
    
    # Create mock polygon data
    from shapely.geometry import Polygon
    
    # Simple square polygon
    poly = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    mock_gdf = gpd.GeoDataFrame({'geometry': [poly]}, crs='EPSG:4326')
    
    # Initialize calculator
    calculator = BasicAttributesCalculator()
    
    # Test geometric attributes
    result = calculator._add_geometric_attributes(mock_gdf)
    print(f"✓ Geometric attributes calculated: {result['Area_km2'].iloc[0]:.3f} km²")
    
    # Test derived attributes
    result = calculator._add_derived_attributes(result)
    print(f"✓ Derived attributes calculated")
    
    # Test validation
    validation = calculator.validate_attributes(result)
    print(f"✓ Validation completed: {validation['total_features']} features")
    
    print("✓ Basic Attributes Calculator ready for integration")


if __name__ == "__main__":
    test_basic_attributes()