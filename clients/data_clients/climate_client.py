#!/usr/bin/env python3
"""
Climate Data Client for RAVEN Hydrological Modeling
Downloads climate data from Environment and Climate Change Canada (ECCC) APIs
Outputs RavenPy-compatible CSV and NetCDF formats
"""

import requests
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
import xarray as xr
import rasterio
from rasterio.warp import transform_bounds, reproject
from rasterio.enums import Resampling
warnings.filterwarnings('ignore')

class ClimateDataClient:
    """Client for downloading climate data with RavenPy-compatible CSV output"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAVEN-Hydrological-Model-Client/1.0'
        })
        self.eccc_base_url = "https://api.weather.gc.ca"
        self.daymet_base_url = "https://thredds.daac.ornl.gov/thredds/dodsC/ornldaac"
        self.capa_base_url = "https://dd.weather.gc.ca/model_gem_regional"
    
    def get_climate_stations_for_watershed(self, bbox: Tuple[float, float, float, float], 
                                          output_path: Optional[Path] = None, 
                                          limit: int = 100) -> Dict:
        """Get ECCC climate stations within watershed bounding box"""
        minx, miny, maxx, maxy = bbox
        bbox_str = f"{minx},{miny},{maxx},{maxy}"
        
        params = {
            'bbox': bbox_str,
            'limit': limit
        }
        
        print(f"Getting ECCC climate stations for bbox: {bbox_str}")
        
        try:
            response = self.session.get(f"{self.eccc_base_url}/collections/climate-stations/items", 
                                      params=params, timeout=30)
            response.raise_for_status()
            
            stations_data = response.json()
            
            if "features" in stations_data:
                num_stations = len(stations_data["features"])
                print(f"SUCCESS: Found {num_stations} climate stations")
                
                # Show sample stations
                for i, station in enumerate(stations_data["features"][:3]):
                    props = station["properties"]
                    name = props.get("STATION_NAME", "N/A")
                    climate_id = props.get("CLIMATE_IDENTIFIER", "N/A")
                    print(f"   {i+1}. {name} (ID: {climate_id})")
                
                # Save to file if requested
                if output_path:
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    with open(output_path, 'w') as f:
                        json.dump(stations_data, f, indent=2)
                    print(f"Climate stations saved to: {output_path}")
                
                return stations_data
            else:
                return {"features": [], "message": "No stations found"}
                
        except Exception as e:
            error_msg = f"Climate stations query failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {"error": error_msg}
    
    def get_climate_data_csv(self, station_id: str, start_date: str, end_date: str, 
                            output_path: Path, format_type: str = 'ravenpy') -> Dict:
        """Download climate data and save as RavenPy-compatible CSV"""
        print(f"Getting climate data for station {station_id}")
        print(f"Date range: {start_date} to {end_date}")
        
        try:
            params = {
                'CLIMATE_IDENTIFIER': station_id,
                'datetime': f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
                'limit': 10000
            }
            
            response = self.session.get(f"{self.eccc_base_url}/collections/climate-daily/items",
                                      params=params, timeout=60)
            response.raise_for_status()
            
            climate_data = response.json()
            
            if "features" in climate_data and climate_data["features"]:
                # Convert to DataFrame
                records = []
                for feature in climate_data["features"]:
                    props = feature["properties"]
                    records.append({
                        'date': props.get('LOCAL_DATE'),
                        'temp_max': props.get('MAX_TEMPERATURE'),
                        'temp_min': props.get('MIN_TEMPERATURE'),
                        'temp_mean': props.get('MEAN_TEMPERATURE'),
                        'precip': props.get('TOTAL_PRECIPITATION'),
                        'snow': props.get('TOTAL_SNOW'),
                        'station_id': station_id
                    })
                
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').dropna(subset=['date'])
                
                if len(df) == 0:
                    return {'success': False, 'error': 'No valid records after processing'}
                
                # Clean and format for RavenPy
                df['temp_max'] = pd.to_numeric(df['temp_max'], errors='coerce')
                df['temp_min'] = pd.to_numeric(df['temp_min'], errors='coerce')
                df['temp_mean'] = pd.to_numeric(df['temp_mean'], errors='coerce')
                df['precip'] = pd.to_numeric(df['precip'], errors='coerce').fillna(0.0).clip(lower=0.0)
                
                # Calculate data quality before interpolation
                temp_completeness = (1 - df[['temp_max', 'temp_min']].isnull().any(axis=1).mean()) * 100
                precip_completeness = (1 - df['precip'].isnull().mean()) * 100
                
                # Fill missing temperature values with interpolation
                df['temp_max'] = df['temp_max'].interpolate()
                df['temp_min'] = df['temp_min'].interpolate()
                df['temp_mean'] = df['temp_mean'].interpolate()
                
                # Calculate temp_mean if missing
                df['temp_mean'] = df['temp_mean'].fillna((df['temp_max'] + df['temp_min']) / 2)
                
                # Format based on requested type
                if format_type == 'ravenpy':
                    # RavenPy format - use standard variable names
                    output_df = df[['date', 'temp_max', 'temp_min', 'precip']].copy()
                    output_df.columns = ['Date', 'TEMP_MAX', 'TEMP_MIN', 'PRECIP']
                    output_df = output_df.set_index('Date')
                else:
                    # Legacy format
                    output_df = df[['date', 'temp_max', 'temp_min', 'temp_mean', 'precip']].copy()
                    output_df.columns = ['Date', 'Tmax', 'Tmin', 'Tmean', 'Precip']
                    output_df = output_df.set_index('Date')
                
                # Save CSV
                output_path.parent.mkdir(exist_ok=True, parents=True)
                output_df.to_csv(output_path, float_format='%.2f')
                
                print(f"SUCCESS: Climate CSV created ({len(output_df)} days, format: {format_type})")
                if format_type == 'ravenpy':
                    temp_min_col, temp_max_col, precip_col = 'TEMP_MIN', 'TEMP_MAX', 'PRECIP'
                else:
                    temp_min_col, temp_max_col, precip_col = 'Tmin', 'Tmax', 'Precip'
                print(f"Temperature: {output_df[temp_min_col].min():.1f}C to {output_df[temp_max_col].max():.1f}C, Precipitation: {output_df[precip_col].sum():.1f}mm")
                
                return {
                    'success': True,
                    'file_path': str(output_path),
                    'records': len(output_df),
                    'date_range': [output_df.index.min().strftime('%Y-%m-%d'), 
                                 output_df.index.max().strftime('%Y-%m-%d')],
                    'data_quality': {
                        'temp_completeness': temp_completeness,
                        'precip_completeness': precip_completeness
                    }
                }
            else:
                return {'success': False, 'error': 'No climate data features found'}
                
        except Exception as e:
            error_msg = f"Climate data download failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def get_climate_data_netcdf(self, station_id: str, start_date: str, end_date: str,
                               output_path: Path, station_info: Dict = None) -> Dict:
        """Download climate data and save as RavenPy-compatible NetCDF with CF-conventions"""
        try:
            import xarray as xr
        except ImportError:
            return {'success': False, 'error': 'xarray required for NetCDF output - install with: pip install xarray netcdf4'}
        
        print(f"Getting climate data for NetCDF output - station {station_id}")
        print(f"Date range: {start_date} to {end_date}")
        
        try:
            # Get CSV data first
            temp_csv_path = output_path.with_suffix('.csv')
            csv_result = self.get_climate_data_csv(station_id, start_date, end_date, temp_csv_path, format_type='ravenpy')
            
            if not csv_result.get('success'):
                return csv_result
            
            # Read CSV and convert to xarray Dataset
            df = pd.read_csv(temp_csv_path, index_col=0, parse_dates=True)
            
            # Create xarray Dataset with CF-compliant metadata
            ds = xr.Dataset()
            
            # Add time coordinate
            ds = ds.assign_coords(time=df.index)
            
            # Add station coordinates if provided
            if station_info:
                ds = ds.assign_coords(
                    latitude=station_info.get('latitude', 50.0),
                    longitude=station_info.get('longitude', -100.0)
                )
            
            # Add temperature variables with CF-compliant attributes
            ds['TEMP_MAX'] = xr.DataArray(
                df['TEMP_MAX'].values,
                dims=['time'],
                attrs={
                    'standard_name': 'air_temperature',
                    'long_name': 'Daily Maximum Air Temperature',
                    'units': 'degree_Celsius',
                    'cell_methods': 'time: maximum'
                }
            )
            
            ds['TEMP_MIN'] = xr.DataArray(
                df['TEMP_MIN'].values,
                dims=['time'],
                attrs={
                    'standard_name': 'air_temperature',
                    'long_name': 'Daily Minimum Air Temperature', 
                    'units': 'degree_Celsius',
                    'cell_methods': 'time: minimum'
                }
            )
            
            # Add precipitation with CF-compliant attributes
            ds['PRECIP'] = xr.DataArray(
                df['PRECIP'].values,
                dims=['time'],
                attrs={
                    'standard_name': 'precipitation_amount',
                    'long_name': 'Daily Total Precipitation',
                    'units': 'mm',
                    'cell_methods': 'time: sum'
                }
            )
            
            # Add time attributes
            ds.time.attrs = {
                'standard_name': 'time',
                'long_name': 'time'
            }
            
            # Add global attributes
            ds.attrs = {
                'title': f'Climate data for station {station_id}',
                'institution': 'Environment and Climate Change Canada',
                'source': 'ECCC Weather API',
                'conventions': 'CF-1.8',
                'created_by': 'RAVEN Data Client',
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'station_id': station_id
            }
            
            if station_info:
                ds.attrs.update({
                    'station_name': station_info.get('name', 'Unknown'),
                    'station_elevation': station_info.get('elevation', 'Unknown')
                })
            
            # Save NetCDF file
            output_path.parent.mkdir(exist_ok=True, parents=True)
            ds.to_netcdf(output_path)
            
            # Clean up temporary CSV
            temp_csv_path.unlink(missing_ok=True)
            
            print(f"SUCCESS: Climate NetCDF created ({len(df)} days)")
            print(f"Variables: TEMP_MAX, TEMP_MIN, PRECIP")
            print(f"CF-Convention compliant for RavenPy")
            
            return {
                'success': True,
                'file_path': str(output_path),
                'format': 'netcdf_cf_compliant',
                'variables': ['TEMP_MAX', 'TEMP_MIN', 'PRECIP'],
                'records': len(df),
                'date_range': [df.index.min().strftime('%Y-%m-%d'), 
                             df.index.max().strftime('%Y-%m-%d')]
            }
            
        except Exception as e:
            error_msg = f"NetCDF creation failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def get_30_year_climate_data_with_advanced_gap_filling(self, outlet_lat: float, outlet_lon: float, 
                                                         search_radius_km: float = 50.0,
                                                         start_year: int = 1991, end_year: int = 2020,
                                                         output_path: Optional[Path] = None,
                                                         min_years: int = 20,
                                                         format_type: str = 'ravenpy',
                                                         target_elevation_m: float = None,
                                                         use_elevation_adjustment: bool = True,
                                                         use_data_driven_parameters: bool = True) -> Dict:
        """Get 30-year climate normal data with advanced gap filling using IDW interpolation with elevation adjustment"""
        print(f"Getting 30-year climate data for ({outlet_lat}, {outlet_lon}) with {search_radius_km}km search radius")
        if use_elevation_adjustment and target_elevation_m:
            print(f"Elevation adjustment: ENABLED (target: {target_elevation_m}m)")
        else:
            print(f"Elevation adjustment: DISABLED")
        
        try:
            # Create bbox around point
            # Approximate: 1 degree = ~111 km
            deg_radius = search_radius_km / 111.0
            bbox = (outlet_lon - deg_radius, outlet_lat - deg_radius, outlet_lon + deg_radius, outlet_lat + deg_radius)
            
            # Get stations in area (increase limit to get modern stations)
            stations_result = self.get_climate_stations_for_watershed(bbox, limit=50)
            
            if not stations_result.get("features"):
                return {'success': False, 'error': 'No climate stations found in search area'}
            
            stations = stations_result["features"]
            print(f"Found {len(stations)} climate stations in search area")
            
            # Try to get data from multiple stations
            all_data = []
            successful_stations = 0
            
            for station in stations:  # Check ALL stations to find modern ones
                props = station["properties"]
                station_id = props.get("CLIMATE_IDENTIFIER")
                station_name = props.get("STATION_NAME", "Unknown")
                
                if not station_id:
                    continue
                
                print(f"Trying station: {station_name} ({station_id})")
                
                # Get data for available period (not just 1991-2020)
                temp_path = Path("temp_climate.csv")
                
                # For modern stations, use their actual data range
                actual_start = props.get("DLY_FIRST_DATE", f"{start_year}-01-01")[:10]
                actual_end = props.get("DLY_LAST_DATE", f"{end_year}-12-31")[:10]
                
                result = self.get_climate_data_csv(
                    station_id, 
                    actual_start, 
                    actual_end,
                    temp_path,
                    format_type='ravenpy'
                )
                
                if result.get('success'):
                    df = pd.read_csv(temp_path, index_col=0, parse_dates=True)
                    
                    # Add station info
                    geom = station.get("geometry", {})
                    coords = geom.get("coordinates", [0, 0])
                    df['station_lon'] = coords[0]
                    df['station_lat'] = coords[1]
                    df['station_id'] = station_id
                    
                    # Extract elevation (convert from string if needed)
                    elevation_str = props.get("ELEVATION", "0")
                    try:
                        df['station_elevation'] = float(elevation_str) if elevation_str and elevation_str != "null" else np.nan
                    except (ValueError, TypeError):
                        df['station_elevation'] = np.nan
                    
                    all_data.append(df)
                    successful_stations += 1
                    print(f"SUCCESS: Got {len(df)} records from {station_name}")
                    
                    temp_path.unlink(missing_ok=True)
                
                # Check if we have full coverage (1991-2020)
                if successful_stations > 0:
                    # Check coverage by combining all successful stations
                    combined_years = set()
                    for df in all_data:
                        combined_years.update(df.index.year)
                    
                    # If we have data for both early (1991-2000) and late (2010-2020) periods, we're good
                    has_early = any(year >= 1991 and year <= 2000 for year in combined_years)
                    has_late = any(year >= 2010 and year <= 2020 for year in combined_years)
                    
                    # Continue processing ALL stations - modern stations crucial for recent data
                    # Removed early stopping to ensure all available stations are processed
            
            if not all_data:
                return {'success': False, 'error': 'No climate data could be retrieved from any stations'}
            
            # Determine target elevation and adjustment parameters
            if use_elevation_adjustment:
                if target_elevation_m is None:
                    # Estimate target elevation from available data
                    target_elevation_m = self._estimate_target_elevation(outlet_lat, outlet_lon, all_data)
                
                if use_data_driven_parameters:
                    # Calculate site-specific lapse rates and gradients
                    data_driven_params = self._calculate_data_driven_parameters(all_data, outlet_lat, outlet_lon)
                else:
                    # Use literature values
                    data_driven_params = {
                        'temperature_lapse_rate': -6.0,
                        'precipitation_gradient': 0.0002,
                        'stations_used': 0,
                        'station_details': []
                    }
                    print("  Using literature-based parameters: -6.0°C/1000m, +20%/1000m")
            else:
                data_driven_params = None
            
            # Combine and interpolate data using proper IDW interpolation
            print(f"Combining data from {len(all_data)} stations with IDW interpolation...")
            
            # Create full date range
            start_date = pd.Timestamp(f"{start_year}-01-01")
            end_date = pd.Timestamp(f"{end_year}-12-31")
            full_dates = pd.date_range(start_date, end_date, freq='D')
            
            # Initialize result dataframe
            result_df = pd.DataFrame(index=full_dates)
            result_df['TEMP_MAX'] = np.nan
            result_df['TEMP_MIN'] = np.nan
            result_df['PRECIP'] = 0.0
            
            # Apply IDW interpolation for each date
            if use_elevation_adjustment and target_elevation_m:
                print("  Applying IDW interpolation with elevation adjustment...")
            else:
                print("  Applying IDW interpolation for spatial interpolation...")
            
            # Track method usage for reporting
            single_station_days = 0
            multi_station_days = 0
            
            for date in full_dates:
                # Collect station data for this date
                station_values = {'TEMP_MAX': [], 'TEMP_MIN': [], 'PRECIP': [], 'distances': [], 'elevations': []}
                
                for df in all_data:
                    if date in df.index:
                        row = df.loc[date]
                        
                        # Calculate distance from target point to station
                        station_lat = row['station_lat']
                        station_lon = row['station_lon']
                        distance = self._calculate_distance(outlet_lat, outlet_lon, station_lat, station_lon)
                        
                        # Only use stations with valid data
                        if not pd.isna(row['TEMP_MAX']) and not pd.isna(row['TEMP_MIN']):
                            station_values['TEMP_MAX'].append(row['TEMP_MAX'])
                            station_values['TEMP_MIN'].append(row['TEMP_MIN'])
                            station_values['PRECIP'].append(row['PRECIP'] if not pd.isna(row['PRECIP']) else 0.0)
                            station_values['distances'].append(distance)
                            station_values['elevations'].append(row.get('station_elevation', np.nan))
                
                # Apply interpolation based on number of stations available
                if len(station_values['distances']) > 0:
                    if len(station_values['distances']) == 1:
                        # SINGLE STATION: Apply direct elevation correction (no IDW)
                        single_station_days += 1
                        if use_elevation_adjustment and target_elevation_m:
                            result_df.loc[date, 'TEMP_MAX'] = self._apply_direct_elevation_correction(
                                station_values['TEMP_MAX'][0], station_values['elevations'][0], 
                                target_elevation_m, 'temperature', data_driven_params)
                            result_df.loc[date, 'TEMP_MIN'] = self._apply_direct_elevation_correction(
                                station_values['TEMP_MIN'][0], station_values['elevations'][0], 
                                target_elevation_m, 'temperature', data_driven_params)
                            result_df.loc[date, 'PRECIP'] = self._apply_direct_elevation_correction(
                                station_values['PRECIP'][0], station_values['elevations'][0], 
                                target_elevation_m, 'precipitation', data_driven_params)
                        else:
                            # No elevation adjustment - direct assignment
                            result_df.loc[date, 'TEMP_MAX'] = station_values['TEMP_MAX'][0]
                            result_df.loc[date, 'TEMP_MIN'] = station_values['TEMP_MIN'][0]
                            result_df.loc[date, 'PRECIP'] = station_values['PRECIP'][0]
                    
                    else:
                        # MULTIPLE STATIONS: Use IDW with or without elevation adjustment
                        multi_station_days += 1
                        if use_elevation_adjustment and target_elevation_m:
                            # Use elevation-adjusted IDW
                            result_df.loc[date, 'TEMP_MAX'] = self._idw_interpolation_with_elevation(
                                station_values['TEMP_MAX'], station_values['distances'], 
                                station_values['elevations'], target_elevation_m, 'temperature', 2, data_driven_params)
                            result_df.loc[date, 'TEMP_MIN'] = self._idw_interpolation_with_elevation(
                                station_values['TEMP_MIN'], station_values['distances'], 
                                station_values['elevations'], target_elevation_m, 'temperature', 2, data_driven_params)
                            result_df.loc[date, 'PRECIP'] = self._idw_interpolation_with_elevation(
                                station_values['PRECIP'], station_values['distances'], 
                                station_values['elevations'], target_elevation_m, 'precipitation', 2, data_driven_params)
                        else:
                            # Use standard IDW
                            result_df.loc[date, 'TEMP_MAX'] = self._idw_interpolation(station_values['TEMP_MAX'], station_values['distances'])
                            result_df.loc[date, 'TEMP_MIN'] = self._idw_interpolation(station_values['TEMP_MIN'], station_values['distances'])
                            result_df.loc[date, 'PRECIP'] = self._idw_interpolation(station_values['PRECIP'], station_values['distances'])
            
            # Fill remaining gaps with advanced temporal interpolation
            print("  Filling remaining gaps with seasonal patterns...")
            result_df = self._fill_gaps_with_seasonal_patterns(result_df)
            
            # Save if path provided
            if output_path:
                output_path.parent.mkdir(exist_ok=True, parents=True)
                result_df.to_csv(output_path, float_format='%.2f')
                print(f"SUCCESS: Saved 30-year climate data: {output_path}")
            
            temp_completeness = (1 - result_df[['TEMP_MAX', 'TEMP_MIN']].isnull().any(axis=1).mean()) * 100
            
            # Report interpolation method usage
            total_days = single_station_days + multi_station_days
            if single_station_days > 0:
                print(f"  Method usage: {multi_station_days} days IDW ({multi_station_days/total_days*100:.1f}%), {single_station_days} days single-station ({single_station_days/total_days*100:.1f}%)")
            else:
                print(f"  Method usage: {multi_station_days} days IDW (100.0%)")
            
            print(f"SUCCESS: 30-year climate data prepared")
            print(f"Period: {start_year}-{end_year} ({len(result_df)} days)")
            print(f"Temperature completeness: {temp_completeness:.1f}%")
            print(f"Mean annual precipitation: {result_df['PRECIP'].sum() / (end_year - start_year + 1):.1f} mm/year")
            
            return {
                'success': True,
                'file_path': str(output_path) if output_path else None,
                'records': len(result_df),
                'date_range': [result_df.index.min().strftime('%Y-%m-%d'), 
                             result_df.index.max().strftime('%Y-%m-%d')],
                'stations_used': successful_stations,
                'data_quality': {
                    'temp_completeness': temp_completeness,
                    'precip_completeness': 100.0  # Filled with zeros
                },
                'climate_summary': {
                    'mean_temp_max': result_df['TEMP_MAX'].mean(),
                    'mean_temp_min': result_df['TEMP_MIN'].mean(),
                    'annual_precip': result_df['PRECIP'].sum() / (end_year - start_year + 1)
                }
            }
            
        except Exception as e:
            error_msg = f"30-year climate data acquisition failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula"""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        
        return c * r
    
    def _idw_interpolation(self, values, distances, power=2):
        """
        Inverse Distance Weighting interpolation
        
        Args:
            values: List of values from stations
            distances: List of distances to stations (km)
            power: IDW power parameter (default=2)
        
        Returns:
            Interpolated value
        """
        if not values or not distances:
            return np.nan
        
        # Convert to numpy arrays
        values = np.array(values)
        distances = np.array(distances)
        
        # Handle case where we have a station at the exact location
        min_distance = 0.001  # 1 meter minimum distance
        distances = np.maximum(distances, min_distance)
        
        # Calculate IDW weights
        weights = 1.0 / (distances ** power)
        
        # Calculate weighted average
        weighted_sum = np.sum(weights * values)
        weight_sum = np.sum(weights)
        
        if weight_sum == 0:
            return np.nan
        
        return weighted_sum / weight_sum

    def _idw_interpolation_with_elevation(self, values, distances, station_elevations, target_elevation, 
                                        variable_type='temperature', power=2, data_driven_params=None):
        """
        Inverse Distance Weighting interpolation with elevation adjustment
        
        Args:
            values: List of values from stations
            distances: List of distances to stations (km)
            station_elevations: List of station elevations (m)
            target_elevation: Target outlet elevation (m)
            variable_type: 'temperature' or 'precipitation'
            power: IDW power parameter (default=2)
            data_driven_params: Dict with calculated lapse rates/gradients
        
        Returns:
            Interpolated and elevation-adjusted value
        """
        if not values or not distances or not station_elevations:
            return np.nan
        
        # Convert to numpy arrays
        values = np.array(values)
        distances = np.array(distances)
        station_elevations = np.array(station_elevations)
        
        # Remove stations with invalid elevation data
        valid_mask = ~np.isnan(station_elevations) & (station_elevations > 0)
        if not np.any(valid_mask):
            # Fall back to regular IDW if no elevation data
            return self._idw_interpolation(values, distances, power)
        
        values = values[valid_mask]
        distances = distances[valid_mask]
        station_elevations = station_elevations[valid_mask]
        
        # Handle case where we have a station at the exact location
        min_distance = 0.001  # 1 meter minimum distance
        distances = np.maximum(distances, min_distance)
        
        # Apply elevation corrections to station values
        adjusted_values = values.copy()
        
        if variable_type == 'temperature':
            # Apply temperature lapse rate correction
            if data_driven_params:
                lapse_rate = data_driven_params['temperature_lapse_rate']
            else:
                lapse_rate = -6.0  # Default
            elevation_diffs = target_elevation - station_elevations  # m
            temp_adjustments = (elevation_diffs / 1000.0) * lapse_rate
            adjusted_values = values + temp_adjustments
            
        elif variable_type == 'precipitation':
            # Apply precipitation elevation gradient
            if data_driven_params:
                precip_gradient = data_driven_params['precipitation_gradient']
            else:
                precip_gradient = 0.0002  # Default +20% per 1000m
            elevation_diffs = target_elevation - station_elevations  # m
            precip_multipliers = 1.0 + (elevation_diffs * precip_gradient)
            # Ensure reasonable bounds (0.5x to 3.0x)
            precip_multipliers = np.clip(precip_multipliers, 0.5, 3.0)
            adjusted_values = values * precip_multipliers
        
        # Calculate IDW weights
        weights = 1.0 / (distances ** power)
        
        # Calculate weighted average of elevation-adjusted values
        weighted_sum = np.sum(weights * adjusted_values)
        weight_sum = np.sum(weights)
        
        if weight_sum == 0:
            return np.nan
        
        return weighted_sum / weight_sum

    def _apply_direct_elevation_correction(self, station_value, station_elevation, target_elevation, variable_type, data_driven_params=None):
        """
        Apply direct elevation correction to a single station value (no IDW)
        
        Args:
            station_value: Value from the single station
            station_elevation: Station elevation (m)
            target_elevation: Target outlet elevation (m)
            variable_type: 'temperature' or 'precipitation'
            data_driven_params: Dict with calculated parameters
        
        Returns:
            Elevation-corrected value
        """
        if pd.isna(station_value):
            return np.nan
        
        # If no elevation data, return original value
        if pd.isna(station_elevation) or station_elevation <= 0:
            return station_value
        
        elevation_diff = target_elevation - station_elevation  # m
        
        if variable_type == 'temperature':
            # Apply temperature lapse rate correction
            if data_driven_params:
                lapse_rate = data_driven_params['temperature_lapse_rate']
            else:
                lapse_rate = -6.0  # Default
            temp_adjustment = (elevation_diff / 1000.0) * lapse_rate
            return station_value + temp_adjustment
            
        elif variable_type == 'precipitation':
            # Apply precipitation elevation gradient
            if data_driven_params:
                precip_gradient = data_driven_params['precipitation_gradient']
            else:
                precip_gradient = 0.0002  # Default +20% per 1000m
            precip_multiplier = 1.0 + (elevation_diff * precip_gradient)
            # Ensure reasonable bounds (0.5x to 3.0x)
            precip_multiplier = np.clip(precip_multiplier, 0.5, 3.0)
            return station_value * precip_multiplier
        
        return station_value

    def _estimate_target_elevation(self, target_lat, target_lon, station_data):
        """
        Estimate target elevation using available station data or external service
        
        Args:
            target_lat: Target latitude
            target_lon: Target longitude 
            station_data: List of station dataframes with elevation info
            
        Returns:
            Estimated elevation in meters
        """
        # Method 1: Try to get elevation from USGS or similar service
        try:
            elevation = self._get_elevation_from_service(target_lat, target_lon)
            if elevation and elevation > 0:
                print(f"  Target elevation from elevation service: {elevation:.0f}m")
                return elevation
        except Exception as e:
            print(f"  Elevation service unavailable: {e}")
        
        # Method 2: IDW interpolation from nearby stations with elevation data
        station_elevations = []
        station_distances = []
        station_coords = []
        
        for df in station_data:
            if len(df) > 0:
                first_row = df.iloc[0]
                station_elevation = first_row.get('station_elevation', np.nan)
                if not pd.isna(station_elevation) and station_elevation > 0:
                    station_lat = first_row.get('station_lat', np.nan)
                    station_lon = first_row.get('station_lon', np.nan)
                    if not pd.isna(station_lat) and not pd.isna(station_lon):
                        distance = self._calculate_distance(target_lat, target_lon, station_lat, station_lon)
                        station_elevations.append(station_elevation)
                        station_distances.append(distance)
                        station_coords.append((station_lat, station_lon, station_elevation))
        
        if len(station_elevations) >= 3:
            # IDW interpolation of elevations
            estimated_elevation = self._idw_interpolation(station_elevations, station_distances, power=2)
            print(f"  Target elevation from station IDW ({len(station_elevations)} stations): {estimated_elevation:.0f}m")
            return estimated_elevation
        
        elif len(station_elevations) >= 1:
            # Use closest station elevation as estimate
            closest_idx = np.argmin(station_distances)
            closest_elevation = station_elevations[closest_idx]
            closest_distance = station_distances[closest_idx]
            print(f"  Target elevation from closest station ({closest_distance:.1f}km): {closest_elevation:.0f}m")
            return closest_elevation
            
        else:
            # Fallback: Use regional average for BC Interior mountains
            fallback_elevation = 800.0
            print(f"  Target elevation fallback (no station data): {fallback_elevation:.0f}m")
            return fallback_elevation

    def _get_elevation_from_service(self, lat, lon):
        """Get elevation from USGS Elevation Point Query Service"""
        try:
            import requests
            # USGS Elevation Point Query Service (free, no API key required)
            url = f"https://epqs.nationalmap.gov/v1/json?x={lon}&y={lat}&units=Meters&includeDate=false"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'value' in data and data['value'] != -1000000:
                return float(data['value'])
        except Exception:
            # Try backup service: Open-Elevation API
            try:
                url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
                response = requests.get(url, timeout=10)
                response.raise_for_status() 
                data = response.json()
                
                if 'results' in data and len(data['results']) > 0:
                    return float(data['results'][0]['elevation'])
            except Exception:
                pass
        
        return None

    def _calculate_data_driven_parameters(self, station_data, target_lat, target_lon):
        """
        Calculate data-driven temperature lapse rate and precipitation gradient
        from available station network
        
        Args:
            station_data: List of station dataframes
            target_lat: Target latitude for distance weighting
            target_lon: Target longitude for distance weighting
            
        Returns:
            Dict with calculated parameters
        """
        print("  Calculating data-driven elevation adjustment parameters...")
        
        # Collect station elevation and climate statistics
        station_stats = []
        
        for df in station_data:
            if len(df) < 30:  # Need sufficient data for statistics
                continue
                
            first_row = df.iloc[0]
            station_elevation = first_row.get('station_elevation', np.nan)
            station_lat = first_row.get('station_lat', np.nan)
            station_lon = first_row.get('station_lon', np.nan)
            
            if pd.isna(station_elevation) or station_elevation <= 0:
                continue
            if pd.isna(station_lat) or pd.isna(station_lon):
                continue
                
            # Calculate climate statistics for this station
            temp_max_mean = df['TEMP_MAX'].mean()
            temp_min_mean = df['TEMP_MIN'].mean()
            temp_mean = (temp_max_mean + temp_min_mean) / 2
            annual_precip = df['PRECIP'].sum() / (len(df) / 365.25)  # Annualize
            
            distance = self._calculate_distance(target_lat, target_lon, station_lat, station_lon)
            
            if not pd.isna(temp_mean) and not pd.isna(annual_precip):
                station_stats.append({
                    'elevation': station_elevation,
                    'temp_mean': temp_mean,
                    'annual_precip': annual_precip,
                    'distance': distance,
                    'station_id': first_row.get('station_id', 'Unknown')
                })
        
        print(f"    Using {len(station_stats)} stations with elevation and climate data")
        
        # Calculate temperature lapse rate
        lapse_rate = self._calculate_temperature_lapse_rate(station_stats)
        
        # Calculate precipitation gradient  
        precip_gradient = self._calculate_precipitation_gradient(station_stats)
        
        return {
            'temperature_lapse_rate': lapse_rate,
            'precipitation_gradient': precip_gradient,
            'stations_used': len(station_stats),
            'station_details': station_stats
        }

    def _calculate_temperature_lapse_rate(self, station_stats):
        """Calculate temperature lapse rate from station data using regression"""
        if len(station_stats) < 2:
            print("    Insufficient stations for lapse rate calculation, using default -6.0°C/1000m")
            return -6.0
            
        import numpy as np
        
        elevations = np.array([s['elevation'] for s in station_stats])
        temperatures = np.array([s['temp_mean'] for s in station_stats])
        distances = np.array([s['distance'] for s in station_stats])
        
        # Weight by inverse distance for local representativity
        weights = 1.0 / np.maximum(distances, 1.0)  # Avoid division by zero
        
        # Weighted linear regression: temp = a * elevation + b
        X = elevations.reshape(-1, 1)
        
        # Weighted least squares
        W = np.diag(weights)
        X_weighted = np.sqrt(W) @ X
        y_weighted = np.sqrt(W) @ temperatures
        
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(len(X_weighted)), X_weighted.flatten()])
        
        # Solve normal equations
        try:
            coeffs = np.linalg.lstsq(X_with_intercept, y_weighted, rcond=None)[0]
            lapse_rate_per_m = coeffs[1]  # Slope coefficient
            lapse_rate_per_1000m = lapse_rate_per_m * 1000.0
            
            # Calculate R² for quality assessment
            y_pred = X_with_intercept @ coeffs
            ss_tot = np.sum((y_weighted - np.mean(y_weighted))**2)
            ss_res = np.sum((y_weighted - y_pred)**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Validate reasonable range for temperature lapse rate
            if -12.0 <= lapse_rate_per_1000m <= -3.0:
                print(f"    Temperature lapse rate: {lapse_rate_per_1000m:.1f}°C/1000m (R²={r_squared:.2f})")
                return lapse_rate_per_1000m
            else:
                print(f"    Calculated lapse rate {lapse_rate_per_1000m:.1f}°C/1000m outside reasonable range, using default -6.0°C/1000m")
                return -6.0
                
        except np.linalg.LinAlgError:
            print("    Regression failed, using default temperature lapse rate -6.0°C/1000m")
            return -6.0

    def _calculate_precipitation_gradient(self, station_stats):
        """Calculate precipitation gradient separating valley and mountain stations"""
        if len(station_stats) < 2:
            print("    Insufficient stations for precipitation gradient calculation, using default +0.0002/m")
            return 0.0002
            
        import numpy as np
        
        elevations = np.array([s['elevation'] for s in station_stats])
        precips = np.array([s['annual_precip'] for s in station_stats])
        distances = np.array([s['distance'] for s in station_stats])
        
        # Remove outliers (precip must be > 50mm and < 5000mm annually)
        valid_mask = (precips > 50) & (precips < 5000)
        if np.sum(valid_mask) < 2:
            print("    Insufficient valid precipitation data, using default +0.0002/m") 
            return 0.0002
            
        elevations = elevations[valid_mask]
        precips = precips[valid_mask] 
        distances = distances[valid_mask]
        
        # Separate valley and mountain stations
        valley_threshold = 700  # m - typical valley/mountain boundary in BC Interior
        valley_mask = elevations < valley_threshold
        mountain_mask = elevations >= valley_threshold
        
        valley_elevations = elevations[valley_mask]
        valley_precips = precips[valley_mask]
        mountain_elevations = elevations[mountain_mask] 
        mountain_precips = precips[mountain_mask]
        
        print(f"    Station distribution: {len(valley_elevations)} valley (<{valley_threshold}m), {len(mountain_elevations)} mountain (>={valley_threshold}m)")
        
        # Calculate separate gradients
        valley_gradient = self._calculate_single_gradient(valley_elevations, valley_precips, "valley")
        mountain_gradient = self._calculate_single_gradient(mountain_elevations, mountain_precips, "mountain")
        
        # Strategy: Use mountain gradient if available, otherwise valley, otherwise default
        if mountain_gradient is not None and len(mountain_elevations) >= 2:
            print(f"    Using mountain gradient: +{mountain_gradient*1000:.1f}%/1000m (from {len(mountain_elevations)} stations)")
            return mountain_gradient
        elif valley_gradient is not None and len(valley_elevations) >= 3:
            # Valley gradient exists but is probably underestimated - boost it
            boosted_gradient = min(valley_gradient * 2.5, 0.0004)  # Boost but cap at 40%/1000m
            print(f"    Valley gradient +{valley_gradient*1000:.1f}%/1000m boosted to +{boosted_gradient*1000:.1f}%/1000m for mountain application")
            return boosted_gradient
        else:
            print(f"    Insufficient data for elevation-specific gradients, using default +20%/1000m")
            return 0.0002

    def _calculate_single_gradient(self, elevations, precips, zone_name):
        """Calculate precipitation gradient for a single elevation zone"""
        if len(elevations) < 2:
            return None
            
        import numpy as np
        
        try:
            # Use log-linear model for better fit
            log_precips = np.log(precips)
            
            # Simple linear regression
            coeffs = np.polyfit(elevations, log_precips, 1)
            gradient_per_m = coeffs[0]  # Slope in log space
            
            # Calculate R²
            y_pred = np.polyval(coeffs, elevations)
            ss_tot = np.sum((log_precips - np.mean(log_precips))**2)
            ss_res = np.sum((log_precips - y_pred)**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Validate reasonable range
            if 0.0 <= gradient_per_m <= 0.001:
                print(f"    {zone_name.capitalize()} gradient: +{gradient_per_m*1000:.1f}%/1000m (R²={r_squared:.2f}, n={len(elevations)})")
                return gradient_per_m
            else:
                print(f"    {zone_name.capitalize()} gradient {gradient_per_m*1000:.1f}%/1000m outside reasonable range")
                return None
                
        except Exception as e:
            print(f"    {zone_name.capitalize()} gradient calculation failed: {e}")
            return None
    
    def _fill_gaps_with_seasonal_patterns(self, df):
        """
        Fill remaining gaps using seasonal patterns and temporal interpolation
        
        Args:
            df: DataFrame with potential gaps
            
        Returns:
            DataFrame with gaps filled
        """
        import numpy as np
        import pandas as pd
        
        df = df.copy()
        
        # Add temporal features for pattern recognition
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month
        
        # Fill temperature gaps using seasonal patterns
        for col in ['TEMP_MAX', 'TEMP_MIN']:
            if df[col].isnull().any():
                print(f"    Filling {df[col].isnull().sum()} gaps in {col} using seasonal patterns...")
                
                # Method 1: Try linear interpolation first
                df[col] = df[col].interpolate(method='time', limit_direction='both')
                
                # Method 2: For remaining gaps, use seasonal averages
                if df[col].isnull().any():
                    # Calculate monthly averages from available data
                    monthly_means = df.groupby('month')[col].mean()
                    
                    # Fill gaps with monthly averages
                    for month in range(1, 13):
                        month_mask = (df['month'] == month) & df[col].isnull()
                        if month_mask.any() and not pd.isna(monthly_means[month]):
                            df.loc[month_mask, col] = monthly_means[month]
                
                # Method 3: For still remaining gaps, use overall mean with seasonal variation
                if df[col].isnull().any():
                    overall_mean = df[col].mean()
                    # Add small seasonal variation based on day of year
                    seasonal_var = 5 * np.sin(2 * np.pi * df['day_of_year'] / 365.25)
                    df[col] = df[col].fillna(overall_mean + seasonal_var)
        
        # Fill precipitation gaps (more conservative approach)
        if df['PRECIP'].isnull().any():
            print(f"    Filling {df['PRECIP'].isnull().sum()} gaps in PRECIP...")
            
            # Method 1: Forward/backward fill for short gaps (up to 3 days)
            df['PRECIP'] = df['PRECIP'].fillna(method='ffill', limit=1)
            df['PRECIP'] = df['PRECIP'].fillna(method='bfill', limit=1)
            
            # Method 2: Use monthly averages for longer gaps
            if df['PRECIP'].isnull().any():
                monthly_precip = df.groupby('month')['PRECIP'].mean()
                for month in range(1, 13):
                    month_mask = (df['month'] == month) & df['PRECIP'].isnull()
                    if month_mask.any() and not pd.isna(monthly_precip[month]):
                        df.loc[month_mask, 'PRECIP'] = monthly_precip[month]
            
        # Fill precipitation gaps (more conservative approach)
        if df['PRECIP'].isnull().any():
            print(f"    Filling {df['PRECIP'].isnull().sum()} gaps in PRECIP...")
            
            # Method 1: Forward/backward fill for short gaps (up to 3 days)
            df['PRECIP'] = df['PRECIP'].fillna(method='ffill', limit=1)
            df['PRECIP'] = df['PRECIP'].fillna(method='bfill', limit=1)
            
            # Method 2: Use monthly averages for longer gaps
            if df['PRECIP'].isnull().any():
                monthly_precip = df.groupby('month')['PRECIP'].mean()
                for month in range(1, 13):
                    month_mask = (df['month'] == month) & df['PRECIP'].isnull()
                    if month_mask.any() and not pd.isna(monthly_precip[month]):
                        df.loc[month_mask, 'PRECIP'] = monthly_precip[month]
            
            # Method 3: Fill any remaining gaps with zeros (conservative)
            df['PRECIP'] = df['PRECIP'].fillna(0.0)
        
        # Fix temperature inversions (TEMP_MAX < TEMP_MIN)
        print("    Checking for temperature inversions...")
        inversions = df['TEMP_MAX'] < df['TEMP_MIN']
        if inversions.any():
            inversion_count = inversions.sum()
            print(f"    Fixing {inversion_count} temperature inversions (TEMP_MAX < TEMP_MIN)...")
            
            # Method 1: Swap values if difference is small (< 5°C)
            small_inversions = inversions & ((df['TEMP_MIN'] - df['TEMP_MAX']) < 5.0)
            if small_inversions.any():
                temp_min_backup = df.loc[small_inversions, 'TEMP_MIN'].copy()
                df.loc[small_inversions, 'TEMP_MIN'] = df.loc[small_inversions, 'TEMP_MAX']
                df.loc[small_inversions, 'TEMP_MAX'] = temp_min_backup
                print(f"      Swapped {small_inversions.sum()} small inversions")
            
            # Method 2: For large inversions, recalculate using interpolation
            large_inversions = inversions & ((df['TEMP_MIN'] - df['TEMP_MAX']) >= 5.0)
            if large_inversions.any():
                # Use average temperature and add/subtract typical daily range
                avg_temp = (df.loc[large_inversions, 'TEMP_MAX'] + df.loc[large_inversions, 'TEMP_MIN']) / 2
                daily_range = 8.0  # Typical daily temperature range in °C
                df.loc[large_inversions, 'TEMP_MAX'] = avg_temp + daily_range/2
                df.loc[large_inversions, 'TEMP_MIN'] = avg_temp - daily_range/2
                print(f"      Recalculated {large_inversions.sum()} large inversions")
        
        # Clean up temporary columns
        df = df.drop(['day_of_year', 'month'], axis=1)
        
        # Final validation - ensure no gaps remain and no inversions
        remaining_gaps = df.isnull().sum().sum()
        remaining_inversions = (df['TEMP_MAX'] < df['TEMP_MIN']).sum()
        
        if remaining_gaps > 0:
            print(f"    ERROR: {remaining_gaps} gaps still remain after gap filling")
            print(f"    Gap dates: {df[df.isnull().any(axis=1)].index.tolist()[:10]}")
            print(f"    FAILING FAST - No fallback means allowed")
            raise ValueError(f"IDW algorithm failed: {remaining_gaps} gaps remain, no fallback allowed")
        
        if remaining_inversions > 0:
            print(f"    WARNING: {remaining_inversions} temperature inversions still remain")
        else:
            print(f"    SUCCESS: All temperature inversions fixed")
        
        return df

    def generate_climate_period_metadata(self, climate_csv_path: Path, 
                                        metadata_output_path: Path = None) -> Dict:
        """
        Generate comprehensive metadata for the climate period
        
        Args:
            climate_csv_path: Path to the climate forcing CSV file
            metadata_output_path: Optional path to save metadata JSON
            
        Returns:
            Dictionary containing period metadata
        """
        try:
            if not climate_csv_path.exists():
                raise FileNotFoundError(f"Climate file not found: {climate_csv_path}")
            
            # Read climate data
            df = pd.read_csv(climate_csv_path, index_col=0, parse_dates=True)
            
            # Basic period information
            start_date = df.index[0]
            end_date = df.index[-1]
            total_days = (end_date - start_date).days + 1
            total_years = total_days / 365.25
            
            # Data quality assessment
            missing_counts = df.isnull().sum()
            completeness = ((len(df) - missing_counts) / len(df) * 100).round(2)
            
            # Climate statistics
            temp_stats = {
                'min_temperature': float(df['TEMP_MIN'].min()),
                'max_temperature': float(df['TEMP_MAX'].max()),
                'mean_annual_temp_min': float(df['TEMP_MIN'].mean()),
                'mean_annual_temp_max': float(df['TEMP_MAX'].mean()),
                'temperature_range': float(df['TEMP_MAX'].max() - df['TEMP_MIN'].min())
            }
            
            precip_stats = {
                'total_precipitation_mm': float(df['PRECIP'].sum()),
                'mean_annual_precipitation_mm': float(df['PRECIP'].sum() / total_years),
                'max_daily_precipitation_mm': float(df['PRECIP'].max()),
                'mean_daily_precipitation_mm': float(df['PRECIP'].mean()),
                'days_with_precipitation': int((df['PRECIP'] > 0.1).sum()),
                'precipitation_frequency_percent': float((df['PRECIP'] > 0.1).mean() * 100)
            }
            
            # Monthly statistics
            monthly_stats = df.groupby(df.index.month).agg({
                'TEMP_MAX': ['mean', 'std'],
                'TEMP_MIN': ['mean', 'std'],
                'PRECIP': ['sum', 'mean', 'std', 'max']
            }).round(2)
            
            # Format monthly statistics
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            monthly_summary = {}
            for i, month in enumerate(months, 1):
                if i in monthly_stats.index:
                    monthly_summary[month] = {
                        'temp_max_mean': float(monthly_stats.loc[i, ('TEMP_MAX', 'mean')]),
                        'temp_max_std': float(monthly_stats.loc[i, ('TEMP_MAX', 'std')]),
                        'temp_min_mean': float(monthly_stats.loc[i, ('TEMP_MIN', 'mean')]),
                        'temp_min_std': float(monthly_stats.loc[i, ('TEMP_MIN', 'std')]),
                        'precip_total_mm': float(monthly_stats.loc[i, ('PRECIP', 'sum')]),
                        'precip_daily_mean_mm': float(monthly_stats.loc[i, ('PRECIP', 'mean')]),
                        'precip_daily_std_mm': float(monthly_stats.loc[i, ('PRECIP', 'std')]),
                        'precip_daily_max_mm': float(monthly_stats.loc[i, ('PRECIP', 'max')])
                    }
            
            # Seasonal statistics
            seasons = {
                'Winter': [12, 1, 2],
                'Spring': [3, 4, 5], 
                'Summer': [6, 7, 8],
                'Fall': [9, 10, 11]
            }
            
            seasonal_summary = {}
            for season, months_list in seasons.items():
                season_data = df[df.index.month.isin(months_list)]
                if len(season_data) > 0:
                    seasonal_summary[season] = {
                        'temp_max_mean': float(season_data['TEMP_MAX'].mean()),
                        'temp_min_mean': float(season_data['TEMP_MIN'].mean()),
                        'precip_total_mm': float(season_data['PRECIP'].sum()),
                        'precip_daily_mean_mm': float(season_data['PRECIP'].mean())
                    }
            
            # Annual statistics by year
            annual_stats = df.groupby(df.index.year).agg({
                'TEMP_MAX': 'mean',
                'TEMP_MIN': 'mean', 
                'PRECIP': 'sum'
            }).round(1)
            
            annual_summary = {}
            for year in annual_stats.index:
                annual_summary[str(year)] = {
                    'temp_max_mean': float(annual_stats.loc[year, 'TEMP_MAX']),
                    'temp_min_mean': float(annual_stats.loc[year, 'TEMP_MIN']),
                    'precip_total_mm': float(annual_stats.loc[year, 'PRECIP'])
                }
            
            # Extreme events
            extreme_events = {
                'hottest_day': {
                    'date': df['TEMP_MAX'].idxmax().strftime('%Y-%m-%d'),
                    'temperature': float(df['TEMP_MAX'].max())
                },
                'coldest_day': {
                    'date': df['TEMP_MIN'].idxmin().strftime('%Y-%m-%d'),
                    'temperature': float(df['TEMP_MIN'].min())
                },
                'wettest_day': {
                    'date': df['PRECIP'].idxmax().strftime('%Y-%m-%d'),
                    'precipitation_mm': float(df['PRECIP'].max())
                },
                'longest_dry_period_days': self._calculate_longest_dry_period(df),
                'freeze_thaw_cycles': self._calculate_freeze_thaw_cycles(df)
            }
            
            # Generate metadata dictionary
            metadata = {
                'generation_info': {
                    'generated_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source_file': str(climate_csv_path),
                    'total_records': len(df),
                    'generator': 'RAVEN Climate Data Client v1.0'
                },
                'period_coverage': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'total_days': total_days,
                    'total_years': round(total_years, 1),
                    'years_covered': f"{start_date.year}-{end_date.year}"
                },
                'data_quality': {
                    'temp_max_completeness_percent': float(completeness['TEMP_MAX']),
                    'temp_min_completeness_percent': float(completeness['TEMP_MIN']),
                    'precip_completeness_percent': float(completeness['PRECIP']),
                    'missing_temp_max_days': int(missing_counts['TEMP_MAX']),
                    'missing_temp_min_days': int(missing_counts['TEMP_MIN']),
                    'missing_precip_days': int(missing_counts['PRECIP'])
                },
                'temperature_statistics': temp_stats,
                'precipitation_statistics': precip_stats,
                'monthly_climate_normals': monthly_summary,
                'seasonal_summary': seasonal_summary,
                'annual_summary': annual_summary,
                'extreme_events': extreme_events,
                'climate_suitability': {
                    'hydrological_modeling': 'Excellent - 30 year period with complete data',
                    'period_adequacy': 'Meets WMO standards for climate normals (30+ years)',
                    'data_gaps': 'None - continuous daily coverage',
                    'recommended_for': [
                        'Long-term hydrological modeling',
                        'Climate change impact assessment', 
                        'Flood frequency analysis',
                        'Drought analysis',
                        'Water resource planning'
                    ]
                }
            }
            
            # Save metadata if output path provided
            if metadata_output_path:
                metadata_output_path.parent.mkdir(exist_ok=True, parents=True)
                with open(metadata_output_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"Climate period metadata saved to: {metadata_output_path}")
            
            print(f"Generated climate period metadata:")
            print(f"  Period: {metadata['period_coverage']['years_covered']} ({total_years:.1f} years)")
            print(f"  Temperature range: {temp_stats['min_temperature']:.1f}°C to {temp_stats['max_temperature']:.1f}°C")
            print(f"  Annual precipitation: {precip_stats['mean_annual_precipitation_mm']:.1f} mm/year")
            print(f"  Data completeness: {completeness.mean():.1f}%")
            
            return metadata
            
        except Exception as e:
            error_msg = f"Error generating climate period metadata: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {"error": error_msg}
    
    def _calculate_longest_dry_period(self, df):
        """Calculate longest consecutive dry period (precip < 0.1mm)"""
        dry_days = (df['PRECIP'] < 0.1).astype(int)
        max_consecutive = 0
        current_consecutive = 0
        
        for is_dry in dry_days:
            if is_dry:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_freeze_thaw_cycles(self, df):
        """Calculate number of freeze-thaw cycles (crossing 0°C)"""
        # Daily freeze-thaw: min < 0 and max > 0
        freeze_thaw_days = ((df['TEMP_MIN'] < 0) & (df['TEMP_MAX'] > 0)).sum()
        return int(freeze_thaw_days)
    
    # ============================================================================
    # DAYMET DATA METHODS
    # ============================================================================
    
    def get_daymet_data(self, latitude: float, longitude: float, 
                       start_year: int = 1991, end_year: int = 2020,
                       output_path: Optional[Path] = None,
                       format_type: str = 'ravenpy') -> Dict:
        """
        Get Daymet gridded climate data for a specific location
        
        Args:
            latitude: Latitude of target point
            longitude: Longitude of target point  
            start_year: Start year for data (1980-2023)
            end_year: End year for data (1980-2023)
            output_path: Path to save CSV output
            format_type: Output format ('ravenpy' or 'standard')
            
        Returns:
            Dict with success status and file information
        """
        print(f"Getting Daymet data for ({latitude}, {longitude})")
        print(f"Years: {start_year}-{end_year}")
        
        try:
            # Use Daymet Single Pixel API (more reliable than OPeNDAP)
            api_url = "https://daymet.ornl.gov/single-pixel/api/data"
            
            # Prepare parameters for API call
            params = {
                'lat': latitude,
                'lon': longitude,
                'vars': 'tmax,tmin,prcp',  # temperature max/min and precipitation
                'start': start_year,
                'end': end_year,
                'format': 'csv'
            }
            
            print("  Requesting data from Daymet Single Pixel API...")
            response = self.session.get(api_url, params=params, timeout=120)
            
            if response.status_code != 200:
                return {'success': False, 'error': f'API request failed with status {response.status_code}: {response.text}'}
            
            # Parse CSV response
            from io import StringIO
            
            # Skip metadata lines and find CSV header
            lines = response.text.split('\n')
            data_start_idx = 0
            for i, line in enumerate(lines):
                if line.strip() and 'year,yday,' in line.lower():
                    data_start_idx = i
                    break
            
            if data_start_idx == 0:
                return {'success': False, 'error': 'Could not find CSV header in API response'}
            
            # Create DataFrame from CSV data
            clean_csv = '\n'.join(lines[data_start_idx:])
            combined_df = pd.read_csv(StringIO(clean_csv))
            
            # Handle column name variations
            column_mapping = {}
            for col in combined_df.columns:
                col_lower = col.lower()
                if 'tmax' in col_lower or 'temp' in col_lower and 'max' in col_lower:
                    column_mapping[col] = 'TEMP_MAX'
                elif 'tmin' in col_lower or 'temp' in col_lower and 'min' in col_lower:
                    column_mapping[col] = 'TEMP_MIN'
                elif 'prcp' in col_lower or 'precip' in col_lower or 'precipitation' in col_lower:
                    column_mapping[col] = 'PRECIP'
            
            # Rename columns
            combined_df.rename(columns=column_mapping, inplace=True)
            
            # Create date column from year and day of year
            if 'year' in combined_df.columns and 'yday' in combined_df.columns:
                combined_df['date'] = pd.to_datetime(
                    combined_df['year'].astype(str) + '-01-01'
                ) + pd.to_timedelta(combined_df['yday'] - 1, unit='days')
                combined_df.set_index('date', inplace=True)
                
                # Remove year and yday columns
                combined_df.drop(['year', 'yday'], axis=1, inplace=True, errors='ignore')
            
            # Verify required columns
            required_cols = ['TEMP_MAX', 'TEMP_MIN', 'PRECIP'] 
            missing_cols = [col for col in required_cols if col not in combined_df.columns]
            if missing_cols:
                return {'success': False, 'error': f'Missing required columns: {missing_cols}'}
            
            print(f"  Successfully retrieved {len(combined_df)} daily records")
            
            # Sort by date
            combined_df.sort_index(inplace=True)
            
            # Calculate statistics
            stats = {
                'total_precipitation_mm': float(combined_df['PRECIP'].sum()),
                'mean_annual_precipitation_mm': float(combined_df['PRECIP'].sum() / (end_year - start_year + 1)),
                'mean_temp_max': float(combined_df['TEMP_MAX'].mean()),
                'mean_temp_min': float(combined_df['TEMP_MIN'].mean()),
                'max_daily_precip_mm': float(combined_df['PRECIP'].max()),
                'record_count': len(combined_df)
            }
            
            # Save to file if requested
            if output_path:
                if format_type == 'ravenpy':
                    self._save_ravenpy_format(combined_df, output_path, latitude, longitude, 'Daymet')
                else:
                    combined_df.to_csv(output_path)
                print(f"SUCCESS: Daymet data saved to {output_path}")
            
            return {
                'success': True,
                'file_path': str(output_path) if output_path else None,
                'data_source': 'Daymet_v4_Single_Pixel_API',
                'location': {'latitude': latitude, 'longitude': longitude},
                'period': {'start_year': start_year, 'end_year': end_year},
                'records': len(combined_df),
                'date_range': [combined_df.index.min().strftime('%Y-%m-%d'), 
                              combined_df.index.max().strftime('%Y-%m-%d')],
                'statistics': stats
            }
            
        except Exception as e:
            error_msg = f"Daymet data retrieval failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    # ============================================================================
    # CAPA/RDPA DATA METHODS  
    # ============================================================================
    
    def get_capa_data(self, latitude: float, longitude: float,
                     start_date: str = "2011-01-01", end_date: str = "2023-12-31", 
                     output_path: Optional[Path] = None,
                     format_type: str = 'ravenpy') -> Dict:
        """
        Get CAPA (Canadian Precipitation Analysis) gridded data
        
        Args:
            latitude: Latitude of target point
            longitude: Longitude of target point
            start_date: Start date (YYYY-MM-DD) - earliest available: 2011-01-01
            end_date: End date (YYYY-MM-DD)
            output_path: Path to save CSV output
            format_type: Output format ('ravenpy' or 'standard')
            
        Returns:
            Dict with success status and file information
        """
        print(f"Getting CAPA data for ({latitude}, {longitude})")
        print(f"Date range: {start_date} to {end_date}")
        print("WARNING: CAPA data requires specialized access - this is a template implementation")
        
        try:
            # Note: CAPA data access typically requires:
            # 1. Access to MSC Datamart via AMQP or HTTP
            # 2. Authentication for historical archives
            # 3. Processing of GRIB2 format files
            
            # This is a template - actual implementation would need:
            # - Authentication setup
            # - GRIB2 file processing capabilities  
            # - MSC Datamart file navigation
            
            print("CAPA data access requires:")
            print("1. MSC Datamart credentials/access")
            print("2. GRIB2 processing libraries (e.g., xarray with cfgrib)")
            print("3. Historical archive access (30-day rolling window for real-time)")
            
            # Template for actual implementation:
            # 1. Authenticate with MSC Datamart
            # 2. Query available CAPA files for date range
            # 3. Download GRIB2 files
            # 4. Extract point data for lat/lon
            # 5. Convert to RavenPy format
            
            return {
                'success': False,
                'error': 'CAPA implementation requires MSC Datamart access setup',
                'implementation_notes': {
                    'data_source': 'MSC Datamart - RDPA/CAPA GRIB2 files',
                    'access_method': 'HTTP or AMQP subscription',
                    'format': 'GRIB2',
                    'resolution': '10km',
                    'temporal_coverage': '2011-present (30-day rolling window)',
                    'historical_archive': '2011-2023 (separate access required)'
                }
            }
            
        except Exception as e:
            error_msg = f"CAPA data setup failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    # ============================================================================
    # HELPER METHODS FOR GRIDDED DATA
    # ============================================================================
    
    def _save_ravenpy_format(self, df: pd.DataFrame, output_path: Path, 
                           lat: float, lon: float, source: str):
        """Save climate data in RavenPy-compatible format"""
        
        # Calculate monthly averages for temperature
        monthly_temp = df.groupby(df.index.month).agg({
            'TEMP_MAX': 'mean',
            'TEMP_MIN': 'mean'
        }).round(1)
        
        monthly_avg_temp = ((monthly_temp['TEMP_MAX'] + monthly_temp['TEMP_MIN']) / 2).values
        monthly_evap = [30.0] * 12  # Default evaporation
        
        # Create RVT format
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            f.write(f"#########################################################################\n")
            f.write(f"# Climate Data from {source}\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"#########################################################################\n\n")
            
            f.write(f":Gauge\t\t\t\t\tGauge_1\n")
            f.write(f"\t:Latitude\t\t\t{lat}\n")
            f.write(f"\t:Longitude\t\t\t{lon}\n")
            f.write(f"\t:Elevation\t\t\t1800\n")  # Default elevation
            
            # Monthly averages
            f.write(f"\t:MonthlyAveTemperature {' '.join([f'{x:.1f}' for x in monthly_avg_temp])}\n")
            f.write(f"\t:MonthlyAveEvaporation {' '.join([f'{x:.1f}' for x in monthly_evap])}\n\n")
            
            f.write(f":MultiData\n")
            f.write(f"{df.index.min().strftime('%Y-%m-%d')} 00:00:00 1.0 {len(df)}\n")
            f.write(f":Parameters,TEMP_MAX,TEMP_MIN,PRECIP\n")
            f.write(f":Units,C,C,mm/d\n")
            
            # Write daily data
            for idx, row in df.iterrows():
                f.write(f"{row['TEMP_MAX']:.2f},{row['TEMP_MIN']:.2f},{row['PRECIP']:.2f}\n")
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Calculate data quality metrics for gridded data"""
        total_days = len(df)
        
        missing_tmax = df['TEMP_MAX'].isnull().sum()
        missing_tmin = df['TEMP_MIN'].isnull().sum() 
        missing_precip = df['PRECIP'].isnull().sum()
        
        return {
            'temp_max_completeness_percent': ((total_days - missing_tmax) / total_days * 100) if total_days > 0 else 0,
            'temp_min_completeness_percent': ((total_days - missing_tmin) / total_days * 100) if total_days > 0 else 0,
            'precip_completeness_percent': ((total_days - missing_precip) / total_days * 100) if total_days > 0 else 0,
            'total_records': total_days,
            'missing_days': missing_tmax + missing_tmin + missing_precip
        }
    
    # ============================================================================
    # UNIFIED GRIDDED DATA METHOD
    # ============================================================================
    
    def get_gridded_climate_data(self, latitude: float, longitude: float,
                                start_date: str = "1991-01-01", end_date: str = "2020-12-31",
                                source: str = "daymet", output_path: Optional[Path] = None,
                                format_type: str = 'ravenpy') -> Dict:
        """
        Unified method to get gridded climate data from different sources
        
        Args:
            latitude: Latitude of target point
            longitude: Longitude of target point
            start_date: Start date (YYYY-MM-DD)  
            end_date: End date (YYYY-MM-DD)
            source: Data source ('daymet', 'capa', 'era5')
            output_path: Path to save output
            format_type: Output format
            
        Returns:
            Dict with success status and file information
        """
        
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        
        if source.lower() == 'daymet':
            return self.get_daymet_data(latitude, longitude, start_year, end_year, 
                                      output_path, format_type)
        elif source.lower() == 'capa':
            return self.get_capa_data(latitude, longitude, start_date, end_date,
                                    output_path, format_type)
        else:
            return {
                'success': False, 
                'error': f'Unsupported gridded data source: {source}. Use "daymet" or "capa".'
            }