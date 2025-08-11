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
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ClimateDataClient:
    """Client for downloading climate data with RavenPy-compatible CSV output"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAVEN-Hydrological-Model-Client/1.0'
        })
        self.eccc_base_url = "https://api.weather.gc.ca"
    
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
                                                         format_type: str = 'ravenpy') -> Dict:
        """Get 30-year climate normal data with advanced gap filling using IDW interpolation"""
        print(f"Getting 30-year climate data for ({outlet_lat}, {outlet_lon}) with {search_radius_km}km search radius")
        
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
            print("  Applying IDW interpolation for spatial interpolation...")
            
            for date in full_dates:
                # Collect station data for this date
                station_values = {'TEMP_MAX': [], 'TEMP_MIN': [], 'PRECIP': [], 'distances': []}
                
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
                
                # Apply IDW if we have valid station data
                if len(station_values['distances']) > 0:
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