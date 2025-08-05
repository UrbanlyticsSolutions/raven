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
                print(f"Temperature: {output_df[temp_min_col].min():.1f}°C to {output_df[temp_max_col].max():.1f}°C, Precipitation: {output_df[precip_col].sum():.1f}mm")
                
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