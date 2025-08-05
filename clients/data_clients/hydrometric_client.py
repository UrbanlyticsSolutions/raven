#!/usr/bin/env python3
"""
Hydrometric Data Client for RAVEN Hydrological Modeling
Downloads streamflow and water level data from Environment and Climate Change Canada (ECCC) APIs
Outputs RavenPy-compatible CSV formats
"""

import requests
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class HydrometricDataClient:
    """Client for downloading hydrometric data with RavenPy-compatible CSV output"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAVEN-Hydrological-Model-Client/1.0'
        })
        self.eccc_base_url = "https://api.weather.gc.ca"
    
    def get_hydrometric_stations_for_watershed(self, bbox: Tuple[float, float, float, float], 
                                              output_path: Optional[Path] = None) -> Dict:
        """Get hydrometric stations within watershed for streamflow data"""
        minx, miny, maxx, maxy = bbox
        bbox_str = f"{minx},{miny},{maxx},{maxy}"
        
        params = {
            'bbox': bbox_str,
            'limit': 100
        }
        
        print(f"Getting hydrometric stations for bbox: {bbox_str}")
        
        try:
            response = self.session.get(f"{self.eccc_base_url}/collections/hydrometric-stations/items", 
                                      params=params, timeout=30)
            response.raise_for_status()
            
            stations_data = response.json()
            
            if "features" in stations_data:
                num_stations = len(stations_data["features"])
                print(f"SUCCESS: Found {num_stations} hydrometric stations")
                
                # Show sample stations
                for i, station in enumerate(stations_data["features"][:3]):
                    props = station["properties"]
                    name = props.get("STATION_NAME", "N/A")
                    station_id = props.get("STATION_NUMBER", "N/A")
                    drainage_area = props.get("DRAINAGE_AREA_GROSS", "N/A")
                    print(f"   {i+1}. {name} (ID: {station_id}, Area: {drainage_area} km²)")
                
                # Save to file if requested
                if output_path:
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    with open(output_path, 'w') as f:
                        json.dump(stations_data, f, indent=2)
                    print(f"Hydrometric stations saved to: {output_path}")
                
                return stations_data
            else:
                return {"features": [], "message": "No stations found"}
                
        except Exception as e:
            error_msg = f"Hydrometric stations query failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {"error": error_msg}
    
    def get_streamflow_data_csv(self, station_id: str, start_date: str, end_date: str,
                               output_path: Path) -> Dict:
        """Download streamflow data and save as RavenPy-compatible CSV"""
        print(f"Getting streamflow data for station {station_id} ({start_date} to {end_date})")
        
        try:
            params = {
                'STATION_NUMBER': station_id,
                'datetime': f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
                'limit': 10000
            }
            
            response = self.session.get(
                f"{self.eccc_base_url}/collections/hydrometric-daily-mean/items",
                params=params, timeout=60)
            response.raise_for_status()
            
            streamflow_data = response.json()
            
            if "features" in streamflow_data and streamflow_data["features"]:
                # Convert to DataFrame
                records = []
                for feature in streamflow_data["features"]:
                    props = feature["properties"]
                    records.append({
                        'date': props.get('DATE'),
                        'discharge': props.get('DISCHARGE'),
                        'water_level': props.get('WATER_LEVEL'),
                        'station_id': station_id
                    })
                
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').dropna(subset=['date'])
                
                if len(df) == 0:
                    return {'success': False, 'error': 'No valid records after processing'}
                
                # Clean and format for RavenPy
                df['discharge'] = pd.to_numeric(df['discharge'], errors='coerce')
                df['water_level'] = pd.to_numeric(df['water_level'], errors='coerce')
                
                # RavenPy compatible format
                output_df = df[['date', 'discharge', 'water_level']].copy()
                output_df.columns = ['Date', 'Discharge_cms', 'WaterLevel_m']
                output_df = output_df.set_index('Date')
                
                # Remove rows where both discharge and water level are NaN
                output_df = output_df.dropna(how='all')
                
                # Save CSV
                output_path.parent.mkdir(exist_ok=True, parents=True)
                output_df.to_csv(output_path, float_format='%.3f')
                
                discharge_count = output_df['Discharge_cms'].notna().sum()
                level_count = output_df['WaterLevel_m'].notna().sum()
                
                print(f"SUCCESS: Streamflow CSV created ({len(output_df)} days)")
                print(f"Discharge: {discharge_count}/{len(output_df)} records ({100*discharge_count/len(output_df):.1f}%)")
                if discharge_count > 0:
                    print(f"Mean discharge: {output_df['Discharge_cms'].mean():.2f} m³/s")
                
                return {
                    'success': True,
                    'file_path': str(output_path),
                    'records': len(output_df),
                    'date_range': [output_df.index.min().strftime('%Y-%m-%d'), 
                                 output_df.index.max().strftime('%Y-%m-%d')],
                    'data_quality': {
                        'discharge_completeness': (discharge_count / len(output_df)) * 100,
                        'water_level_completeness': (level_count / len(output_df)) * 100
                    },
                    'statistics': {
                        'mean_discharge': output_df['Discharge_cms'].mean() if discharge_count > 0 else None,
                        'max_discharge': output_df['Discharge_cms'].max() if discharge_count > 0 else None,
                        'min_discharge': output_df['Discharge_cms'].min() if discharge_count > 0 else None
                    }
                }
            else:
                return {'success': False, 'error': 'No streamflow data features found'}
                
        except Exception as e:
            error_msg = f"Streamflow data download failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'success': False, 'error': error_msg}