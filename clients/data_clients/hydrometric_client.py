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
import math
warnings.filterwarnings('ignore')

class HydrometricDataClient:
    """Client for downloading hydrometric data with RavenPy-compatible CSV output"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAVEN-Hydrological-Model-Client/1.0'
        })
        self.eccc_base_url = "https://api.weather.gc.ca"
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of Earth in kilometers
        r = 6371
        
        return c * r
    
    def get_hydrometric_stations_for_watershed(self, bbox: Tuple[float, float, float, float], 
                                              output_path: Optional[Path] = None) -> Dict:
        """Get hydrometric stations within watershed for streamflow data"""
        minx, miny, maxx, maxy = bbox
        bbox_str = f"{minx},{miny},{maxx},{maxy}"
        
        params = {
            'bbox': bbox_str,
            'limit': 5000
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
                    print(f"   {i+1}. {name} (ID: {station_id}, Area: {drainage_area} km2)")
                
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
                               output_path: Path, debug: bool = False) -> Dict:
        """Download streamflow data and save as RavenPy-compatible CSV"""
        print(f"Getting streamflow data for station {station_id} ({start_date} to {end_date})")
        
        try:
            # Use pagination to get ALL data (API limits to 10,000 per request)
            all_features = []
            offset = 0
            limit = 10000  # API maximum
            
            while True:
                params = {
                    'STATION_NUMBER': station_id,
                    'datetime': f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
                    'limit': limit,
                    'offset': offset
                }
                
                response = self.session.get(
                    f"{self.eccc_base_url}/collections/hydrometric-daily-mean/items",
                    params=params, timeout=60)
                response.raise_for_status()
                
                batch_data = response.json()
                batch_features = batch_data.get('features', [])
                
                if not batch_features:
                    break  # No more data
                
                all_features.extend(batch_features)
                print(f"   Downloaded {len(batch_features)} records (total: {len(all_features)})")
                
                # Check if we got fewer records than requested (end of data)
                if len(batch_features) < limit:
                    break
                    
                offset += limit
            
            # Combine all paginated data
            streamflow_data = {'features': all_features}
            
            # DEBUG: Show raw response structure
            if debug:
                print(f"DEBUG: DEBUG - Station {station_id}:")
                print(f"   Response keys: {list(streamflow_data.keys())}")
                if "features" in streamflow_data:
                    print(f"   Number of features: {len(streamflow_data.get('features', []))}")
                    if streamflow_data["features"]:
                        # Show first feature structure
                        first_feature = streamflow_data["features"][0]
                        print(f"   First feature keys: {list(first_feature.keys())}")
                        if "properties" in first_feature:
                            props = first_feature["properties"]
                            print(f"   Properties keys: {list(props.keys())}")
                            print(f"   Sample date: {props.get('DATE')}")
                            print(f"   Sample discharge: {props.get('DISCHARGE')}")
                            print(f"   Sample water level: {props.get('WATER_LEVEL')}")
                        
                        # Show date range in response
                        dates = [f["properties"].get("DATE") for f in streamflow_data["features"] if f.get("properties", {}).get("DATE")]
                        if dates:
                            print(f"   Available date range: {min(dates)} to {max(dates)}")
                            print(f"   Total records in response: {len(dates)}")
            
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
                    if debug:
                        print(f"   ERROR: No valid records after date processing")
                    return {'success': False, 'error': 'No valid records after processing'}
                
                # Clean and format for RavenPy
                df['discharge'] = pd.to_numeric(df['discharge'], errors='coerce')
                df['water_level'] = pd.to_numeric(df['water_level'], errors='coerce')
                
                # DEBUG: Show data quality
                if debug:
                    discharge_valid = df['discharge'].notna().sum()
                    level_valid = df['water_level'].notna().sum()
                    print(f"   INFO: Data quality after processing:")
                    print(f"      Valid discharge records: {discharge_valid}/{len(df)} ({100*discharge_valid/len(df):.1f}%)")
                    print(f"      Valid water level records: {level_valid}/{len(df)} ({100*level_valid/len(df):.1f}%)")
                    print(f"      Date range after processing: {df['date'].min()} to {df['date'].max()}")
                
                # RavenPy compatible format
                output_df = df[['date', 'discharge', 'water_level']].copy()
                output_df.columns = ['Date', 'Discharge_cms', 'WaterLevel_m']
                output_df = output_df.set_index('Date')
                
                # Remove rows where both discharge and water level are NaN
                output_df = output_df.dropna(how='all')
                
                if len(output_df) == 0:
                    if debug:
                        print(f"   ERROR: No valid records after removing NaN rows")
                    return {'success': False, 'error': 'No valid records after removing NaN values'}
                
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
                if debug:
                    print(f"   ERROR: No features in response or features list is empty")
                    if "features" in streamflow_data:
                        print(f"   Features list length: {len(streamflow_data['features'])}")
                return {'success': False, 'error': 'No streamflow data features found'}
                
        except Exception as e:
            error_msg = f"Streamflow data download failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            if debug:
                print(f"DEBUG: DEBUG - Exception details: {type(e).__name__}: {str(e)}")
            return {'success': False, 'error': error_msg}
    
    def find_best_hydrometric_stations_with_data(self, bbox: Tuple[float, float, float, float] = None,
                                                outlet_lat: float = None, outlet_lon: float = None, 
                                                search_range_km: float = 100.0,
                                                max_stations: int = 15,
                                                min_years: int = 10,
                                                start_year: int = 1990, end_year: int = 2024) -> List[Dict]:
        """Find closest hydrometric stations with 10+ years of actual data using proper distance calculation"""
        
        # Handle bbox input or create from lat/lon
        if bbox is not None:
            minx, miny, maxx, maxy = bbox
            center_lat = (miny + maxy) / 2
            center_lon = (minx + maxx) / 2
        elif outlet_lat is not None and outlet_lon is not None:
            center_lat, center_lon = outlet_lat, outlet_lon
            # Create larger bbox around point for wider search
            deg_radius = search_range_km / 111.0
            bbox = (center_lon - deg_radius, center_lat - deg_radius, 
                   center_lon + deg_radius, center_lat + deg_radius)
        else:
            return []
        
        print(f"INFO: Finding CLOSEST hydrometric stations to ({center_lat:.4f}, {center_lon:.4f})")
        print(f"   Search radius: {search_range_km} km")
        print(f"   Minimum data requirement: {min_years}+ years")
        print(f"   Date range: {start_year}-{end_year}")
        
        try:
            # Get all stations in the expanded search area
            stations_result = self.get_hydrometric_stations_for_watershed(bbox)
            
            if not stations_result.get("features"):
                print("ERROR: No stations found in search area")
                return []
            
            stations = stations_result["features"]
            print(f"INFO: Found {len(stations)} total stations in search area")
            
            # Calculate distances and sort by proximity
            station_distances = []
            
            for station in stations:
                props = station["properties"]
                station_id = props.get("STATION_NUMBER")
                station_name = props.get("STATION_NAME", "Unknown")
                
                # Get station coordinates
                geometry = station.get("geometry", {})
                coordinates = geometry.get("coordinates", [])
                
                if not station_id or not coordinates or len(coordinates) < 2:
                    continue
                
                station_lon, station_lat = coordinates[0], coordinates[1]
                
                # Calculate actual distance using Haversine formula
                distance_km = self.calculate_distance(center_lat, center_lon, station_lat, station_lon)
                
                # Only include stations within search radius
                if distance_km <= search_range_km:
                    station_distances.append({
                        'station_id': station_id,
                        'station_name': station_name,
                        'distance_km': distance_km,
                        'latitude': station_lat,
                        'longitude': station_lon,
                        'drainage_area': props.get("DRAINAGE_AREA_GROSS", 0),
                        'properties': props
                    })
            
            # Sort by distance (closest first)
            station_distances.sort(key=lambda x: x['distance_km'])
            
            print(f"INFO: Found {len(station_distances)} stations within {search_range_km} km radius")
            
            if not station_distances:
                print(f"ERROR: No stations found within {search_range_km} km. Try increasing search_range_km.")
                return []
            
            # Check stations one by one, stopping at first suitable station
            stations_with_data = []
            max_to_check = len(station_distances)
            
            print(f"\nDEBUG: Checking data availability - will STOP at first suitable station...")
            
            for i, station_info in enumerate(station_distances):
                station_id = station_info['station_id']
                station_name = station_info['station_name']
                distance = station_info['distance_km']
                
                print(f"\n   Station {i+1}/{max_to_check}: {station_name} ({station_id})")
                print(f"   Distance: {distance:.2f} km")
                
                # Check actual data range
                data_metadata = self.check_station_data_range(station_id, extensive_check=True)
                
                # Check if meets minimum years requirement
                if data_metadata['total_years_with_data'] >= min_years:
                    # Calculate comprehensive score
                    distance_score = max(0, 1 - (distance / search_range_km))  # Closer = better
                    years_score = min(data_metadata['total_years_with_data'] / 20.0, 1.0)  # More years = better
                    quality_score = data_metadata['data_quality_score']
                    drainage_score = min(station_info['drainage_area'] / 1000.0, 1.0) if station_info['drainage_area'] else 0.5
                    
                    # Weighted comprehensive score (distance is most important)
                    comprehensive_score = (distance_score * 0.4 + 
                                         years_score * 0.3 + 
                                         quality_score * 0.2 + 
                                         drainage_score * 0.1)
                    
                    station_result = {
                        'id': station_id,
                        'name': station_name,
                        'distance_km': distance,
                        'latitude': station_info['latitude'],
                        'longitude': station_info['longitude'],
                        'drainage_area_km2': station_info['drainage_area'],
                        'years_with_data': data_metadata['total_years_with_data'],
                        'data_range': f"{data_metadata['earliest_year']}-{data_metadata['latest_year']}" if data_metadata['earliest_year'] else "Unknown",
                        'total_discharge_records': data_metadata['total_discharge_records'],
                        'data_quality_score': data_metadata['data_quality_score'],
                        'comprehensive_score': comprehensive_score,
                        'continuous_periods': data_metadata['continuous_periods']
                    }
                    
                    stations_with_data.append(station_result)
                    
                    print(f"   SUCCESS: QUALIFIED: {data_metadata['total_years_with_data']} years of data")
                    print(f"   INFO: Data range: {station_result['data_range']}")
                    print(f"   Score: {comprehensive_score:.3f}")
                    print(f"   INFO: STOPPING - Found suitable station!")
                    
                    # STOP immediately after finding first suitable station
                    break
                else:
                    print(f"   ERROR: Only {data_metadata['total_years_with_data']} years (need {min_years}+)")
                    print(f"   INFO: Continuing to next station...")
            
            if not stations_with_data:
                print(f"\nERROR: No stations found with {min_years}+ years of data within {search_range_km} km")
                print("   Consider:")
                print(f"   - Increasing search radius beyond {search_range_km} km")
                print(f"   - Reducing minimum years requirement below {min_years}")
                return []
            
            # Sort by comprehensive score (best stations first)
            stations_with_data.sort(key=lambda x: x['comprehensive_score'], reverse=True)
            best_stations = stations_with_data[:max_stations]
            
            print(f"\nINFO: BEST STATIONS FOUND ({len(best_stations)}):")
            print("=" * 80)
            
            for i, station in enumerate(best_stations):
                print(f"{i+1}. {station['name']} ({station['id']})")
                print(f"   INFO: Distance: {station['distance_km']:.2f} km")
                print(f"   INFO: Data: {station['years_with_data']} years ({station['data_range']})")
                print(f"   INFO: Records: {station['total_discharge_records']:,} discharge measurements")
                drainage_area = station['drainage_area_km2'] or 0
                print(f"   INFO:  Drainage: {drainage_area:,.0f} km2")
                print(f"   Score: {station['comprehensive_score']:.3f}")
                
                if station['continuous_periods']:
                    longest = max(station['continuous_periods'], key=lambda x: x['years_count'])
                    print(f"   INFO: Longest period: {longest['start_year']}-{longest['end_year']} ({longest['years_count']} years)")
                print()
            
            return best_stations
            
        except Exception as e:
            error_msg = f"Station search failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return []
    
    def check_station_data_range(self, station_id: str, extensive_check: bool = True) -> Dict:
        """Check the actual data range available for a station"""
        print(f"DEBUG: Checking data range for station {station_id}...")
        
        metadata = {
            'station_id': station_id,
            'total_years_with_data': 0,
            'earliest_year': None,
            'latest_year': None,
            'continuous_periods': [],
            'total_discharge_records': 0,
            'data_quality_score': 0.0
        }
        
        try:
            # First, try to get ANY data for this station to find the actual range
            params = {
                'STATION_NUMBER': station_id,
                'limit': 100000  # Large limit to get full range
            }
            
            print(f"   Getting full data range for station {station_id}...")
            response = self.session.get(
                f"{self.eccc_base_url}/collections/hydrometric-daily-mean/items",
                params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                features = data.get("features", [])
                
                if features:
                    # Extract all dates and discharge values
                    all_dates = []
                    discharge_records = []
                    
                    for feature in features:
                        props = feature.get("properties", {})
                        date_str = props.get("DATE")
                        discharge = props.get("DISCHARGE")
                        
                        if date_str:
                            all_dates.append(date_str)
                            
                        if discharge is not None:
                            discharge_records.append({
                                'date': date_str,
                                'discharge': discharge
                            })
                    
                    if discharge_records:
                        # Convert to DataFrame for easier analysis
                        import pandas as pd
                        df = pd.DataFrame(discharge_records)
                        df['date'] = pd.to_datetime(df['date'])
                        df['year'] = df['date'].dt.year
                        
                        # Get actual data range
                        metadata['earliest_year'] = int(df['year'].min())
                        metadata['latest_year'] = int(df['year'].max())
                        metadata['total_discharge_records'] = len(df)
                        
                        # Count years with data (years with at least 30 days of data)
                        years_with_data = df.groupby('year').size()
                        significant_years = years_with_data[years_with_data >= 30].index.tolist()
                        metadata['total_years_with_data'] = len(significant_years)
                        
                        # Find continuous periods
                        if extensive_check and significant_years:
                            continuous_periods = []
                            current_period_start = significant_years[0]
                            current_period_end = significant_years[0]
                            
                            for i in range(1, len(significant_years)):
                                if significant_years[i] == current_period_end + 1:
                                    # Continuous
                                    current_period_end = significant_years[i]
                                else:
                                    # Gap found, save current period
                                    continuous_periods.append({
                                        'start_year': current_period_start,
                                        'end_year': current_period_end,
                                        'years_count': current_period_end - current_period_start + 1
                                    })
                                    current_period_start = significant_years[i]
                                    current_period_end = significant_years[i]
                            
                            # Add final period
                            continuous_periods.append({
                                'start_year': current_period_start,
                                'end_year': current_period_end,
                                'years_count': current_period_end - current_period_start + 1
                            })
                            
                            metadata['continuous_periods'] = continuous_periods
                        
                        # Calculate data quality score
                        if metadata['total_years_with_data'] > 0:
                            years_span = metadata['latest_year'] - metadata['earliest_year'] + 1
                            completeness = metadata['total_years_with_data'] / years_span
                            metadata['data_quality_score'] = completeness * min(metadata['total_years_with_data'] / 10.0, 1.0)
                        
                        print(f"   INFO: Station {station_id} Summary:")
                        print(f"      Years with significant data (30+ days): {metadata['total_years_with_data']}")
                        print(f"      Actual data range: {metadata['earliest_year']} - {metadata['latest_year']}")
                        print(f"      Total discharge records: {metadata['total_discharge_records']:,}")
                        
                        if metadata['continuous_periods']:
                            longest_period = max(metadata['continuous_periods'], key=lambda x: x['years_count'])
                            print(f"      Longest continuous period: {longest_period['start_year']}-{longest_period['end_year']} ({longest_period['years_count']} years)")
                        
                        print(f"      Data quality score: {metadata['data_quality_score']:.3f}")
                        
                        # Show year-by-year breakdown for debugging
                        if len(significant_years) > 0:
                            print(f"      Years with data: {min(significant_years)}-{max(significant_years)} (sample: {significant_years[:5]}{'...' if len(significant_years) > 5 else ''})")
                        
                    else:
                        print(f"   ERROR: No discharge data found in {len(features)} total records")
                
                else:
                    print(f"   ERROR: No data features found for station {station_id}")
            
            else:
                print(f"   ERROR: HTTP {response.status_code} error for station {station_id}")
                
        except Exception as e:
            print(f"   ERROR: Error checking station {station_id}: {str(e)}")
        
        return metadata
    
    def _check_year_has_data(self, station_id: str, year: int) -> bool:
        """Quick check if a specific year has discharge data"""
        try:
            params = {
                'STATION_NUMBER': station_id,
                'datetime': f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
                'limit': 100
            }
            
            response = self.session.get(
                f"{self.eccc_base_url}/collections/hydrometric-daily-mean/items",
                params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                features = data.get("features", [])
                discharge_records = len([f for f in features 
                                       if f.get("properties", {}).get("DISCHARGE") is not None])
                return discharge_records > 50  # At least 50 days of data
            
        except Exception:
            pass
        
        return False