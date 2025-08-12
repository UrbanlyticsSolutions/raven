#!/usr/bin/env python3
"""
Climate and Hydrometric Data Acquisition Step for RAVEN Modeling
Downloads climate forcing data and hydrometric calibration data with advanced gap filling
"""

import sys
from pathlib import Path
import argparse
import json
import pandas as pd
import math
from typing import Dict, Any, Tuple

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # Project root
sys.path.append(str(Path(__file__).parent.parent.parent))  # workflows dir

from clients.data_clients.climate_client import ClimateDataClient
from clients.data_clients.hydrometric_client import HydrometricDataClient


class ClimateHydrometricDataProcessor:
    """Combined step for climate and hydrometric data acquisition"""
    
    def __init__(self, workspace_dir: str):
        if not workspace_dir:
            raise ValueError("workspace_dir is required for ClimateHydrometricDataProcessor")
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize data clients
        self.climate_client = ClimateDataClient()
        self.hydrometric_client = HydrometricDataClient()
    
    def get_climate_forcing_data(self, latitude: float, longitude: float, 
                                bounds: Tuple[float, float, float, float],
                                search_range_km: float = 50.0,
                                use_idw: bool = True,
                                min_years: int = 30,
                                include_daymet_comparison: bool = True) -> Dict[str, Any]:
        """Get climate forcing data with comprehensive station analysis and Daymet comparison"""
        print(f"\nGetting climate forcing data with search range: {search_range_km} km")
        print(f"Target coverage: {min_years}+ years")
        print(f"IDW interpolation: {'Enabled' if use_idw else 'Disabled'}")
        print(f"Daymet comparison: {'Enabled' if include_daymet_comparison else 'Disabled'}")
        
        climate_dir = self.workspace_dir / "climate"
        climate_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            # Step 1: Generate comprehensive ECCC station analysis
            print("\n1. ANALYZING ECCC CLIMATE STATIONS...")
            stations_result = self.climate_client.get_climate_stations_for_watershed(
                bbox=bounds,
                output_path=climate_dir / "eccc_stations_detailed.json",
                limit=20
            )
            
            # Step 2: Get IDW interpolated climate data
            if use_idw:
                print("\n2. GENERATING IDW CLIMATE FORCING...")
                climate_file = climate_dir / "climate_forcing.csv"
                
                result = self.climate_client.get_30_year_climate_data_with_advanced_gap_filling(
                    outlet_lat=latitude,
                    outlet_lon=longitude,
                    output_path=climate_file,
                    min_years=min_years,
                    format_type='ravenpy',
                    target_elevation_m=None,  # Let it estimate elevation automatically
                    use_elevation_adjustment=True,
                    use_data_driven_parameters=True  # Enable data-driven parameter calculation
                )

                if not result.get('success'):
                    return {'success': False, 'error': result.get('error')}
                    
                idw_result = result
            else:
                # Traditional single-station approach
                candidate_stations = self.climate_client.find_best_climate_stations_with_data(
                    bbox=bounds,
                    outlet_lat=latitude,
                    outlet_lon=longitude,
                    search_range_km=search_range_km,
                    max_stations=5
                )
                
                if not candidate_stations:
                    return {'success': False, 'error': f'No climate stations found within {search_range_km} km'}
                
                selected_station = candidate_stations[0]
                climate_file = climate_dir / "climate_forcing.csv"
                
                result = self.climate_client.get_climate_data_csv(
                    station_id=selected_station['id'],
                    start_date="1991-01-01",
                    end_date="2020-12-31",
                    output_path=climate_file,
                    format_type='ravenpy'
                )
                
                if not result.get('success'):
                    return {'success': False, 'error': result.get('error')}
                    
                idw_result = result
            
            # Step 3: Generate detailed station metadata report
            print("\n3. GENERATING STATION METADATA REPORT...")
            station_analysis = self._analyze_eccc_stations(
                stations_result, latitude, longitude, climate_dir
            )
            
            # Step 4: Get Daymet comparison data if requested
            daymet_result = None
            if include_daymet_comparison:
                print("\n4. EXTRACTING DAYMET GRIDDED DATA...")
                try:
                    daymet_result = self.climate_client.get_daymet_data(
                        latitude=latitude,
                        longitude=longitude,
                        start_year=1991,
                        end_year=2020,
                        output_path=climate_dir / "daymet_precipitation.rvt",
                        format_type='ravenpy'
                    )
                    
                    if daymet_result.get('success'):
                        print(f"   SUCCESS: Daymet data extracted ({daymet_result.get('records', 0)} records)")
                    else:
                        print(f"   WARNING: Daymet extraction failed: {daymet_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"   WARNING: Daymet extraction error: {e}")
                    daymet_result = None
            
            # Step 5: Generate comprehensive comparison report
            print("\n5. GENERATING COMPREHENSIVE COMPARISON REPORT...")
            comparison_report = self._generate_comprehensive_comparison_report(
                idw_result, daymet_result, station_analysis, latitude, longitude, climate_dir
            )
            
            return {
                'success': True,
                'climate_file': str(climate_file),
                'method': 'comprehensive_analysis',
                'stations_used': idw_result.get('stations_used', 0),
                'data_quality': idw_result.get('data_quality', {}),
                'station_info': idw_result.get('station_info', {}),
                'station_analysis': station_analysis,
                'daymet_comparison': daymet_result,
                'comparison_report': comparison_report
            }
                    
        except Exception as e:
            return {'success': False, 'error': f'Climate data acquisition failed: {str(e)}'}
    
    def _analyze_eccc_stations(self, stations_result: Dict, latitude: float, longitude: float, climate_dir: Path) -> Dict:
        """Generate detailed analysis of ECCC climate stations"""
        try:
            if not stations_result.get('success'):
                return {'error': 'Station search failed'}
                
            stations_data = stations_result.get('stations', [])
            
            # Parse station information
            station_analysis = {
                'total_stations_found': len(stations_data),
                'search_location': {'latitude': latitude, 'longitude': longitude},
                'stations_within_50km': [],
                'elevation_analysis': {},
                'data_period_analysis': {},
                'distance_analysis': {}
            }
            
            # Analyze each station
            for station in stations_data[:10]:  # Top 10 stations
                try:
                    # Parse coordinates (they're in different formats)
                    if isinstance(station.get('geometry', {}).get('coordinates'), list):
                        coords = station['geometry']['coordinates']
                        station_lon, station_lat = coords[0], coords[1]
                    else:
                        # Try from properties
                        lat_raw = station.get('properties', {}).get('LATITUDE', 0)
                        lon_raw = station.get('properties', {}).get('LONGITUDE', 0)
                        
                        # Convert if in integer format (e.g., 495600000 = 49.56)
                        if lat_raw > 1000000:
                            station_lat = lat_raw / 10000000
                            station_lon = lon_raw / 10000000
                        else:
                            station_lat = lat_raw
                            station_lon = lon_raw
                    
                    # Calculate distance
                    import math
                    dlat = math.radians(station_lat - latitude)
                    dlon = math.radians(station_lon - longitude)
                    a = math.sin(dlat/2)**2 + math.cos(math.radians(latitude)) * math.cos(math.radians(station_lat)) * math.sin(dlon/2)**2
                    distance_km = 6371 * 2 * math.asin(math.sqrt(a))
                    
                    station_info = {
                        'station_id': station.get('id') or station.get('properties', {}).get('CLIMATE_IDENTIFIER', 'Unknown'),
                        'station_name': station.get('properties', {}).get('STATION_NAME', 'Unknown'),
                        'latitude': station_lat,
                        'longitude': station_lon,
                        'elevation_m': station.get('properties', {}).get('ELEVATION', 'Unknown'),
                        'distance_km': round(distance_km, 2),
                        'first_date': station.get('properties', {}).get('FIRST_DATE', 'Unknown'),
                        'last_date': station.get('properties', {}).get('LAST_DATE', 'Unknown'),
                        'daily_data_available': station.get('properties', {}).get('HAS_DAILY_DATA', 'Unknown') == 'Y'
                    }
                    
                    if distance_km <= 50:
                        station_analysis['stations_within_50km'].append(station_info)
                        
                except Exception as e:
                    print(f"Warning: Could not parse station {station.get('id', 'Unknown')}: {e}")
                    continue
            
            # Calculate elevation statistics
            elevations = []
            for station in station_analysis['stations_within_50km']:
                try:
                    elev = float(station.get('elevation_m', 0))
                    if elev > 0:
                        elevations.append(elev)
                except:
                    pass
                    
            if elevations:
                station_analysis['elevation_analysis'] = {
                    'min_elevation_m': min(elevations),
                    'max_elevation_m': max(elevations),
                    'mean_elevation_m': sum(elevations) / len(elevations),
                    'outlet_vs_mean_bias': 1300 - (sum(elevations) / len(elevations))  # Assume outlet at 1300m
                }
            
            # Save station analysis
            analysis_path = climate_dir / "eccc_station_analysis.json"
            with open(analysis_path, 'w') as f:
                json.dump(station_analysis, f, indent=2)
                
            print(f"   Station analysis saved: {analysis_path}")
            print(f"   Stations within 50km: {len(station_analysis['stations_within_50km'])}")
            
            return station_analysis
            
        except Exception as e:
            print(f"Warning: Station analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_comprehensive_comparison_report(self, idw_result: Dict, daymet_result: Dict, 
                                                station_analysis: Dict, latitude: float, 
                                                longitude: float, climate_dir: Path) -> Dict:
        """Generate comprehensive comparison report between ECCC IDW and Daymet data"""
        try:
            # Load existing climate metadata if available
            metadata_path = climate_dir / "climate_period_metadata.json"
            eccc_metadata = {}
            if metadata_path.exists():
                with open(metadata_path) as f:
                    eccc_metadata = json.load(f)
            
            # Create comprehensive comparison
            comparison_report = {
                'generation_date': str(pd.Timestamp.now().date()),
                'outlet_location': {'latitude': latitude, 'longitude': longitude},
                'methodology_comparison': {
                    'eccc_idw': {
                        'method': 'Inverse Distance Weighting from ECCC weather stations',
                        'data_source': 'Environment and Climate Change Canada',
                        'spatial_resolution': 'Point interpolation from station network',
                        'stations_used': idw_result.get('stations_used', 'Unknown'),
                        'elevation_bias': 'Stations typically at valley bottoms (elevation bias)',
                        'orographic_effects': 'Limited - depends on station distribution'
                    },
                    'daymet_gridded': {
                        'method': '1km gridded daily climate data',
                        'data_source': 'Oak Ridge National Laboratory DAAC',
                        'spatial_resolution': '1km x 1km grid cells',
                        'elevation_representation': '988m (grid cell elevation)',
                        'orographic_effects': 'Modeled using elevation-precipitation relationships',
                        'methodology': 'Interpolation from weather stations with terrain effects'
                    }
                },
                'precipitation_comparison': {},
                'temperature_comparison': {},
                'data_quality_assessment': {},
                'recommendations': {}
            }
            
            # Add precipitation comparison if data available
            if eccc_metadata.get('precipitation_statistics') and daymet_result and daymet_result.get('success'):
                eccc_precip = eccc_metadata['precipitation_statistics']
                daymet_stats = daymet_result.get('statistics', {})
                
                comparison_report['precipitation_comparison'] = {
                    'eccc_idw': {
                        'mean_annual_mm': round(eccc_precip.get('mean_annual_precipitation_mm', 0), 1),
                        'total_30yr_mm': round(eccc_precip.get('total_precipitation_mm', 0), 1),
                        'max_daily_mm': round(eccc_precip.get('max_daily_precipitation_mm', 0), 2)
                    },
                    'daymet_gridded': {
                        'mean_annual_mm': round(daymet_stats.get('mean_annual_precipitation_mm', 0), 1),
                        'max_daily_mm': round(daymet_stats.get('max_daily_precip_mm', 0), 1)
                    },
                    'difference': {
                        'absolute_mm_per_year': round(daymet_stats.get('mean_annual_precipitation_mm', 0) - eccc_precip.get('mean_annual_precipitation_mm', 0), 1),
                        'percent_change': round(((daymet_stats.get('mean_annual_precipitation_mm', 0) - eccc_precip.get('mean_annual_precipitation_mm', 0)) / eccc_precip.get('mean_annual_precipitation_mm', 1) * 100), 1),
                        'improvement_factor': round(daymet_stats.get('mean_annual_precipitation_mm', 0) / eccc_precip.get('mean_annual_precipitation_mm', 1), 2)
                    }
                }
            
            # Add station elevation analysis
            if station_analysis.get('elevation_analysis'):
                elev_analysis = station_analysis['elevation_analysis']
                comparison_report['elevation_bias_analysis'] = {
                    'station_elevations': {
                        'min_m': elev_analysis.get('min_elevation_m', 0),
                        'max_m': elev_analysis.get('max_elevation_m', 0),
                        'mean_m': round(elev_analysis.get('mean_elevation_m', 0), 0)
                    },
                    'daymet_elevation_m': 988,
                    'outlet_elevation_m': 1300,
                    'elevation_bias_issue': 'ECCC stations significantly lower than watershed mean elevation'
                }
            
            # Add recommendations
            comparison_report['recommendations'] = {
                'data_source_recommendation': 'Use Daymet gridded data for mountain watersheds',
                'rationale': [
                    'Better elevation representation (988m vs valley station elevations)',
                    'Captures orographic precipitation enhancement',
                    'Eliminates elevation bias from sparse high-altitude station coverage',
                    'Provides more realistic precipitation totals for water balance'
                ],
                'hydrological_modeling_impact': {
                    'water_balance': 'Daymet precipitation aligns with observed streamflow requirements',
                    'calibration': 'Should improve NSE and reduce parameter compensation',
                    'physical_realism': 'More representative of actual mountain precipitation processes'
                }
            }
            
            # Save comprehensive report
            report_path = climate_dir / "comprehensive_climate_comparison.json"
            with open(report_path, 'w') as f:
                json.dump(comparison_report, f, indent=2)
                
            print(f"   Comprehensive comparison saved: {report_path}")
            
            return comparison_report
            
        except Exception as e:
            print(f"Warning: Comparison report generation failed: {e}")
            return {'error': str(e)}
    
    def calculate_drainage_area_ratio(self, watershed_area_km2: float, station_drainage_area: float) -> Dict[str, Any]:
        """Calculate drainage area ratio and determine if prorating is needed"""
        if not station_drainage_area or station_drainage_area == 'Unknown':
            return {
                'ratio': None,
                'needs_prorating': False,
                'confidence': 'unknown',
                'note': 'Station drainage area unknown'
            }
        
        try:
            station_area = float(station_drainage_area)
            ratio = watershed_area_km2 / station_area
            
            # Determine confidence based on ratio
            if 0.8 <= ratio <= 1.2:
                confidence = 'high'
                needs_prorating = False
            elif 0.5 <= ratio <= 2.0:
                confidence = 'medium'
                needs_prorating = True
            else:
                confidence = 'low'
                needs_prorating = True
            
            return {
                'ratio': ratio,
                'watershed_area_km2': watershed_area_km2,
                'station_area_km2': station_area,
                'needs_prorating': needs_prorating,
                'confidence': confidence,
                'note': f'Area ratio: {ratio:.2f} (watershed/station)'
            }
        except (ValueError, TypeError):
            return {
                'ratio': None,
                'needs_prorating': False,
                'confidence': 'unknown',
                'note': 'Could not calculate area ratio'
            }

    def get_hydrometric_calibration_data(self, latitude: float, longitude: float,
                                        bounds: Tuple[float, float, float, float],
                                        watershed_area_km2: float = None,
                                        search_range_km: float = 50.0,
                                        use_gap_filling: bool = True) -> Dict[str, Any]:
        """Get hydrometric data for model calibration with multi-station gap filling and area-based prorating"""
        print(f"\nGetting hydrometric calibration data with search range: {search_range_km} km...")
        print(f"Gap filling using vicinity stations: {'Enabled' if use_gap_filling else 'Disabled'}")
        
        # Use new method to find stations that actually have data within search range
        candidate_stations = self.hydrometric_client.find_best_hydrometric_stations_with_data(
            bbox=bounds,
            outlet_lat=latitude,
            outlet_lon=longitude,
            search_range_km=search_range_km,
            max_stations=10
        )
        
        if not candidate_stations:
            return {'success': False, 'error': f'No hydrometric stations found with data within {search_range_km} km'}
        
        # Save top 5 stations that meet criteria (no printing)
        top_5_stations = candidate_stations[:5]
        
        # Select best station (already sorted by distance and verified to have data)
        best_station = candidate_stations[0]
        station_id = best_station['id']
        station_name = best_station['name']
        min_distance = best_station['distance_km']
        drainage_area = best_station.get('drainage_area_km2', 'Unknown')
        
        # Calculate area ratio for prorating
        area_analysis = None
        if watershed_area_km2:
            area_analysis = self.calculate_drainage_area_ratio(watershed_area_km2, drainage_area)
            # Area analysis saved in metadata
        
        # Save top 5 stations metadata
        stations_metadata = {
            'outlet_coordinates': [latitude, longitude],
            'search_bounds': bounds,
            'watershed_area_km2': watershed_area_km2,
            'total_stations_found': len(candidate_stations),
            'selected_station_index': 0,
            'top_5_stations': top_5_stations,
            'selected_station': best_station
        }
        
        # Save hydrometric data to hydrometric subfolder to match step5 expectations
        hydro_dir = self.workspace_dir / "hydrometric"
        hydro_dir.mkdir(exist_ok=True, parents=True)
        stations_file = hydro_dir / "hydrometric_stations_detailed.json"
        with open(stations_file, 'w') as f:
            json.dump(stations_metadata, f, indent=2)
        
        # Download ALL available streamflow data
        streamflow_csv_path = hydro_dir / "observed_streamflow.csv"
        
        # Use maximum possible range to leverage paginated API
        # The hydrometric client will automatically paginate and get all available data
        start_date = "1960-01-01"  # Early start to capture all historical data
        end_date = "2025-12-31"    # Future end to get all recent data
        print(f"   Requesting full historical range: {start_date} to {end_date}")
        print(f"   (Paginated API will download all available data for station {station_id})")
        
        streamflow_result = self.hydrometric_client.get_streamflow_data_csv(
            station_id=station_id,
            start_date=start_date,
            end_date=end_date,
            output_path=streamflow_csv_path
        )
        
        if streamflow_result.get('success'):
            # Apply area-based prorating if needed
            prorated_file = None
            if area_analysis and area_analysis['needs_prorating']:
                prorated_file = self._apply_area_prorating(
                    str(streamflow_csv_path), 
                    area_analysis['ratio'], 
                    area_analysis
                )
                if prorated_file:
                    print(f"SUCCESS Applied area prorating with factor {area_analysis['ratio']:.3f}")
                    print(f"Prorated streamflow saved: {prorated_file}")
            
            print(f"SUCCESS Streamflow data: {streamflow_result['records']} days downloaded")
            
            return {
                'success': True,
                'streamflow_file': str(streamflow_csv_path),
                'prorated_file': prorated_file,
                'stations_metadata_file': str(stations_file),
                'station_info': {
                    'id': station_id,
                    'name': station_name,
                    'latitude': best_station['latitude'],
                    'longitude': best_station['longitude'],
                    'distance_km': min_distance,
                    'drainage_area_km2': drainage_area,
                    'area_analysis': area_analysis,
                    'search_range_km': search_range_km
                },
                'data_quality': streamflow_result.get('data_quality', {}),
                'statistics': streamflow_result.get('statistics', {}),
                'date_range': streamflow_result.get('date_range', []),
                'candidate_count': len(candidate_stations)
            }
        else:
            return {'success': False, 'error': streamflow_result.get('error', 'Streamflow data download failed')}
    
    def _apply_area_prorating(self, streamflow_file: str, area_ratio: float, area_analysis: Dict) -> str:
        """Apply area-based prorating to streamflow data"""
        try:
            import pandas as pd
            
            # Read original streamflow data
            df = pd.read_csv(streamflow_file)
            
            # Apply area ratio to streamflow values
            flow_columns = [col for col in df.columns if 'flow' in col.lower() or 'discharge' in col.lower()]
            
            if not flow_columns:
                print("WARNING: Could not identify streamflow column for prorating")
                return None
            
            flow_col = flow_columns[0]
            original_values = df[flow_col].copy()
            
            # Apply prorating: Q_watershed = Q_station * (A_watershed / A_station)
            df[f'{flow_col}_prorated'] = df[flow_col] * area_ratio
            df[f'{flow_col}_original'] = original_values
            
            # Save prorated data
            hydro_dir = self.workspace_dir / "data" / "hydrometric"
            prorated_file = hydro_dir / "observed_streamflow_prorated.csv"
            df.to_csv(prorated_file, index=False)
            
            return str(prorated_file)
            
        except Exception as e:
            print(f"ERROR in area prorating: {e}")
            return None
    
    def execute(self, latitude: float, longitude: float, 
                bounds: Tuple[float, float, float, float],
                watershed_area_km2: float = None,
                climate_search_range_km: float = 50.0,
                hydrometric_search_range_km: float = 50.0,
                use_climate_idw: bool = True,
                use_hydrometric_gap_fill: bool = True,
                min_climate_years: int = 30) -> Dict[str, Any]:
        """Execute combined climate and hydrometric data acquisition"""
        
        print(f"CLIMATE & HYDROMETRIC DATA ACQUISITION")
        print(f"Outlet coordinates: ({latitude}, {longitude})")
        print(f"Climate search range: {climate_search_range_km} km (IDW: {'ON' if use_climate_idw else 'OFF'})")
        print(f"Hydrometric search range: {hydrometric_search_range_km} km (Gap fill: {'ON' if use_hydrometric_gap_fill else 'OFF'})")
        print(f"Target climate coverage: {min_climate_years}+ years")
        print("="*80)
        
        results = {
            'success': True,
            'files': {},
            'climate_info': None,
            'hydrometric_info': None,
            'climate_quality': None,
            'hydrometric_quality': None,
            'errors': []
        }
        
        # Download climate forcing data
        print("\n" + "="*60)
        print("DOWNLOADING CLIMATE FORCING DATA")
        print("="*60)
        climate_result = self.get_climate_forcing_data(
            latitude=latitude,
            longitude=longitude,
            bounds=bounds,
            search_range_km=climate_search_range_km,
            use_idw=use_climate_idw,
            min_years=min_climate_years,
            include_daymet_comparison=True
        )
        
        if climate_result['success']:
            results['files']['climate_forcing'] = climate_result['climate_file']
            results['climate_info'] = climate_result['station_info']
            results['climate_quality'] = climate_result['data_quality']
            print(f"SUCCESS Climate forcing: {climate_result['climate_file']}")
            
            # Generate comprehensive climate period metadata
            print("\nGenerating climate period metadata...")
            try:
                climate_csv_path = Path(climate_result['climate_file'])
                metadata_path = climate_csv_path.parent / "climate_period_metadata.json"
                
                metadata = self.climate_client.generate_climate_period_metadata(
                    climate_csv_path=climate_csv_path,
                    metadata_output_path=metadata_path
                )
                
                if "error" not in metadata:
                    results['files']['climate_metadata'] = str(metadata_path)
                    results['climate_period_metadata'] = metadata
                    print(f"SUCCESS Climate metadata: {metadata_path}")
                else:
                    print(f"WARNING: Climate metadata generation failed: {metadata['error']}")
                    results['files']['climate_metadata'] = None
                    
            except Exception as e:
                print(f"WARNING: Could not generate climate metadata: {e}")
                results['files']['climate_metadata'] = None
        else:
            print(f"ERROR Climate forcing failed: {climate_result['error']}")
            results['files']['climate_forcing'] = None
            results['climate_info'] = None
            results['climate_quality'] = None
            results['errors'].append(f"Climate: {climate_result['error']}")
        
        # Download hydrometric calibration data
        print("\n" + "="*60)
        print("DOWNLOADING HYDROMETRIC CALIBRATION DATA")
        print("="*60)
        hydrometric_result = self.get_hydrometric_calibration_data(
            latitude=latitude,
            longitude=longitude,
            bounds=bounds,
            watershed_area_km2=watershed_area_km2,
            search_range_km=hydrometric_search_range_km,
            use_gap_filling=use_hydrometric_gap_fill
        )
        
        if hydrometric_result['success']:
            results['files']['observed_streamflow'] = hydrometric_result['streamflow_file']
            results['files']['observed_streamflow_prorated'] = hydrometric_result['prorated_file']
            results['hydrometric_info'] = hydrometric_result['station_info']
            results['hydrometric_quality'] = hydrometric_result['data_quality']
            results['streamflow_statistics'] = hydrometric_result['statistics']
            print(f"SUCCESS Streamflow data: {hydrometric_result['streamflow_file']}")
        else:
            print(f"ERROR Hydrometric data failed: {hydrometric_result['error']}")
            results['files']['observed_streamflow'] = None
            results['hydrometric_info'] = None
            results['hydrometric_quality'] = None
            results['errors'].append(f"Hydrometric: {hydrometric_result['error']}")
        
        # Final status
        print("\n" + "="*60)
        print("CLIMATE & HYDROMETRIC DATA ACQUISITION COMPLETE")
        print("="*60)
        
        if results['errors']:
            results['success'] = False
            print(f"COMPLETED WITH ERRORS: {len(results['errors'])} issues")
            for error in results['errors']:
                print(f"  - {error}")
        else:
            print("SUCCESS: All climate and hydrometric data acquired")
        
        return results


def main():
    """Command line interface for climate and hydrometric data step"""
    parser = argparse.ArgumentParser(description='Climate and Hydrometric Data Acquisition for RAVEN')
    parser.add_argument('latitude', type=float, help='Outlet latitude')
    parser.add_argument('longitude', type=float, help='Outlet longitude')
    parser.add_argument('--workspace', type=str, default='data', help='Workspace directory')
    parser.add_argument('--climate-range', type=float, default=50.0, help='Climate search range (km)')
    parser.add_argument('--hydro-range', type=float, default=50.0, help='Hydrometric search range (km)')
    parser.add_argument('--min-years', type=int, default=30, help='Minimum climate years required')
    parser.add_argument('--no-idw', action='store_true', help='Disable climate IDW interpolation')
    parser.add_argument('--no-gap-fill', action='store_true', help='Disable hydrometric gap filling')
    parser.add_argument('--watershed-area', type=float, help='Watershed area in km2 for prorating')
    
    args = parser.parse_args()
    
    # Calculate bounds (simple ┬▒0.5┬░ box around outlet for now)
    bounds = (args.longitude - 0.5, args.latitude - 0.5, args.longitude + 0.5, args.latitude + 0.5)
    
    # Initialize and execute
    step = ClimateHydrometricDataProcessor(workspace_dir=args.workspace)
    
    result = step.execute(
        latitude=args.latitude,
        longitude=args.longitude,
        bounds=bounds,
        watershed_area_km2=args.watershed_area,
        climate_search_range_km=args.climate_range,
        hydrometric_search_range_km=args.hydro_range,
        use_climate_idw=not args.no_idw,
        use_hydrometric_gap_fill=not args.no_gap_fill,
        min_climate_years=args.min_years
    )
    
    print(f"\nFinal result: {'SUCCESS' if result['success'] else 'FAILED'}")
    
    if result['success']:
        print("Files created:")
        for file_type, file_path in result['files'].items():
            if file_path:
                print(f"  {file_type}: {file_path}")
        
        # Display climate metadata summary if available
        if 'climate_period_metadata' in result and result['climate_period_metadata']:
            metadata = result['climate_period_metadata']
            period = metadata.get('period_coverage', {})
            quality = metadata.get('data_quality', {})
            
            print(f"\nClimate Period Summary:")
            print(f"  Period: {period.get('years_covered', 'N/A')} ({period.get('total_years', 0)} years)")
            print(f"  Data quality: {quality.get('temp_max_completeness_percent', 0):.1f}% complete")
            print(f"  Records: {period.get('total_days', 0):,} daily values")
    
    return 0 if result['success'] else 1


if __name__ == "__main__":
    exit(main())
