#!/usr/bin/env python3
"""
Climate and Hydrometric Data Acquisition Step for RAVEN Modeling
Downloads climate forcing data and hydrometric calibration data with advanced gap filling
"""

import sys
from pathlib import Path
import argparse
import json
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
                                min_years: int = 30) -> Dict[str, Any]:
        """Get climate forcing data with 30+ year coverage using 50km search limit and IDW interpolation"""
        print(f"\nGetting climate forcing data with search range: {search_range_km} km")
        print(f"Target coverage: {min_years}+ years")
        print(f"IDW interpolation: {'Enabled' if use_idw else 'Disabled'}")
        
        try:
            if use_idw:
                # Use 30-year coverage method with advanced gap filling
                # Save climate data to climate subfolder to match step5 expectations
                climate_dir = self.workspace_dir / "climate"
                climate_dir.mkdir(exist_ok=True, parents=True)
                climate_file = climate_dir / "climate_forcing.csv" # Changed to .csv for RAVEN
                
                # Use the actual IDW gap filling method with multiple stations
                result = self.climate_client.get_30_year_climate_data_with_advanced_gap_filling(
                    outlet_lat=latitude,
                    outlet_lon=longitude,
                    output_path=climate_file,
                    min_years=min_years,
                    format_type='ravenpy'
                )

                if result.get('success'):
                    return {
                        'success': True,
                        'climate_file': result['file_path'],
                        'method': 'idw_multi_station',
                        'stations_used': result.get('stations_used', 0),
                        'data_quality': result.get('data_quality', {}),
                        'station_info': result.get('station_info', {})
                    }
                else:
                    return {'success': False, 'error': result.get('error')}

            else:
                # Use traditional single-station approach, but output to NetCDF
                candidate_stations = self.climate_client.find_best_climate_stations_with_data(
                    bbox=bounds,
                    outlet_lat=latitude,
                    outlet_lon=longitude,
                    search_range_km=search_range_km,
                    max_stations=5
                )
                
                if not candidate_stations:
                    return {'success': False, 'error': f'No climate stations found within {search_range_km} km'}
                
                # Use best station
                selected_station = candidate_stations[0]
                # Save climate data to data/climate subfolder to match step5 expectations
                climate_dir = self.workspace_dir / "climate"
                climate_dir.mkdir(exist_ok=True, parents=True)
                climate_file = climate_dir / "climate_forcing.csv" # Changed to .csv for RAVEN
                
                result = self.climate_client.get_climate_data_csv(
                    station_id=selected_station['id'],
                    start_date="1991-01-01",
                    end_date="2020-12-31",
                    output_path=climate_file,
                    format_type='ravenpy'
                )
                
                if result.get('success'):
                    return {
                        'success': True,
                        'climate_file': result['file_path'],
                        'method': 'netcdf_single_station',
                        'stations_used': 1,
                        'data_quality': result.get('data_quality', {}),
                        'station_info': {
                            'id': selected_station['id'],
                            'name': selected_station['name'],
                            'distance_km': selected_station['distance_km']
                        }
                    }
                else:
                    return {'success': False, 'error': result.get('error')}
                    
        except Exception as e:
            return {'success': False, 'error': f'Climate data acquisition failed: {str(e)}'}
    
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
            min_years=min_climate_years
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
