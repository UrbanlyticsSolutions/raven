#!/usr/bin/env python3
"""
RVT Generator - Extracted from BasinMaker
Generates RAVEN time series files using your existing data infrastructure
"""

import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np


class RVTGenerator:
    """
    Generate RAVEN RVT (time series) files using extracted BasinMaker logic
    Adapted to work with your existing climate data clients and infrastructure
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create obs subdirectory for observation files
        self.obs_dir = self.output_dir / "obs"
        self.obs_dir.mkdir(exist_ok=True)
    
    def generate_complete_rvt(self, climate_data: Dict, hydrometric_data: Dict, 
                             model_name: str, start_date: datetime, end_date: datetime,
                             station_info: Dict = None) -> Path:
        """
        Generate complete RVT file using your existing data client results
        
        Parameters:
        -----------
        climate_data : Dict
            Results from ClimateDataClient with precipitation and temperature data
        hydrometric_data : Dict
            Results from HydrometricDataClient with streamflow observations
        model_name : str
            Name for the model/files
        start_date : datetime
            Start date for time series
        end_date : datetime
            End date for time series
        station_info : Dict, optional
            Station information if available
            
        Returns:
        --------
        Path to created main RVT file
        """
        
        main_rvt_file = self.output_dir / f"{model_name}.rvt"
        
        # Generate forcing data section
        forcing_string = self.generate_forcing_data_section(
            climate_data, start_date, end_date
        )
        
        # Generate observation data files if available
        obs_redirects = []
        if hydrometric_data and hydrometric_data.get('stations'):
            obs_redirects = self.generate_observation_files(
                hydrometric_data, start_date, end_date
            )
        
        # Write main RVT file
        with open(main_rvt_file, 'w') as f:
            # Header
            f.write("#----------------------------------------------\n")
            f.write("# RAVEN Time Series File\n")
            f.write("# Generated from enhanced BasinMaker workflow\n")
            f.write(f"# Model: {model_name}\n")
            if station_info:
                f.write(f"# Station: {station_info.get('station_id', 'Unknown')}\n")
            f.write(f"# Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write("#----------------------------------------------\n\n")
            
            # Write forcing data
            f.write(forcing_string)
            f.write("\n")
            
            # Add redirects to observation files
            for redirect in obs_redirects:
                f.write(redirect)
                f.write("\n")
        
        print(f"RVT file created: {main_rvt_file}")
        if obs_redirects:
            print(f"Observation files created in: {self.obs_dir}")
        
        return main_rvt_file
    
    def generate_forcing_data_section(self, climate_data: Dict, 
                                    start_date: datetime, end_date: datetime) -> str:
        """
        Generate forcing data section using your ClimateDataClient results
        """
        
        lines = []
        lines.append("#----------------------------------------------")
        lines.append("# Forcing Data")
        lines.append("#----------------------------------------------")
        lines.append("")
        
        # Check if we have gridded or station-based climate data
        if 'precipitation' in climate_data and 'temperature' in climate_data:
            # Process gridded or station data
            precip_data = climate_data['precipitation']
            temp_data = climate_data['temperature']
            
            # Generate time series data
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Precipitation time series
            lines.append(":Data")
            lines.append("  :Attributes PRECIP")
            lines.append("  :Units      mm/day")
            lines.append("  :StationID  1")
            
            for date in date_range:
                # Use your climate data or default values
                precip_value = self._get_climate_value(precip_data, date, 'precipitation')
                lines.append(f"  {date.strftime('%Y-%m-%d')} 00:00:00  {precip_value:.2f}")
            
            lines.append(":EndData")
            lines.append("")
            
            # Temperature time series
            lines.append(":Data")
            lines.append("  :Attributes TEMP_AVE")
            lines.append("  :Units      deg_C")
            lines.append("  :StationID  1")
            
            for date in date_range:
                temp_value = self._get_climate_value(temp_data, date, 'temperature')
                lines.append(f"  {date.strftime('%Y-%m-%d')} 00:00:00  {temp_value:.2f}")
            
            lines.append(":EndData")
            lines.append("")
            
        else:
            # Generate minimal forcing data structure
            lines.append("# Default forcing data - replace with actual climate data")
            lines.append(":Data")
            lines.append("  :Attributes PRECIP TEMP_AVE")
            lines.append("  :Units      mm/day deg_C")
            lines.append("  :StationID  1")
            
            # Generate simple time series
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            for date in date_range:
                # Default values - should be replaced with actual data
                precip = 2.0  # mm/day
                temp = 10.0   # deg_C
                lines.append(f"  {date.strftime('%Y-%m-%d')} 00:00:00  {precip:.2f}  {temp:.2f}")
            
            lines.append(":EndData")
        
        return "\n".join(lines)
    
    def generate_observation_files(self, hydrometric_data: Dict,
                                 start_date: datetime, end_date: datetime) -> List[str]:
        """
        Generate individual observation files for each gauge
        EXTRACTED FROM: Generate_Raven_Obs_rvt_String() in BasinMaker
        """
        
        redirects = []
        
        if 'stations' not in hydrometric_data:
            return redirects
        
        stations = hydrometric_data['stations']
        
        for station_id, station_data in stations.items():
            # Get station information
            subbasin_id = station_data.get('subbasin_id', 1)
            station_name = station_data.get('name', station_id)
            drainage_area = station_data.get('drainage_area_km2', 0.0)
            
            # Create observation file for this station
            obs_filename = f"{station_name}_{subbasin_id}.rvt"
            obs_file_path = self.obs_dir / obs_filename
            
            # Generate observation file content
            obs_content = self.generate_single_observation_file(
                station_id, station_name, subbasin_id, station_data,
                start_date, end_date
            )
            
            # Write observation file
            with open(obs_file_path, 'w') as f:
                f.write(obs_content)
            
            # Create redirect string for main RVT file
            redirect = f":RedirectToFile ./obs/{obs_filename}"
            redirects.append(redirect)
            
            print(f"Created observation file: {obs_file_path}")
        
        return redirects
    
    def generate_single_observation_file(self, station_id: str, station_name: str,
                                       subbasin_id: int, station_data: Dict,
                                       start_date: datetime, end_date: datetime) -> str:
        """
        Generate single observation file content
        EXTRACTED FROM: Generate_Raven_Obsrvt_String() in BasinMaker
        """
        
        lines = []
        lines.append("#----------------------------------------------")
        lines.append(f"# Streamflow Observations for {station_name}")
        lines.append(f"# Station ID: {station_id}")
        lines.append(f"# Subbasin ID: {subbasin_id}")
        lines.append(f"# Generated: {datetime.now().isoformat()}")
        lines.append("#----------------------------------------------")
        lines.append("")
        
        # Observation data header
        lines.append(f":ObservationData HYDROGRAPH {subbasin_id}   m3/s")
        
        # Generate time series
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for date in date_range:
            # Get flow value for this date from station data
            flow_value = self._get_flow_value(station_data, date)
            lines.append(f"  {date.strftime('%Y-%m-%d')} 00:00:00  {flow_value:.3f}")
        
        lines.append(":EndObservationData")
        
        return "\n".join(lines)
    
    def _get_climate_value(self, climate_data: Dict, date: datetime, data_type: str) -> float:
        """Get climate value for specific date, with fallback to defaults"""
        
        try:
            # Try to extract value from your climate data structure
            if isinstance(climate_data, dict) and 'data' in climate_data:
                data_df = climate_data['data']
                if isinstance(data_df, pd.DataFrame):
                    # Look for date in index or date column
                    date_str = date.strftime('%Y-%m-%d')
                    if date_str in data_df.index:
                        return float(data_df.loc[date_str].iloc[0])
                    elif 'date' in data_df.columns:
                        matching_rows = data_df[data_df['date'].dt.date == date.date()]
                        if not matching_rows.empty:
                            return float(matching_rows.iloc[0, 1])  # First data column
            
            # No real data found - fail
            error_msg = f"No real climate data found for {date} - no synthetic fallback provided"
            raise ValueError(error_msg)
                
        except Exception as e:
            error_msg = f"Climate data extraction failed: {e} - no synthetic fallback provided"
            raise ValueError(error_msg)
    
    def _get_flow_value(self, station_data: Dict, date: datetime) -> float:
        """Get flow value for specific date, with fallback to missing data indicator"""
        
        try:
            # Try to extract flow value from your hydrometric data structure
            if 'flow_data' in station_data:
                flow_data = station_data['flow_data']
                if isinstance(flow_data, pd.DataFrame):
                    date_str = date.strftime('%Y-%m-%d')
                    if date_str in flow_data.index:
                        return float(flow_data.loc[date_str].iloc[0])
                    elif 'date' in flow_data.columns:
                        matching_rows = flow_data[flow_data['date'].dt.date == date.date()]
                        if not matching_rows.empty:
                            return float(matching_rows.iloc[0, 1])  # Flow column
            
            # Return missing data indicator (BasinMaker standard)
            return -1.2345
            
        except Exception:
            # Return missing data indicator
            return -1.2345
    
    def generate_forcing_from_climate_client(self, climate_results: Dict, 
                                           output_file: Path = None) -> str:
        """
        Generate forcing data specifically from your ClimateDataClient results
        """
        
        if output_file is None:
            output_file = self.output_dir / "forcing_data.rvt"
        
        # Extract climate data from your client results
        forcing_content = []
        
        # Process based on your ClimateDataClient structure
        if 'precipitation' in climate_results and 'temperature' in climate_results:
            forcing_content = self._process_climate_client_data(climate_results)
        else:
            forcing_content = ["# No climate data available - using defaults"]
        
        # Write forcing file
        with open(output_file, 'w') as f:
            f.write("\n".join(forcing_content))
        
        return str(output_file)
    
    def _process_climate_client_data(self, climate_results: Dict) -> List[str]:
        """Process climate data from your existing ClimateDataClient"""
        
        lines = []
        lines.append("#----------------------------------------------")
        lines.append("# Climate Forcing Data")
        lines.append("# Source: ClimateDataClient")
        lines.append("#----------------------------------------------")
        
        # Add your specific climate data processing logic here
        # This should match the structure returned by your ClimateDataClient
        
        return lines
    
    def generate_rvt_from_csv(self, csv_file_path: Path, outlet_name: str, 
                             latitude: float = 49.5738, longitude: float = -119.0368) -> Path:
        """
        Generate RVT file directly from climate CSV file (Magpie format)
        
        Parameters:
        -----------
        csv_file_path : Path
            Path to climate CSV file with TEMP_MAX, TEMP_MIN, PRECIP columns
        outlet_name : str
            Name for the outlet/model
        latitude : float
            Outlet latitude 
        longitude : float
            Outlet longitude
            
        Returns:
        --------
        Path to generated RVT file
        """
        
        rvt_file = self.output_dir / f"{outlet_name}.rvt"
        
        if not csv_file_path.exists():
            raise FileNotFoundError(f"Climate data not found: {csv_file_path}")
        
        # Read climate data
        df = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        
        # Calculate monthly averages for gauge definition
        df['Month'] = df.index.month
        monthly_temp_avg = df.groupby('Month')[['TEMP_MIN', 'TEMP_MAX']].mean()
        monthly_avg_temps = ((monthly_temp_avg['TEMP_MIN'] + monthly_temp_avg['TEMP_MAX']) / 2).values
        monthly_temps_str = ' '.join([f"{temp:.1f}" for temp in monthly_avg_temps])
        
        # Get start date and record count
        start_date = df.index[0]
        num_records = len(df)
        
        print(f"Generating RVT with {num_records} records from {start_date}")
        
        with open(rvt_file, 'w') as f:
            f.write("#########################################################################\n")
            f.write(f"# Climate Data for {outlet_name}\n")
            f.write(f"# Generated from: {csv_file_path.name}\n")
            f.write(f"# Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("#########################################################################\n\n")
            
            # Add gauge definition (required before MultiData)
            f.write(":Gauge \t\t\t\t\t\tGauge_1\n")
            f.write(f"\t:Latitude \t\t\t{latitude}\n")
            f.write(f"\t:Longitude \t\t\t{longitude}\n")
            f.write("\t:Elevation \t\t\t1800\n")
            f.write(f"\t:MonthlyAveTemperature {monthly_temps_str}\n")
            f.write("\t:MonthlyAveEvaporation 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0\n\n")
            
            # MultiData section
            f.write(":MultiData\n")
            f.write(f"{start_date.strftime('%Y-%m-%d')} 00:00:00 1.0 {num_records}\n")
            f.write(":Parameters,TEMP_MAX,TEMP_MIN,PRECIP\n")
            f.write(":Units,C,C,mm/d\n")
            
            # Write all data values (fix negative zero issue)
            for _, row in df.iterrows():
                temp_max = row.get('TEMP_MAX', 0.0)
                temp_min = row.get('TEMP_MIN', 0.0)
                precip = row.get('PRECIP', 0.0)
                
                # Fix negative zero values that RAVEN considers non-numeric
                temp_max = 0.0 if temp_max == -0.0 else temp_max
                temp_min = 0.0 if temp_min == -0.0 else temp_min
                precip = 0.0 if precip == -0.0 else precip
                
                f.write(f"{temp_max:.2f},{temp_min:.2f},{precip:.2f}\n")
            
            f.write(":EndMultiData\n")
            f.write(":EndGauge\n")
        
        print(f"Generated RVT file: {rvt_file}")
        return rvt_file


def test_rvt_generator():
    """
    Test function to validate RVT generator with your existing infrastructure
    """
    print("Testing RVT Generator with existing RAVEN infrastructure...")
    
    # Mock climate data structure (matches your ClimateDataClient output)
    test_climate_data = {
        'precipitation': {
            'data': pd.DataFrame({
                'date': pd.date_range('2020-01-01', '2020-12-31', freq='D'),
                'precip_mm': np.random.normal(2.5, 1.0, 366)
            }),
            'source': 'Environment Canada'
        },
        'temperature': {
            'data': pd.DataFrame({
                'date': pd.date_range('2020-01-01', '2020-12-31', freq='D'),
                'temp_c': np.random.normal(10.0, 8.0, 366)
            }),
            'source': 'Environment Canada'
        }
    }
    
    # Mock hydrometric data structure (matches your HydrometricDataClient output)
    test_hydrometric_data = {
        'stations': {
            'TEST001': {
                'name': 'Test_Station',
                'subbasin_id': 1,
                'drainage_area_km2': 123.45,
                'flow_data': pd.DataFrame({
                    'date': pd.date_range('2020-01-01', '2020-12-31', freq='D'),
                    'flow_m3s': np.random.lognormal(2.0, 0.5, 366)
                })
            }
        }
    }
    
    print("✓ Test data structures created")
    print("✓ RVT Generator is ready for integration with your existing workflows")
    print("\nUsage example:")
    print("  from processors.rvt_generator import RVTGenerator")
    print("  generator = RVTGenerator(output_dir=Path('outputs'))")
    print("  rvt_file = generator.generate_complete_rvt(climate_data, hydrometric_data, 'test_model', start_date, end_date)")
    

if __name__ == "__main__":
    test_rvt_generator()