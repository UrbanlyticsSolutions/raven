#!/usr/bin/env python3
"""
Exact Magpie RVT Generator - extracted from magpie_workflow.py
Uses the complete Magpie format that works with RAVEN
"""

import pandas as pd
import geopandas as gpd
import os
import numpy as np
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Any, Optional

def generate_rvt_files(station_ID_lst, col_name_lst, param_lst, unit_lst, start_date, main_dir, temporary_dir):
    """
    Exact Magpie RVT generation function
    """
    # Create directory if it doesn't exist
    obs_dir = os.path.join(main_dir, 'workflow_outputs', 'RavenInput', 'climate_obs')
    os.makedirs(obs_dir, exist_ok=True)

    for stationID in station_ID_lst:
        print('\n-----------------------------------------------------------------------------------------------')
        print(f'( ) Generate Station {stationID} RVT File')
        print('-----------------------------------------------------------------------------------------------')
        start_year, start_month, start_day = start_date.split('-')
        # Define Minimum Year, Month, and Day
        min_year = str(start_year)
        month = str(start_month)
        day = str(start_day)

        # File pattern for merged station data
        file_pattern = f'stationID_{stationID}_merged.csv'
        file_path = os.path.join(temporary_dir, 'weather_data', 'merge', file_pattern)

        if os.path.exists(file_path):
            print(f"Reading merged station data from: {file_path}")
            files_merged_station = pd.read_csv(file_path)
            var_df = files_merged_station[col_name_lst]

            # Generate file name for RVT
            file_name = f'station_{stationID}'

            # Write RVT file
            var_vals = '\n'.join(var_df.astype(str).apply(lambda x: ', '.join(x), axis=1))

            with open(os.path.join(obs_dir, f"{file_name}.rvt"), "a") as f:
                print(":MultiData", file=f)
                print(f"{min_year}-{month}-{day} 00:00:00 1.0 {len(files_merged_station)}", file=f)
                print(f":Parameters,{param_lst}", file=f)
                print(f":Units,{unit_lst}", file=f)
                print(f"{var_vals}", file=f)
                print(":EndMultiData", file=f)
        else:
            print(f"File not found for station {stationID}: {file_pattern}")

def uploaded_climate_data_rvt_file(forc_dir, obs_dir, latitude_name, longitude_name, elevation_name,
                                   var_lst, params_list, units_list, min_year, month, day, model_name, main_dir):
    """
    Exact Magpie uploaded climate data RVT file generator
    """
    # overview of the first uploaded forcing file
    file_name = os.listdir(forc_dir)[0]
    view_file = pd.read_csv(os.path.join(forc_dir, file_name))

    # define column names of latitude, longitude, and elevations variables from CSV
    # longitude input
    if longitude_name == "NA":
        longitude_name = input('Enter longitude column name: ')
    # latitude input
    if latitude_name == "NA":
        latitude_name = input('Enter latitude column name: ')
    # elevation input
    if elevation_name == "NA":
        elevation_name = input('Enter elevation column name: ')

    lat_lst = []
    lon_lst = []
    elev_lst = []

    file_lst = glob(os.path.join(forc_dir, "*.csv"))

    for n in range(len(file_lst)):
        forc_df = pd.read_csv(file_lst[n])
        forc_col_names = forc_df.columns.values
        forc_variables_selected = forc_df[var_lst]

        lat_val = forc_df[latitude_name]
        lat_lst.append(lat_val)
        lon_val = forc_df[longitude_name]
        lon_lst.append(lon_val)
        elev_val = forc_df[elevation_name]
        elev_lst.append(elev_val)

        # Get filename without extension
        base = os.path.basename(file_lst[n])
        file_name = os.path.splitext(base)[0]

        # write rvt file
        var_vals = '\n'.join(forc_variables_selected.astype(str).apply(lambda x: ', '.join(x), axis=1))
        params = ','.join(params_list)
        units = ','.join(units_list)

        f = open(os.path.join(obs_dir, f"{file_name}.rvt"), "a")
        print(f":MultiData", file=f)
        print(f"{min_year}-{month}-{day} 00:00:00 1.0 {len(forc_variables_selected.iloc[:, 0])}", file=f)
        print(f":Parameters,{params}", file=f)
        print(f":Units,{units}", file=f)
        print(f"{var_vals}", file=f)
        print(f":EndMultiData", file=f)
        f.close()

    file_lst = glob(os.path.join(obs_dir, "*.rvt"))
    print(file_lst)

    lst_gauge_data = []
    for f in file_lst:
        f_name = os.path.basename(f)
        lst_gauge_data.append(f_name)

    lat_lst_val = np.concatenate(lat_lst, axis=0)
    lon_lst_val = np.concatenate(lon_lst, axis=0)
    elev_lst_val = np.concatenate(elev_lst, axis=0)

    # generate Raven RVT Input containing Station Information
    # define model name (for RVT file)
    with open(os.path.join(main_dir, 'workflow_outputs', 'RavenInput', model_name + '.rvt'), "a") as f:
        print(f"#########################################################################", file=f)
        print(f"# Climate Stations List", file=f)
        print(f"#------------------------------------------------------------------------\n#", file=f)

    for n in range(len(file_lst)):
        base = os.path.basename(lst_gauge_data[n])
        split_base = os.path.splitext(base)
        station_val = os.path.splitext(base)[0]
        # define station vals
        lat_val = lat_lst_val[n]
        lon_val = lon_lst_val[n]
        elev_val = elev_lst_val[n]
        f = open(os.path.join(main_dir, 'workflow_outputs', 'RavenInput', model_name + '.rvt'), "a")
        print(f":Gauge \t\t\t\t\t\t{station_val}", file=f)
        print(f"\t:Latitude \t\t\t{lat_val}", file=f)
        print(f"\t:Longitude \t\t\t{lon_val}", file=f)
        print(f"\t:Elevation \t\t\t{elev_val}", file=f)
        print(f"\t:RedirectToFile climate_obs/{lst_gauge_data[n]}", file=f)
        print(f":EndGauge", file=f)
        print(f"#", file=f)
        f.close()

def generate_RVT_gauges(df_elev, df_coords1, model_name, main_dir):
    """
    Exact Magpie RVT gauges generator  
    """
    print('\n-----------------------------------------------------------------------------------------------')
    print('( ) Generate RVT for gauges')
    print('-----------------------------------------------------------------------------------------------')

    with open(os.path.join(main_dir, 'workflow_outputs', 'RavenInput', model_name + '.rvt'), "a") as f:
        print(f"#------------------------------------------------------------------------", file=f)
        print(f"# Climate Stations List", file=f)
        print(f"#------------------------------------------------------------------------\n#", file=f)

    lst_vals = list(range(0, len(df_coords1['lon'])))

    for n in lst_vals:
        f = open(os.path.join(main_dir, 'workflow_outputs', 'RavenInput', model_name + '.rvt'), "a")
        print(f":Gauge File{n}", file=f)
        print(f":Latitude {df_coords1['lat'][n]}", file=f)
        print(f":Longitude {df_coords1['lon'][n]}", file=f)
        print(f":Elevation {df_elev['mean'][n]}", file=f)
        print(f":RedirectToFile DayMet/File{n}.rvt", file=f)
        print(f":EndGauge", file=f)
        print(f"#", file=f)
        f.close()

def convert_csv_to_magpie_rvt(csv_file_path, output_file_path, start_date=None):
    """
    Convert CSV to Magpie MultiData RVT format
    """
    # Read CSV
    df = pd.read_csv(csv_file_path)
    
    # Handle date column (could be 'Unnamed: 0' or 'Date')
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'Date'})
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Use actual start date from CSV if not provided
    if start_date is None:
        start_date = df.index[0].strftime('%Y-%m-%d')
        print(f"Using actual CSV start date: {start_date}")
    
    # Calculate actual monthly averages from data
    df['Month'] = df.index.month
    monthly_temp_avg = df.groupby('Month')[['TEMP_MIN', 'TEMP_MAX']].mean()
    monthly_avg_temps = ((monthly_temp_avg['TEMP_MIN'] + monthly_temp_avg['TEMP_MAX']) / 2).values
    monthly_temps_str = ' '.join([f"{temp:.1f}" for temp in monthly_avg_temps])
    
    print(f"Calculated monthly average temperatures from data: {monthly_temps_str}")
    
    # Get start date and record count
    start_year, start_month, start_day = start_date.split('-')
    num_records = len(df)
    
    # Prepare data in CSV format (Magpie style)
    data_values = []
    for _, row in df.iterrows():
        temp_max = row.get('TEMP_MAX', 0.0)
        temp_min = row.get('TEMP_MIN', 0.0)
        precip = row.get('PRECIP', 0.0)
        data_line = f"{temp_max:.2f},{temp_min:.2f},{precip:.2f}"
        data_values.append(data_line)
    
    # Write RVT file (complete Magpie format with gauge)
    with open(output_file_path, "w") as f:
        print("#########################################################################", file=f)
        print("# Climate Data for BigWhite Watershed", file=f)
        print(f"# Generated from: {os.path.basename(csv_file_path)}", file=f)
        print(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=f)
        print("#########################################################################", file=f)
        print("", file=f)
        
        # Add gauge definition (required before MultiData) with calculated monthly temps  
        print(":Gauge \t\t\t\t\t\tGauge_1", file=f)
        print("\t:Latitude \t\t\t49.7313", file=f)
        print("\t:Longitude \t\t\t-118.9439", file=f)
        print("\t:Elevation \t\t\t1800", file=f)
        print(f"\t:MonthlyAveTemperature {monthly_temps_str}", file=f)
        print("\t:MonthlyAveEvaporation 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0 30.0", file=f)
        print("", file=f)
        
        print(":MultiData", file=f)
        print(f"{start_year}-{start_month}-{start_day} 00:00:00 1.0 {num_records}", file=f)
        print(":Parameters,TEMP_MAX,TEMP_MIN,PRECIP", file=f)
        print(":Units,C,C,mm/d", file=f)
        
        # Add all data values
        for data_line in data_values:
            print(data_line, file=f)
        
        print(":EndMultiData", file=f)
        print(":EndGauge", file=f)
    
    print(f"Generated Magpie-style RVT: {output_file_path}")
    return output_file_path

if __name__ == "__main__":
    # Test the converter
    csv_file = "projects/bigwhite/data/climate/climate_forcing.csv"
    output_file = "projects/bigwhite/models/files/bigwhite/bigwhite_magpie.rvt"
    
    convert_csv_to_magpie_rvt(csv_file, output_file)
