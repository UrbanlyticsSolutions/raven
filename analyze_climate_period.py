#!/usr/bin/env python3
"""
Analyze the total period and coverage of climate data
"""

import pandas as pd
import os
from datetime import datetime

def analyze_climate_period():
    """Analyze the climate data period and provide detailed statistics"""
    
    # Path to climate forcing data
    csv_file = 'outlet_49.5738_-119.0368/climate/climate_forcing.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: Climate file not found at {csv_file}")
        return
    
    try:
        # Read the climate forcing data
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        
        print('=' * 60)
        print('CLIMATE DATA TOTAL PERIOD SUMMARY')
        print('=' * 60)
        
        # Basic period information
        start_date = df.index[0]
        end_date = df.index[-1]
        total_days = (end_date - start_date).days + 1
        total_years = total_days / 365.25
        
        print(f'Total records: {len(df):,} days')
        print(f'Start date: {start_date.strftime("%Y-%m-%d")}')
        print(f'End date: {end_date.strftime("%Y-%m-%d")}')
        print(f'Total period: {total_days:,} days ({total_years:.1f} years)')
        print()
        
        # Year-by-year breakdown
        print('=' * 40)
        print('YEAR-BY-YEAR BREAKDOWN')
        print('=' * 40)
        yearly_counts = df.groupby(df.index.year).size()
        for year, count in yearly_counts.items():
            print(f'{year}: {count:3d} days')
        print()
        
        # Data quality overview
        print('=' * 40)
        print('DATA QUALITY OVERVIEW')
        print('=' * 40)
        missing_temp_max = df['TEMP_MAX'].isnull().sum()
        missing_temp_min = df['TEMP_MIN'].isnull().sum()
        missing_precip = df['PRECIP'].isnull().sum()
        
        print(f'Missing TEMP_MAX: {missing_temp_max} days')
        print(f'Missing TEMP_MIN: {missing_temp_min} days')
        print(f'Missing PRECIP: {missing_precip} days')
        print()
        
        # Climate statistics
        print('=' * 40)
        print('CLIMATE STATISTICS')
        print('=' * 40)
        temp_min_range = df['TEMP_MIN'].min()
        temp_max_range = df['TEMP_MAX'].max()
        total_precip = df['PRECIP'].sum()
        avg_annual_precip = total_precip / total_years
        
        print(f'Temperature range: {temp_min_range:.1f}°C to {temp_max_range:.1f}°C')
        print(f'Total precipitation: {total_precip:.1f} mm')
        print(f'Average annual precipitation: {avg_annual_precip:.1f} mm/year')
        print()
        
        # Monthly averages
        print('=' * 40)
        print('MONTHLY CLIMATE AVERAGES')
        print('=' * 40)
        monthly_stats = df.groupby(df.index.month).agg({
            'TEMP_MAX': 'mean',
            'TEMP_MIN': 'mean', 
            'PRECIP': 'mean'
        }).round(1)
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for i, month in enumerate(months, 1):
            if i in monthly_stats.index:
                row = monthly_stats.loc[i]
                print(f'{month}: Temp {row["TEMP_MIN"]:.1f}°C to {row["TEMP_MAX"]:.1f}°C, Avg Precip {row["PRECIP"]:.2f}mm/day')
        
        print()
        print('=' * 60)
        print(f'SUMMARY: {total_years:.1f} years of climate data ({start_date.year}-{end_date.year})')
        print('=' * 60)
        
    except Exception as e:
        print(f"Error analyzing climate data: {e}")

if __name__ == "__main__":
    analyze_climate_period()
