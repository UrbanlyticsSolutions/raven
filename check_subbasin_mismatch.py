#!/usr/bin/env python3
"""
Check subbasin-HRU mismatch and fix the issue
"""

import geopandas as gpd
import pandas as pd

def check_subbasin_hru_mismatch():
    """Check which subbasins are missing HRUs"""
    
    print("=== SUBBASIN-HRU MISMATCH ANALYSIS ===")
    
    # Check different subbasin files
    files = [
        'outlet_49.5738_-119.0368/data/subbasins.shp',
        'outlet_49.5738_-119.0368/data/subbasins_with_lakes.shp'
    ]
    
    for file in files:
        try:
            gdf = gpd.read_file(file)
            print(f'\n{file}: {len(gdf)} subbasins')
            print(f'  SubId range: {gdf["SubId"].min()} to {gdf["SubId"].max()}')
            subids = sorted(gdf["SubId"].unique())
            print(f'  SubIds: {subids[:10]}{"..." if len(subids) > 10 else ""}')
        except Exception as e:
            print(f'\n{file}: ERROR - {e}')
    
    # Check HRU assignments
    try:
        hrus = gpd.read_file('outlet_49.5738_-119.0368/data/hrus.geojson')
        print(f'\nHRUs file: {len(hrus)} HRUs')
        
        if 'SubId' in hrus.columns:
            hru_subids = sorted(hrus['SubId'].unique())
            print(f'  HRU SubIds: {hru_subids}')
            
            # Compare with subbasin files
            for file in files:
                try:
                    subbasins = gpd.read_file(file)
                    subbasin_ids = set(subbasins['SubId'])
                    hru_ids = set(hrus['SubId'])
                    
                    missing_hrus = subbasin_ids - hru_ids
                    extra_hrus = hru_ids - subbasin_ids
                    
                    print(f'\n  Comparison with {file}:')
                    print(f'    Subbasins without HRUs: {sorted(missing_hrus) if missing_hrus else "None"}')
                    print(f'    HRUs without subbasins: {sorted(extra_hrus) if extra_hrus else "None"}')
                    
                except Exception as e:
                    print(f'    Error comparing with {file}: {e}')
        else:
            print(f'  ERROR: No SubId column in HRUs. Columns: {list(hrus.columns)}')
            
    except Exception as e:
        print(f'\nHRU file error: {e}')

if __name__ == "__main__":
    check_subbasin_hru_mismatch()
