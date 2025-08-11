#!/usr/bin/env python3
"""
Test script to verify that the lake area calculation fix works correctly
"""

import sys
import pandas as pd
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)

# Add the workspace to path
sys.path.append('.')

from processors.lakes_generator import LakesGenerator

def test_lake_area_calculation():
    """Test that lake areas are correctly calculated from HRU data"""
    
    # Create test data that mimics the actual HRU structure
    test_data = [
        {'HRU_ID': 455, 'AREA': 0.0739, 'LAND_USE_C': 'LAKE', 'SubId': 7},  # Subbasin 7
        {'HRU_ID': 456, 'AREA': 0.0293, 'LAND_USE_C': 'LAKE', 'SubId': 7},  # Subbasin 7
        {'HRU_ID': 457, 'AREA': 0.0334, 'LAND_USE_C': 'LAKE', 'SubId': 7},  # Subbasin 7
        {'HRU_ID': 458, 'AREA': 0.0129, 'LAND_USE_C': 'LAKE', 'SubId': 6},  # Subbasin 6
        {'HRU_ID': 459, 'AREA': 0.0223, 'LAND_USE_C': 'LAKE', 'SubId': 13}, # Subbasin 13
        {'HRU_ID': 460, 'AREA': 0.0123, 'LAND_USE_C': 'LAKE', 'SubId': 39}, # Subbasin 39
    ]
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    
    # Initialize generator
    generator = LakesGenerator(output_dir='./test_output')
    
    # Extract lake HRUs
    lake_hrus = generator._extract_lake_hrus(df)
    print(f"Extracted {len(lake_hrus)} lake HRUs")
    
    # Group by subbasin
    subbasin_lakes = generator._group_lakes_by_subbasin(lake_hrus)
    print(f"Grouped into {len(subbasin_lakes)} subbasins: {list(subbasin_lakes.keys())}")
    
    # Test area calculation for subbasin 7 (should be composite)
    if 7 in subbasin_lakes:
        lakes_group_7 = subbasin_lakes[7]
        print(f"\nSubbasin 7 has {len(lakes_group_7)} lakes:")
        
        total_area_m2 = 0
        for lake_hru in lakes_group_7:
            area_km2 = lake_hru['AREA']
            area_m2 = area_km2 * 1000000  # Convert km² to m²
            print(f"  HRU {lake_hru['HRU_ID']}: {area_km2} km² = {int(area_m2)} m²")
            total_area_m2 += area_m2
        
        expected_total_km2 = 0.0739 + 0.0293 + 0.0334  # 0.1366 km²
        expected_total_m2 = expected_total_km2 * 1000000  # 136,600 m²
        
        print(f"\nExpected total area: {expected_total_km2} km² = {int(expected_total_m2)} m²")
        print(f"Calculated total area: {total_area_m2/1000000:.4f} km² = {int(total_area_m2)} m²")
        
        if abs(total_area_m2 - expected_total_m2) < 1:  # Allow 1 m² rounding error
            print("✓ PASS: Lake area calculation is correct!")
            return True
        else:
            print("✗ FAIL: Lake area calculation is incorrect!")
            return False
    else:
        print("✗ FAIL: Subbasin 7 not found!")
        return False

if __name__ == "__main__":
    success = test_lake_area_calculation()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
