#!/usr/bin/env python3
"""
Quick test of the Soil Data Client
"""

import sys
import os
sys.path.append('/workspaces/raven')

from clients.data_clients.soil_client import SoilDataClient

def quick_test():
    print("Quick Soil Data Client Test...")
    
    client = SoilDataClient()
    
    # Test location in Canada (Vancouver - known to have data)
    coords = (49.25, -123.25)
    
    print(f"\nTesting SoilGrids API for Saskatoon: {coords}")
    
    # Test with minimal properties to speed up
    result = client.get_soilgrids_data(coords, properties=['clay'], depths=['0-5cm'])
    
    if result['success']:
        print(f"SUCCESS: Got soil data")
        print(f"Source: {result['source']}")
        if 'soil_properties_by_depth' in result:
            props = result['soil_properties_by_depth']
            for prop, depths in props.items():
                for depth, data in depths.items():
                    value = data['value']
                    if value is not None:
                        print(f"  {prop} ({depth}): {value:.1f} {data['unit']}")
                    else:
                        print(f"  {prop} ({depth}): NULL")
    else:
        print(f"FAILED: {result['error']}")
    
    # Test Canadian soil data
    print(f"\nTesting Canadian soil data sources...")
    canadian_result = client.get_canadian_soil_data(coords)
    
    if canadian_result['success']:
        print(f"SUCCESS: Got Canadian soil data")
        print(f"Source: {canadian_result['source']}")
        print(f"Properties: {canadian_result['soil_properties']}")
    else:
        print(f"Canadian sources failed: {canadian_result['error']}")
    
    print("\nQuick test complete!")

if __name__ == '__main__':
    quick_test()
