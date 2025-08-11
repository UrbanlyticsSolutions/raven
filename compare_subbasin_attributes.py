#!/usr/bin/env python3
"""
Compare attributes between subbasins.shp and subbasins_with_lakes.shp
to ensure subbasins.shp has all necessary attributes for RAVEN modeling
"""

import geopandas as gpd
from pathlib import Path
import pandas as pd

def compare_subbasin_attributes():
    """Compare attributes between the two subbasin files"""
    
    workspace_dir = Path("outlet_49.5738_-119.0368/data")
    
    # File paths
    subbasins_file = workspace_dir / "subbasins.shp"
    subbasins_with_lakes_file = workspace_dir / "subbasins_with_lakes.shp"
    
    print("=" * 80)
    print("COMPARING SUBBASIN ATTRIBUTES")
    print("=" * 80)
    
    # Check if files exist
    if not subbasins_file.exists():
        print(f"ERROR: {subbasins_file} not found")
        return
    
    if not subbasins_with_lakes_file.exists():
        print(f"ERROR: {subbasins_with_lakes_file} not found")
        return
    
    # Load both files
    print("Loading subbasin files...")
    gdf_basic = gpd.read_file(subbasins_file)
    gdf_with_lakes = gpd.read_file(subbasins_with_lakes_file)
    
    print(f"subbasins.shp: {len(gdf_basic)} features")
    print(f"subbasins_with_lakes.shp: {len(gdf_with_lakes)} features")
    print()
    
    # Compare column sets
    cols_basic = set(gdf_basic.columns)
    cols_with_lakes = set(gdf_with_lakes.columns)
    
    # Remove geometry column for comparison
    cols_basic.discard('geometry')
    cols_with_lakes.discard('geometry')
    
    print("=" * 50)
    print("ATTRIBUTE COMPARISON")
    print("=" * 50)
    
    print(f"subbasins.shp attributes ({len(cols_basic)}):")
    for col in sorted(cols_basic):
        print(f"  - {col}")
    print()
    
    print(f"subbasins_with_lakes.shp attributes ({len(cols_with_lakes)}):")
    for col in sorted(cols_with_lakes):
        print(f"  - {col}")
    print()
    
    # Find differences
    missing_in_basic = cols_with_lakes - cols_basic
    missing_in_lakes = cols_basic - cols_with_lakes
    common_attrs = cols_basic & cols_with_lakes
    
    print("=" * 50)
    print("ATTRIBUTE DIFFERENCES")
    print("=" * 50)
    
    if missing_in_basic:
        print(f"❌ MISSING in subbasins.shp ({len(missing_in_basic)} attributes):")
        for attr in sorted(missing_in_basic):
            print(f"   - {attr}")
        print()
    
    if missing_in_lakes:
        print(f"❌ MISSING in subbasins_with_lakes.shp ({len(missing_in_lakes)} attributes):")
        for attr in sorted(missing_in_lakes):
            print(f"   - {attr}")
        print()
    
    if common_attrs:
        print(f"✅ COMMON attributes ({len(common_attrs)}):")
        for attr in sorted(common_attrs):
            print(f"   - {attr}")
        print()
    
    # Check required RAVEN attributes
    raven_required_attrs = [
        'SubId', 'DowSubId', 'DrainArea', 'RivLength', 'RivSlope', 
        'BkfWidth', 'BkfDepth', 'Ch_n', 'FloodP_n', 'MeanElev', 
        'BasArea', 'Lake_Cat', 'Has_POI'
    ]
    
    print("=" * 50)
    print("RAVEN REQUIRED ATTRIBUTES CHECK")
    print("=" * 50)
    
    print("Checking subbasins.shp for RAVEN requirements:")
    basic_missing_raven = []
    for attr in raven_required_attrs:
        if attr in cols_basic:
            print(f"   ✅ {attr}")
        else:
            print(f"   ❌ {attr} - MISSING")
            basic_missing_raven.append(attr)
    
    print()
    print("Checking subbasins_with_lakes.shp for RAVEN requirements:")
    lakes_missing_raven = []
    for attr in raven_required_attrs:
        if attr in cols_with_lakes:
            print(f"   ✅ {attr}")
        else:
            print(f"   ❌ {attr} - MISSING")
            lakes_missing_raven.append(attr)
    
    print()
    print("=" * 50)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 50)
    
    if len(basic_missing_raven) == 0:
        print("✅ subbasins.shp has ALL required RAVEN attributes")
    else:
        print(f"❌ subbasins.shp is MISSING {len(basic_missing_raven)} required RAVEN attributes:")
        for attr in basic_missing_raven:
            print(f"   - {attr}")
    
    if len(lakes_missing_raven) == 0:
        print("✅ subbasins_with_lakes.shp has ALL required RAVEN attributes")
    else:
        print(f"❌ subbasins_with_lakes.shp is MISSING {len(lakes_missing_raven)} required RAVEN attributes:")
        for attr in lakes_missing_raven:
            print(f"   - {attr}")
    
    print()
    
    # Check if we can copy missing attributes
    if missing_in_basic and len(gdf_basic) <= len(gdf_with_lakes):
        print("🔧 POTENTIAL FIX:")
        print("   Missing attributes in subbasins.shp could be copied from subbasins_with_lakes.shp")
        print("   if we can establish a proper mapping between the features.")
        
        # Check if SubId exists in both for mapping
        if 'SubId' in cols_basic and 'SubId' in cols_with_lakes:
            basic_subids = set(gdf_basic['SubId'].unique())
            lakes_subids = set(gdf_with_lakes['SubId'].unique())
            
            common_subids = basic_subids & lakes_subids
            print(f"   Common SubIds: {len(common_subids)} out of {len(basic_subids)} in subbasins.shp")
            
            if len(common_subids) == len(basic_subids):
                print("   ✅ All SubIds in subbasins.shp exist in subbasins_with_lakes.shp")
                print("   ✅ Attribute copying is POSSIBLE")
            else:
                missing_subids = basic_subids - lakes_subids
                print(f"   ❌ Missing SubIds in lakes file: {missing_subids}")
    
    # Show sample data for key attributes
    print()
    print("=" * 50)
    print("SAMPLE DATA COMPARISON")
    print("=" * 50)
    
    if 'SubId' in cols_basic and 'SubId' in cols_with_lakes:
        print("Sample SubIds:")
        print(f"  subbasins.shp: {sorted(gdf_basic['SubId'].unique())[:10]}...")
        print(f"  subbasins_with_lakes.shp: {sorted(gdf_with_lakes['SubId'].unique())[:10]}...")
    
    # Check data completeness for key attributes
    for attr in ['RivLength', 'RivSlope', 'DrainArea']:
        if attr in cols_basic:
            null_count = gdf_basic[attr].isnull().sum()
            print(f"  subbasins.shp {attr}: {null_count}/{len(gdf_basic)} null values")
        
        if attr in cols_with_lakes:
            null_count = gdf_with_lakes[attr].isnull().sum()
            print(f"  subbasins_with_lakes.shp {attr}: {null_count}/{len(gdf_with_lakes)} null values")

if __name__ == "__main__":
    compare_subbasin_attributes()
