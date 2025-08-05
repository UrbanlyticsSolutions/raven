#!/usr/bin/env python3
"""
Test script for fixed workflows with proper area calculations
"""

import sys
from pathlib import Path
import json
import traceback
import math
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_area_calculations():
    """Test the fixed area calculations"""
    print("=" * 60)
    print("TESTING FIXED AREA CALCULATIONS")
    print("=" * 60)
    
    import math
    
    # Test coordinates
    test_coords = [
        (45.5017, -73.5673, "Montreal"),
        (45.4215, -75.6972, "Ottawa"),
        (46.8139, -71.2080, "Quebec_City")
    ]
    
    for lat, lon, name in test_coords:
        print(f"\n{name} ({lat}, {lon}):")
        
        # Test different buffer sizes
        for buffer_km in [1.0, 2.0, 5.0]:
            # Proper buffer calculation
            lat_buffer = buffer_km / 111.0
            lon_buffer = buffer_km / (111.0 * abs(math.cos(math.radians(lat))))
            
            bounds = (
                lon - lon_buffer,
                lat - lat_buffer,
                lon + lon_buffer,
                lat + lat_buffer
            )
            
            width_deg = bounds[2] - bounds[0]
            height_deg = bounds[3] - bounds[1]
            width_km = width_deg * 111.0 * abs(math.cos(math.radians(lat)))
            height_km = height_deg * 111.0
            area_km2 = width_km * height_km
            
            print(f"  {buffer_km}km buffer: {width_km:.1f}x{height_km:.1f}km = {area_km2:.1f} km²")

def test_gauge_discovery():
    """Test gauge discovery without full processing"""
    print("\n" + "=" * 60)
    print("TESTING GAUGE DISCOVERY ONLY")
    print("=" * 60)
    
    try:
        from workflows.multi_gauge_delineation import MultiGaugeDelineation
        
        # Small test region around Montreal
        bbox = (-73.8, 45.4, -73.4, 45.6)  # Much smaller area
        
        workflow = MultiGaugeDelineation(
            workspace_dir="test_outputs/gauge_discovery_test"
        )
        
        print(f"Testing gauge discovery for bbox: {bbox}")
        
        # Just test gauge discovery
        gauges = workflow.discover_gauges_in_region(bbox, buffer_km=0.1)
        
        print(f"✅ Found {len(gauges)} gauges")
        
        # Show details of first few gauges
        for i, gauge in enumerate(gauges[:5]):
            print(f"  {i+1}. {gauge['station_id']}: {gauge['station_name']}")
            print(f"     Area: {gauge['drainage_area_km2']} km²")
            print(f"     Location: ({gauge['latitude']:.4f}, {gauge['longitude']:.4f})")
        
        return gauges
        
    except Exception as e:
        print(f"❌ Gauge discovery failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return []

def test_dem_download_only():
    """Test just DEM download with proper area calculations"""
    print("\n" + "=" * 60)
    print("TESTING DEM DOWNLOAD WITH FIXED AREAS")
    print("=" * 60)
    
    try:
        from workflows.steps import DEMClippingStep
        
        # Test small areas around different cities
        test_areas = [
            ((-73.6, 45.4, -73.5, 45.5), "Montreal_Small"),
            ((-75.8, 45.3, -75.6, 45.5), "Ottawa_Small")
        ]
        
        for bounds, name in test_areas:
            print(f"\nTesting DEM download for {name}: {bounds}")
            
            width_deg = bounds[2] - bounds[0]
            height_deg = bounds[3] - bounds[1]
            center_lat = (bounds[1] + bounds[3]) / 2
            width_km = width_deg * 111.0 * abs(math.cos(math.radians(center_lat)))
            height_km = height_deg * 111.0
            area_km2 = width_km * height_km
            
            print(f"  Area: {width_km:.1f}x{height_km:.1f}km = {area_km2:.1f} km²")
            
            try:
                dem_step = DEMClippingStep(workspace_dir=f"test_outputs/dem_test_{name.lower()}")
                
                result = dem_step.execute(
                    bounds=bounds,
                    resolution=30,
                    output_filename=f"{name.lower()}_dem.tif"
                )
                
                if result['success']:
                    dem_file = Path(result['dem_file'])
                    if dem_file.exists():
                        size_mb = dem_file.stat().st_size / (1024 * 1024)
                        print(f"  ✅ DEM downloaded: {dem_file.name} ({size_mb:.1f} MB)")
                    else:
                        print(f"  ❌ DEM file not found: {result['dem_file']}")
                else:
                    print(f"  ❌ DEM download failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"  ❌ Exception: {str(e)}")
        
    except Exception as e:
        print(f"❌ DEM test failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

def test_refactored_workflow_minimal():
    """Test refactored workflow with minimal processing to avoid PROJ issues"""
    print("\n" + "=" * 60)
    print("TESTING REFACTORED WORKFLOW (MINIMAL)")
    print("=" * 60)
    
    try:
        from workflows.refactored_full_delineation import RefactoredFullDelineation
        
        # Test with very small area
        lat, lon = 45.5017, -73.5673
        
        print(f"Testing minimal workflow for Montreal: ({lat}, {lon})")
        
        workflow = RefactoredFullDelineation(
            workspace_dir="test_outputs/minimal_refactored"
        )
        
        # Test just the bounds calculation
        buffer_km = 1.0
        lat_buffer = buffer_km / 111.0
        lon_buffer = buffer_km / (111.0 * abs(math.cos(math.radians(lat))))
        
        bounds = (
            lon - lon_buffer,
            lat - lat_buffer,
            lon + lon_buffer,
            lat + lat_buffer
        )
        
        width_deg = bounds[2] - bounds[0]
        height_deg = bounds[3] - bounds[1]
        width_km = width_deg * 111.0 * abs(math.cos(math.radians(lat)))
        height_km = height_deg * 111.0
        area_km2 = width_km * height_km
        
        print(f"Calculated bounds: {bounds}")
        print(f"Area: {width_km:.1f}x{height_km:.1f}km = {area_km2:.1f} km²")
        
        # Test just DEM preparation
        try:
            shared_prep = workflow.prepare_shared_datasets(bounds, buffer_km=0.1)
            
            if shared_prep['success']:
                print(f"✅ Shared datasets prepared successfully")
                print(f"  Bounds used: {shared_prep['bounds']}")
                
                for key, path in shared_prep['datasets'].items():
                    file_path = Path(path)
                    if file_path.exists():
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        print(f"  {key}: {file_path.name} ({size_mb:.1f} MB)")
                    else:
                        print(f"  {key}: FILE NOT FOUND - {path}")
            else:
                print(f"❌ Shared datasets failed: {shared_prep.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Shared datasets exception: {str(e)}")
        
    except Exception as e:
        print(f"❌ Refactored workflow test failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

def analyze_existing_outputs():
    """Analyze what was already generated"""
    print("\n" + "=" * 60)
    print("ANALYZING EXISTING OUTPUTS")
    print("=" * 60)
    
    test_outputs = Path("test_outputs")
    if not test_outputs.exists():
        print("❌ No test outputs found")
        return
    
    # Look for DEM files that were successfully downloaded
    dem_files = list(test_outputs.rglob("*.tif"))
    
    print(f"Found {len(dem_files)} DEM/raster files:")
    
    for dem_file in dem_files:
        try:
            size_mb = dem_file.stat().st_size / (1024 * 1024)
            print(f"  {dem_file.relative_to(test_outputs)} ({size_mb:.1f} MB)")
            
            # Try to read basic info
            try:
                import rasterio
                with rasterio.open(dem_file) as src:
                    print(f"    Shape: {src.shape}, CRS: {src.crs.to_string() if src.crs else 'None'}")
                    bounds = src.bounds
                    width_deg = bounds.right - bounds.left
                    height_deg = bounds.top - bounds.bottom
                    print(f"    Bounds: {width_deg:.3f}° x {height_deg:.3f}°")
            except Exception as e:
                print(f"    Could not read raster info: {str(e)}")
                
        except Exception as e:
            print(f"  Error reading {dem_file}: {str(e)}")
    
    # Look for gauge files
    gauge_files = list(test_outputs.rglob("*.geojson"))
    
    print(f"\nFound {len(gauge_files)} gauge files:")
    
    for gauge_file in gauge_files:
        try:
            size_mb = gauge_file.stat().st_size / (1024 * 1024)
            print(f"  {gauge_file.relative_to(test_outputs)} ({size_mb:.2f} MB)")
            
            # Try to read gauge count
            try:
                with open(gauge_file) as f:
                    data = json.load(f)
                    gauge_count = len(data.get('features', []))
                    print(f"    Contains {gauge_count} gauges")
            except Exception as e:
                print(f"    Could not read gauge data: {str(e)}")
                
        except Exception as e:
            print(f"  Error reading {gauge_file}: {str(e)}")

def main():
    """Main test function"""
    print("🚀 TESTING FIXED WORKFLOWS")
    print(f"Timestamp: {datetime.now()}")
    print(f"Working directory: {Path.cwd()}")
    
    # Test area calculations
    test_area_calculations()
    
    # Test gauge discovery
    test_gauge_discovery()
    
    # Test DEM download with proper areas
    test_dem_download_only()
    
    # Test minimal refactored workflow
    test_refactored_workflow_minimal()
    
    # Analyze existing outputs
    analyze_existing_outputs()
    
    print("\n🏁 FIXED WORKFLOW TESTS COMPLETED")

if __name__ == "__main__":
    main()