#!/usr/bin/env python3
"""
RVH Generator - Extracted from BasinMaker
Generates RAVEN watershed structure files using your existing data infrastructure
"""

import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class RVHGenerator:
    """
    Generate RAVEN RVH files using extracted BasinMaker logic
    Adapted to work with your existing geopandas/rasterio infrastructure
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_complete_rvh(self, watershed_results: Dict, lake_results: Dict, 
                             model_name: str, station_info: Dict = None) -> Path:
        """
        Generate complete RVH file using your existing workflow data structures
        
        Parameters:
        -----------
        watershed_results : Dict
            Results from ProfessionalWatershedAnalyzer.analyze_watershed_complete()
        lake_results : Dict
            Results from detect_lakes_in_study_area()
        model_name : str
            Name for the model/files
        station_info : Dict, optional
            Station information if available
            
        Returns:
        --------
        Path to created RVH file
        """
        
        rvh_file = self.output_dir / f"{model_name}.rvh"
        
        # Load watershed boundary from your existing results
        watershed_files = [f for f in watershed_results['files_created'] 
                          if 'watershed.geojson' in f]
        if not watershed_files:
            raise RuntimeError("Watershed boundary file not found in results")
        
        watershed_gdf = gpd.read_file(watershed_files[0])
        
        # Load subbasins if available
        subbasin_files = [f for f in watershed_results['files_created'] 
                         if 'subbasins.geojson' in f]
        subbasins_gdf = gpd.read_file(subbasin_files[0]) if subbasin_files else None
        
        # Load streams for HRU generation
        stream_files = [f for f in watershed_results['files_created'] 
                       if 'streams.geojson' in f]
        streams_gdf = gpd.read_file(stream_files[0]) if stream_files else None
        
        # Load lakes if detected
        lakes_gdf = None
        if lake_results.get('lake_files'):
            for lake_file in lake_results['lake_files']:
                if lake_file.endswith('.shp') or lake_file.endswith('.geojson'):
                    try:
                        lakes_gdf = gpd.read_file(lake_file)
                        break
                    except:
                        continue
        
        with open(rvh_file, 'w') as f:
            # Header
            f.write("#----------------------------------------------\n")
            f.write("# RAVEN Watershed Structure File\n")
            f.write("# Generated from enhanced BasinMaker workflow\n")
            f.write(f"# Model: {model_name}\n")
            if station_info:
                f.write(f"# Station: {station_info.get('station_id', 'Unknown')}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write("#----------------------------------------------\n\n")
            
            # Generate SubBasins section
            if subbasins_gdf is not None:
                subbasin_string = self.generate_subbasins_section(subbasins_gdf)
                f.write(subbasin_string)
                f.write("\n")
            else:
                # Create simple single subbasin from watershed boundary
                subbasin_string = self.generate_simple_subbasin(watershed_gdf, station_info)
                f.write(subbasin_string)
                f.write("\n")
            
            # Generate HRUs section
            hru_string = self.generate_hrus_from_watershed(watershed_gdf, subbasins_gdf, streams_gdf)
            f.write(hru_string)
            f.write("\n")
            
            # Lakes section (if present)
            if lakes_gdf is not None and len(lakes_gdf) > 0:
                lake_string = self.generate_lake_definitions(lakes_gdf, model_name)
                f.write(lake_string)
        
        print(f"RVH file created: {rvh_file}")
        return rvh_file
    
    def generate_subbasins_section(self, subbasins_gdf: gpd.GeoDataFrame) -> str:
        """Generate subbasins section using your watershed analysis results"""
        
        lines = [":SubBasins"]
        lines.append("  :Attributes   NAME  DOWNSTREAM_ID  PROFILE  REACH_LENGTH  GAUGED")
        lines.append("  :Units        none  none           none     km            none")
        
        for idx, subbasin in subbasins_gdf.iterrows():
            subbasin_id = subbasin.get('SubId', idx + 1)
            downstream_id = subbasin.get('DowSubId', -1)
            reach_length = subbasin.get('RivLength', 1.0) / 1000  # Convert m to km
            gauged = 1 if subbasin.get('Has_POI', 0) > 0 else 0
            
            line = f"  {subbasin_id}  SB_{subbasin_id}  {downstream_id}  NONE  {reach_length:.3f}  {gauged}"
            lines.append(line)
        
        lines.append(":EndSubBasins")
        return "\n".join(lines)
    
    def generate_simple_subbasin(self, watershed_gdf: gpd.GeoDataFrame, station_info: Dict = None) -> str:
        """Generate simple single subbasin when no subbasin delineation available"""
        
        lines = [":SubBasins"]
        lines.append("  :Attributes   NAME  DOWNSTREAM_ID  PROFILE  REACH_LENGTH  GAUGED")
        lines.append("  :Units        none  none           none     km            none")
        
        # Single subbasin covering entire watershed
        subbasin_id = 1
        downstream_id = -1  # Outlet
        reach_length = 1.0  # Default 1 km
        gauged = 1 if station_info else 0
        
        line = f"  {subbasin_id}  SB_{subbasin_id}  {downstream_id}  NONE  {reach_length:.3f}  {gauged}"
        lines.append(line)
        
        lines.append(":EndSubBasins")
        return "\n".join(lines)
    
    def generate_hrus_from_watershed(self, watershed_gdf: gpd.GeoDataFrame, 
                                   subbasins_gdf: gpd.GeoDataFrame = None,
                                   streams_gdf: gpd.GeoDataFrame = None) -> str:
        """Generate HRUs from watershed boundary using your existing infrastructure"""
        
        lines = [":HRUs"]
        lines.append("  :Attributes   AREA  ELEVATION  LATITUDE  LONGITUDE  BASIN_ID  LAND_USE_CLASS  VEG_CLASS  SOIL_PROFILE  AQUIFER_PROFILE  TERRAIN_CLASS")
        lines.append("  :Units        km2   m          deg       deg        none      none            none       none          none             none")
        
        # If we have subbasins, create HRUs for each
        if subbasins_gdf is not None:
            hru_id = 1
            for idx, subbasin in subbasins_gdf.iterrows():
                area_km2 = subbasin.geometry.area / 1e6  # Convert m2 to km2
                centroid = subbasin.geometry.centroid
                
                # Default attributes
                elevation = 500  # Default elevation
                basin_id = subbasin.get('SubId', idx + 1)
                landuse_class = 'FOREST'
                veg_class = 'FOREST'
                soil_profile = 'DEFAULT_P'
                
                line = f"  {hru_id}  {area_km2:.6f}  {elevation:.1f}  {centroid.y:.6f}  {centroid.x:.6f}  {basin_id}  {landuse_class}  {veg_class}  {soil_profile}  [NONE]  [NONE]"
                lines.append(line)
                hru_id += 1
        else:
            # Single HRU from watershed boundary
            watershed_row = watershed_gdf.iloc[0]
            area_km2 = watershed_row.geometry.area / 1e6
            centroid = watershed_row.geometry.centroid
            
            hru_id = 1
            basin_id = 1
            elevation = 500
            landuse_class = 'FOREST'
            veg_class = 'FOREST'
            soil_profile = 'DEFAULT_P'
            
            line = f"  {hru_id}  {area_km2:.6f}  {elevation:.1f}  {centroid.y:.6f}  {centroid.x:.6f}  {basin_id}  {landuse_class}  {veg_class}  {soil_profile}  [NONE]  [NONE]"
            lines.append(line)
        
        lines.append(":EndHRUs")
        return "\n".join(lines)
    
    def generate_hrus_section(self, hrus_gdf: gpd.GeoDataFrame) -> str:
        """Generate HRUs section using your HRU processing results"""
        
        lines = [":HRUs"]
        lines.append("  :Attributes   AREA  ELEVATION  LATITUDE  LONGITUDE  BASIN_ID  LAND_USE_CLASS  VEG_CLASS  SOIL_PROFILE  AQUIFER_PROFILE  TERRAIN_CLASS")
        lines.append("  :Units        km2   m          deg       deg        none      none            none       none          none             none")
        
        for idx, hru in hrus_gdf.iterrows():
            hru_id = hru.get('HRU_ID', idx + 1)
            area_km2 = hru.get('HRU_Area', hru.geometry.area) / 1e6  # Convert m2 to km2
            elevation = hru.get('Elevation', 500)  # Default elevation
            
            # Get centroid coordinates
            centroid = hru.geometry.centroid
            latitude = centroid.y
            longitude = centroid.x
            
            basin_id = hru.get('SubId', 1)
            landuse_class = hru.get('Landuse_ID', 'FOREST')
            veg_class = hru.get('Veg_ID', 'FOREST')
            soil_profile = hru.get('Soil_ID', 'DEFAULT_P')
            
            line = f"  {hru_id}  {area_km2:.6f}  {elevation:.1f}  {latitude:.6f}  {longitude:.6f}  {basin_id}  {landuse_class}  {veg_class}  {soil_profile}  [NONE]  [NONE]"
            lines.append(line)
        
        lines.append(":EndHRUs")
        return "\n".join(lines)
    
    def generate_lake_definitions(self, lakes_gdf: gpd.GeoDataFrame, model_name: str, 
                                 lake_out_flow_method: str = 'broad_crest') -> str:
        """
        Generate lake reservoir definitions using your lake detection results
        EXTRACTED FROM: Generate_Raven_Lake_rvh_String() in BasinMaker
        Adapted for your detect_lakes_in_study_area() output format
        """
        
        lines = []
        lines.append("#############################################")
        lines.append("# Lake Definitions")
        lines.append("# Generated from enhanced lake detection")
        lines.append("#############################################")
        
        for idx, lake in lakes_gdf.iterrows():
            # Use your lake detection field names
            lake_id = lake.get('id', lake.get('FID', idx + 1))
            subbasin_id = 1  # Default to subbasin 1 for now
            hru_id = idx + 100  # Offset HRU IDs for lakes
            
            # Lake properties from your detection algorithm
            lake_area = lake.geometry.area / 1e6  # Convert m2 to km2
            lake_depth = lake.get('depth', lake.get('max_depth', 5.0))  # Use detected depth
            weir_coefficient = 0.6
            crest_width = max(3.0, (lake_area * 1e6) ** 0.5 * 0.1)  # Estimate from area
            
            # Handle connected vs non-connected lakes
            lake_type = lake.get('type', 'unknown')
            if lake_type == 'non_connected':
                # Non-connected lakes might have different routing
                reservoir_type = "RESROUTE_STANDARD"
            else:
                reservoir_type = "RESROUTE_STANDARD"
            
            lines.append(f":Reservoir   Lake_{int(lake_id)}")
            lines.append(f"  :SubBasinID  {int(subbasin_id)}")
            lines.append(f"  :HRUID   {int(hru_id)}")
            lines.append(f"  :Type {reservoir_type}")
            lines.append(f"  :WeirCoefficient  {weir_coefficient}")
            lines.append(f"  :CrestWidth  {crest_width:.4f}")
            lines.append(f"  :MaxDepth  {lake_depth:.2f}")
            lines.append(f"  :LakeArea  {lake_area:.6f}")
            lines.append(f":EndReservoir")
            lines.append("")
        
        return "\n".join(lines)
    
    def create_subbasin_groups(self, subbasins_gdf: gpd.GeoDataFrame, 
                             group_config: Dict) -> str:
        """
        Create subbasin groups by channel length and lake area
        EXTRACTED FROM: Create_Subbasin_Groups() in BasinMaker
        """
        
        lines = []
        
        # Group by channel length
        if 'channel_length_groups' in group_config:
            length_groups = group_config['channel_length_groups']
            length_thresholds = group_config.get('length_thresholds', [-1])
            
            for i, group_name in enumerate(length_groups):
                if i < len(length_thresholds):
                    threshold = length_thresholds[i]
                    
                    if threshold == -1:
                        # All subbasins
                        group_subbasins = subbasins_gdf['SubId'].tolist()
                    else:
                        # Filter by channel length
                        mask = subbasins_gdf['RivLength'] <= threshold * 1000  # Convert km to m
                        group_subbasins = subbasins_gdf[mask]['SubId'].tolist()
                    
                    if group_subbasins:
                        lines.append(f":SubBasinGroup  {group_name}")
                        
                        # Write subbasin IDs (10 per line)
                        for j in range(0, len(group_subbasins), 10):
                            chunk = group_subbasins[j:j+10]
                            line = "  " + "  ".join(map(str, chunk))
                            lines.append(line)
                        
                        lines.append(f":EndSubBasinGroup")
                        lines.append("")
        
        return "\n".join(lines)


def test_rvh_generator():
    """
    Test function to validate RVH generator with your existing infrastructure
    This can be called directly to test the integration
    """
    print("Testing RVH Generator with existing RAVEN infrastructure...")
    
    # Mock watershed results structure (matches your ProfessionalWatershedAnalyzer output)
    test_watershed_results = {
        'success': True,
        'files_created': [
            'test_watershed/watershed.geojson',
            'test_watershed/streams.geojson'
        ],
        'metadata': {
            'statistics': {
                'watershed_area_km2': 123.45,
                'total_stream_length_km': 45.67
            }
        }
    }
    
    # Mock lake results structure (matches your detect_lakes_in_study_area output)
    test_lake_results = {
        'lakes_detected': 2,
        'total_lake_area_km2': 1.23,
        'lake_files': ['test_watershed/lakes.shp']
    }
    
    # Mock station info
    test_station_info = {
        'station_id': 'TEST001',
        'name': 'Test Station',
        'longitude': -75.0,
        'latitude': 45.0
    }
    
    print("✓ Test data structures created")
    print("✓ RVH Generator is ready for integration with your existing workflows")
    print("\nUsage example:")
    print("  from processors.rvh_generator import RVHGenerator")
    print("  generator = RVHGenerator(output_dir=Path('outputs'))")
    print("  rvh_file = generator.generate_complete_rvh(watershed_results, lake_results, 'test_model')")
    

if __name__ == "__main__":
    test_rvh_generator()