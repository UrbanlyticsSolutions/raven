#!/usr/bin/env python3
"""
Lakes Generator - Generates RAVEN Lakes.rvh files
Detects lakes in HRU data and generates proper reservoir definitions
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging


class LakesGenerator:
    """
    Generate RAVEN Lakes.rvh files when lakes are detected in HRU data
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detect_lakes_in_hru_data(self, hru_gdf: gpd.GeoDataFrame) -> bool:
        """
        Detect if there are lakes/water bodies in HRU data
        
        Parameters:
        -----------
        hru_gdf : gpd.GeoDataFrame
            HRU geodataframe to check for lakes
            
        Returns:
        --------
        bool
            True if lakes are detected, False otherwise
        """
        
        # Check specific lake indicator columns first
        if 'HRU_IsLake' in hru_gdf.columns:
            # Check if any HRU is explicitly marked as a lake
            lake_mask = (hru_gdf['HRU_IsLake'].notna()) & (hru_gdf['HRU_IsLake'] != 0) & (hru_gdf['HRU_IsLake'] != False)
            if lake_mask.any():
                return True
        
        if 'lake_type' in hru_gdf.columns:
            # Check if any HRU has a lake type defined
            lake_type_mask = hru_gdf['lake_type'].notna() & (hru_gdf['lake_type'] != '') & (hru_gdf['lake_type'] != 'None')
            if lake_type_mask.any():
                return True
        
        # Check landuse classes for water/lake bodies
        landuse_columns = ['Landuse', 'LAND_USE_C', 'LandCover']
        for col in landuse_columns:
            if col in hru_gdf.columns:
                water_mask = hru_gdf[col].astype(str).str.contains('LAKE|WATER|lake|water|11', case=False, na=False)
                if water_mask.any():
                    return True
                
        return False
    
    def generate_lakes_rvh(self, hru_gdf: gpd.GeoDataFrame, outlet_name: str) -> Optional[Path]:
        """
        Generate Lakes.rvh file if lakes are detected in HRU data
        
        Parameters:
        -----------
        hru_gdf : gpd.GeoDataFrame
            HRU data containing potential lake information
        outlet_name : str
            Name for the outlet/model
            
        Returns:
        --------
        Optional[Path]
            Path to generated Lakes.rvh file, None if no lakes detected
        """
        
        lakes_file = self.output_dir / "Lakes.rvh"
        
        # Check if lakes are present
        if not self.detect_lakes_in_hru_data(hru_gdf):
            self.logger.info("No lakes detected in HRU data - skipping Lakes.rvh")
            return None
        
        self.logger.info("Lakes detected - generating Lakes.rvh")
        
        # Extract lake HRUs
        lake_hrus = self._extract_lake_hrus(hru_gdf)
        
        if lake_hrus.empty:
            self.logger.warning("Lake detection positive but no lake HRUs found")
            return None
        
        # Filter out lake HRUs with invalid SubIds before processing
        valid_lake_hrus = []
        for _, lake_hru in lake_hrus.iterrows():
            subid_columns = ['SubId', 'SubBasin_ID', 'Subbasin_ID', 'SUBBASIN_ID']
            valid_subid = False
            for col in subid_columns:
                if col in lake_hru.index and pd.notna(lake_hru[col]):
                    if isinstance(lake_hru[col], (int, float)) and lake_hru[col] > 0:
                        valid_subid = True
                        break
            if valid_subid:
                valid_lake_hrus.append(lake_hru)
            else:
                self.logger.warning(f"Filtering out lake HRU {lake_hru.get('HRU_ID', 'unknown')} - invalid SubId: {lake_hru.get('SubId', 'missing')}")
        
        if not valid_lake_hrus:
            self.logger.warning("All lake HRUs have invalid SubIds - skipping Lakes.rvh")
            return None
        
        lake_hrus = gpd.GeoDataFrame(valid_lake_hrus)
        
        # Group lakes by subbasin (RAVEN constraint: one reservoir per subbasin)
        subbasin_lakes = self._group_lakes_by_subbasin(lake_hrus)
        
        # Generate Lakes.rvh file
        with open(lakes_file, 'w') as f:
            f.write("#----------------------------------------------\n")
            f.write("# Lakes/Reservoirs Definition File\n")
            f.write(f"# Generated for {outlet_name}\n")
            f.write(f"# Number of composite reservoirs: {len(subbasin_lakes)}\n")
            f.write("#----------------------------------------------\n\n")
            
            for subbasin_id, lakes_group in subbasin_lakes.items():
                self._write_composite_reservoir_definition(f, lakes_group, subbasin_id)
        
        self.logger.info(f"Generated Lakes.rvh file: {lakes_file} with {len(lake_hrus)} lakes")
        return lakes_file
    
    def _extract_lake_hrus(self, hru_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Extract HRUs that represent lakes/water bodies"""
        
        lake_hrus = gpd.GeoDataFrame()
        
        # Priority 1: Check HRU_IsLake column (most reliable)
        if 'HRU_IsLake' in hru_gdf.columns:
            lake_mask = (hru_gdf['HRU_IsLake'] == 1) | (hru_gdf['HRU_IsLake'] == True)
            lake_hrus = hru_gdf[lake_mask]
            if not lake_hrus.empty:
                return lake_hrus
        
        # Priority 2: Check LAND_USE_C for LAKE class
        if 'LAND_USE_C' in hru_gdf.columns:
            lake_mask = hru_gdf['LAND_USE_C'].str.upper() == 'LAKE'
            lake_hrus = hru_gdf[lake_mask]
            if not lake_hrus.empty:
                return lake_hrus
        
        # Priority 3: Check Landuse column 
        if 'Landuse' in hru_gdf.columns:
            water_mask = hru_gdf['Landuse'].str.contains('LAKE|WATER', case=False, na=False)
            lake_hrus = hru_gdf[water_mask]
            if not lake_hrus.empty:
                return lake_hrus
        
        return lake_hrus
    
    def _group_lakes_by_subbasin(self, lake_hrus: gpd.GeoDataFrame) -> dict:
        """Group lake HRUs by subbasin ID to create composite reservoirs"""
        
        subbasin_lakes = {}
        
        for _, lake_hru in lake_hrus.iterrows():
            # Get SubBasin ID
            subbasin_id = None
            subid_columns = ['SubId', 'SubBasin_ID', 'Subbasin_ID', 'SUBBASIN_ID']
            for col in subid_columns:
                if col in lake_hru.index and pd.notna(lake_hru[col]):
                    potential_subid = lake_hru[col]
                    if isinstance(potential_subid, (int, float)) and potential_subid > 0:
                        subbasin_id = int(potential_subid)
                        break
            
            if subbasin_id is not None:
                if subbasin_id not in subbasin_lakes:
                    subbasin_lakes[subbasin_id] = []
                subbasin_lakes[subbasin_id].append(lake_hru)
        
        return subbasin_lakes
    
    def _write_composite_reservoir_definition(self, f, lakes_group: list, subbasin_id: int, subbasin_gdf=None):
        """Write a single composite reservoir definition for multiple lakes in the same subbasin"""
        
        if not lakes_group:
            return
        
        # Use the first lake's HRU ID as the primary HRU ID
        primary_lake = lakes_group[0]
        hru_id = primary_lake.get('HRU_ID', primary_lake.get('HRUID', subbasin_id))
        
        # Use primary HRU area to match RVH file (not composite sum)
        # This prevents the "LakeArea and corresponding HRU area do not agree" warning
        primary_lake_area_m2 = None
        area_columns = ['HRU_Area', 'LakeArea', 'Area_km2', 'AREA', 'area', 'area_km2']
        for col in area_columns:
            if col in primary_lake.index and pd.notna(primary_lake[col]):
                area_value = float(primary_lake[col])
                # Convert km² columns to m²
                if col in ['Area_km2', 'HRU_Area', 'area_km2', 'AREA']:  # These are in km²
                    primary_lake_area_m2 = area_value * 1000000  # Convert km² to m²
                else:
                    primary_lake_area_m2 = area_value  # Already in m² (BasinMaker format)
                break
        
        # If no area column found, calculate from geometry
        if primary_lake_area_m2 is None:
            if hasattr(primary_lake, 'geometry') and primary_lake.geometry is not None:
                primary_lake_area_m2 = primary_lake.geometry.area  # Keep in m²
            else:
                primary_lake_area_m2 = 100000  # Default 0.1 km² = 100,000 m²
        
        # For logging purposes, still calculate total area
        total_area_m2 = primary_lake_area_m2
        for lake_hru in lakes_group[1:]:  # Skip first (primary) lake
            lake_area_m2 = None
            for col in area_columns:
                if col in lake_hru.index and pd.notna(lake_hru[col]):
                    area_value = float(lake_hru[col])
                    if col in ['Area_km2', 'HRU_Area', 'area_km2', 'AREA']:
                        lake_area_m2 = area_value * 1000000
                    else:
                        lake_area_m2 = area_value
                    break
            if lake_area_m2 is None:
                if hasattr(lake_hru, 'geometry') and lake_hru.geometry is not None:
                    lake_area_m2 = lake_hru.geometry.area
                else:
                    lake_area_m2 = 100000
            total_area_m2 += lake_area_m2
        
        # BasinMaker logic: Get hydraulic parameters
        crest_width = 10.0  # Default
        max_depth = 2.0     # Default
        has_gauge = 0       # Default
        
        # Use primary lake's lake ID or default to subbasin ID
        lake_id_from_data = primary_lake.get('HyLakeId', subbasin_id)
        
        # Generate reservoir name
        num_lakes = len(lakes_group)
        if num_lakes == 1:
            reservoir_name = f"Lake_{lake_id_from_data}"
        else:
            reservoir_name = f"Composite_Lake_{subbasin_id}"  # Composite lake name
        
        # Write composite reservoir definition
        f.write(f":Reservoir {reservoir_name}\n")
        f.write(f"  :SubBasinID {int(subbasin_id)}\n")
        f.write(f"  :HRUID {int(hru_id)}\n")  # Primary HRU ID
        f.write("  :WeirCoefficient 0.6\n")                           # BasinMaker standard
        f.write(f"  :CrestWidth {crest_width}\n")                     # Default
        f.write(f"  :MaxDepth {max_depth}\n")                         # Default  
        f.write(f"  :LakeArea {int(primary_lake_area_m2)}\n")          # Primary HRU area in m² (matches RVH)
        f.write("  :SeepageParameters 0 0\n")                        # BasinMaker always includes
        
        # BasinMaker power law for gauged lakes
        if has_gauge > 0:
            f.write("  :OutflowStageRelation POWER_LAW\n")
            
        f.write(":EndReservoir\n\n")
        
        area_km2 = total_area_m2 / 1000000  # For logging
        self.logger.info(f"Added {reservoir_name}: {num_lakes} lakes, Total Area={area_km2:.3f} km² ({int(total_area_m2)} m²)")
    
    def _write_reservoir_definition(self, f, lake_hru, lake_id: int, subbasin_gdf=None):
        """Write a single reservoir definition using BasinMaker hydraulic logic"""
        
        # Get SubBasin ID - try multiple column names, filter invalid values
        subbasin_id = None
        subid_columns = ['SubId', 'SubBasin_ID', 'Subbasin_ID', 'SUBBASIN_ID']
        for col in subid_columns:
            if col in lake_hru.index and pd.notna(lake_hru[col]):
                potential_subid = lake_hru[col]
                if isinstance(potential_subid, (int, float)) and potential_subid > 0:
                    subbasin_id = int(potential_subid)
                    break
        
        # If no valid SubId found, skip this lake (prevents invalid -1 SubIds)
        if subbasin_id is None or subbasin_id <= 0:
            self.logger.warning(f"Skipping Lake_{lake_id} - invalid or missing SubId: {subbasin_id}")
            return
            
        # Get HRU ID 
        hru_id = lake_hru.get('HRU_ID', lake_hru.get('HRUID', lake_id))
        
        # BasinMaker logic: Get hydraulic parameters from subbasin data
        crest_width = 10.0  # Default
        max_depth = 2.0     # Default
        has_gauge = 0       # Default
        lake_id_from_data = lake_hru.get('HyLakeId', lake_id)  # BasinMaker lake ID
        
        if subbasin_gdf is not None:
            subbasin_match = subbasin_gdf[subbasin_gdf['SubId'] == subbasin_id]
            if len(subbasin_match) > 0:
                subbasin = subbasin_match.iloc[0]
                # Use actual hydraulic parameters (BasinMaker approach)
                crest_width = float(subbasin.get('BkfWidth', 10.0))    # Bankfull width as crest width
                channel_depth = float(subbasin.get('BkfDepth', 2.0))   # Channel depth
                has_gauge = int(subbasin.get('Has_POI', 0))
                
                # Lake depth priority: LakeDepth from data, then channel depth
                max_depth = float(lake_hru.get('LakeDepth', channel_depth))
        
        # Calculate area - BasinMaker uses m² directly
        lake_area_m2 = None
        area_columns = ['HRU_Area', 'LakeArea', 'Area_km2', 'AREA', 'area', 'area_km2']
        for col in area_columns:
            if col in lake_hru.index and pd.notna(lake_hru[col]):
                area_value = float(lake_hru[col])
                # Convert km² columns to m²
                if col in ['Area_km2', 'HRU_Area', 'area_km2', 'AREA']:  # These are in km²
                    lake_area_m2 = area_value * 1000000  # Convert km² to m²
                else:
                    lake_area_m2 = area_value  # Already in m² (BasinMaker format)
                break
        
        # If no area column found, calculate from geometry
        if lake_area_m2 is None:
            if hasattr(lake_hru, 'geometry') and lake_hru.geometry is not None:
                lake_area_m2 = lake_hru.geometry.area  # Keep in m²
            else:
                lake_area_m2 = 1000000  # Default 1 km² = 1,000,000 m²
        
        # BasinMaker reservoir format (no :Type, includes :SeepageParameters)
        reservoir_name = f"Lake_{lake_id_from_data}" if lake_id_from_data != lake_id else f"Lake_{lake_id}"
        
        f.write(f":Reservoir {reservoir_name}\n")
        f.write(f"  :SubBasinID {int(subbasin_id)}\n")
        f.write(f"  :HRUID {int(hru_id)}\n")
        f.write("  :WeirCoefficient 0.6\n")                           # BasinMaker standard
        f.write(f"  :CrestWidth {crest_width}\n")                     # From actual BkfWidth
        f.write(f"  :MaxDepth {max_depth}\n")                         # From actual LakeDepth/BkfDepth  
        f.write(f"  :LakeArea {int(lake_area_m2)}\n")                 # In m² (BasinMaker format)
        f.write("  :SeepageParameters 0 0\n")                        # BasinMaker always includes
        
        # BasinMaker power law for gauged lakes
        if has_gauge > 0:
            f.write("  :OutflowStageRelation POWER_LAW\n")
            
        f.write(":EndReservoir\n\n")
        
        area_km2 = lake_area_m2 / 1000000  # For logging
        self.logger.debug(f"Added {reservoir_name}: Area={area_km2:.3f} km² ({int(lake_area_m2)} m²), Depth={max_depth:.1f}m, CrestWidth={crest_width:.1f}m")


def test_lakes_generator():
    """Test function for lakes generator"""
    
    print("Testing Lakes Generator...")
    
    # Create test HRU data with lakes
    test_data = {
        'HRU_ID': [1, 2, 3, 4, 5],
        'SubBasin_ID': [1, 1, 2, 2, 3],
        'Landuse': ['FOREST', 'LAKE', 'GRASSLAND', 'WATER', 'FOREST'],
        'HRU_Area': [5.2, 0.8, 3.1, 1.5, 4.0]
    }
    
    # Create simple geometries
    from shapely.geometry import Point
    geometries = [Point(i, i).buffer(0.01) for i in range(5)]
    
    test_hru_gdf = gpd.GeoDataFrame(test_data, geometry=geometries)
    
    # Test lakes generator
    generator = LakesGenerator(Path("test_output"))
    
    # Test lake detection
    has_lakes = generator.detect_lakes_in_hru_data(test_hru_gdf)
    print(f"Lakes detected: {has_lakes}")
    
    # Test lakes file generation
    if has_lakes:
        lakes_file = generator.generate_lakes_rvh(test_hru_gdf, "test_outlet")
        if lakes_file:
            print(f"Generated lakes file: {lakes_file}")
        else:
            print("No lakes file generated")
    
    print("✓ Lakes generator test completed")


if __name__ == "__main__":
    test_lakes_generator()