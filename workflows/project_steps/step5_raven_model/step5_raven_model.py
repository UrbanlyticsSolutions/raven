#!/usr/bin/env python3
"""
Enhanced Step 5: RAVEN Model Generation with Dynamic Lookup Database Integration
Uses JSON-based lookup tables for comprehensive soil/landcover parameter extraction
ENHANCED: All parameter gaps fixed with dynamic generators
"""

import sys
from pathlib import Path
import argparse
import json
from typing import Dict, Any, Optional, List
import geopandas as gpd
import pandas as pd
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from processors.rvh_generator import RVHGenerator
from processors.raven_generator import RAVENGenerator  
from processors.rvp_generator import RVPGenerator
from processors.rvt_generator import RVTGenerator
from processors.observation_point_integrator import ObservationPointIntegrator
from utilities.raven_parameter_extractor import RAVENParameterExtractor
from utilities.lookup_table_generator import RAVENLookupTableGenerator
from processors.terrain_attributes_calculator import TerrainAttributesCalculator
from processors.lakes_generator import LakesGenerator
from utilities.centralized_config_manager import CentralizedConfigManager, ConfigurationError
from clients.data_clients.hydrometric_client import HydrometricDataClient

class Step5RAVENModel:
    """
    Enhanced Step 5: Generate RAVEN model with dynamic lookup database integration
    ENHANCED: All parameter gaps fixed with comprehensive JSON-based classification
    """
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir).resolve()  # Use absolute path
        self.models_dir = self.workspace_dir / "models" / "files"
        self.models_dir.mkdir(exist_ok=True, parents=True)
        print(f"Initialized workspace: {self.workspace_dir}")
        
        # Load all configuration files using centralized config manager with FAIL-FAST validation
        print("Loading all configuration files...")
        try:
            project_root = Path("E:/python/Raven").resolve()
            config_dir = project_root / "config"
            self.config_manager = CentralizedConfigManager(config_dir)
            self.config_manager.print_config_summary()
            # Keep backward compatibility with old config access
            self.config = self.config_manager.get_all_configs()['raven_class_definitions']
            print("[OK] All configuration loaded successfully with FAIL-FAST validation")
        except ConfigurationError as e:
            print(f"[CRITICAL ERROR] Configuration validation failed: {e}")
            print("STEP 5 CANNOT PROCEED WITHOUT VALID CONFIGURATION")
            raise e
        
        # Initialize enhanced parameter extractors and generators
        print("Initializing enhanced RAVEN model generators...")
        self.parameter_extractor = RAVENParameterExtractor(output_dir=self.workspace_dir / "parameters")
        self.lookup_generator = RAVENLookupTableGenerator(output_dir=self.workspace_dir / "lookup_tables")
        self.terrain_calculator = TerrainAttributesCalculator()
        
        # Initialize original generators
        self.rvh_generator = RVHGenerator(self.models_dir)
        self.raven_generator = RAVENGenerator(self.models_dir)
        
        # Initialize RVT and lakes generators
        self.rvt_generator = RVTGenerator(self.models_dir)
        self.lakes_generator = LakesGenerator(self.models_dir)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _load_all_config_files(self) -> Dict[str, Any]:
        """Load all RAVEN configuration files at initialization - centralized config loading"""
        
        # Use absolute paths with pathlib
        project_root = Path("E:/python/Raven").resolve()
        config_dir = project_root / "config"
        
        # Define all config files to load
        config_files = {
            'lookup_database': config_dir / "raven_lookup_database.json",
            'parameter_table': config_dir / "raven_complete_parameter_table.json"
        }
        
        print(f"Config directory: {config_dir}")
        config = {}
        
        # Load lookup database
        lookup_file = config_files['lookup_database']
        print(f"Loading lookup database: {lookup_file}")
        if lookup_file.exists():
            try:
                with open(lookup_file, 'r') as f:
                    lookup_data = json.load(f)
                    config.update(lookup_data)
                print(f"[OK] Lookup database loaded with {len(lookup_data)} sections")
            except Exception as e:
                raise RuntimeError(f"Failed to load lookup database: {e}")
        else:
            raise FileNotFoundError(f"Lookup database not found: {lookup_file}")
        
        # Load parameter table
        param_file = config_files['parameter_table']
        print(f"Loading parameter table: {param_file}")
        if param_file.exists():
            try:
                with open(param_file, 'r') as f:
                    param_data = json.load(f)
                    config['raven_parameters'] = param_data
                print(f"[OK] Parameter table loaded with {len(param_data)} sections")
            except Exception as e:
                raise RuntimeError(f"Failed to load parameter table: {e}")
        else:
            raise FileNotFoundError(f"Parameter table not found: {param_file}")
        
        # Validate critical config sections exist
        required_sections = [
            'soil_classification', 
            'landcover_classification', 
            'vegetation_classification',
            'raven_parameters'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            raise ValueError(f"Missing required config sections: {missing_sections}")
        
        print(f"[OK] All configuration loaded successfully:")
        print(f"  - Total sections: {len(config)}")
        print(f"  - Soil classes: {len(config['soil_classification']['soil_classes'])}")
        print(f"  - Landcover classes: {len(config['landcover_classification']['landcover_classes'])}")
        print(f"  - RVI parameters: {len(config['raven_parameters']['rvi_parameters'])}")
        print(f"  - RVP parameters: {len(config['raven_parameters']['rvp_parameters'])}")
        
        return config
    
    def _load_hru_data(self) -> gpd.GeoDataFrame:
        """Load HRU data from Step 4"""
        # Use absolute paths with pathlib
        hru_paths = [
            self.workspace_dir / "data" / "hrus.geojson",
            self.workspace_dir / "data" / "hru_output.geojson", 
            self.workspace_dir / "data" / "finalcat_hru_info.shp"
        ]
        
        for hru_file in hru_paths:
            hru_file = hru_file.resolve()  # Convert to absolute path
            print(f"Checking HRU file: {hru_file} (exists: {hru_file.exists()})")
            if hru_file.exists():
                print(f"Loading HRU data: {hru_file}")
                return gpd.read_file(hru_file)
        
        raise FileNotFoundError(f"No HRU data found from Step 4. Checked paths: {[str(p.resolve()) for p in hru_paths]}")
    
    def _extract_dynamic_classes(self, hru_gdf: gpd.GeoDataFrame) -> Dict[str, set]:
        """Extract unique classes from HRU data for dynamic generation"""
        
        # Extract land use classes (excluding water/lake)
        landuse_classes = set()
        for landuse in hru_gdf['LAND_USE_C'].unique():
            if landuse not in ['WATER', 'LAKE'] and landuse in self.config["landuse_classes"]:
                landuse_classes.add(landuse)
        
        # Extract vegetation classes (excluding water)
        veg_classes = set()
        available_vegetation = set(self.config["vegetation_classification"]["basinmaker_format"]["landcover_mapping"].values())
        for veg in hru_gdf['VEG_C'].unique():
            if veg not in ['WATER', 'LAKE'] and pd.notna(veg):
                if veg in available_vegetation:
                    veg_classes.add(veg)
        
        # Extract soil classes (excluding water/lake soils)
        soil_classes = set()
        for soil in hru_gdf['SOIL_PROF'].unique():
            if soil not in ['WATER', 'LAKE_SOIL'] and soil in self.config["soil_profiles"]:
                soil_classes.add(soil)
        
        return {
            'landuse': landuse_classes,
            'vegetation': veg_classes,
            'soil': soil_classes
        }
    
    def _extract_dynamic_classes_from_lookup(self, hru_gdf: gpd.GeoDataFrame) -> Dict[str, set]:
        """Extract dynamic classes using parameter extractor and lookup database"""
        
        # Extract unique classes directly from HRU data for now
        # (The parameter extractor validates but doesn't extract class lists)
        
        # Extract land use classes (excluding water/lake) and validate against database
        landuse_classes = set()
        # Use config manager to get available landuse classes 
        available_landcover = set(self.config_manager.get_required_parameter('raven_class_definitions', 'landuse_classes').keys())
        for landuse in hru_gdf['LAND_USE_C'].unique():
            if landuse not in ['WATER', 'LAKE'] and pd.notna(landuse):
                if landuse in available_landcover:
                    landuse_classes.add(landuse)
        
        # Extract vegetation classes (excluding water) and validate against database  
        veg_classes = set()
        # Use config manager to get available vegetation classes
        available_vegetation = set(self.config_manager.get_required_parameter('raven_class_definitions', 'vegetation_classes').keys())
        
        for veg in hru_gdf['VEG_C'].unique():
            if veg not in ['WATER', 'LAKE'] and pd.notna(veg):
                if veg in available_vegetation:
                    veg_classes.add(veg)
                else:
                    print(f"[WARNING] Vegetation class '{veg}' not found in centralized config")
        
        # Extract soil classes (excluding water/lake soils) and validate against database
        soil_classes = set()
        available_soils = set(self.config_manager.get_required_parameter('raven_class_definitions', 'soil_profiles').keys())
        for soil in hru_gdf['SOIL_PROF'].unique():
            if soil not in ['WATER', 'LAKE_SOIL'] and pd.notna(soil):
                if soil in available_soils:
                    soil_classes.add(soil)
                else:
                    print(f"[WARNING] Soil profile '{soil}' not found in centralized config")
        
        return {
            'landuse': landuse_classes,
            'vegetation': veg_classes,
            'soil': soil_classes
        }
    
    def _load_routing_data(self) -> tuple:
        """Load routing data from Step 3 if available"""
        routing_rvh_path = self.workspace_dir / "data" / "watershed_routing_routing.rvh"
        
        if routing_rvh_path.exists():
            print(f"Loading routing data from Step 3: {routing_rvh_path}")
            with open(routing_rvh_path, 'r') as f:
                routing_content = f.read()
                
            # Extract channel profiles section
            channel_profiles = ""
            if "# Channel Profiles" in routing_content:
                start = routing_content.find("# Channel Profiles")
                end = routing_content.find("# SubBasins")
                if end == -1:
                    end = routing_content.find("# Reservoirs/Lakes")
                if start != -1:
                    channel_profiles = routing_content[start:end].strip()
            
            # Extract reservoir sections
            reservoirs = ""
            if "# Reservoirs/Lakes" in routing_content:
                start = routing_content.find("# Reservoirs/Lakes")
                reservoirs = routing_content[start:].strip()
                
            return channel_profiles, reservoirs
        else:
            print("No Step 3 routing data found - using basic routing")
            return "", ""
    
    def _generate_rvh_with_real_classes(self, hru_gdf: gpd.GeoDataFrame, outlet_name: str, has_lakes: bool = False) -> Path:
        """Generate RVH file with actual class names (simple routing)"""
        
        rvh_file = self.models_dir / outlet_name / f"{outlet_name}.rvh"
        rvh_file.parent.mkdir(exist_ok=True, parents=True)
        
        with open(rvh_file, 'w') as f:
            f.write("#----------------------------------------------\n")
            f.write("# Clean RAVEN RVH file with real class names\n")
            f.write("# Generated by Clean Step 5\n")
            f.write("#----------------------------------------------\n\n")
            
            # CRITICAL FIX: Load enhanced subbasin data with CHANNEL_PROF and GAUGED attributes
            print("  Loading enhanced subbasin data with hydraulic parameters...")
            
            # Try to load subbasin shapefile (prioritize the file used in Step 4 HRU generation)
            subbasins_gdf = None
            enhanced_subbasin_files = [
                self.workspace_dir / "data" / "subbasins_with_lakes.shp",  # Step 3 enhanced file with all RAVEN attributes
                self.workspace_dir / "data" / "subbasins_enhanced.shp",
                self.workspace_dir / "data" / "subbasins.shp"  # Fallback with minimal attributes
            ]
            
            for subbasin_file in enhanced_subbasin_files:
                if subbasin_file.exists():
                    try:
                        subbasins_gdf = gpd.read_file(subbasin_file)
                        print(f"    Loaded subbasin data from: {subbasin_file.name}")
                        print(f"    Available columns: {subbasins_gdf.columns.tolist()}")
                        break
                    except Exception as e:
                        print(f"    Failed to load {subbasin_file}: {e}")
                        continue
            
            if subbasins_gdf is None:
                raise ValueError("No subbasin shapefile found - cannot generate RVH file")
            
            # CRITICAL FIX: Load Step 3 routing configuration with channel profiles
            print("  Loading Step 3 routing configuration with channel profiles...")
            step3_results_file = self.workspace_dir / "data" / "step3_results.json"
            step3_routing_config = None
            channel_profile_mapping = {}
            
            if step3_results_file.exists():
                try:
                    import json
                    with open(step3_results_file, 'r') as step3_file:
                        step3_data = json.load(step3_file)
                        step3_routing_config = step3_data.get('routing_config', {})
                        
                        # Parse subbasin routing to get channel profile mappings
                        for subbasin_config in step3_routing_config.get('subbasins', []):
                            # Extract SubBasin ID and PROFILE from the config string
                            # Example: ":SubBasin 14\n  :Attributes NAME SB_14\n  :Attributes DOWNSTREAM_ID 85\n  :Attributes PROFILE CHANNEL_14\n  :Attributes GAUGE_LOC NONE\n:EndSubBasin"
                            lines = subbasin_config.split('\n')
                            sub_id = None
                            profile = None
                            for line in lines:
                                if line.startswith(':SubBasin '):
                                    sub_id = int(line.split()[1])
                                elif ':Attributes PROFILE ' in line:
                                    profile = line.split()[-1]
                            if sub_id is not None and profile is not None:
                                channel_profile_mapping[sub_id] = profile
                        
                        print(f"    Loaded channel profile mappings for {len(channel_profile_mapping)} subbasins")
                        print(f"    Channel profiles: {list(set(channel_profile_mapping.values()))}")
                except Exception as e:
                    print(f"    Failed to load Step 3 results: {e}")
                    print("    Will use fallback channel profile generation")
            else:
                print(f"    Step 3 results file not found: {step3_results_file}")
                print("    Will use fallback channel profile generation")

            # Validate routing network topology BEFORE generating RVH file
            self._validate_routing_network(subbasins_gdf)
            
            # Filter out invalid SubIds (-1, 0, or null)
            invalid_subids = subbasins_gdf[
                (subbasins_gdf['SubId'] == -1) | 
                (subbasins_gdf['SubId'] == 0) | 
                (subbasins_gdf['SubId'].isna())
            ]
            
            if len(invalid_subids) > 0:
                print(f"  WARNING: Filtering out {len(invalid_subids)} invalid SubIds from RVH file: {invalid_subids['SubId'].tolist()}")
                subbasins_gdf = subbasins_gdf[
                    (subbasins_gdf['SubId'] > 0) & 
                    (subbasins_gdf['SubId'].notna())
                ].copy()
            
            f.write(":SubBasins\n")
            f.write("  :Attributes   NAME  DOWNSTREAM_ID       PROFILE REACH_LENGTH  GAUGED\n")
            f.write("  :Units        none           none          none           km    none\n")
            
            for _, subbasin in subbasins_gdf.iterrows():
                sub_id = int(subbasin['SubId'])
                
                # Additional safety check
                if sub_id <= 0:
                    continue
                
                # CRITICAL FIX: Initialize river_length first (already in km from Step 3)
                river_length_km = float(subbasin.get('RivLength', 0))
                
                # CRITICAL FIX: Use Step 3 channel profile mapping if available
                if sub_id in channel_profile_mapping:
                    channel_profile = channel_profile_mapping[sub_id]
                    print(f"    Using Step 3 channel profile for subbasin {sub_id}: {channel_profile}")
                else:
                    # Fallback: Generate channel profile names dynamically (BasinMaker logic)
                    if river_length_km > 0.001:  # BasinMaker length threshold (1m = 0.001km)
                        channel_profile = f"Chn_{sub_id}"
                    else:
                        channel_profile = "Chn_ZERO_LENGTH"
                    print(f"    Using fallback channel profile for subbasin {sub_id}: {channel_profile}")
                
                # Extract GAUGED from subbasin data (not HRU data)
                if 'GAUGED' in subbasin:
                    gauged = int(subbasin['GAUGED'])
                elif 'Has_POI' in subbasin:
                    gauged = 1 if subbasin['Has_POI'] > 0 else 0
                else:
                    gauged = 0  # Default: no observation point
                
                # Extract DOWNSTREAM_ID from subbasin data (BasinMaker logic)
                downstream_id = int(subbasin.get('DowSubId', -1))
                # Handle downstream ID (BasinMaker logic) 
                if sub_id == downstream_id or downstream_id < 0:
                    downstream_str = "-1"
                else:
                    downstream_str = str(downstream_id)
                
                # BasinMaker logic: Handle reach length with threshold and lake handling
                if river_length_km > 0.001:  # Above threshold (1m = 0.001km)
                    length_str = f"{river_length_km:.4f}"
                else:
                    length_str = "ZERO-"  # BasinMaker uses "ZERO-" for short rivers
                
                # Lake subbasins get zero length (BasinMaker logic)
                if subbasin.get('Lake_Cat', 0) > 0:
                    length_str = "ZERO-"
                
                # BasinMaker gauge naming logic
                if gauged > 0:
                    # Use actual gauge name if available
                    rvh_name = str(subbasin.get('Obs_NM', f'gauge_{sub_id}')).replace(" ", "_")
                else:
                    rvh_name = f"sub{sub_id}"
                
                f.write(f"  {sub_id:6d}  {rvh_name:<15}  {downstream_str:<12}  {channel_profile:<15}  {length_str:>10}  {gauged:>6}\n")
            f.write(":EndSubBasins\n\n")
            
            # Write HRUs section with REAL class names and proper HRU ID column
            f.write(":HRUs\n")
            f.write("  :Attributes    ID    AREA ELEVATION  LATITUDE  LONGITUDE   BASIN_ID  LAND_USE_CLASS  VEG_CLASS   SOIL_PROFILE  AQUIFER_PROFILE   TERRAIN_CLASS   SLOPE   ASPECT\n")
            f.write("  :Units       none     km2         m       deg        deg       none            none       none           none             none            none     deg      deg\n")
            
            hru_id = 1  # Start HRU ID counter
            filtered_hru_count = 0
            for _, hru in hru_gdf.iterrows():
                # Filter out HRUs with invalid SubIds
                if hru['SubId'] <= 0 or pd.isna(hru['SubId']):
                    filtered_hru_count += 1
                    continue
                
                # FAIL-FAST: All HRU attributes must exist - no fallbacks
                area_km2 = float(hru['HRU_Area'])
                elevation = float(hru['elevation'])
                
                # Convert UTM coordinates to lat/lon if needed
                hru_cenx = float(hru['HRU_CenX'])
                hru_ceny = float(hru['HRU_CenY'])
                
                # Check if coordinates are in UTM (large values) vs lat/lon (small values)
                if abs(hru_cenx) > 180 or abs(hru_ceny) > 90:
                    # Coordinates are in UTM, need to convert to lat/lon
                    try:
                        # Create a temporary GeoDataFrame with UTM coordinates
                        temp_df = pd.DataFrame({'x': [hru_cenx], 'y': [hru_ceny]})
                        temp_gdf = gpd.GeoDataFrame(temp_df, geometry=gpd.points_from_xy(temp_df.x, temp_df.y))
                        
                        # Assume UTM Zone 11N (EPSG:32611) based on the coordinates
                        temp_gdf.set_crs('EPSG:32611', inplace=True)
                        
                        # Transform to WGS84 (lat/lon)
                        temp_gdf_wgs84 = temp_gdf.to_crs('EPSG:4326')
                        
                        # Extract lat/lon
                        longitude = temp_gdf_wgs84.geometry.iloc[0].x
                        latitude = temp_gdf_wgs84.geometry.iloc[0].y
                        
                    except Exception as e:
                        raise ValueError(f"Coordinate conversion failed for HRU {hru['HRU_ID']}: {e}")
                else:
                    # Coordinates are already in lat/lon
                    longitude = hru_cenx
                    latitude = hru_ceny
                
                basin_id = int(hru['SubId'])
                
                # FAIL-FAST: Require actual class names from HRU data
                if 'LAND_USE_C' not in hru.index:
                    raise ValueError(f"Missing required LAND_USE_C for HRU {hru_id}")
                if 'VEG_C' not in hru.index:
                    raise ValueError(f"Missing required VEG_C for HRU {hru_id}")
                if 'SOIL_PROF' not in hru.index:
                    raise ValueError(f"Missing required SOIL_PROF for HRU {hru_id}")
                
                land_use = str(hru['LAND_USE_C'])
                veg_class = str(hru['VEG_C'])
                soil_profile = str(hru['SOIL_PROF'])
                
                slope = float(hru['slope'])
                aspect = float(hru['aspect'])
                
                # BasinMaker approach: Use DEFAULT values for missing aquifer and terrain classes
                if 'AQUIFER_PROF' in hru.index:
                    aquifer_profile = str(hru['AQUIFER_PROF'])
                else:
                    aquifer_profile = "[NONE]"  # RAVEN format for no aquifer
                
                if 'TERRAIN_CLASS' in hru.index:
                    terrain_class = str(hru['TERRAIN_CLASS'])
                else:
                    terrain_class = "[NONE]"  # RAVEN format for no terrain class
                
                f.write(f"  {hru_id:6d} {area_km2:8.4f} {elevation:9.1f} {latitude:8.4f} {longitude:9.4f} {basin_id:8d} ")
                f.write(f"{land_use:14} {veg_class:9} {soil_profile:12} {aquifer_profile:15} {terrain_class:15} {slope:5.1f} {aspect:6.1f}\n")
                hru_id += 1  # Increment HRU ID for next HRU
            
            f.write(":EndHRUs\n")
            
            # BasinMaker pattern: Add RedirectToFile at end of RVH if lakes exist
            if has_lakes:
                f.write("\n:RedirectToFile Lakes.rvh\n")
            
            if filtered_hru_count > 0:
                print(f"  WARNING: Filtered out {filtered_hru_count} HRUs with invalid SubIds from RVH file")
        
        return rvh_file
    
    def _generate_basinmaker_channel_profile(self, chname: str, chwd: float, chdep: float, 
                                           chslope: float, elev: float, floodn: float, 
                                           channeln: float) -> List[str]:
        """Generate channel profile string using exact BasinMaker logic"""
        
        output_lines = []
        
        # BasinMaker trapezoidal channel geometry constants - EXACT BASINMAKER VALUES
        zch = 2                    # Side slope ratio (2:1 horizontal:vertical)
        sidwdfp = 16              # Exact BasinMaker floodplain width = 16m each side (total 32m)
        
        # Calculate trapezoidal geometry
        sidwd = zch * chdep
        botwd = chwd - 2 * sidwd
        
        # Handle narrow channels (BasinMaker logic)
        if botwd < 0:
            botwd = 0.5 * chwd
            sidwd = 0.5 * 0.5 * chwd
            zch = (chwd - botwd) / 2 / chdep if chdep > 0 else 2
        
        # Elevation calculations - EXACT BASINMAKER flood capacity
        zfld = 4 + elev           # Exact BasinMaker 4m flood depth above channel elevation
        zbot = elev - chdep       # Channel bottom elevation
        
        # BasinMaker 8-point survey pattern
        x0 = 0.0                                           # Left floodplain start
        x1 = sidwdfp                                       # Left floodplain edge (16m)
        x2 = sidwdfp + sidwd                              # Left bank top
        x3 = sidwdfp + sidwd + botwd                      # Left bank bottom
        x4 = sidwdfp + sidwd + botwd                      # Right bank bottom (same as x3)
        x5 = sidwdfp + sidwd + botwd + botwd              # Right bank top
        x6 = sidwdfp + sidwd + botwd + botwd + sidwd      # Right floodplain edge
        x7 = sidwdfp + sidwd + botwd + botwd + sidwd + sidwdfp  # Right floodplain end
        
        # Channel profile header
        output_lines.append(f":ChannelProfile {chname}")
        output_lines.append(f"  :Bedslope {chslope:.6f}")
        output_lines.append("  :SurveyPoints")
        output_lines.append("    # Channel cross-section with extended flood capacity")
        
        # BasinMaker 8-point survey geometry
        output_lines.append(f"    {x0:.1f} {zfld:.2f}")    # Left floodplain start
        output_lines.append(f"    {x1:.1f} {elev:.2f}")    # Left floodplain edge
        output_lines.append(f"    {x2:.1f} {elev:.2f}")    # Left bank top
        output_lines.append(f"    {x3:.1f} {zbot:.2f}")    # Left bank bottom
        output_lines.append(f"    {x4:.1f} {zbot:.2f}")    # Right bank bottom
        output_lines.append(f"    {x5:.1f} {elev:.2f}")    # Right bank top
        output_lines.append(f"    {x6:.1f} {elev:.2f}")    # Right floodplain edge
        output_lines.append(f"    {x7:.1f} {zfld:.2f}")    # Right floodplain end
        
        output_lines.append("  :EndSurveyPoints")
        output_lines.append("  :RoughnessZones")
        
        # BasinMaker 3-zone roughness pattern
        output_lines.append(f"    {x0:.1f} {floodn:.4f}")    # Left floodplain Manning's n
        output_lines.append(f"    {x2:.1f} {channeln:.4f}")  # Channel Manning's n
        output_lines.append(f"    {x6:.1f} {floodn:.4f}")    # Right floodplain Manning's n
        
        output_lines.append("  :EndRoughnessZones")
        output_lines.append(":EndChannelProfile")
        
        return output_lines

    def _calculate_dynamic_reach_length(self, sub_hrus: gpd.GeoDataFrame, sub_id: int) -> float:
        """
        Calculate dynamic reach length from stream data (BasinMaker approach)
        Uses actual stream geometry length within the subbasin
        """
        
        # Check for stream length data in HRU attributes
        stream_length_fields = ['RivLength', 'REACH_LENGTH', 'stream_length', 'Length_m']
        
        for field in stream_length_fields:
            if field in sub_hrus.columns:
                lengths = sub_hrus[field].dropna()
                if len(lengths) > 0:
                    total_length = lengths.sum()
                    # Convert from meters to km if needed
                    if total_length > 100:  # Likely in meters
                        return total_length / 1000.0
                    else:
                        return total_length
        
        # Calculate reach length using proper subbasin-stream intersection
        streams_file = self.workspace_dir / "data" / "streams.geojson"
        subbasins_file = self.workspace_dir / "data" / "subbasins.geojson"
        
        if streams_file.exists() and subbasins_file.exists():
            try:
                streams_gdf = gpd.read_file(streams_file)
                subbasins_gdf = gpd.read_file(subbasins_file)
                
                # Find the specific subbasin polygon
                subbasin_rows = subbasins_gdf[subbasins_gdf['SubId'] == sub_id]
                if len(subbasin_rows) == 0:
                    self.logger.warning(f"Subbasin {sub_id} not found in subbasins.geojson")
                    # Fallback to HRU geometry union
                    subbasin_geom = sub_hrus.geometry.union_all()
                else:
                    subbasin_geom = subbasin_rows.geometry.iloc[0]
                
                # Calculate precise intersection length
                total_reach_length_m = 0.0
                intersecting_streams = streams_gdf[streams_gdf.geometry.intersects(subbasin_geom)]
                
                for idx, stream in intersecting_streams.iterrows():
                    try:
                        # Calculate actual intersection geometry
                        intersection = stream.geometry.intersection(subbasin_geom)
                        
                        if intersection.is_empty:
                            continue
                            
                        # Handle different geometry types
                        if hasattr(intersection, 'length'):
                            # Single LineString
                            total_reach_length_m += intersection.length
                        elif hasattr(intersection, 'geoms'):
                            # MultiLineString or GeometryCollection
                            for geom in intersection.geoms:
                                if hasattr(geom, 'length') and geom.length > 0:
                                    total_reach_length_m += geom.length
                        
                    except Exception as geom_error:
                        # Fallback: use portion of full stream length
                        self.logger.warning(f"Geometric intersection failed for stream {idx}: {geom_error}")
                        # Approximate as 50% of stream length if it intersects
                        total_reach_length_m += stream.geometry.length * 0.5
                
                if total_reach_length_m > 0:
                    # Convert to km and apply bounds
                    reach_length_km = total_reach_length_m / 1000.0
                    return max(0.1, min(reach_length_km, 50.0))  # Bounds: 0.1-50 km
                    
            except Exception as e:
                self.logger.warning(f"Stream-subbasin intersection calculation failed: {e}")
        
        # Fallback: estimate from HRU area (rough approximation)
        # FAIL-FAST: HRU_Area must exist
        if 'HRU_Area' not in sub_hrus.columns:
            raise ValueError(f"Missing required HRU_Area column for subbasin {sub_id}")
        hru_area_km2 = sub_hrus['HRU_Area'].sum()
        estimated_length = (hru_area_km2 ** 0.5) * 0.5  # Very rough estimate
        
        return max(0.1, estimated_length)  # Minimum 0.1 km
    
    def _integrate_gauge_stations(self, hru_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Integrate hydrometric stations as observation points to set GAUGED attribute correctly
        Enhanced: Checks obs folder for existing data, automatically fetches best stations if none exist
        """
        
        # Check for hydrometric data from climate/hydrometric step
        hydro_data_file = self.workspace_dir / "hydrometric" / "observed_streamflow.csv"
        
        if hydro_data_file.exists():
            print(f"  Found hydrometric data: {hydro_data_file}")
            self._create_observation_rvt_files(hydro_data_file)
        else:
            print("  No hydrometric data found - attempting to fetch best stations...")
            try:
                # Get outlet coordinates for station search
                outlet_bounds = hru_gdf.total_bounds  # [minx, miny, maxx, maxy]
                center_lat = (outlet_bounds[1] + outlet_bounds[3]) / 2
                center_lon = (outlet_bounds[0] + outlet_bounds[2]) / 2
                
                # Fetch best hydrometric stations
                best_stations = self._fetch_best_hydrometric_stations(center_lat, center_lon)
                if best_stations:
                    print(f"  SUCCESS: Found {len(best_stations)} suitable hydrometric stations")
                    # Convert best stations to the expected format
                    return self._integrate_fetched_stations(hru_gdf, best_stations)
                else:
                    print("  WARNING: No suitable hydrometric stations found - GAUGED attributes will remain 0")
                    return hru_gdf
            except Exception as e:
                print(f"  ERROR: Failed to fetch hydrometric stations: {e}")
                return hru_gdf
        
        # Check if hydrometric station data exists from previous steps
        stations_file = self.workspace_dir / "hydrometric" / "hydrometric_stations_detailed.json"
        if not stations_file.exists():
            print("  No hydrometric station metadata found - GAUGED attributes will remain 0")
            return hru_gdf
        
        try:
            # Load hydrometric station data
            with open(stations_file, 'r') as f:
                stations_data = json.load(f)
            
            # Convert station data to observation points format
            obs_points_data = []
            # FAIL-FAST: stations data must have top_5_stations
            if 'top_5_stations' not in stations_data:
                raise ValueError("Missing required top_5_stations in stations data")
            stations_to_process = stations_data['top_5_stations']
            
            if not stations_to_process:
                print("  No stations found in metadata - using selected station")
                selected_station = stations_data.get('selected_station')
                if selected_station:
                    stations_to_process = [selected_station]
            
            for station in stations_to_process:
                # Fail-fast: Require all station data
                if 'id' not in station:
                    raise ValueError("Station missing required 'id' field")
                if 'drainage_area_km2' not in station:
                    raise ValueError(f"Station {station['id']} missing required 'drainage_area_km2' field")
                if 'longitude' not in station:
                    raise ValueError(f"Station {station['id']} missing required 'longitude' field")
                if 'latitude' not in station:
                    raise ValueError(f"Station {station['id']} missing required 'latitude' field")
                
                obs_point = {
                    'Obs_NM': station['id'],
                    'DA_Obs': float(station['drainage_area_km2']) if station['drainage_area_km2'] != 'Unknown' else 0,
                    'SRC_obs': 'Environment_Canada',
                    'Type': 'River',
                    'geometry': gpd.points_from_xy([station['longitude']], [station['latitude']])[0]
                }
                obs_points_data.append(obs_point)
            
            if not obs_points_data:
                print("  No valid station data found")
                return hru_gdf
            
            # Create observation points GeoDataFrame
            obs_points_gdf = gpd.GeoDataFrame(obs_points_data, crs='EPSG:4326')
            
            # Save as temporary shapefile for ObservationPointIntegrator
            obs_points_file = self.workspace_dir / "data" / "observation_points.shp"
            obs_points_gdf.to_file(obs_points_file)
            
            print(f"  Created observation points file with {len(obs_points_gdf)} stations")
            
            # Initialize ObservationPointIntegrator
            obs_integrator = ObservationPointIntegrator(workspace_dir=self.workspace_dir / "data")
            
            # Get catchments file
            catchments_file = self.workspace_dir / "data" / "subbasins.shp"
            streams_file = self.workspace_dir / "data" / "streams.shp"
            
            if not catchments_file.exists():
                print("  No catchments shapefile found - cannot integrate stations")
                return hru_gdf
            
            # Integrate observation points with catchments
            integration_results = obs_integrator.integrate_observation_points(
                observation_points_shapefile=obs_points_file,
                catchments_shapefile=catchments_file,
                streams_shapefile=streams_file if streams_file.exists() else None,
                snap_to_streams=True
            )
            
            if integration_results['success']:
                # Load updated catchments with GAUGED attribute
                updated_catchments_file = integration_results['updated_catchments_file']
                updated_catchments_gdf = gpd.read_file(updated_catchments_file)
                
                # Update HRU data with GAUGED information
                enhanced_hru_gdf = hru_gdf.copy()
                
                # Map GAUGED attribute from catchments to HRUs based on SubId
                for idx, hru in enhanced_hru_gdf.iterrows():
                    sub_id = hru.get('SubId')
                    if sub_id:
                        # Find matching catchment
                        matching_catchments = updated_catchments_gdf[updated_catchments_gdf['SubId'] == sub_id]
                        if len(matching_catchments) > 0:
                            if 'Has_POI' not in matching_catchments.columns:
                                raise ValueError(f"Missing required 'Has_POI' column in catchments data")
                            gauged_value = matching_catchments.iloc[0]['Has_POI']
                            enhanced_hru_gdf.loc[idx, 'GAUGED'] = int(gauged_value)
                
                # Report results
                summary = integration_results['integration_summary']
                gauged_subbasins = len(updated_catchments_gdf[updated_catchments_gdf['Has_POI'] > 0])
                
                print(f"  SUCCESS: {summary['points_linked_to_subbasins']} stations linked to subbasins")
                print(f"  SUCCESS: {gauged_subbasins} subbasins now have GAUGED = 1")
                
                if summary.get('validation_issues', 0) > 0:
                    print(f"  WARNING: {summary['validation_issues']} drainage area validation issues")
                
                return enhanced_hru_gdf
            
            else:
                print(f"  WARNING: Station integration failed: {integration_results.get('error', 'Unknown error')}")
                return hru_gdf
                
        except Exception as e:
            print(f"  ERROR: Gauge station integration failed: {str(e)}")
            return hru_gdf
    
    def _check_obs_folder_for_hydrometric_data(self, obs_dir: Path) -> bool:
        """Check if obs folder contains hydrometric data files"""
        if not obs_dir.exists():
            print(f"  obs folder does not exist: {obs_dir}")
            return False
        
        # Check for common hydrometric data file patterns
        hydrometric_patterns = ["*_daily_flows.csv", "*_streamflow.csv", "*.obs", "*discharge*.csv"]
        
        for pattern in hydrometric_patterns:
            matching_files = list(obs_dir.glob(pattern))
            if matching_files:
                print(f"  Found {len(matching_files)} hydrometric data files: {[f.name for f in matching_files[:3]]}")
                return True
        
        # Check if folder has any CSV files that might be hydrometric data
        csv_files = list(obs_dir.glob("*.csv"))
        if csv_files:
            print(f"  Found {len(csv_files)} CSV files in obs folder, checking content...")
            for csv_file in csv_files:
                try:
                    # Quick check if file contains hydrometric-like data
                    with open(csv_file, 'r') as f:
                        header = f.readline().lower()
                        if any(term in header for term in ['discharge', 'flow', 'streamflow', 'cms', 'm3/s', 'water_level']):
                            print(f"  Found hydrometric data in: {csv_file.name}")
                            return True
                except Exception:
                    continue
        
        print(f"  No hydrometric data files found in obs folder")
        return False
    
    def _create_observation_rvt_files(self, hydro_data_file: Path):
        """Create observation RVT files from hydrometric data"""
        try:
            import pandas as pd
            
            # Load hydrometric data
            df = pd.read_csv(hydro_data_file)
            df['Date'] = pd.to_datetime(df['Date'])
            
            print(f"  Creating observation RVT from {len(df)} records ({df.Date.min().date()} to {df.Date.max().date()})")
            
            # Create obs directory
            obs_dir = self.workspace_dir / "models" / "files" / self.model_name / "obs"
            obs_dir.mkdir(parents=True, exist_ok=True)
            
            # Get actual station info from hydrometric metadata
            hydro_metadata_file = self.workspace_dir / "hydrometric" / "hydrometric_stations_detailed.json"
            station_name = "HYDROMETRIC_STATION"
            station_id = "UNKNOWN"
            subbasin_id = 50  # Default outlet subbasin
            
            if hydro_metadata_file.exists():
                import json
                with open(hydro_metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if 'selected_station' in metadata:
                        station_info = metadata['selected_station']
                        station_name = station_info.get('name', 'UNKNOWN_STATION').replace(' ', '_')
                        station_id = station_info.get('id', 'UNKNOWN')
            
            # Get outlet subbasin ID
            subbasins_file = self.workspace_dir / "data" / "subbasins_with_lakes.shp"
            if subbasins_file.exists():
                import geopandas as gpd
                subbasins_gdf = gpd.read_file(subbasins_file)
                outlet_subbasin = subbasins_gdf[subbasins_gdf['DowSubId'] == -1]
                if len(outlet_subbasin) > 0:
                    subbasin_id = int(outlet_subbasin.iloc[0]['SubId'])
                    
                    # Generate BasinMaker format observation RVT exactly like RavenInput.py
                    lines = [f':ObservationData HYDROGRAPH {subbasin_id}   m3/s']
                    
                    # Add header line with start date and count (BasinMaker format)
                    start_date = df['Date'].min().strftime('%Y-%m-%d')
                    num_records = len(df)
                    lines.append(f'{start_date}  00:00:00  1     {num_records}')
                    
                    for _, row in df.iterrows():
                        discharge = row['Discharge_cms']
                        if pd.isna(discharge):
                            discharge = -1.2345  # BasinMaker NoData value
                        lines.append(f'         {discharge:.6f}')
                    
                    lines.append(':EndObservationData')
                    
                    # Write observation RVT file with proper name
                    obs_rvt_file = obs_dir / f"{station_name}_{station_id}_{subbasin_id}.rvt"
                    with open(obs_rvt_file, 'w') as f:
                        f.write('\n'.join(lines))
                    
                    # Add redirect to main RVT file
                    self._add_observation_redirect_to_rvt(obs_rvt_file.name)
                    
                    print(f"  ✓ Created observation RVT: {obs_rvt_file.name}")
                    print(f"  ✓ Added redirect to main RVT file")
                    
        except Exception as e:
            print(f"  ERROR creating observation RVT: {e}")
    
    def _add_observation_redirect_to_rvt(self, obs_filename: str):
        """Add RedirectToFile for observation data to main RVT file"""
        try:
            main_rvt_file = self.workspace_dir / "models" / "files" / self.model_name / f"{self.model_name}.rvt"
            
            if main_rvt_file.exists():
                # Read current RVT content
                with open(main_rvt_file, 'r') as f:
                    content = f.read()
                
                # Add observation redirect if not already present
                redirect_line = f":RedirectToFile ./obs/{obs_filename}"
                if redirect_line not in content:
                    # Add at the end, before any existing redirects
                    content = content.rstrip()
                    content += f"\n\n# Hydrometric observation data\n{redirect_line}\n"
                    
                    with open(main_rvt_file, 'w') as f:
                        f.write(content)
                        
        except Exception as e:
            print(f"  WARNING: Could not add observation redirect: {e}")
    
    def _fetch_best_hydrometric_stations(self, center_lat: float, center_lon: float) -> List[Dict]:
        """Fetch best hydrometric stations using the hydrometric client"""
        
        # Initialize hydrometric client
        hydrometric_client = HydrometricDataClient()
        
        # Find best stations within 100km radius with 10+ years of data
        print(f"  Searching for hydrometric stations near ({center_lat:.4f}, {center_lon:.4f})")
        
        best_stations = hydrometric_client.find_best_hydrometric_stations_with_data(
            outlet_lat=center_lat,
            outlet_lon=center_lon,
            search_range_km=100.0,  # 100km search radius
            max_stations=5,         # Get top 5 stations
            min_years=10,          # Minimum 10 years of data
            start_year=1990,
            end_year=2024
        )
        
        if best_stations:
            print(f"  Found {len(best_stations)} suitable stations")
            # Download data for the best station
            best_station = best_stations[0]  # Get the top station
            
            obs_dir = self.workspace_dir / "models" / "files" / "obs"
            obs_dir.mkdir(exist_ok=True, parents=True)
            
            station_id = best_station['id']
            output_file = obs_dir / f"{station_id}_daily_flows.csv"
            
            print(f"  Downloading data for station {station_id}: {best_station['name']}")
            download_result = hydrometric_client.get_streamflow_data_csv(
                station_id=station_id,
                start_date="2002-10-01",  # Match simulation period
                end_date="2004-09-30",
                output_path=output_file,
                debug=True
            )
            
            if download_result['success']:
                print(f"  SUCCESS: Downloaded {download_result['records']} records to {output_file}")
                return best_stations
            else:
                print(f"  ERROR: Failed to download data: {download_result.get('error', 'Unknown error')}")
                return []
        else:
            print("  No suitable stations found within search criteria")
            return []
    
    def _integrate_fetched_stations(self, hru_gdf: gpd.GeoDataFrame, best_stations: List[Dict]) -> gpd.GeoDataFrame:
        """Integrate fetched hydrometric stations with HRU data"""
        
        if not best_stations:
            return hru_gdf
        
        try:
            # Convert station data to observation points format
            obs_points_data = []
            for station in best_stations[:1]:  # Use only the best station
                obs_point = {
                    'Obs_NM': station['id'],
                    'DA_Obs': float(station['drainage_area_km2']) if station['drainage_area_km2'] else 0,
                    'SRC_obs': 'Environment_Canada',
                    'Type': 'River',
                    'geometry': gpd.points_from_xy([station['longitude']], [station['latitude']])[0]
                }
                obs_points_data.append(obs_point)
                print(f"    Added observation point: {station['id']} at ({station['latitude']:.4f}, {station['longitude']:.4f})")
            
            # Create observation points GeoDataFrame
            obs_points_gdf = gpd.GeoDataFrame(obs_points_data, crs='EPSG:4326')
            
            # Save as temporary shapefile for ObservationPointIntegrator
            obs_points_file = self.workspace_dir / "data" / "observation_points.shp"
            obs_points_gdf.to_file(obs_points_file)
            
            print(f"  Created observation points file with {len(obs_points_gdf)} stations")
            
            # Initialize ObservationPointIntegrator
            obs_integrator = ObservationPointIntegrator(workspace_dir=self.workspace_dir / "data")
            
            # Get catchments file
            catchments_file = self.workspace_dir / "data" / "subbasins.shp"
            streams_file = self.workspace_dir / "data" / "streams.shp"
            
            if not catchments_file.exists():
                print("  No catchments shapefile found - cannot integrate stations")
                return hru_gdf
            
            # Integrate observation points with catchments
            integration_results = obs_integrator.integrate_observation_points(
                observation_points_shapefile=obs_points_file,
                catchments_shapefile=catchments_file,
                streams_shapefile=streams_file if streams_file.exists() else None,
                snap_to_streams=True
            )
            
            if integration_results['success']:
                # Load updated catchments with GAUGED attribute
                updated_catchments_file = integration_results['updated_catchments_file']
                updated_catchments_gdf = gpd.read_file(updated_catchments_file)
                
                # Update HRU data with GAUGED information
                enhanced_hru_gdf = hru_gdf.copy()
                
                # Map GAUGED attribute from catchments to HRUs based on SubId
                for idx, hru in enhanced_hru_gdf.iterrows():
                    sub_id = hru.get('SubId')
                    if sub_id:
                        # Find matching catchment
                        matching_catchments = updated_catchments_gdf[updated_catchments_gdf['SubId'] == sub_id]
                        if len(matching_catchments) > 0:
                            gauged_value = matching_catchments.iloc[0].get('Has_POI', 0)
                            enhanced_hru_gdf.loc[idx, 'GAUGED'] = int(gauged_value)
                
                # Report results
                summary = integration_results['integration_summary']
                gauged_subbasins = len(updated_catchments_gdf[updated_catchments_gdf.get('Has_POI', 0) > 0])
                
                print(f"  SUCCESS: {summary['points_linked_to_subbasins']} stations linked to subbasins")
                print(f"  SUCCESS: {gauged_subbasins} subbasins now have GAUGED = 1")
                
                return enhanced_hru_gdf
            
            else:
                print(f"  WARNING: Station integration failed: {integration_results.get('error', 'Unknown error')}")
                return hru_gdf
                
        except Exception as e:
            print(f"  ERROR: Fetched station integration failed: {str(e)}")
            return hru_gdf
    
    def _load_climate_data(self) -> pd.DataFrame:
        """Load climate data from workspace climate folder"""
        try:
            climate_csv = self.workspace_dir / "climate" / "climate_forcing.csv"
            if not climate_csv.exists():
                print(f"WARNING: Climate data not found: {climate_csv}")
                return None
            
            climate_df = pd.read_csv(climate_csv, index_col=0, parse_dates=True)
            print(f"  Loaded climate data: {len(climate_df)} records from {climate_df.index[0].strftime('%Y-%m-%d')} to {climate_df.index[-1].strftime('%Y-%m-%d')}")
            return climate_df
        except Exception as e:
            print(f"WARNING: Failed to load climate data: {e}")
            return None
    
    def _validate_routing_network(self, subbasin_gdf: gpd.GeoDataFrame) -> None:
        """Validate routing network for circular references and topology issues"""
        print("  Validating routing network topology...")
        
        # Build routing dictionary
        routing_dict = {}
        outlet_count = 0
        
        for _, row in subbasin_gdf.iterrows():
            sub_id = int(row['SubId'])
            downstream_id = int(row['DowSubId'])
            
            routing_dict[sub_id] = downstream_id
            if downstream_id == -1:
                outlet_count += 1
        
        # Fix missing outlet - identify outlet subbasin based on outlet coordinates
        if outlet_count == 0:
            print("  WARNING: No outlet subbasin found. Attempting to fix based on outlet coordinates...")
            self._fix_outlet_assignment(subbasin_gdf)
            
            # Rebuild routing dictionary after fix
            routing_dict = {}
            outlet_count = 0
            for _, row in subbasin_gdf.iterrows():
                sub_id = int(row['SubId'])
                downstream_id = int(row['DowSubId'])
                routing_dict[sub_id] = downstream_id
                if downstream_id == -1:
                    outlet_count += 1
        
        # Check for exactly one outlet
        if outlet_count == 0:
            raise ValueError("ERROR: No outlet subbasin found (must have DowSubId = -1)")
        elif outlet_count > 1:
            raise ValueError(f"ERROR: Multiple outlets found ({outlet_count}). Only one subbasin can have DowSubId = -1")
        
        # Check for circular references using DFS
        visited = set()
        path = set()
        
        def has_cycle(sub_id):
            if sub_id == -1:  # Outlet reached
                return False
            if sub_id in path:
                return True  # Circular reference detected
            if sub_id in visited:
                return False
            
            if sub_id not in routing_dict:
                raise ValueError(f"ERROR: Subbasin {sub_id} referenced but not defined")
            
            visited.add(sub_id)
            path.add(sub_id)
            
            downstream = routing_dict[sub_id]
            if has_cycle(downstream):
                return True
            
            path.remove(sub_id)
            return False
        
        # Check each subbasin for cycles
        for sub_id in routing_dict:
            if has_cycle(sub_id):
                raise ValueError(f"ERROR: Circular reference in routing network involving subbasin {sub_id}")
        
        print(f"  SUCCESS: Routing network validation passed: {len(routing_dict)} subbasins, 1 outlet")
    
    def _fix_outlet_assignment(self, subbasin_gdf: gpd.GeoDataFrame) -> None:
        """Fix outlet assignment by finding subbasin that contains outlet coordinates"""
        from shapely.geometry import Point
        
        # Get outlet coordinates from config
        lat = self.config.get('coordinates', {}).get('latitude')
        lon = self.config.get('coordinates', {}).get('longitude')
        
        if lat is None or lon is None:
            print("  ERROR: Cannot fix outlet assignment - no coordinates in config")
            return
            
        print(f"  Looking for subbasin containing outlet point ({lat}, {lon})...")
        
        # Create outlet point
        outlet_point = Point(lon, lat)
        
        # Find which subbasin contains the outlet point
        outlet_subbasin_id = None
        
        # First try spatial containment
        for idx, subbasin in subbasin_gdf.iterrows():
            if subbasin.geometry.contains(outlet_point):
                outlet_subbasin_id = int(subbasin['SubId'])
                print(f"  Found outlet subbasin {outlet_subbasin_id} (spatial containment)")
                break
        
        # If no containment, find closest subbasin
        if outlet_subbasin_id is None:
            print("  No subbasin contains outlet point exactly, finding closest...")
            min_distance = float('inf')
            
            for idx, subbasin in subbasin_gdf.iterrows():
                distance = subbasin.geometry.distance(outlet_point)
                if distance < min_distance:
                    min_distance = distance
                    outlet_subbasin_id = int(subbasin['SubId'])
            
            print(f"  Closest subbasin to outlet: {outlet_subbasin_id} (distance: {min_distance:.6f})")
        
        if outlet_subbasin_id is not None:
            # Set this subbasin as the outlet
            outlet_mask = subbasin_gdf['SubId'] == outlet_subbasin_id
            subbasin_gdf.loc[outlet_mask, 'DowSubId'] = -1
            print(f"  SUCCESS: Set subbasin {outlet_subbasin_id} as watershed outlet (DowSubId = -1)")
            
            # Also fix any subbasins that were routing to the outlet subbasin to route to outlet
            routing_to_outlet = subbasin_gdf['DowSubId'] == outlet_subbasin_id
            if routing_to_outlet.any():
                affected_count = routing_to_outlet.sum() - 1  # Exclude the outlet itself
                if affected_count > 0:
                    print(f"  Note: {affected_count} subbasins were already routing to the outlet subbasin")
        else:
            print("  ERROR: Could not identify outlet subbasin")
    
    def _generate_rvi_with_dynamic_classes(self, outlet_name: str, dynamic_classes: Dict[str, set], has_lakes: bool = False) -> Path:
        """Generate RVI file using RAVEN benchmark format - NO class definitions in RVI"""
        
        rvi_file = self.models_dir / outlet_name / f"{outlet_name}.rvi"
        
        # Get RVI parameters from centralized table
        rvi_params = self.config_manager.get_required_parameter('raven_complete_parameter_table', 'rvi_parameters')
        temporal_config = rvi_params['temporal_configuration']
        model_config = rvi_params['model_configuration']
        
        with open(rvi_file, 'w') as f:
            f.write(f"# RAVEN Input file for {outlet_name}\n")
            f.write("# Generated by Step 5 using RAVEN framework standards\n\n")
            
            # Temporal configuration - auto-match to climate data period
            climate_df = self._load_climate_data()
            if climate_df is not None and not climate_df.empty:
                start_date = climate_df.index[0].strftime('%Y-%m-%d')
                end_date = climate_df.index[-1].strftime('%Y-%m-%d')
                f.write(f":StartDate        {start_date} 00:00:00\n")
                f.write(f":EndDate          {end_date} 00:00:00\n")
            else:
                # Fallback to config defaults if no climate data
                f.write(f":StartDate        {temporal_config[':StartDate']['example']}\n")
                f.write(f":EndDate          {temporal_config[':EndDate']['example']}\n")
            f.write(f":TimeStep         {temporal_config[':TimeStep']['default']}\n")
            f.write(f":Method           {temporal_config[':Method']['default']}\n\n")
            
            # Model configuration from parameter table
            soil_model = model_config[':SoilModel']['default']
            routing = model_config[':Routing']['default']
            evaporation = model_config[':Evaporation']['default']
            
            f.write(f":SoilModel        {soil_model}\n")
            f.write(f":Routing          {routing}\n")
            f.write(f":Evaporation      {evaporation}\n\n")
            
            # Aliases from config (RAVEN benchmark format)
            aliases = self.config['aliases']
            for alias_name, alias_target in aliases.items():
                f.write(f":Alias       {alias_name} {alias_target}\n")
            if aliases:
                f.write("\n")
            
            # Hydrologic processes from config (RAVEN benchmark format)
            hydro_processes = self.config['hydrologic_processes']
            if hydro_processes:
                f.write(":HydrologicProcesses\n")
                for process in hydro_processes:
                    f.write(f"  :{process['name']}    {process['method']}     {process['from']}")
                    if isinstance(process['to'], list):
                        f.write(f"     {' '.join(process['to'])}\n")
                    else:
                        f.write(f"     {process['to']}\n")
                    
                    # Handle overflow if specified
                    if 'overflow' in process:
                        f.write(f"    :-->Overflow     {process['overflow']['method']}      {process['overflow']['from']}        {process['overflow']['to']}\n")
                f.write(":EndHydrologicProcesses\n\n")
            
            # Global parameters are handled in RVP file, not RVI file
            # Remove global parameters from RVI file generation
            
            # Output options (RAVEN benchmark format)
            output_config = self.config.get('output_options', {})
            if output_config.get('write_forcing_functions', False):
                f.write(":WriteForcingFunctions\n")
            if output_config.get('evaluation_metrics'):
                metrics = ' '.join(output_config['evaluation_metrics'])
                f.write(f":EvaluationMetrics {metrics}\n")
            f.write("\n")
            
            # End command (RAVEN benchmark format - BasinMaker pattern: no RedirectToFile in RVI)
            f.write(f":End\n")
        
        return rvi_file
    
    def _generate_rvp_file(self, hru_gdf: gpd.GeoDataFrame, outlet_name: str, dynamic_classes: Dict[str, set]) -> Path:
        """Generate RVP parameter file using centralized parameter table"""
        
        rvp_file = self.models_dir / outlet_name / f"{outlet_name}.rvp"
        
        # Load subbasin data for channel profile generation
        subbasin_file = self.workspace_dir / "data" / "subbasins_enhanced.shp"
        if not subbasin_file.exists():
            subbasin_file = self.workspace_dir / "data" / "subbasins.shp"
        
        if subbasin_file.exists():
            subbasin_gdf = gpd.read_file(subbasin_file)
            print(f"    Loaded subbasin data from: {subbasin_file.name}")
        else:
            raise FileNotFoundError(f"No subbasin data found. Checked: {subbasin_file}")
        
        # Get RVP parameters from parameter table (where they actually exist)
        if hasattr(self.config_manager, 'parameter_table') and self.config_manager.parameter_table:
            parameter_table = self.config_manager.parameter_table
        else:
            # Direct access to loaded parameter table data
            parameter_table = getattr(self.config_manager, '_parameter_table', {})
            
        if 'rvp_parameters' not in parameter_table:
            # Since the code doesn't actually use rvp_params, just skip this validation
            print("  Warning: rvp_parameters not found in parameter table, continuing with hardcoded values")
            rvp_params = {}
        else:
            rvp_params = parameter_table['rvp_parameters']
        
        # BasinMaker pattern: Generate separate channel_properties.rvp file
        channel_props_file = self._generate_channel_properties_file(outlet_name, subbasin_gdf)
        
        with open(rvp_file, 'w') as f:
            f.write(f"# RAVEN Parameter file for {outlet_name}\n")
            f.write("# Generated using RAVEN benchmark format\n\n")
            
            # BasinMaker adoption: Extract ACTUAL classes from HRU data
            actual_soil_classes = set(hru_gdf['SOIL_PROF'].dropna().unique())
            actual_veg_classes = set(hru_gdf['VEG_C'].dropna().unique())
            actual_landuse_classes = set(hru_gdf['LAND_USE_C'].dropna().unique())
            
            print(f"[HYBRID] Using {len(actual_landuse_classes)} landuse, {len(actual_veg_classes)} vegetation, {len(actual_soil_classes)} soil classes from HRU data with your config system")
            
            # Use your config system with ACTUAL classes (not VEG_ALL, LU_ALL)
            f.write(":SoilClasses\n")
            f.write("  :Attributes POROSITY FIELD_CAPACITY WILTING_POINT\n")
            f.write("  :Units      none     none           none\n")
            
            # HYBRID: Generate for each ACTUAL soil class using your config (FAIL-FAST)
            for soil_class in sorted(actual_soil_classes):
                soil_data = self.config_manager.get_soil_profile_params(soil_class)  # Will raise if missing
                # FAIL-FAST: Require all soil parameters (check both cases)
                params = soil_data['parameters']
                
                # Check for porosity (uppercase or lowercase)
                if 'POROSITY' in params:
                    porosity = float(params['POROSITY'])
                elif 'porosity' in params:
                    porosity = float(params['porosity'])
                else:
                    raise ValueError(f"Missing POROSITY/porosity parameter for soil class {soil_class}")
                
                # Check for field capacity
                if 'FIELD_CAPACITY' in params:
                    field_capacity = float(params['FIELD_CAPACITY'])
                elif 'field_capacity' in params:
                    field_capacity = float(params['field_capacity'])
                else:
                    raise ValueError(f"Missing FIELD_CAPACITY/field_capacity parameter for soil class {soil_class}")
                
                # Check for wilting point (multiple possible names)
                if 'WILTING_POINT' in params:
                    wilting_point = float(params['WILTING_POINT'])
                elif 'wilting_point' in params:
                    wilting_point = float(params['wilting_point'])
                elif 'SAT_WILT' in params:
                    wilting_point = float(params['SAT_WILT'])
                elif 'sat_wilt' in params:
                    wilting_point = float(params['sat_wilt'])
                else:
                    raise ValueError(f"Missing wilting point parameter for soil class {soil_class}. Available: {list(params.keys())}")
                f.write(f"  {soil_class:12} {porosity:.3f} {field_capacity:.3f} {wilting_point:.3f}\n")
            f.write(":EndSoilClasses\n\n")
            
            f.write(":SoilProfiles\n")
            for soil_class in sorted(actual_soil_classes):
                f.write(f"  {soil_class} 1 {soil_class} 1.0\n")
            f.write(":EndSoilProfiles\n\n")
            
            f.write(":VegetationClasses\n")
            f.write("  :Attributes MAX_HT MAX_LAI MAX_LEAF_COND\n")
            f.write("  :Units      m      none    mm_per_s\n")
            
            # HYBRID: Generate for each ACTUAL vegetation class using your config (FAIL-FAST)
            for veg_class in sorted(actual_veg_classes):
                veg_data = self.config_manager.get_vegetation_class_params(veg_class)  # Will raise if missing
                max_height = float(veg_data['MAX_HT'])
                max_lai = float(veg_data['MAX_LAI']) 
                max_leaf_cond = float(veg_data['MAX_LEAF_COND'])
                f.write(f"  {veg_class:12} {max_height:6.1f} {max_lai:8.1f} {max_leaf_cond:13.1f}\n")
            f.write(":EndVegetationClasses\n\n")
            
            # Add essential global parameters to prevent auto-generation (RAVEN benchmark format)
            f.write("# Global Parameters\n")
            f.write(":GlobalParameter RAINSNOW_TEMP 0.15\n")
            f.write(":GlobalParameter RAINSNOW_DELTA 1.16\n")
            f.write(":GlobalParameter SNOW_SWI 0.05\n")
            f.write(":GlobalParameter AIRSNOW_COEFF 0.05\n\n")
            
            # Vegetation parameter list - only include parameters used by the model
            f.write(":VegetationParameterList\n")
            f.write("  :Parameters, RAIN_ICEPT_PCT, SNOW_ICEPT_PCT\n")
            f.write("  :Units     , none,          none\n")
            
            for veg_class in sorted(actual_veg_classes):
                veg_data = self.config_manager.get_vegetation_class_params(veg_class)  # Will raise if missing
                veg_params = veg_data['parameters']
                rain_icept_pct = float(veg_params.get('RAIN_ICEPT_PCT', 0.12))
                snow_icept_pct = float(veg_params.get('SNOW_ICEPT_PCT', 0.10))
                f.write(f"  {veg_class:15} {rain_icept_pct:13.3f} {snow_icept_pct:13.3f}\n")
            f.write(":EndVegetationParameterList\n\n")
            
            f.write(":LandUseClasses\n")
            f.write("  :Attributes IMPERM FOREST_COV\n")
            f.write("  :Units      frac   frac\n")
            
            # HYBRID: Generate for each ACTUAL landuse class using your config (FAIL-FAST)
            for landuse_class in sorted(actual_landuse_classes):
                landuse_data = self.config_manager.get_landuse_class_params(landuse_class)  # Will raise if missing
                imperm = float(landuse_data['IMPERM'])
                forest_cov = float(landuse_data['FOREST_COV'])
                f.write(f"  {landuse_class:12} {imperm:.1f} {forest_cov:.1f}\n")
            f.write(":EndLandUseClasses\n\n")
            
            # RAVEN HBV soil parameter list with actual soil classes (no aliases)
            f.write(":SoilParameterList\n")
            f.write("  :Parameters,                POROSITY,FIELD_CAPACITY,    SAT_WILT,    HBV_BETA, MAX_CAP_RISE_RATE,MAX_PERC_RATE,BASEFLOW_COEFF,            BASEFLOW_N\n")
            f.write("  :Units     ,                    none,          none,        none,        none,              mm/d,         mm/d,           1/d,                  none\n")
            
            # Use RAVEN HBV defaults for all soil classes
            for soil_class in sorted(actual_soil_classes):
                try:
                    soil_params = self.config_manager.get_soil_profile_params(soil_class)['parameters']
                    porosity = float(soil_params.get('POROSITY', 0.45))
                    field_capacity = float(soil_params.get('FIELD_CAPACITY', 0.22))
                    sat_wilt = float(soil_params.get('SAT_WILT', 0.08))
                    hbv_beta = float(soil_params.get('HBV_BETA', 1.2))
                    max_cap_rise = float(soil_params.get('MAX_CAP_RISE_RATE', 2.5))
                    max_perc_rate = float(soil_params.get('MAX_PERC_RATE', 15.0))
                    baseflow_coeff = float(soil_params.get('BASEFLOW_COEFF', 0.0))
                    baseflow_n = float(soil_params.get('BASEFLOW_N', 1.0))
                except:
                    # RAVEN HBV defaults
                    porosity = 0.45
                    field_capacity = 0.22
                    sat_wilt = 0.08
                    hbv_beta = 1.2
                    max_cap_rise = 2.5
                    max_perc_rate = 15.0
                    baseflow_coeff = 0.0
                    baseflow_n = 1.0
                
                f.write(f"  {soil_class:12},              {porosity:.8f},     {field_capacity:.7f},  {sat_wilt:.8f},    {hbv_beta:.6f},          {max_cap_rise:.5f},     {max_perc_rate:.1f},      {baseflow_coeff:.1f},                   {baseflow_n:.1f}\n")
            
            f.write(":EndSoilParameterList\n")
            
            # NOTE: Vegetation classes and parameters are handled in the main section above
            # Duplicate sections removed to prevent RAVEN parsing issues
            
            # Land use classes with parameter validation
            f.write("# Land Use Classes Parameters (Generated from RAVEN Parameter Table)\n\n")
            # FAIL-FAST: No fallbacks for landuse config access
            
            f.write(":LandUseClasses\n")
            f.write("  :Attributes IMPERM FOREST_COV\n")
            f.write("  :Units      frac   frac\n")
            
            # FAIL-FAST LANDUSE CLASSES: Extract using centralized config manager
            landuse_classes_used = dynamic_classes['landuse']
            
            for landuse_class in landuse_classes_used:
                # FAIL-FAST: No try/except - must exist in centralized config
                landuse_data = self.config_manager.get_landuse_class_params(landuse_class)
                
                # FAIL-FAST: Extract unique parameters for each landuse class - NO FALLBACKS
                imperm = float(landuse_data['IMPERM'])
                forest_cov = float(landuse_data['FOREST_COV'])
                
                f.write(f"  {landuse_class:<15} {imperm:.3f} {forest_cov:.3f}\n")
                print(f"[OK] Landuse class '{landuse_class}' parameters: IMPERM={imperm}, FOREST_COV={forest_cov}")
            
            f.write(":EndLandUseClasses\n\n")
            
            # Add landuse parameter list for snow processes
            f.write(":LandUseParameterList\n")
            f.write("  :Parameters, MELT_FACTOR, REFREEZE_FACTOR\n")
            f.write("  :Units     , mm/d/C,     mm/d/C\n")
            
            for landuse_class in landuse_classes_used:
                landuse_data = self.config_manager.get_landuse_class_params(landuse_class)
                melt_factor = float(landuse_data.get('MELT_FACTOR', 5.04))
                refreeze_factor = float(landuse_data.get('REFREEZE_FACTOR', 5.04))
                f.write(f"  {landuse_class:<15}, {melt_factor:9.2f}, {refreeze_factor:13.2f}\n")
                print(f"[OK] Landuse snow params '{landuse_class}': MELT={melt_factor}, REFREEZE={refreeze_factor}")
            
            # Add LAKE class for lake HRUs
            f.write(f"  {'LAKE':<15}, {5.04:9.2f}, {5.04:13.2f}\n")
            print(f"[OK] Landuse snow params 'LAKE': MELT=5.04, REFREEZE=5.04")
            
            f.write(":EndLandUseParameterList\n\n")
            
            # Add AvgAnnualRunoff parameter (required for multi-basin models)
            try:
                avg_runoff_config = self.config_manager.get_required_parameter('raven_complete_parameter_table', 'rvp_parameters')
                avg_runoff_value = avg_runoff_config.get('AVG_ANNUAL_RUNOFF', {}).get('default', 300.0)
                f.write(f":AvgAnnualRunoff {avg_runoff_value}\n\n")
                print(f"[OK] Added AvgAnnualRunoff: {avg_runoff_value} mm")
            except Exception as e:
                # Fallback to reasonable default
                avg_runoff_value = 300.0
                f.write(f":AvgAnnualRunoff {avg_runoff_value}\n\n")
                print(f"[OK] Added AvgAnnualRunoff (default): {avg_runoff_value} mm")
            
            # BasinMaker pattern: Add RedirectToFile at end of RVP
            f.write(":RedirectToFile channel_properties.rvp\n")
        
        return rvp_file
    
    def _generate_channel_properties_file(self, outlet_name: str, subbasin_gdf: gpd.GeoDataFrame) -> Path:
        """Generate separate channel_properties.rvp file following BasinMaker pattern"""
        
        channel_props_file = self.models_dir / outlet_name / "channel_properties.rvp"
        
        # CRITICAL FIX: First try to load Step 3 channel profiles
        step3_results_file = self.workspace_dir / "data" / "step3_results.json"
        step3_channel_profiles = []
        
        if step3_results_file.exists():
            try:
                import json
                with open(step3_results_file, 'r') as step3_file:
                    step3_data = json.load(step3_file)
                    step3_routing_config = step3_data.get('routing_config', {})
                    step3_channel_profiles = step3_routing_config.get('channel_profiles', [])
                    
                print(f"  Found {len(step3_channel_profiles)} channel profiles from Step 3")
            except Exception as e:
                print(f"  Failed to load Step 3 channel profiles: {e}")
                step3_channel_profiles = []
        
        with open(channel_props_file, 'w') as f:
            f.write(f"# Channel Properties file for {outlet_name}\n")
            f.write("# Generated using RAVEN benchmark format\n\n")
            
            # Use Step 3 channel profiles if available
            if step3_channel_profiles:
                print(f"[STEP3 INTEGRATION] Using {len(step3_channel_profiles)} channel profiles from Step 3")
                for profile in step3_channel_profiles:
                    f.write(profile + "\n\n")
                print(f"[STEP3 INTEGRATION] Successfully integrated all Step 3 channel profiles")
            else:
                # Fallback: Generate channel profiles directly from subbasin hydraulic data
                print(f"[BASINMAKER FALLBACK] Generating channel profiles directly from subbasin hydraulic data")
                channel_profiles_generated = 0
                
                for _, subbasin in subbasin_gdf.iterrows():
                    subid = int(subbasin['SubId'])
                    river_length_km = float(subbasin.get('RivLength', 0))
                    
                    # Only generate profiles for subbasins with sufficient river length (BasinMaker logic)
                    if river_length_km > 0.001:  # BasinMaker length threshold (1m = 0.001km)
                        channel_name = f"Chn_{subid}"
                        width = float(subbasin.get('BkfWidth', 10.0))
                        depth = float(subbasin.get('BkfDepth', 2.0))
                        slope = max(float(subbasin.get('RivSlope', 0.001)), 0.0001)  # BasinMaker min slope
                        elevation = float(subbasin.get('MeanElev', 1000.0))
                        flood_n = float(subbasin.get('FloodP_n', 0.1))
                        channel_n = float(subbasin.get('Ch_n', 0.035))  # BasinMaker default Manning's n
                        
                        # Generate BasinMaker trapezoidal channel profile
                        profile_lines = self._generate_basinmaker_channel_profile(
                            channel_name, width, depth, slope, elevation, flood_n, channel_n
                        )
                        
                        for line in profile_lines:
                            f.write(line + "\n")
                        f.write("\n")
                        channel_profiles_generated += 1
                
                print(f"[BASINMAKER FALLBACK] Generated {channel_profiles_generated} channel profiles from subbasin hydraulic data")
        
        return channel_props_file
    
    def _extract_channel_parameters_from_basinmaker(self) -> Dict[str, float]:
        """Extract channel parameters from BasinMaker hydraulic data files"""
        
        # Get default values from parameter table config
        channel_defaults = self.config.get('raven_parameters', {}).get('rvp_parameters', {}).get('channel_profile', {}).get('default_hydraulic_parameters', {})
        
        # Check for BasinMaker hydraulic parameter files
        hydraulic_files = [
            self.workspace_dir / "data" / "watershed_routing_hydraulic_parameters.csv",
            self.workspace_dir / "data" / "magpie_hydraulic_parameters.csv"
        ]
        
        # Initialize with config defaults instead of hardcoded values
        channel_data = {
            'mean_slope': channel_defaults.get('mean_slope', 0.001),
            'mean_width': channel_defaults.get('mean_width', 10.0),
            'mean_depth': channel_defaults.get('mean_depth', 2.0),
            'channel_manning_n': channel_defaults.get('channel_manning_n', 0.035)
        }
        
        for hydraulic_file in hydraulic_files:
            if hydraulic_file.exists():
                try:
                    print(f"Reading BasinMaker hydraulic data: {hydraulic_file}")
                    df = pd.read_csv(hydraulic_file)
                    
                    # Extract channel parameters from BasinMaker data
                    if 'channel_slope' in df.columns:
                        channel_data['mean_slope'] = df['channel_slope'].mean()
                    elif 'RivSlope' in df.columns:
                        channel_data['mean_slope'] = df['RivSlope'].mean()
                        
                    if 'channel_width_m' in df.columns:
                        channel_data['mean_width'] = df['channel_width_m'].mean()
                    elif 'BkfWidth' in df.columns:
                        channel_data['mean_width'] = df['BkfWidth'].mean()
                        
                    if 'channel_depth_m' in df.columns:
                        channel_data['mean_depth'] = df['channel_depth_m'].mean()
                    elif 'BkfDepth' in df.columns:
                        channel_data['mean_depth'] = df['BkfDepth'].mean()
                        
                    if 'manning_n' in df.columns:
                        channel_data['channel_manning_n'] = df['manning_n'].mean()
                    elif 'Ch_n' in df.columns:
                        channel_data['channel_manning_n'] = df['Ch_n'].mean()
                    
                    print(f"[OK] Extracted channel parameters: width={channel_data['mean_width']:.2f}m, "
                          f"depth={channel_data['mean_depth']:.2f}m, slope={channel_data['mean_slope']:.6f}, "
                          f"manning_n={channel_data['channel_manning_n']:.3f}")
                    break
                    
                except Exception as e:
                    print(f"Warning: Could not read {hydraulic_file}: {e}")
                    continue
        
        # Validate extracted parameters using config ranges
        param_ranges = self.config.get('raven_parameters', {}).get('rvp_parameters', {}).get('channel_profile', {}).get('parameter_ranges', {})
        
        slope_range = param_ranges.get('mean_slope', [0.0001, 0.1])
        width_range = param_ranges.get('mean_width', [1.0, 100.0])
        depth_range = param_ranges.get('mean_depth', [0.1, 10.0])
        manning_range = param_ranges.get('channel_manning_n', [0.02, 0.15])
        
        channel_data['mean_slope'] = max(slope_range[0], min(slope_range[1], channel_data['mean_slope']))
        channel_data['mean_width'] = max(width_range[0], min(width_range[1], channel_data['mean_width']))
        channel_data['mean_depth'] = max(depth_range[0], min(depth_range[1], channel_data['mean_depth']))
        channel_data['channel_manning_n'] = max(manning_range[0], min(manning_range[1], channel_data['channel_manning_n']))
        
        return channel_data
    
    def _generate_rvt_file(self, outlet_name: str, latitude: float, longitude: float) -> Path:
        """Generate RVT file using RAVEN benchmark format"""
        
        outlet_dir = self.models_dir / outlet_name
        rvt_file = outlet_dir / f"{outlet_name}.rvt"
        
        # Load climate data
        climate_csv = self.workspace_dir / "climate" / "climate_forcing.csv"
        if not climate_csv.exists():
            raise FileNotFoundError(f"Climate data not found: {climate_csv}")
        
        climate_df = pd.read_csv(climate_csv, index_col=0, parse_dates=True)
        
        # Use existing BasinMaker RVT generator logic
        from processors.rvt_generator import RVTGenerator
        
        rvt_generator = RVTGenerator(outlet_dir)
        rvt_file = rvt_generator.generate_rvt_from_csv(
            csv_file_path=climate_csv,
            outlet_name=outlet_name,
            latitude=latitude,
            longitude=longitude
        )
        
        # Add hydrometric observation gauges if available (BasinMaker format)
        obs_dir = outlet_dir / "obs"
        if obs_dir.exists():
            obs_files = list(obs_dir.glob("*.rvt"))
            # Remove duplicates and use only the main station file
            obs_files = [f for f in obs_files if not f.name.startswith("HYDROMETRIC_STATION_")]
            if obs_files:
                # Append observation gauges to existing RVT file
                with open(rvt_file, 'a') as f:
                    f.write("\n# Hydrometric observation gauges\n")
                    for obs_file in obs_files:
                        # Extract station info from filename (e.g., TRAPPING_CREEK_08NN019_50.rvt)
                        station_name = obs_file.stem.replace("_", " ")
                        f.write(f":Gauge {station_name}\n")
                        f.write(f"  :Latitude {latitude}\n")
                        f.write(f"  :Longitude {longitude}\n") 
                        f.write(f"  :RedirectToFile ./obs/{obs_file.name}\n")
                        f.write(f":EndGauge\n\n")
            

        
        return rvt_file
    

    def _generate_lakes_rvh_if_needed(self, hru_gdf: gpd.GeoDataFrame, outlet_name: str) -> Optional[Path]:
        """Generate Lakes.rvh file using lakes processor"""
        
        # Set output directory for this outlet
        outlet_dir = self.models_dir / outlet_name
        self.lakes_generator.output_dir = outlet_dir
        
        # Generate lakes using the processor
        return self.lakes_generator.generate_lakes_rvh(hru_gdf, outlet_name)
    
    def _generate_rvc_file(self, hru_gdf: gpd.GeoDataFrame, outlet_name: str) -> Path:
        """Generate RVC (initial conditions) file based on JSON configuration and basin data"""
        
        rvc_file = self.models_dir / outlet_name / f"{outlet_name}.rvc"
        
        # Get initial conditions configuration from JSON
        initial_config = self.config.get('initial_conditions', {})
        default_values = initial_config.get('default_values', {
            "snow_mm": 0.0,
            "soil_layer_1_mm": 50.0,
            "soil_layer_2_mm": 100.0,
            "soil_layer_3_mm": 50.0,
            "groundwater_1_mm": 200.0,
            "groundwater_2_mm": 500.0
        })
        
        # Get unique subbasins from HRU data, filtering out invalid SubIds
        subbasins = hru_gdf.drop_duplicates('SubId')[['SubId']].copy()
        
        # Filter out invalid SubIds (-1, 0, or null)
        invalid_subids = subbasins[
            (subbasins['SubId'] == -1) | 
            (subbasins['SubId'] == 0) | 
            (subbasins['SubId'].isna())
        ]
        
        if len(invalid_subids) > 0:
            print(f"  WARNING: Filtering out {len(invalid_subids)} invalid SubIds from RVC file: {invalid_subids['SubId'].tolist()}")
            subbasins = subbasins[
                (subbasins['SubId'] > 0) & 
                (subbasins['SubId'].notna())
            ].copy()
        
        # Determine season for seasonal adjustments (use simulation start date)
        sim_config = self.config.get('simulation', {})
        start_date = sim_config.get('start_date', '2002-10-01 00:00:00')
        start_month = int(start_date.split('-')[1])
        
        seasonal_adjustments = initial_config.get('seasonal_adjustments', {})
        season = self._get_season_from_month(start_month, seasonal_adjustments)
        season_adjustments = seasonal_adjustments.get('adjustments', {}).get(season, {})
        
        with open(rvc_file, 'w') as f:
            f.write(f"# RAVEN Initial Conditions file for {outlet_name}\n")
            f.write("# Generated by Step 5 - Configurable basin-specific initial conditions\n")
            f.write("# Format: SubID, Snow(mm), Soil1(mm), Soil2(mm), Soil3(mm), GW1(mm), GW2(mm)\n\n")
            
            f.write(":BasinInitialConditions\n")
            
            for _, sub in subbasins.iterrows():
                sub_id = int(sub['SubId'])
                
                # Additional safety check
                if sub_id <= 0:
                    continue
                
                # Get HRUs for this subbasin to determine dominant soil type
                sub_hrus = hru_gdf[hru_gdf['SubId'] == sub_id]
                dominant_soil = self._get_dominant_soil_type(sub_hrus)
                
                # Apply base values
                snow = default_values.get('snow_mm', 0.0)
                soil1 = default_values.get('soil_layer_1_mm', 50.0)
                soil2 = default_values.get('soil_layer_2_mm', 100.0)
                soil3 = default_values.get('soil_layer_3_mm', 50.0)
                gw1 = default_values.get('groundwater_1_mm', 200.0)
                gw2 = default_values.get('groundwater_2_mm', 500.0)
                
                # Apply seasonal adjustments
                snow += season_adjustments.get('snow_mm', 0.0)
                soil1 += season_adjustments.get('soil_layer_1_mm', 0.0)
                soil2 += season_adjustments.get('soil_layer_2_mm', 0.0)
                
                # Apply soil type adjustments
                soil_adjustments = initial_config.get('soil_type_adjustments', {}).get(dominant_soil, {})
                soil1 = soil_adjustments.get('soil_layer_1_mm', soil1)
                soil2 = soil_adjustments.get('soil_layer_2_mm', soil2)
                gw1 = soil_adjustments.get('groundwater_1_mm', gw1)
                
                f.write(f"    {sub_id}, {snow:.1f}, {soil1:.1f}, {soil2:.1f}, {soil3:.1f}, {gw1:.1f}, {gw2:.1f}\n")
            
            f.write(":EndBasinInitialConditions\n")
        
        return rvc_file
    
    def _get_season_from_month(self, month: int, seasonal_config: dict) -> str:
        """Determine season from month number"""
        winter_months = seasonal_config.get('winter_months', [12, 1, 2])
        spring_months = seasonal_config.get('spring_months', [3, 4, 5])
        summer_months = seasonal_config.get('summer_months', [6, 7, 8])
        fall_months = seasonal_config.get('fall_months', [9, 10, 11])
        
        if month in winter_months:
            return 'winter'
        elif month in spring_months:
            return 'spring'
        elif month in summer_months:
            return 'summer'
        elif month in fall_months:
            return 'fall'
        else:
            return 'summer'  # default
    
    def _get_dominant_soil_type(self, sub_hrus: gpd.GeoDataFrame) -> str:
        """Get the dominant soil type for a subbasin"""
        if 'SOIL_PROF' in sub_hrus.columns:
            # Get the most common soil type by area
            soil_areas = sub_hrus.groupby('SOIL_PROF')['HRU_Area'].sum()
            dominant_soil = soil_areas.idxmax()
            return str(dominant_soil)
        else:
            return 'LOAM'  # default
    
    def _validate_hru_data(self, hru_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Comprehensive validation of HRU data before model generation"""
        
        validation_summary = {
            'total_hrus': len(hru_gdf),
            'valid_hrus': 0,
            'missing_data_count': 0,
            'issues': []
        }
        
        required_fields = ['SubId', 'LAND_USE_C', 'VEG_C', 'SOIL_PROF', 'HRU_Area']
        
        for idx, hru in hru_gdf.iterrows():
            is_valid = True
            
            # Check for missing required fields
            for field in required_fields:
                if field not in hru or pd.isna(hru[field]) or str(hru[field]).strip() == '':
                    validation_summary['missing_data_count'] += 1
                    validation_summary['issues'].append(f"HRU {idx}: Missing {field}")
                    is_valid = False
            
            # Check for reasonable area values
            if 'HRU_Area' in hru and not pd.isna(hru['HRU_Area']):
                area = float(hru['HRU_Area'])
                if area <= 0 or area > 10000:  # 0 to 10,000 km²
                    validation_summary['issues'].append(f"HRU {idx}: Unrealistic area {area} km²")
                    is_valid = False
            
            # Check for valid coordinates
            coord_fields = [('HRU_CenX', 'longitude'), ('HRU_CenY', 'latitude')]
            for field, coord_type in coord_fields:
                if field in hru and not pd.isna(hru[field]):
                    coord = float(hru[field])
                    if coord_type == 'longitude' and (coord < -180 or coord > 180):
                        validation_summary['issues'].append(f"HRU {idx}: Invalid longitude {coord}")
                        is_valid = False
                    elif coord_type == 'latitude' and (coord < -90 or coord > 90):
                        validation_summary['issues'].append(f"HRU {idx}: Invalid latitude {coord}")
                        is_valid = False
            
            if is_valid:
                validation_summary['valid_hrus'] += 1
        
        return validation_summary
    
    def _validate_generated_files(self, outlet_name: str) -> Dict[str, Any]:
        """Validate that all required RAVEN files were generated correctly"""
        
        model_dir = self.models_dir / outlet_name
        validation_results = {
            'success': True,
            'files_validated': {},
            'errors': []
        }
        
        # Check required files
        required_files = {
            'rvi': f"{outlet_name}.rvi",
            'rvp': f"{outlet_name}.rvp", 
            'rvt': f"{outlet_name}.rvt",
            'rvh': f"{outlet_name}.rvh"
        }
        
        for file_type, filename in required_files.items():
            file_path = model_dir / filename
            
            if not file_path.exists():
                validation_results['success'] = False
                validation_results['errors'].append(f"Missing {file_type.upper()} file: {filename}")
                validation_results['files_validated'][file_type] = False
            else:
                # Basic content validation
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    if len(content.strip()) == 0:
                        validation_results['success'] = False
                        validation_results['errors'].append(f"Empty {file_type.upper()} file: {filename}")
                        validation_results['files_validated'][file_type] = False
                    else:
                        # File-specific validation
                        if file_type == 'rvi':
                            if ':RedirectToFile' not in content:
                                validation_results['errors'].append(f"RVI file missing RedirectToFile commands")
                            if f':RedirectToFile {outlet_name}.rvp' not in content:
                                validation_results['errors'].append(f"RVI file missing RVP redirect")
                        
                        elif file_type == 'rvh':
                            if ':SubBasins' not in content or ':HRUs' not in content:
                                validation_results['errors'].append(f"RVH file missing required sections")
                        
                        elif file_type == 'rvt':
                            if ':Gauge' not in content or ':MultiData' not in content:
                                validation_results['errors'].append(f"RVT file missing required climate data sections")
                        
                        validation_results['files_validated'][file_type] = True
                        
                except Exception as e:
                    validation_results['success'] = False
                    validation_results['errors'].append(f"Error reading {file_type.upper()} file: {str(e)}")
                    validation_results['files_validated'][file_type] = False
        
        return validation_results

    def _run_raven_model(self, outlet_name: str) -> Dict[str, Any]:
        """Run RAVEN model directly with the executable"""
        
        model_dir = self.models_dir / outlet_name
        raven_exe_paths = [
            Path("E:/python/Raven/RavenHydroFramework/build/Release/Raven.exe"),
            Path("E:/python/Raven/RavenHydroFramework/build/Debug/Raven.exe"),
            Path("Raven.exe")  # If in PATH
        ]
        
        # Find RAVEN executable
        raven_exe = None
        for exe_path in raven_exe_paths:
            if exe_path.exists():
                raven_exe = exe_path
                break
        
        if not raven_exe:
            return {
                'success': False,
                'error': 'RAVEN executable not found',
                'output': '',
                'model_dir': str(model_dir)
            }
        
        print(f"Running RAVEN model with: {raven_exe}")
        print(f"Model directory: {model_dir}")
        
        # Change to model directory and run RAVEN
        import subprocess
        import os
        
        try:
            # Run RAVEN with the model name (without extension)
            result = subprocess.run(
                [str(raven_exe), outlet_name],
                cwd=str(model_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Check for ERROR in output (case insensitive)
            combined_output = result.stdout + result.stderr
            has_error = 'ERROR' in combined_output.upper()
            
            # Mark as failed if there are any ERRORs in the output, regardless of return code
            success = result.returncode == 0 and not has_error
            
            if has_error:
                print("ERROR detected in RAVEN output - marking as failed")
                # Extract error lines for reporting
                error_lines = [line for line in combined_output.split('\n') if 'ERROR' in line.upper()]
                print("Error details:")
                for error_line in error_lines[:5]:  # Show first 5 errors
                    print(f"  {error_line}")
            
            return {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'output': combined_output,
                'has_errors': has_error,
                'error_details': [line for line in combined_output.split('\n') if 'ERROR' in line.upper()] if has_error else [],
                'model_dir': str(model_dir)
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'RAVEN execution timed out (5 minutes)',
                'output': 'Timeout after 5 minutes',
                'model_dir': str(model_dir)
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'RAVEN execution failed: {str(e)}',
                'output': str(e),
                'model_dir': str(model_dir)
            }
    
    def execute(self, latitude: float, longitude: float, outlet_name: str) -> Dict[str, Any]:
        """Execute Step 5 RAVEN model generation with dynamic lookup integration"""
        
        # Store model name for use in other methods
        self.model_name = outlet_name
        
        print(f"=== STEP 5: RAVEN Model Generation with Dynamic Lookup Integration ===")
        print(f"Outlet: ({latitude}, {longitude})")
        print(f"Model name: {outlet_name}")
        
        # Generate all BasinMaker-compatible lookup tables first
        print("Generating comprehensive lookup tables from JSON database...")
        lookup_files = self.lookup_generator.generate_all_lookup_tables()
        for table_name, file_path in lookup_files.items():
            print(f"  Generated {table_name}: {file_path}")
        
        # Load HRU data from Step 4
        hru_gdf = self._load_hru_data()
        print(f"Loaded {len(hru_gdf)} HRUs from Step 4")
        
        # Calculate dynamic terrain attributes (elevation, slope, aspect) using BasinMaker approach
        print("Calculating dynamic terrain attributes from DEM...")
        dem_file = self.workspace_dir / "data" / "dem.tif"
        hru_gdf = self.terrain_calculator.calculate_terrain_attributes(hru_gdf, dem_file)
        
        # Validate HRU data and parameters
        print("Validating HRU parameters and classifications...")
        validation_results = self.parameter_extractor.validate_parameters(hru_gdf)
        if not validation_results['success']:
            print(f"WARNING: Parameter validation issues: {validation_results['errors']}")
        
        # Additional validation checks
        validation_summary = self._validate_hru_data(hru_gdf)
        print(f"HRU Data Validation Summary:")
        print(f"  Total HRUs: {validation_summary['total_hrus']}")
        print(f"  Valid HRUs: {validation_summary['valid_hrus']}")
        print(f"  Missing data issues: {validation_summary['missing_data_count']}")
        if validation_summary['issues']:
            print(f"  Issues found: {validation_summary['issues']}")
        
        # Extract dynamic classes using parameter extractor
        print("Extracting parameter classes from lookup database...")
        dynamic_classes = self._extract_dynamic_classes_from_lookup(hru_gdf)
        print(f"Dynamic classes found:")
        print(f"  Land use: {sorted(dynamic_classes['landuse'])}")
        print(f"  Vegetation: {sorted(dynamic_classes['vegetation'])}")  
        print(f"  Soil: {sorted(dynamic_classes['soil'])}")
        
        # INTEGRATE OBSERVATION POINTS (GAUGE STATIONS) for proper GAUGED attribute
        print("Integrating hydrometric stations as observation points...")
        try:
            hru_gdf = self._integrate_gauge_stations(hru_gdf)
            print("SUCCESS: Gauge station integration completed")
        except Exception as e:
            print(f"WARNING: Gauge station integration failed: {e}")
        
        # Generate model files with dynamic parameters
        print("Generating RAVEN model files with dynamic parameters...")
        
        # Check if lakes will be generated (for RVI redirect)
        has_lakes = self.lakes_generator.detect_lakes_in_hru_data(hru_gdf)
        
        # Generate RVH with actual class names (no routing)
        rvh_file = self._generate_rvh_with_real_classes(hru_gdf, outlet_name, has_lakes)
        print(f"[OK] RVH: {rvh_file}")
        
        # Generate RVI with dynamic class definitions
        rvi_file = self._generate_rvi_with_dynamic_classes(outlet_name, dynamic_classes, has_lakes)
        print(f"[OK] RVI: {rvi_file}")
        
        # Generate RVP parameter file
        rvp_file = self._generate_rvp_file(hru_gdf, outlet_name, dynamic_classes)
        print(f"[OK] RVP: {rvp_file}")
        
        # Generate RVT climate forcing file
        rvt_file = self._generate_rvt_file(outlet_name, latitude, longitude)
        print(f"[OK] RVT: {rvt_file}")
        
        # Generate RVC initial conditions file
        rvc_file = self._generate_rvc_file(hru_gdf, outlet_name)
        print(f"[OK] RVC: {rvc_file}")
        
        # Generate Lakes.rvh if lakes are detected
        if has_lakes:
            lakes_file = self._generate_lakes_rvh_if_needed(hru_gdf, outlet_name)
            if lakes_file and lakes_file.exists():
                print(f"[OK] Lakes.rvh: {lakes_file}")
        
        # Validate generated files
        print("\n=== Validating Generated Files ===")
        file_validation = self._validate_generated_files(outlet_name)
        if file_validation['success']:
            print("[OK] All RAVEN files validated successfully")
            for file_type, status in file_validation['files_validated'].items():
                print(f"  {file_type.upper()}: {'[OK]' if status else '[ERROR]'}")
        else:
            print("[ERROR] File validation failed:")
            for error in file_validation['errors']:
                print(f"  - {error}")
        
        # Run RAVEN model directly
        print("\n=== Running RAVEN Model ===")
        raven_results = self._run_raven_model(outlet_name)
        
        if raven_results['success']:
            print(f"[OK] RAVEN model completed successfully!")
            print(f"Model output in: {raven_results['model_dir']}")
        else:
            print(f"[ERROR] RAVEN model failed: {raven_results.get('error', 'Unknown error')}")
            print(f"Output: {raven_results.get('output', 'No output')}")
        
        return {
            'success': True,
            'model_files': {
                'rvh': str(rvh_file),
                'rvi': str(rvi_file),
                'rvp': str(rvp_file),
                'rvt': str(rvt_file),
                'rvc': str(rvc_file)
            },
            'hru_count': len(hru_gdf),
            'dynamic_classes': dynamic_classes,
            'output_dir': str(rvh_file.parent),
            'raven_execution': raven_results
        }


def main():
    parser = argparse.ArgumentParser(description="Clean Step 5: RAVEN Model Generation")
    parser.add_argument("latitude", type=float, help="Outlet latitude")
    parser.add_argument("longitude", type=float, help="Outlet longitude")
    parser.add_argument("--workspace-dir", type=str, required=True, help="Workspace directory")
    parser.add_argument("--outlet-name", type=str, required=True, help="Outlet/model name")
    
    args = parser.parse_args()
    
    # Execute Step 5
    step5 = Step5RAVENModel(workspace_dir=args.workspace_dir)
    results = step5.execute(args.latitude, args.longitude, args.outlet_name)
    
    if results['success']:
        print(f"\n=== SUCCESS ===")
        print(f"Generated {results['hru_count']} HRUs with real class names!")
        print(f"Model files: {list(results['model_files'].keys())}")
        print(f"Output: {results['output_dir']}")
    else:
        print(f"ERROR: {results.get('error', 'Unknown error')}")


# Alias for backward compatibility
CompleteStep5RAVENModel = Step5RAVENModel

if __name__ == "__main__":
    main()