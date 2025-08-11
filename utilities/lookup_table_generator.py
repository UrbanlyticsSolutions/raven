#!/usr/bin/env python3
"""
Dynamic Lookup Table Generator
Generates BasinMaker-compatible CSV files from the comprehensive JSON lookup database
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

class RAVENLookupTableGenerator:
    """
    Generates all required lookup tables dynamically from the JSON database
    Fixes all gaps identified in Step 4 and Step 5
    """
    
    def __init__(self, json_database_path: str = None, output_dir: str = None):
        """
        Initialize the lookup table generator
        
        Args:
            json_database_path: Path to raven_lookup_database.json
            output_dir: Output directory for generated CSV files
        """
        if json_database_path is None:
            json_database_path = Path(__file__).parent.parent / "config" / "raven_lookup_database.json"
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "generated_lookup_tables"
        
        self.json_db_path = Path(json_database_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load the JSON database
        self.load_database()
    
    def load_database(self):
        """Load the JSON lookup database"""
        try:
            with open(self.json_db_path, 'r', encoding='utf-8') as f:
                self.database = json.load(f)
            self.logger.info(f"Loaded lookup database from: {self.json_db_path}")
        except Exception as e:
            raise FileNotFoundError(f"Could not load database: {self.json_db_path}. Error: {e}")
    
    def generate_soil_info_csv(self, output_file: str = None) -> str:
        """
        Generate soil_info.csv for BasinMaker compatibility
        Format: Soil_ID,SOIL_PROF
        """
        if output_file is None:
            output_file = self.output_dir / "soil_info.csv"
        
        # Get soil classification data
        soil_classes = self.database["soil_classification"]["soil_classes"]
        mapping = self.database["soil_classification"]["basinmaker_format"]["example_mapping"]
        
        # Create DataFrame
        data = []
        for soil_id, soil_name in mapping.items():
            data.append({
                'Soil_ID': int(soil_id),
                'SOIL_PROF': soil_name
            })
        
        df = pd.DataFrame(data).sort_values('Soil_ID')
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Generated soil_info.csv: {output_file} ({len(df)} soil classes)")
        return str(output_file)
    
    def generate_landuse_info_csv(self, output_file: str = None) -> str:
        """
        Generate landuse_info.csv for BasinMaker compatibility  
        Format: Landuse_ID,LAND_USE_C
        """
        if output_file is None:
            output_file = self.output_dir / "landuse_info.csv"
        
        # Get landcover classification data
        mapping = self.database["landcover_classification"]["basinmaker_format"]["example_mapping"]
        
        # Create DataFrame
        data = []
        for landuse_id, land_use_c in mapping.items():
            data.append({
                'Landuse_ID': int(landuse_id),
                'LAND_USE_C': land_use_c
            })
        
        df = pd.DataFrame(data).sort_values('Landuse_ID')
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Generated landuse_info.csv: {output_file} ({len(df)} landuse classes)")
        return str(output_file)
    
    def generate_veg_info_csv(self, output_file: str = None) -> str:
        """
        Generate veg_info.csv for BasinMaker compatibility
        Format: Veg_ID,VEG_C
        """
        if output_file is None:
            output_file = self.output_dir / "veg_info.csv"
        
        # Get vegetation mapping from landcover classes
        landcover_mapping = self.database["landcover_classification"]["basinmaker_format"]["example_mapping"]
        veg_mapping = self.database["vegetation_classification"]["basinmaker_format"]["landcover_mapping"]
        
        # Create DataFrame
        data = []
        for landuse_id, land_use_c in landcover_mapping.items():
            if land_use_c in veg_mapping:
                veg_c = veg_mapping[land_use_c]
                data.append({
                    'Veg_ID': int(landuse_id),
                    'VEG_C': veg_c
                })
        
        df = pd.DataFrame(data).sort_values('Veg_ID')
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Generated veg_info.csv: {output_file} ({len(df)} vegetation classes)")
        return str(output_file)
    
    def generate_manning_roughness_csv(self, output_file: str = None) -> str:
        """
        Generate Manning's roughness CSV (equivalent to BasinMaker Landuse_info3.csv)
        Format: RasterV,MannV
        """
        if output_file is None:
            output_file = self.output_dir / "manning_roughness.csv"
        
        # Get Manning's roughness values
        roughness_values = self.database["manning_roughness_database"]["roughness_values"]["landcover_based"]
        landcover_mapping = self.database["landcover_classification"]["basinmaker_format"]["example_mapping"]
        
        # Create DataFrame
        data = []
        for landuse_id, land_use_c in landcover_mapping.items():
            # Map land use class to roughness category
            roughness_category = self._map_landuse_to_roughness_category(land_use_c)
            if roughness_category in roughness_values:
                manning_n = roughness_values[roughness_category]
                # Convert to Manning's n * 100 (BasinMaker format)
                mann_v = int(manning_n * 100)
                data.append({
                    'RasterV': int(landuse_id),
                    'MannV': mann_v
                })
        
        df = pd.DataFrame(data).sort_values('RasterV')
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Generated manning_roughness.csv: {output_file} ({len(df)} roughness values)")
        return str(output_file)
    
    def _map_landuse_to_roughness_category(self, land_use_c: str) -> str:
        """Map landuse class to Manning's roughness category"""
        mapping = {
            'FOREST_CONIFEROUS': 'FOREST',
            'FOREST_DECIDUOUS': 'FOREST', 
            'FOREST_MIXED': 'FOREST',
            'AGRICULTURE': 'AGRICULTURE',
            'URBAN': 'URBAN',
            'GRASSLAND': 'GRASSLAND',
            'BARREN': 'BARREN',
            'WATER': 'WATER',
            'WETLAND': 'WETLAND',
            'SHRUBLAND': 'SHRUBLAND',
            'LAKE': 'WATER'
        }
        return mapping.get(land_use_c, 'GRASSLAND')  # Default to grassland
    
    def classify_soil_texture_from_percentages(self, sand_pct: float, silt_pct: float, clay_pct: float) -> str:
        """
        Classify soil texture using USDA texture triangle rules from JSON database
        
        Args:
            sand_pct: Sand percentage (0-100)
            silt_pct: Silt percentage (0-100) 
            clay_pct: Clay percentage (0-100)
            
        Returns:
            Soil texture class name (e.g., 'LOAM', 'CLAY', etc.)
        """
        # Normalize percentages
        total = sand_pct + silt_pct + clay_pct
        if total > 0:
            sand_pct = (sand_pct / total) * 100
            silt_pct = (silt_pct / total) * 100
            clay_pct = (clay_pct / total) * 100
        else:
            return 'UNKNOWN'
        
        # Apply USDA classification rules from JSON database
        rules = self.database["usda_soil_texture_classification"]["classification_rules"]
        
        # Check clay group first (highest clay content)
        if clay_pct >= 40:
            if sand_pct >= 45:
                return 'SANDY_CLAY'
            elif silt_pct >= 40:
                return 'SILTY_CLAY'
            else:
                return 'CLAY'
        
        # Check clay loam group
        elif clay_pct >= 27:
            if sand_pct >= 45:
                return 'SANDY_CLAY_LOAM'
            elif silt_pct >= 28:
                return 'SILTY_CLAY_LOAM'
            else:
                return 'CLAY_LOAM'
        
        # Check loam group
        elif clay_pct >= 7:
            if sand_pct >= 52:
                return 'SANDY_LOAM'
            elif silt_pct >= 50:
                return 'SILT_LOAM'
            else:
                return 'LOAM'
        
        # Check sand group (low clay)
        else:
            if silt_pct >= 80:
                return 'SILT'
            elif sand_pct >= 85:
                return 'SAND'
            elif sand_pct >= 70:
                return 'LOAMY_SAND'
            else:
                return 'LOAM'
    
    def get_soil_texture_id(self, texture_class: str) -> int:
        """Get soil texture ID for given texture class"""
        soil_classes = self.database["soil_classification"]["soil_classes"]
        for class_name, class_data in soil_classes.items():
            if class_name == texture_class:
                return class_data["id"]
        return 1  # Default to CLAY
    
    def classify_landcover_from_worldcover(self, worldcover_id: int) -> Dict[str, Any]:
        """
        Classify landcover from ESA WorldCover ID to RAVEN classes
        
        Args:
            worldcover_id: ESA WorldCover class ID
            
        Returns:
            Dictionary with raven_class, raven_id, and parameters
        """
        mapping = self.database["worldcover_to_raven_mapping"]["mappings"]
        worldcover_str = str(worldcover_id)
        
        if worldcover_str in mapping:
            mapped = mapping[worldcover_str]
            raven_class = mapped["raven_class"]
            
            # Get full parameters from landcover database
            landcover_classes = self.database["landcover_classification"]["landcover_classes"]
            if raven_class in landcover_classes:
                return {
                    "raven_class": raven_class,
                    "raven_id": mapped["raven_id"],
                    "parameters": landcover_classes[raven_class]
                }
        
        # Default to grassland if not found
        return {
            "raven_class": "GRASSLAND",
            "raven_id": 60,
            "parameters": self.database["landcover_classification"]["landcover_classes"]["GRASSLAND"]
        }
    
    def generate_all_lookup_tables(self) -> Dict[str, str]:
        """
        Generate all lookup tables needed for BasinMaker integration
        
        Returns:
            Dictionary with file paths for all generated tables
        """
        results = {}
        
        self.logger.info("Generating all BasinMaker lookup tables...")
        
        # Generate core lookup tables
        results['soil_info'] = self.generate_soil_info_csv()
        results['landuse_info'] = self.generate_landuse_info_csv()
        results['veg_info'] = self.generate_veg_info_csv()
        results['manning_roughness'] = self.generate_manning_roughness_csv()
        
        self.logger.info(f"Generated {len(results)} lookup tables in: {self.output_dir}")
        return results
    
    def get_soil_parameters_for_raven(self, soil_class: str) -> Dict[str, float]:
        """
        Get soil hydraulic parameters for RAVEN RVP file generation
        
        Args:
            soil_class: Soil texture class name
            
        Returns:
            Dictionary of hydraulic parameters
        """
        soil_classes = self.database["soil_classification"]["soil_classes"]
        
        if soil_class in soil_classes:
            hydraulic_props = soil_classes[soil_class]["hydraulic_properties"]
            hbv_params = soil_classes[soil_class]["hbv_parameters"]
            
            return {
                "POROSITY": hydraulic_props["porosity"],
                "FIELD_CAPACITY": hydraulic_props["field_capacity"] * 1000,  # Convert to mm
                "WILTING_POINT": hydraulic_props["wilting_point"] * 1000,    # Convert to mm
                "SAT_WILT": hydraulic_props["wilting_point"] * 1000,         # Convert to mm
                "HBV_BETA": hbv_params["beta"],
                "HBV_LP": hbv_params["lp"],
                "SAT_HYDRAULIC_CONDUCTIVITY": hydraulic_props["saturated_hydraulic_conductivity_mm_hr"]
            }
        
        # Return default values if not found
        return {
            "POROSITY": 0.50,
            "FIELD_CAPACITY": 250.0,
            "WILTING_POINT": 130.0,
            "SAT_WILT": 130.0,
            "HBV_BETA": 2.0,
            "HBV_LP": 0.65,
            "SAT_HYDRAULIC_CONDUCTIVITY": 25.0
        }
    
    def get_vegetation_parameters_for_raven(self, landcover_class: str) -> Dict[str, float]:
        """
        Get vegetation parameters for RAVEN RVP file generation
        
        Args:
            landcover_class: Land cover class name
            
        Returns:
            Dictionary of vegetation parameters
        """
        landcover_classes = self.database["landcover_classification"]["landcover_classes"]
        
        if landcover_class in landcover_classes:
            raven_params = landcover_classes[landcover_class]["raven_parameters"]
            hydro_props = landcover_classes[landcover_class]["hydrological_properties"]
            
            return {
                "MAX_HEIGHT": raven_params["max_height_m"],
                "MAX_LAI": raven_params["max_lai"],
                "MAX_LEAF_CONDUCTANCE": raven_params["max_leaf_conductance"],
                "ALBEDO": hydro_props["albedo"]
            }
        
        # Return default values if not found
        return {
            "MAX_HEIGHT": 1.0,
            "MAX_LAI": 3.0,
            "MAX_LEAF_CONDUCTANCE": 20.0,
            "ALBEDO": 0.20
        }
    
    def get_landuse_parameters_for_raven(self, landcover_class: str) -> Dict[str, float]:
        """
        Get landuse parameters for RAVEN RVP file generation
        
        Args:
            landcover_class: Land cover class name
            
        Returns:
            Dictionary of landuse parameters
        """
        landcover_classes = self.database["landcover_classification"]["landcover_classes"]
        
        if landcover_class in landcover_classes:
            raven_params = landcover_classes[landcover_class]["raven_parameters"]
            hydro_props = landcover_classes[landcover_class]["hydrological_properties"]
            
            return {
                "IMPERMEABLE_FRAC": raven_params["imperviousness"],
                "FOREST_COV": raven_params["forest_coverage"],
                "MANNING_N": hydro_props["manning_roughness"],
                "INTERCEPTION_CAPACITY": hydro_props["interception_capacity_mm"]
            }
        
        # Return default values if not found
        return {
            "IMPERMEABLE_FRAC": 0.0,
            "FOREST_COV": 0.0,
            "MANNING_N": 0.25,
            "INTERCEPTION_CAPACITY": 1.0
        }


def main():
    """Main function for testing the lookup table generator"""
    generator = RAVENLookupTableGenerator()
    
    # Generate all lookup tables
    results = generator.generate_all_lookup_tables()
    
    print("Generated lookup tables:")
    for table_name, file_path in results.items():
        print(f"  {table_name}: {file_path}")
    
    # Test soil texture classification
    sand_pct, silt_pct, clay_pct = 65, 23, 12
    soil_class = generator.classify_soil_texture_from_percentages(sand_pct, silt_pct, clay_pct)
    print(f"\nSoil texture classification test:")
    print(f"  Sand: {sand_pct}%, Silt: {silt_pct}%, Clay: {clay_pct}%")
    print(f"  Classified as: {soil_class}")
    
    # Test landcover classification  
    worldcover_id = 10  # Forest
    landcover_result = generator.classify_landcover_from_worldcover(worldcover_id)
    print(f"\nLandcover classification test:")
    print(f"  WorldCover ID: {worldcover_id}")
    print(f"  RAVEN class: {landcover_result['raven_class']}")
    
    # Test parameter extraction
    soil_params = generator.get_soil_parameters_for_raven('SANDY_LOAM')
    print(f"\nSoil parameters for SANDY_LOAM:")
    for param, value in soil_params.items():
        print(f"  {param}: {value}")


if __name__ == "__main__":
    main()