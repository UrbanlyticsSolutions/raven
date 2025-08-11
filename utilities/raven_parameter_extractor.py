#!/usr/bin/env python3
"""
RAVEN Parameter Extractor
Extracts parameters from lookup database and generates RAVEN RVP/RVH file sections
Fixes Step 5 parameter extraction gaps
"""

import json
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from .lookup_table_generator import RAVENLookupTableGenerator

class RAVENParameterExtractor:
    """
    Extracts parameters from JSON database and generates RAVEN model file sections
    Integrates with Step 5 to fix parameter extraction gaps
    """
    
    def __init__(self, json_database_path: str = None, output_dir: str = None):
        """
        Initialize the RAVEN parameter extractor
        
        Args:
            json_database_path: Path to raven_lookup_database.json
            output_dir: Output directory for generated parameter files
        """
        self.lookup_generator = RAVENLookupTableGenerator(json_database_path, output_dir)
        self.database = self.lookup_generator.database
        self.output_dir = self.lookup_generator.output_dir
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_soil_classes_section(self, hru_gdf: gpd.GeoDataFrame = None, 
                                    unique_soil_classes: List[str] = None) -> str:
        """
        Generate :SoilClasses section for RAVEN RVP file
        
        Args:
            hru_gdf: HRU GeoDataFrame with soil class information
            unique_soil_classes: List of unique soil classes (if HRU data not available)
            
        Returns:
            Formatted :SoilClasses section for RVP file
        """
        # Determine unique soil classes
        if hru_gdf is not None and 'SOIL_PROF' in hru_gdf.columns:
            unique_classes = hru_gdf['SOIL_PROF'].unique().tolist()
        elif unique_soil_classes is not None:
            unique_classes = unique_soil_classes
        else:
            # Use all soil classes from database
            unique_classes = list(self.database["soil_classification"]["soil_classes"].keys())
            # Remove lake soil for land HRUs
            unique_classes = [sc for sc in unique_classes if sc != 'LAKE_SOIL']
        
        # Get parameter extraction config
        soil_config = self.database["integration_utilities"]["raven_parameter_extraction"]["soil_parameters"]
        attributes = soil_config["attributes"]
        units = soil_config["units"]
        
        # Generate section header
        section_lines = [
            ":SoilClasses",
            f"  :Attributes {' '.join(attributes)}",
            f"  :Units      {' '.join(units)}"
        ]
        
        # Generate parameter lines for each soil class
        for soil_class in unique_classes:
            params = self.lookup_generator.get_soil_parameters_for_raven(soil_class)
            
            # Format parameter values
            param_values = []
            for attr in attributes:
                value = params.get(attr, 0.0)
                param_values.append(f"{value:.3f}")
            
            # Add soil class line
            section_lines.append(f"  {soil_class:<15} {' '.join(param_values)}")
        
        section_lines.append(":EndSoilClasses")
        
        self.logger.info(f"Generated :SoilClasses section with {len(unique_classes)} soil classes")
        return "\n".join(section_lines)
    
    def generate_vegetation_classes_section(self, hru_gdf: gpd.GeoDataFrame = None,
                                          unique_veg_classes: List[str] = None) -> str:
        """
        Generate :VegetationClasses section for RAVEN RVP file
        
        Args:
            hru_gdf: HRU GeoDataFrame with vegetation class information
            unique_veg_classes: List of unique vegetation classes
            
        Returns:
            Formatted :VegetationClasses section for RVP file
        """
        # Determine unique vegetation classes
        if hru_gdf is not None and 'VEG_C' in hru_gdf.columns:
            unique_classes = hru_gdf['VEG_C'].unique().tolist()
        elif unique_veg_classes is not None:
            unique_classes = unique_veg_classes
        else:
            # Map from landcover classes
            landcover_classes = list(self.database["landcover_classification"]["landcover_classes"].keys())
            veg_mapping = self.database["vegetation_classification"]["basinmaker_format"]["landcover_mapping"]
            unique_classes = [veg_mapping.get(lc, lc) for lc in landcover_classes if lc != 'LAKE']
        
        # Get parameter extraction config
        veg_config = self.database["integration_utilities"]["raven_parameter_extraction"]["vegetation_parameters"]
        attributes = veg_config["attributes"]
        units = veg_config["units"]
        
        # Generate section header
        section_lines = [
            ":VegetationClasses",
            f"  :Attributes {' '.join(attributes)}",
            f"  :Units      {' '.join(units)}"
        ]
        
        # Generate parameter lines for each vegetation class
        for veg_class in unique_classes:
            # Map vegetation class back to landcover class for parameter lookup
            landcover_class = self._map_veg_to_landcover_class(veg_class)
            params = self.lookup_generator.get_vegetation_parameters_for_raven(landcover_class)
            
            # Format parameter values
            param_values = []
            for attr in attributes:
                value = params.get(attr, 0.0)
                param_values.append(f"{value:.3f}")
            
            # Add vegetation class line
            section_lines.append(f"  {veg_class:<20} {' '.join(param_values)}")
        
        section_lines.append(":EndVegetationClasses")
        
        self.logger.info(f"Generated :VegetationClasses section with {len(unique_classes)} vegetation classes")
        return "\n".join(section_lines)
    
    def generate_landuse_classes_section(self, hru_gdf: gpd.GeoDataFrame = None,
                                       unique_landuse_classes: List[str] = None) -> str:
        """
        Generate :LandUseClasses section for RAVEN RVP file
        
        Args:
            hru_gdf: HRU GeoDataFrame with land use class information
            unique_landuse_classes: List of unique land use classes
            
        Returns:
            Formatted :LandUseClasses section for RVP file
        """
        # Determine unique land use classes
        if hru_gdf is not None and 'LAND_USE_C' in hru_gdf.columns:
            unique_classes = hru_gdf['LAND_USE_C'].unique().tolist()
        elif unique_landuse_classes is not None:
            unique_classes = unique_landuse_classes
        else:
            # Use all landcover classes from database
            unique_classes = list(self.database["landcover_classification"]["landcover_classes"].keys())
            unique_classes = [lc for lc in unique_classes if lc != 'LAKE']
        
        # Get parameter extraction config
        landuse_config = self.database["integration_utilities"]["raven_parameter_extraction"]["landuse_parameters"]
        attributes = landuse_config["attributes"]
        units = landuse_config["units"]
        
        # Generate section header
        section_lines = [
            ":LandUseClasses",
            f"  :Attributes {' '.join(attributes)}",
            f"  :Units      {' '.join(units)}"
        ]
        
        # Generate parameter lines for each land use class
        for landuse_class in unique_classes:
            params = self.lookup_generator.get_landuse_parameters_for_raven(landuse_class)
            
            # Format parameter values
            param_values = []
            for attr in attributes:
                value = params.get(attr, 0.0)
                param_values.append(f"{value:.3f}")
            
            # Add land use class line
            section_lines.append(f"  {landuse_class:<20} {' '.join(param_values)}")
        
        section_lines.append(":EndLandUseClasses")
        
        self.logger.info(f"Generated :LandUseClasses section with {len(unique_classes)} landuse classes")
        return "\n".join(section_lines)
    
    def generate_hru_section(self, hru_gdf: gpd.GeoDataFrame) -> str:
        """
        Generate :HRUs section for RAVEN RVH file with proper class assignments
        
        Args:
            hru_gdf: HRU GeoDataFrame with all required attributes
            
        Returns:
            Formatted :HRUs section for RVH file
        """
        if hru_gdf is None or len(hru_gdf) == 0:
            raise ValueError("HRU GeoDataFrame is required and cannot be empty")
        
        # Required columns for RAVEN HRUs
        required_cols = ['AREA', 'ELEVATION', 'LATITUDE', 'LONGITUDE', 'BASIN_ID']
        hru_class_cols = ['LAND_USE_CLASS', 'VEG_CLASS', 'SOIL_PROFILE']
        
        # Check if we have the required columns
        missing_cols = [col for col in required_cols if col not in hru_gdf.columns]
        if missing_cols:
            # Try alternative column names
            col_mapping = {
                'AREA': ['Area_km2', 'area', 'Area'],
                'ELEVATION': ['Elevation', 'elevation', 'ELEV'],
                'LATITUDE': ['Latitude', 'latitude', 'lat'],
                'LONGITUDE': ['Longitude', 'longitude', 'lon', 'long'],
                'BASIN_ID': ['SubId', 'BasinID', 'basin_id', 'subbasin_id']
            }
            
            for req_col in required_cols:
                if req_col not in hru_gdf.columns:
                    for alt_col in col_mapping.get(req_col, []):
                        if alt_col in hru_gdf.columns:
                            hru_gdf = hru_gdf.rename(columns={alt_col: req_col})
                            break
        
        # Generate section header
        section_lines = [
            ":HRUs",
            "  :Attributes AREA ELEVATION LATITUDE LONGITUDE BASIN_ID LAND_USE_CLASS VEG_CLASS SOIL_PROFILE",
            "  :Units      km2  masl      deg      deg       none     none           none      none"
        ]
        
        # Generate HRU lines
        for idx, hru in hru_gdf.iterrows():
            # Get basic attributes
            hru_id = idx + 1
            area = hru.get('AREA', hru.geometry.area / 1e6 if hasattr(hru, 'geometry') else 1.0)
            elevation = hru.get('ELEVATION', 1000.0)
            
            # Calculate centroid if lat/lon not available
            if 'LATITUDE' not in hru_gdf.columns or 'LONGITUDE' not in hru_gdf.columns:
                if hasattr(hru, 'geometry') and hru.geometry is not None:
                    # Reproject to WGS84 for lat/lon
                    hru_gdf_wgs84 = hru_gdf.to_crs('EPSG:4326')
                    centroid = hru_gdf_wgs84.geometry.iloc[idx].centroid
                    latitude = centroid.y
                    longitude = centroid.x
                else:
                    latitude = 49.0  # Default
                    longitude = -120.0
            else:
                latitude = hru.get('LATITUDE', 49.0)
                longitude = hru.get('LONGITUDE', -120.0)
            
            basin_id = hru.get('BASIN_ID', 1)
            
            # Get class assignments (use existing or assign from lookup)
            land_use_class = hru.get('LAND_USE_C', hru.get('LAND_USE_CLASS', 'GRASSLAND'))
            veg_class = hru.get('VEG_C', hru.get('VEG_CLASS', 'GRASSLAND_VEG'))
            soil_profile = hru.get('SOIL_PROF', hru.get('SOIL_PROFILE', 'LOAM'))
            
            # Format HRU line
            hru_line = f"  {hru_id:<3} {area:.2f} {elevation:.1f} {latitude:.6f} {longitude:.6f} {basin_id} {land_use_class} {veg_class} {soil_profile}"
            section_lines.append(hru_line)
        
        section_lines.append(":EndHRUs")
        
        self.logger.info(f"Generated :HRUs section with {len(hru_gdf)} HRUs")
        return "\n".join(section_lines)
    
    def _map_veg_to_landcover_class(self, veg_class: str) -> str:
        """Map vegetation class back to landcover class for parameter lookup"""
        veg_mapping = self.database["vegetation_classification"]["basinmaker_format"]["landcover_mapping"]
        
        # Reverse lookup - exact match first
        for landcover_class, veg_c in veg_mapping.items():
            if veg_c == veg_class:
                return landcover_class
        
        # Enhanced mapping for actual HRU vegetation class names
        veg_upper = veg_class.upper()
        
        # Map specific vegetation classes to landcover classes
        if veg_upper == 'CONIFEROUS':
            return 'FOREST_CONIFEROUS'
        elif veg_upper == 'DECIDUOUS':
            return 'FOREST_DECIDUOUS'
        elif veg_upper == 'MIXED_FOREST':
            return 'FOREST_MIXED'
        elif veg_upper == 'MIXED_SHRUBLAND' or 'SHRUB' in veg_upper:
            return 'SHRUBLAND'
        elif veg_upper == 'GRASSLAND' or 'GRASS' in veg_upper:
            return 'GRASSLAND'
        elif 'FOREST' in veg_upper or 'CONIFER' in veg_upper:
            return 'FOREST_CONIFEROUS'
        elif 'CROP' in veg_upper or 'AGRICULTURAL' in veg_upper:
            return 'AGRICULTURE'
        elif 'URBAN' in veg_upper:
            return 'URBAN'
        elif 'WATER' in veg_upper:
            return 'WATER'
        else:
            # Default fallback
            return 'GRASSLAND'
    
    def generate_complete_rvp_parameters(self, hru_gdf: gpd.GeoDataFrame, 
                                       output_file: str = None) -> str:
        """
        Generate complete RVP parameter sections for RAVEN model
        
        Args:
            hru_gdf: HRU GeoDataFrame with class assignments
            output_file: Optional output file path
            
        Returns:
            Complete RVP parameter sections as string
        """
        self.logger.info("Generating complete RVP parameter sections...")
        
        # Generate all parameter sections
        soil_section = self.generate_soil_classes_section(hru_gdf)
        vegetation_section = self.generate_vegetation_classes_section(hru_gdf)
        landuse_section = self.generate_landuse_classes_section(hru_gdf)
        
        # Combine sections
        rvp_content = "\n\n".join([
            "# Soil Classes Parameters (Generated from RAVEN Lookup Database)",
            soil_section,
            "",
            "# Vegetation Classes Parameters (Generated from RAVEN Lookup Database)", 
            vegetation_section,
            "",
            "# Land Use Classes Parameters (Generated from RAVEN Lookup Database)",
            landuse_section
        ])
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(output_path, 'w') as f:
                f.write(rvp_content)
            
            self.logger.info(f"Saved RVP parameters to: {output_file}")
        
        return rvp_content
    
    def generate_complete_rvh_hrus(self, hru_gdf: gpd.GeoDataFrame,
                                  output_file: str = None) -> str:
        """
        Generate complete RVH HRU section for RAVEN model
        
        Args:
            hru_gdf: HRU GeoDataFrame with all attributes
            output_file: Optional output file path
            
        Returns:
            Complete RVH HRU section as string
        """
        self.logger.info("Generating complete RVH HRU section...")
        
        # Generate HRU section
        hru_section = self.generate_hru_section(hru_gdf)
        
        # Add header comment
        rvh_content = "\n".join([
            "# HRU Definitions (Generated from RAVEN Lookup Database)",
            hru_section
        ])
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(output_path, 'w') as f:
                f.write(rvh_content)
            
            self.logger.info(f"Saved RVH HRUs to: {output_file}")
        
        return rvh_content
    
    def extract_seasonal_parameters(self, landcover_classes: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Extract seasonal parameter variations for given landcover classes
        
        Args:
            landcover_classes: List of landcover class names
            
        Returns:
            Dictionary with monthly parameter variations
        """
        seasonal_data = {}
        
        for lc_class in landcover_classes:
            if lc_class in self.database["landcover_classification"]["landcover_classes"]:
                seasonal_var = self.database["landcover_classification"]["landcover_classes"][lc_class].get("seasonal_variation", {})
                
                if seasonal_var:
                    seasonal_data[lc_class] = {
                        "lai_winter": seasonal_var.get("lai_winter", 1.0),
                        "lai_summer": seasonal_var.get("lai_summer", 3.0),
                        "canopy_cover_winter": seasonal_var.get("canopy_cover_winter", 0.3),
                        "canopy_cover_summer": seasonal_var.get("canopy_cover_summer", 0.5)
                    }
        
        self.logger.info(f"Extracted seasonal parameters for {len(seasonal_data)} landcover classes")
        return seasonal_data
    
    def validate_basinmaker_compatibility(self, hru_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Validate HRU data against BasinMaker requirements
        Based on BasinMaker's 26+ required field specifications
        
        Args:
            hru_gdf: HRU GeoDataFrame to validate
            
        Returns:
            BasinMaker compatibility validation results
        """
        validation_results = {
            "basinmaker_compatible": True,
            "warnings": [],
            "errors": [],
            "field_mapping": {},
            "statistics": {}
        }
        
        # BasinMaker required subbasin fields (from raveninput.py lines 53-95)
        required_subbasin_fields = [
            'SubId', 'DowSubId', 'RivLength', 'RivSlope', 
            'BkfWidth', 'BkfDepth', 'Ch_n', 'FloodP_n',
            'MeanElev', 'BasArea', 'Lake_Cat', 'Has_POI'
        ]
        
        # BasinMaker required HRU fields
        required_hru_fields = [
            'HRU_ID', 'HRU_Area', 'HRU_S_mean', 'HRU_A_mean', 
            'HRU_E_mean', 'HRU_IsLake', 'LAND_USE_C', 'VEG_C', 
            'SOIL_PROF', 'HRU_CenX', 'HRU_CenY'
        ]
        
        # Check for required subbasin fields
        missing_subbasin_fields = []
        for field in required_subbasin_fields:
            if field not in hru_gdf.columns:
                # Try common alternative names
                alternatives = self._get_basinmaker_field_alternatives(field)
                found_alternative = None
                for alt in alternatives:
                    if alt in hru_gdf.columns:
                        found_alternative = alt
                        validation_results['field_mapping'][field] = alt
                        break
                
                if not found_alternative:
                    missing_subbasin_fields.append(field)
        
        # Check for required HRU fields  
        missing_hru_fields = []
        for field in required_hru_fields:
            if field not in hru_gdf.columns:
                alternatives = self._get_basinmaker_field_alternatives(field)
                found_alternative = None
                for alt in alternatives:
                    if alt in hru_gdf.columns:
                        found_alternative = alt
                        validation_results['field_mapping'][field] = alt
                        break
                
                if not found_alternative:
                    missing_hru_fields.append(field)
        
        # Report missing critical fields
        if missing_subbasin_fields:
            validation_results['errors'].append(f"Missing critical subbasin fields: {missing_subbasin_fields}")
            validation_results['basinmaker_compatible'] = False
            
        if missing_hru_fields:
            validation_results['errors'].append(f"Missing critical HRU fields: {missing_hru_fields}")
            validation_results['basinmaker_compatible'] = False
        
        # Validate data ranges (BasinMaker constraints)
        if validation_results['basinmaker_compatible']:
            self._validate_basinmaker_data_ranges(hru_gdf, validation_results)
        
        # Calculate BasinMaker-specific statistics
        validation_results['statistics'] = {
            'total_hrus': len(hru_gdf),
            'total_subbasins': hru_gdf['SubId'].nunique() if 'SubId' in hru_gdf.columns else 0,
            'lake_hrus': len(hru_gdf[hru_gdf['HRU_IsLake'] == 1]) if 'HRU_IsLake' in hru_gdf.columns else 0,
            'gauged_subbasins': len(hru_gdf[hru_gdf.get('Has_POI', 0) > 0]) if 'Has_POI' in hru_gdf.columns else 0
        }
        
        return validation_results
    
    def _get_basinmaker_field_alternatives(self, field: str) -> List[str]:
        """Get alternative field names for BasinMaker compatibility"""
        alternatives = {
            'SubId': ['SUBID', 'Sub_Id', 'SubbasinId', 'Subbasin_ID'],
            'DowSubId': ['DowSub', 'DownstreamId', 'Downstream_ID', 'DOWSUBID'],
            'RivLength': ['Rivlen', 'River_Length', 'RIVLENGTH', 'StreamLength'],
            'RivSlope': ['River_Slope', 'RIVSLOPE', 'StreamSlope', 'Slope'],
            'BkfWidth': ['Bankfull_Width', 'BKFWIDTH', 'ChannelWidth'],
            'BkfDepth': ['Bankfull_Depth', 'BKFDEPTH', 'ChannelDepth'],
            'Ch_n': ['Channel_n', 'CHN', 'ManningN', 'Manning_n'],
            'FloodP_n': ['Floodplain_n', 'FLOODPN', 'FloodplainManning'],
            'HRU_Area': ['HRU_Area_k', 'Area', 'AREA', 'HRU_AREA'],
            'LAND_USE_C': ['LANDUSE', 'Land_Use', 'LU_Class', 'LAND_USE'],
            'VEG_C': ['Vegetation', 'VEG_CLASS', 'Veg_Class'],
            'SOIL_PROF': ['SOIL_PROFILE', 'Soil_Profile', 'SOIL_TYPE'],
            'HRU_S_mean': ['Slope_mean', 'SLOPE_MEAN', 'HRU_SLOPE'],
            'HRU_A_mean': ['Aspect_mean', 'ASPECT_MEAN', 'HRU_ASPECT'],
            'HRU_E_mean': ['Elevation_mean', 'ELEVATION_MEAN', 'HRU_ELEVATION'],
            'HRU_CenX': ['Centroid_X', 'CENTROID_X', 'HRU_X'],
            'HRU_CenY': ['Centroid_Y', 'CENTROID_Y', 'HRU_Y']
        }
        return alternatives.get(field, [])
    
    def _validate_basinmaker_data_ranges(self, hru_gdf: gpd.GeoDataFrame, 
                                       validation_results: Dict[str, Any]):
        """Validate data ranges following BasinMaker constraints"""
        
        # Check river lengths (must be positive)
        if 'RivLength' in hru_gdf.columns:
            negative_lengths = hru_gdf[hru_gdf['RivLength'] < 0]
            if len(negative_lengths) > 0:
                validation_results['warnings'].append(f"Found {len(negative_lengths)} subbasins with negative river lengths")
        
        # Check slopes (must be positive, BasinMaker minimum: 0.0001)
        if 'RivSlope' in hru_gdf.columns:
            zero_slopes = hru_gdf[hru_gdf['RivSlope'] <= 0]
            if len(zero_slopes) > 0:
                validation_results['warnings'].append(f"Found {len(zero_slopes)} subbasins with zero/negative slopes")
            
            min_slope_violations = hru_gdf[hru_gdf['RivSlope'] < 0.0001]
            if len(min_slope_violations) > 0:
                validation_results['warnings'].append(f"Found {len(min_slope_violations)} subbasins below BasinMaker minimum slope (0.0001)")
        
        # Check HRU areas (must be positive)
        if 'HRU_Area' in hru_gdf.columns:
            zero_areas = hru_gdf[hru_gdf['HRU_Area'] <= 0]
            if len(zero_areas) > 0:
                validation_results['errors'].append(f"Found {len(zero_areas)} HRUs with zero/negative areas")
                validation_results['basinmaker_compatible'] = False
        
        # Check Manning's n values (BasinMaker valid range: 0.025-0.15)
        if 'Ch_n' in hru_gdf.columns:
            invalid_manning = hru_gdf[(hru_gdf['Ch_n'] <= 0) | (hru_gdf['Ch_n'] > 0.15)]
            if len(invalid_manning) > 0:
                validation_results['warnings'].append(f"Found {len(invalid_manning)} subbasins with invalid Manning's n values")
        
        # Check bankfull dimensions (must be positive)
        if 'BkfWidth' in hru_gdf.columns:
            invalid_width = hru_gdf[hru_gdf['BkfWidth'] <= 0]
            if len(invalid_width) > 0:
                validation_results['warnings'].append(f"Found {len(invalid_width)} subbasins with invalid bankfull width")
        
        if 'BkfDepth' in hru_gdf.columns:
            invalid_depth = hru_gdf[hru_gdf['BkfDepth'] <= 0]
            if len(invalid_depth) > 0:
                validation_results['warnings'].append(f"Found {len(invalid_depth)} subbasins with invalid bankfull depth")
    
    def validate_parameters(self, hru_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Validate parameter consistency and ranges
        
        Args:
            hru_gdf: HRU GeoDataFrame to validate
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            "success": True,
            "warnings": [],
            "errors": [],
            "statistics": {}
        }
        
        # Get validation ranges from database
        validation_config = self.database["integration_utilities"]["validation"]
        soil_ranges = validation_config["parameter_range_validator"]["soil_ranges"]
        veg_ranges = validation_config["parameter_range_validator"]["vegetation_ranges"]
        
        # Check if required columns exist
        required_cols = ['SOIL_PROF', 'LAND_USE_C', 'VEG_C']
        missing_cols = [col for col in required_cols if col not in hru_gdf.columns]
        
        if missing_cols:
            validation_results["errors"].append(f"Missing required columns: {missing_cols}")
            validation_results["success"] = False
        
        # Validate soil classes exist in database
        if 'SOIL_PROF' in hru_gdf.columns:
            soil_classes = set(hru_gdf['SOIL_PROF'].unique())
            db_soil_classes = set(self.database["soil_classification"]["soil_classes"].keys())
            missing_soil = soil_classes - db_soil_classes
            
            if missing_soil:
                validation_results["warnings"].append(f"Soil classes not in database: {missing_soil}")
            
            validation_results["statistics"]["unique_soil_classes"] = len(soil_classes)
        
        # Validate landcover classes
        if 'LAND_USE_C' in hru_gdf.columns:
            landcover_classes = set(hru_gdf['LAND_USE_C'].unique())
            db_landcover_classes = set(self.database["landcover_classification"]["landcover_classes"].keys())
            missing_landcover = landcover_classes - db_landcover_classes
            
            if missing_landcover:
                validation_results["warnings"].append(f"Landcover classes not in database: {missing_landcover}")
            
            validation_results["statistics"]["unique_landcover_classes"] = len(landcover_classes)
        
        # Calculate basic statistics
        if 'AREA' in hru_gdf.columns or hasattr(hru_gdf, 'geometry'):
            if 'AREA' in hru_gdf.columns:
                areas = hru_gdf['AREA']
            else:
                areas = hru_gdf.geometry.area / 1e6  # Convert to km2
            
            validation_results["statistics"]["total_area_km2"] = areas.sum()
            validation_results["statistics"]["mean_hru_area_km2"] = areas.mean()
            validation_results["statistics"]["num_hrus"] = len(hru_gdf)
        
        # Add BasinMaker compatibility check
        basinmaker_validation = self.validate_basinmaker_compatibility(hru_gdf)
        validation_results["basinmaker_compatibility"] = basinmaker_validation
        
        # Merge warnings and errors
        validation_results["warnings"].extend(basinmaker_validation["warnings"])
        validation_results["errors"].extend(basinmaker_validation["errors"])
        
        if not basinmaker_validation["basinmaker_compatible"]:
            validation_results["success"] = False
        
        self.logger.info(f"Parameter validation completed: {validation_results['success']}")
        if validation_results["warnings"]:
            self.logger.warning(f"Validation warnings: {len(validation_results['warnings'])}")
        if validation_results["errors"]:
            self.logger.error(f"Validation errors: {len(validation_results['errors'])}")
        
        return validation_results
    
    def apply_basinmaker_parameter_constraints(self, hru_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Apply BasinMaker parameter constraints to ensure valid ranges
        Based on BasinMaker's parameter bounds and requirements
        
        Args:
            hru_gdf: HRU GeoDataFrame to constrain
            
        Returns:
            HRU GeoDataFrame with constrained parameters
        """
        hru_constrained = hru_gdf.copy()
        
        # Apply BasinMaker slope constraints (minimum 0.0001)
        if 'RivSlope' in hru_constrained.columns:
            min_slope = 0.0001
            hru_constrained['RivSlope'] = hru_constrained['RivSlope'].apply(
                lambda x: max(min_slope, x) if pd.notnull(x) and x > 0 else min_slope
            )
            self.logger.info(f"Applied minimum slope constraint: {min_slope}")
        
        # Apply Manning's n constraints (BasinMaker range: 0.025-0.15)
        if 'Ch_n' in hru_constrained.columns:
            min_manning = 0.025
            max_manning = 0.15
            default_manning = 0.035
            
            hru_constrained['Ch_n'] = hru_constrained['Ch_n'].apply(
                lambda x: min(max_manning, max(min_manning, x)) 
                if pd.notnull(x) and x > 0 else default_manning
            )
            self.logger.info(f"Applied Manning's n constraints: [{min_manning}, {max_manning}]")
        
        # Apply floodplain Manning's n constraints
        if 'FloodP_n' in hru_constrained.columns:
            min_flood_manning = 0.03
            max_flood_manning = 0.20
            default_flood_manning = 0.05
            
            hru_constrained['FloodP_n'] = hru_constrained['FloodP_n'].apply(
                lambda x: min(max_flood_manning, max(min_flood_manning, x)) 
                if pd.notnull(x) and x > 0 else default_flood_manning
            )
            self.logger.info(f"Applied floodplain Manning's n constraints: [{min_flood_manning}, {max_flood_manning}]")
        
        # Apply bankfull width constraints (must be positive)
        if 'BkfWidth' in hru_constrained.columns:
            min_width = 0.5  # Minimum 0.5m width
            hru_constrained['BkfWidth'] = hru_constrained['BkfWidth'].apply(
                lambda x: max(min_width, x) if pd.notnull(x) and x > 0 else min_width
            )
            self.logger.info(f"Applied minimum bankfull width constraint: {min_width}m")
        
        # Apply bankfull depth constraints (must be positive)
        if 'BkfDepth' in hru_constrained.columns:
            min_depth = 0.1  # Minimum 0.1m depth
            max_depth = 50.0  # Maximum 50m depth
            hru_constrained['BkfDepth'] = hru_constrained['BkfDepth'].apply(
                lambda x: min(max_depth, max(min_depth, x)) 
                if pd.notnull(x) and x > 0 else min_depth
            )
            self.logger.info(f"Applied bankfull depth constraints: [{min_depth}, {max_depth}]m")
        
        # Apply HRU area constraints (must be positive)
        if 'HRU_Area' in hru_constrained.columns:
            min_area = 0.0001  # Minimum 0.1m² area
            hru_constrained['HRU_Area'] = hru_constrained['HRU_Area'].apply(
                lambda x: max(min_area, x) if pd.notnull(x) and x > 0 else min_area
            )
            self.logger.info(f"Applied minimum HRU area constraint: {min_area}m²")
        
        # Apply slope constraints (degrees, 0-90)
        if 'HRU_S_mean' in hru_constrained.columns:
            min_slope_deg = 0.0
            max_slope_deg = 89.0  # Avoid 90 degrees (vertical)
            hru_constrained['HRU_S_mean'] = hru_constrained['HRU_S_mean'].apply(
                lambda x: min(max_slope_deg, max(min_slope_deg, x)) 
                if pd.notnull(x) else 5.0  # Default 5 degrees
            )
            self.logger.info(f"Applied slope constraints: [{min_slope_deg}, {max_slope_deg}] degrees")
        
        # Apply aspect constraints (degrees, 0-360)
        if 'HRU_A_mean' in hru_constrained.columns:
            hru_constrained['HRU_A_mean'] = hru_constrained['HRU_A_mean'].apply(
                lambda x: x % 360.0 if pd.notnull(x) else 180.0  # Default south-facing
            )
            self.logger.info("Applied aspect constraints: [0, 360] degrees")
        
        # Apply elevation constraints (reasonable range)
        if 'HRU_E_mean' in hru_constrained.columns:
            min_elevation = -500.0  # Below sea level (Death Valley)
            max_elevation = 9000.0  # High mountains
            hru_constrained['HRU_E_mean'] = hru_constrained['HRU_E_mean'].apply(
                lambda x: min(max_elevation, max(min_elevation, x)) 
                if pd.notnull(x) else 1000.0  # Default 1000m
            )
            self.logger.info(f"Applied elevation constraints: [{min_elevation}, {max_elevation}]m")
        
        self.logger.info("Applied BasinMaker parameter constraints to all fields")
        return hru_constrained
    
    def generate_basinmaker_compatible_parameters(self, hru_gdf: gpd.GeoDataFrame,
                                                 output_file: str = None) -> str:
        """
        Generate BasinMaker-compatible RVP parameters with proper constraints
        
        Args:
            hru_gdf: HRU GeoDataFrame (will be constrained)
            output_file: Optional output file path
            
        Returns:
            Complete RVP parameter sections with BasinMaker compatibility
        """
        self.logger.info("Generating BasinMaker-compatible RVP parameters...")
        
        # Apply BasinMaker constraints first
        constrained_hru_gdf = self.apply_basinmaker_parameter_constraints(hru_gdf)
        
        # Validate BasinMaker compatibility
        validation_results = self.validate_basinmaker_compatibility(constrained_hru_gdf)
        
        if not validation_results["basinmaker_compatible"]:
            self.logger.warning("HRU data not fully BasinMaker compatible despite constraints")
            for error in validation_results["errors"]:
                self.logger.error(f"BasinMaker compatibility error: {error}")
        
        # Generate standard RVP content with constrained data
        rvp_content = self.generate_complete_rvp_parameters(constrained_hru_gdf, output_file)
        
        # Add BasinMaker compatibility header
        basinmaker_header = "\n".join([
            "# BasinMaker-Compatible RAVEN Parameters",
            "# Generated with BasinMaker field validation and constraints",
            f"# Total HRUs: {validation_results['statistics']['total_hrus']}",
            f"# Total Subbasins: {validation_results['statistics']['total_subbasins']}",
            f"# Lake HRUs: {validation_results['statistics']['lake_hrus']}",
            f"# Gauged Subbasins: {validation_results['statistics']['gauged_subbasins']}",
            "# Parameters constrained to BasinMaker valid ranges",
            ""
        ])
        
        final_content = basinmaker_header + "\n" + rvp_content
        
        # Save with BasinMaker suffix if output requested
        if output_file:
            basinmaker_file = output_file.replace('.rvp', '_basinmaker.rvp')
            output_path = Path(basinmaker_file)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(output_path, 'w') as f:
                f.write(final_content)
            
            self.logger.info(f"Saved BasinMaker-compatible RVP parameters to: {basinmaker_file}")
        
        return final_content


def main():
    """Main function for testing the parameter extractor"""
    extractor = RAVENParameterExtractor()
    
    # Test parameter section generation (without actual HRU data)
    print("Testing parameter section generation...")
    
    # Test soil classes section
    soil_section = extractor.generate_soil_classes_section(unique_soil_classes=['SAND', 'LOAM', 'CLAY'])
    print("\nGenerated Soil Classes Section:")
    print(soil_section[:500] + "..." if len(soil_section) > 500 else soil_section)
    
    # Test vegetation classes section  
    veg_section = extractor.generate_vegetation_classes_section(unique_veg_classes=['CONIFEROUS_FOREST', 'AGRICULTURAL_CROPS', 'GRASSLAND_VEG'])
    print("\nGenerated Vegetation Classes Section:")
    print(veg_section[:500] + "..." if len(veg_section) > 500 else veg_section)
    
    # Test landuse classes section
    landuse_section = extractor.generate_landuse_classes_section(unique_landuse_classes=['FOREST_CONIFEROUS', 'AGRICULTURE', 'GRASSLAND'])
    print("\nGenerated LandUse Classes Section:")
    print(landuse_section[:500] + "..." if len(landuse_section) > 500 else landuse_section)
    
    print("\nRAVEN Parameter Extractor test completed successfully!")


if __name__ == "__main__":
    main()