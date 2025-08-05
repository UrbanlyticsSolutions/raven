"""
RAVEN Generation Steps for RAVEN Workflows

This module contains steps for generating RAVEN model files.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from workflows.steps.base_step import WorkflowStep

class GenerateRAVENModelFiles(WorkflowStep):
    """
    Step 4A: Generate complete RAVEN model files
    Used in Approach A (Routing Product Workflow)
    """
    
    def __init__(self):
        super().__init__(
            step_name="generate_raven_files",
            step_category="raven_generation",
            description="Generate complete 5-file RAVEN model from routing product data"
        )
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._log_step_start()
        
        try:
            # Validate required inputs
            required_inputs = ['final_hrus']
            self.validate_inputs(inputs, required_inputs)
            
            final_hrus = self.validate_file_exists(inputs['final_hrus'])
            
            # Create workspace
            workspace_dir = inputs.get('workspace_dir', final_hrus.parent)
            workspace = Path(workspace_dir)
            
            # Step 1: Select model type
            self.logger.info("Selecting RAVEN model type...")
            model_info = self._select_model_type(inputs)
            
            # Step 2: Generate all RAVEN files
            self.logger.info("Generating RAVEN model files...")
            raven_files = self._generate_all_raven_files(final_hrus, model_info, workspace)
            
            outputs = {
                'rvh_file': raven_files['rvh_file'],
                'rvp_file': raven_files['rvp_file'],
                'rvi_file': raven_files['rvi_file'],
                'rvt_file': raven_files['rvt_file'],
                'rvc_file': raven_files['rvc_file'],
                'selected_model': model_info['model_type'],
                'model_description': model_info['description'],
                'model_files_count': 5,
                'success': True
            }
            
            created_files = [raven_files[key] for key in ['rvh_file', 'rvp_file', 'rvi_file', 'rvt_file', 'rvc_file']]
            self._log_step_complete(created_files)
            return outputs
            
        except Exception as e:
            error_msg = f"RAVEN model file generation failed: {str(e)}"
            self._log_step_failed(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _select_model_type(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Select appropriate RAVEN model type"""
        
        watershed_area_km2 = inputs.get('total_area_km2', 100)
        has_lakes = inputs.get('lake_hru_count', 0) > 0
        
        # Model selection logic
        if watershed_area_km2 < 100:
            model_type = "GR4JCN"
            description = "Simple conceptual model for small watersheds"
        elif has_lakes:
            model_type = "HBVEC"
            description = "HBV-EC model with lake routing capabilities"
        else:
            model_type = "HMETS"
            description = "HMETS model optimized for cold regions"
        
        return {
            'model_type': model_type,
            'description': description
        }
    
    def _generate_all_raven_files(self, hru_file: Path, model_info: Dict[str, str], 
                                 workspace: Path) -> Dict[str, str]:
        """Generate all 5 RAVEN model files"""
        
        import geopandas as gpd
        
        # Load HRU data
        hru_gdf = gpd.read_file(hru_file)
        
        # Handle shapefile column name truncation (10-char limit)
        column_mapping = {
            'landuse_cl': 'landuse_class',
            'subbasin_i': 'subbasin_id',
            'elevation_': 'elevation_m',
            'slope_perc': 'slope_percent',
            'vegetation': 'vegetation_class'
        }
        
        # Rename truncated columns
        for old_name, new_name in column_mapping.items():
            if old_name in hru_gdf.columns and new_name not in hru_gdf.columns:
                hru_gdf = hru_gdf.rename(columns={old_name: new_name})
        
        # Check for required columns and add defaults if missing
        required_columns = ['landuse_class', 'soil_class', 'hru_type', 'area_km2']
        for col in required_columns:
            if col not in hru_gdf.columns:
                self.logger.warning(f"Missing column {col}, adding default values")
                if col == 'landuse_class':
                    hru_gdf[col] = 'MIXED'
                elif col == 'soil_class':
                    hru_gdf[col] = 'LOAM'
                elif col == 'hru_type':
                    hru_gdf[col] = 'LAND'
                elif col == 'area_km2':
                    hru_gdf[col] = 25.0
        
        # Generate each file
        files = {}
        
        # RVH file (spatial structure)
        files['rvh_file'] = str(self._generate_rvh_file(hru_gdf, workspace))
        
        # RVP file (parameters)
        files['rvp_file'] = str(self._generate_rvp_file(hru_gdf, model_info, workspace))
        
        # RVI file (instructions)
        files['rvi_file'] = str(self._generate_rvi_file(model_info, workspace))
        
        # RVT file (time series template)
        files['rvt_file'] = str(self._generate_rvt_file(hru_gdf, workspace))
        
        # RVC file (initial conditions)
        files['rvc_file'] = str(self._generate_rvc_file(hru_gdf, workspace))
        
        return files
    
    def _generate_rvh_file(self, hru_gdf, workspace: Path) -> Path:
        """Generate RVH file (spatial structure)"""
        
        rvh_file = workspace / "model.rvh"
        
        # Get unique subbasins - handle missing subbasin_id column
        if 'subbasin_id' in hru_gdf.columns:
            subbasins = hru_gdf[hru_gdf['hru_type'] == 'LAND']['subbasin_id'].unique()
        else:
            # Create default subbasin IDs if missing
            land_hrus = hru_gdf[hru_gdf['hru_type'] == 'LAND']
            subbasins = list(range(1, len(land_hrus) + 1))
        
        content = [
            "# RAVEN Watershed Structure File",
            "# Generated from routing product workflow",
            "",
            ":SubBasins",
            "#   SubID, Name, Area, DownstreamID, Profile, Terrain"
        ]
        
        # Add subbasins
        for i, subbasin_id in enumerate(subbasins):
            if 'subbasin_id' in hru_gdf.columns:
                subbasin_hrus = hru_gdf[hru_gdf['subbasin_id'] == subbasin_id]
            else:
                # Use index-based selection if subbasin_id column doesn't exist
                land_hrus = hru_gdf[hru_gdf['hru_type'] == 'LAND']
                subbasin_hrus = land_hrus.iloc[i:i+1] if i < len(land_hrus) else land_hrus.iloc[-1:]
            
            area = subbasin_hrus['area_km2'].sum()
            downstream_id = 0 if i == len(subbasins) - 1 else subbasins[i + 1]
            
            content.append(f"  {subbasin_id}, SubBasin_{subbasin_id}, {area:.3f}, {downstream_id}, DEFAULT_P, DEFAULT_T")
        
        content.extend([
            ":EndSubBasins",
            "",
            ":HRUs",
            "#   HRUID, SubID, Area, Elevation, Slope, Aspect, LandUse, Soil, Aquifer"
        ])
        
        # Add HRUs
        for idx, hru in hru_gdf.iterrows():
            hru_id = idx + 1 if hru['hru_type'] == 'LAND' else -1
            subbasin_id = hru.get('subbasin_id', 1)  # Default to 1 if missing
            content.append(
                f"  {hru_id}, {subbasin_id}, {hru['area_km2']:.3f}, "
                f"{hru.get('elevation_m', 500):.1f}, {hru.get('slope_percent', 5):.3f}, "
                f"180, {hru['landuse_class']}, {hru['soil_class']}, AQUIFER_1"
            )
        
        content.append(":EndHRUs")
        
        rvh_file.write_text("\n".join(content))
        return rvh_file
    
    def _generate_rvp_file(self, hru_gdf, model_info: Dict[str, str], workspace: Path) -> Path:
        """Generate RVP file (parameters)"""
        
        rvp_file = workspace / "model.rvp"
        
        # Get unique classes
        landuse_classes = hru_gdf['landuse_class'].unique()
        soil_classes = hru_gdf['soil_class'].unique()
        
        content = [
            "# RAVEN Parameters File",
            f"# Generated for {model_info['model_type']} model",
            "",
            ":LandUseClasses",
            "#   Name, ForestFrac, ImperviousFrac, LAImax, LAImin"
        ]
        
        # Add land use classes
        for landuse in landuse_classes:
            if landuse == 'WATER':
                content.append("  WATER, 0.00, 0.00, 0.0, 0.0")
            elif landuse == 'FOREST':
                content.append("  FOREST, 0.95, 0.05, 6.0, 1.0")
            else:
                content.append(f"  {landuse}, 0.80, 0.10, 4.0, 1.5")
        
        content.extend([
            ":EndLandUseClasses",
            "",
            ":SoilProfiles",
            "#   Name, NumLayers, Thickness1, Thickness2, Thickness3"
        ])
        
        # Add soil profiles
        for soil in soil_classes:
            if soil == 'WATER':
                content.append("  WATER, 1, 1000.0")
            elif soil == 'LOAM':
                content.append("  LOAM, 3, 0.1, 0.5, 2.0")
            else:
                content.append(f"  {soil}, 3, 0.1, 0.4, 1.5")
        
        content.append(":EndSoilProfiles")
        
        rvp_file.write_text("\n".join(content))
        return rvp_file
    
    def _generate_rvi_file(self, model_info: Dict[str, str], workspace: Path) -> Path:
        """Generate RVI file (model instructions)"""
        
        rvi_file = workspace / "model.rvi"
        
        content = [
            "# RAVEN Instructions File",
            f"# {model_info['description']}",
            "",
            ":SimulationPeriod 2020-01-01 2022-12-31",
            ":TimeStep 1.0",
            ":Method ORDERED_SERIES",
            "",
            ":Evaporation PET_PENMAN_MONTEITH",
            ":RainSnowFraction RAINSNOW_DINGMAN",
            ":PotentialMeltMethod POTMELT_DEGREE_DAY",
            "",
            ":HydrologicProcesses"
        ]
        
        # Add processes based on model type
        if model_info['model_type'] == 'HMETS':
            content.extend([
                "  :SnowBalance SNOBAL_SIMPLE_MELT SNOW PONDED_WATER",
                "  :Precipitation PRECIP_RAVEN ATMOS_PRECIP MULTIPLE",
                "  :Infiltration INF_GREEN_AMPT PONDED_WATER SOIL[0]",
                "  :Baseflow BASE_LINEAR SOIL[2] SURFACE_WATER"
            ])
        else:
            content.extend([
                "  :Precipitation PRECIP_RAVEN ATMOS_PRECIP MULTIPLE",
                "  :Infiltration INF_GREEN_AMPT PONDED_WATER SOIL[0]",
                "  :Baseflow BASE_LINEAR SOIL[1] SURFACE_WATER"
            ])
        
        content.append(":EndHydrologicProcesses")
        
        rvi_file.write_text("\n".join(content))
        return rvi_file
    
    def _generate_rvt_file(self, hru_gdf, workspace: Path) -> Path:
        """Generate RVT file (time series template)"""
        
        rvt_file = workspace / "model.rvt"
        
        # Get unique subbasins for climate stations - handle missing subbasin_id
        if 'subbasin_id' in hru_gdf.columns:
            subbasins = hru_gdf[hru_gdf['hru_type'] == 'LAND']['subbasin_id'].unique()
        else:
            land_hrus = hru_gdf[hru_gdf['hru_type'] == 'LAND']
            subbasins = list(range(1, len(land_hrus) + 1))
        
        content = [
            "# RAVEN Time Series File",
            "# Climate data template",
            ""
        ]
        
        # Add gauge for each subbasin
        for subbasin_id in subbasins:
            content.extend([
                f":Gauge Station_{subbasin_id} PRECIP TEMP_AVE",
                f"  :Latitude 45.5",
                f"  :Longitude -73.5",
                f"  :Elevation 200.0",
                f"  # Add climate data here",
                f"  # Format: YYYY-MM-DD HH:MM:SS PRECIP(mm/day) TEMP(°C)",
                f":EndGauge",
                ""
            ])
        
        rvt_file.write_text("\n".join(content))
        return rvt_file
    
    def _generate_rvc_file(self, hru_gdf, workspace: Path) -> Path:
        """Generate RVC file (initial conditions)"""
        
        rvc_file = workspace / "model.rvc"
        
        # Get unique subbasins - handle missing subbasin_id
        if 'subbasin_id' in hru_gdf.columns:
            subbasins = hru_gdf[hru_gdf['hru_type'] == 'LAND']['subbasin_id'].unique()
        else:
            land_hrus = hru_gdf[hru_gdf['hru_type'] == 'LAND']
            subbasins = list(range(1, len(land_hrus) + 1))
        
        content = [
            "# RAVEN Initial Conditions File",
            "# Generated automatically - adjust as needed",
            "",
            ":BasinInitialConditions",
            "#   SubID, Snow(mm), Soil1(mm), Soil2(mm), Soil3(mm), GW1(mm), GW2(mm)"
        ]
        
        # Add initial conditions for each subbasin
        for subbasin_id in subbasins:
            content.append(f"  {subbasin_id}, 0.0, 50.0, 100.0, 50.0, 200.0, 500.0")
        
        content.extend([
            ":EndBasinInitialConditions",
            "",
            "# Optional lake initial conditions",
            "# :LakeInitialConditions",
            "# :EndLakeInitialConditions"
        ])
        
        rvc_file.write_text("\n".join(content))
        return rvc_file


class SelectModelAndGenerateStructure(WorkflowStep):
    """
    Step 6B: Select model and generate structure files
    Used in Approach B (Full Delineation Workflow)
    """
    
    def __init__(self):
        super().__init__(
            step_name="select_model_structure",
            step_category="raven_generation",
            description="Select RAVEN model type and generate spatial structure files"
        )
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._log_step_start()
        
        try:
            # Validate required inputs
            required_inputs = ['final_hrus', 'sub_basins']
            self.validate_inputs(inputs, required_inputs)
            
            final_hrus = self.validate_file_exists(inputs['final_hrus'])
            sub_basins = self.validate_file_exists(inputs['sub_basins'])
            
            # Create workspace
            workspace_dir = inputs.get('workspace_dir', final_hrus.parent)
            workspace = Path(workspace_dir)
            
            # Select model type
            model_info = self._select_model_type(inputs)
            
            # Generate structure files
            structure_files = self._generate_structure_files(final_hrus, sub_basins, model_info, workspace)
            
            outputs = {
                'selected_model': model_info['model_type'],
                'model_description': model_info['description'],
                'rvh_file': structure_files['rvh_file'],
                'rvp_file': structure_files['rvp_file'],
                'hru_count': structure_files['hru_count'],
                'subbasin_count': structure_files['subbasin_count'],
                'parameter_count': structure_files['parameter_count'],
                'success': True
            }
            
            self._log_step_complete([structure_files['rvh_file'], structure_files['rvp_file']])
            return outputs
            
        except Exception as e:
            error_msg = f"Model selection and structure generation failed: {str(e)}"
            self._log_step_failed(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _select_model_type(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Select appropriate RAVEN model type (same logic as GenerateRAVENModelFiles)"""
        
        watershed_area_km2 = inputs.get('total_area_km2', 100)
        has_lakes = inputs.get('lake_hru_count', 0) > 0
        
        if watershed_area_km2 < 100:
            model_type = "GR4JCN"
            description = "Simple conceptual model for small watersheds"
        elif has_lakes:
            model_type = "HBVEC"
            description = "HBV-EC model with lake routing capabilities"
        else:
            model_type = "HMETS"
            description = "HMETS model optimized for cold regions"
        
        return {
            'model_type': model_type,
            'description': description
        }
    
    def _generate_structure_files(self, hru_file: Path, subbasins_file: Path, 
                                 model_info: Dict[str, str], workspace: Path) -> Dict[str, Any]:
        """Generate RVH and RVP files"""
        
        import geopandas as gpd
        
        # Load data
        hru_gdf = gpd.read_file(hru_file)
        subbasins_gdf = gpd.read_file(subbasins_file)
        
        # Generate RVH file (reuse logic from GenerateRAVENModelFiles)
        generator = GenerateRAVENModelFiles()
        rvh_file = generator._generate_rvh_file(hru_gdf, workspace)
        rvp_file = generator._generate_rvp_file(hru_gdf, model_info, workspace)
        
        return {
            'rvh_file': str(rvh_file),
            'rvp_file': str(rvp_file),
            'hru_count': len(hru_gdf),
            'subbasin_count': len(subbasins_gdf),
            'parameter_count': len(hru_gdf['landuse_class'].unique()) + len(hru_gdf['soil_class'].unique())
        }


class GenerateModelInstructions(WorkflowStep):
    """
    Step 7B: Generate model instruction files
    Used in Approach B (Full Delineation Workflow)
    """
    
    def __init__(self):
        super().__init__(
            step_name="generate_model_instructions",
            step_category="raven_generation",
            description="Generate RAVEN model execution and climate data files"
        )
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._log_step_start()
        
        try:
            # Validate required inputs
            required_inputs = ['selected_model', 'final_hrus']
            self.validate_inputs(inputs, required_inputs)
            
            selected_model = inputs['selected_model']
            final_hrus = self.validate_file_exists(inputs['final_hrus'])
            
            # Create workspace
            workspace_dir = inputs.get('workspace_dir', final_hrus.parent)
            workspace = Path(workspace_dir)
            
            # Generate instruction files
            instruction_files = self._generate_instruction_files(selected_model, final_hrus, workspace)
            
            outputs = {
                'rvi_file': instruction_files['rvi_file'],
                'rvt_file': instruction_files['rvt_file'],
                'rvc_file': instruction_files['rvc_file'],
                'simulation_period': '2020-01-01 to 2022-12-31',
                'time_step': '1.0 day',
                'climate_stations': instruction_files['climate_stations'],
                'process_count': instruction_files['process_count'],
                'success': True
            }
            
            created_files = [instruction_files['rvi_file'], instruction_files['rvt_file'], instruction_files['rvc_file']]
            self._log_step_complete(created_files)
            return outputs
            
        except Exception as e:
            error_msg = f"Model instruction generation failed: {str(e)}"
            self._log_step_failed(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _generate_instruction_files(self, selected_model: str, hru_file: Path, 
                                  workspace: Path) -> Dict[str, Any]:
        """Generate RVI, RVT, and RVC files"""
        
        import geopandas as gpd
        
        # Load HRU data
        hru_gdf = gpd.read_file(hru_file)
        
        # Generate files (reuse logic from GenerateRAVENModelFiles)
        generator = GenerateRAVENModelFiles()
        model_info = {'model_type': selected_model, 'description': f'{selected_model} model'}
        
        rvi_file = generator._generate_rvi_file(model_info, workspace)
        rvt_file = generator._generate_rvt_file(hru_gdf, workspace)
        rvc_file = generator._generate_rvc_file(hru_gdf, workspace)
        
        # Count climate stations (unique subbasins) - handle missing subbasin_id
        if 'subbasin_id' in hru_gdf.columns:
            climate_stations = len(hru_gdf[hru_gdf['hru_type'] == 'LAND']['subbasin_id'].unique())
        else:
            climate_stations = len(hru_gdf[hru_gdf['hru_type'] == 'LAND'])
        
        return {
            'rvi_file': str(rvi_file),
            'rvt_file': str(rvt_file),
            'rvc_file': str(rvc_file),
            'climate_stations': climate_stations,
            'process_count': 4  # Typical number of processes
        }