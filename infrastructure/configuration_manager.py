"""
ConfigurationManager for parameterized and reproducible workflows.
"""

import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
from dataclasses import dataclass, asdict, field
import logging

from .path_manager import AbsolutePathManager, PathResolutionError, FileAccessError

logger = logging.getLogger(__name__)


@dataclass
class StepConfiguration:
    """Configuration for a single workflow step"""
    enabled: bool
    parameters: Dict[str, Any]
    inputs: Dict[str, str]  # Relative to workspace, resolved to absolute
    outputs: Dict[str, str]  # Relative to workspace, resolved to absolute
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StepConfiguration':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class BasinMakerConfig:
    """Configuration for BasinMaker integration parameters"""
    # Lake area thresholds (km²)
    thres_area_conn_lakes_km2: float = 0.5
    thres_area_non_conn_lakes_km2: float = 1.0
    
    # Selected lakes to force-keep regardless of thresholds
    selected_lake_list: List[int] = field(default_factory=list)
    
    # Stream processing parameters
    stream_initiation_threshold: float = 1.0
    minimum_hru_area_km2: float = 0.001
    
    # Climate data parameters
    climate_station_search_radius_km: float = 50.0
    climate_record_completeness_threshold: float = 0.8
    
    # Refinement options (Step 4.1)
    enable_refinement: bool = False
    merge_tiny_catchments: bool = False
    update_stream_order: bool = False
    update_routing_attributes: bool = False
    generate_routing_table: bool = False
    
    # Validation tolerances
    area_conservation_tolerance: float = 0.01  # 1% tolerance
    drainage_area_ratio_tolerance: float = 0.2  # 20% tolerance for observed vs delineated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BasinMakerConfig':
        """Create from dictionary"""
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate BasinMaker configuration parameters"""
        errors = []
        
        # Validate area thresholds
        if self.thres_area_conn_lakes_km2 < 0:
            errors.append("thres_area_conn_lakes_km2 cannot be negative")
        
        if self.thres_area_non_conn_lakes_km2 < 0:
            errors.append("thres_area_non_conn_lakes_km2 cannot be negative")
        
        if self.stream_initiation_threshold <= 0:
            errors.append("stream_initiation_threshold must be positive")
        
        if self.minimum_hru_area_km2 <= 0:
            errors.append("minimum_hru_area_km2 must be positive")
        
        if self.climate_station_search_radius_km <= 0:
            errors.append("climate_station_search_radius_km must be positive")
        
        if not (0 < self.climate_record_completeness_threshold <= 1):
            errors.append("climate_record_completeness_threshold must be between 0 and 1")
        
        if not (0 < self.area_conservation_tolerance <= 1):
            errors.append("area_conservation_tolerance must be between 0 and 1")
        
        if not (0 < self.drainage_area_ratio_tolerance <= 1):
            errors.append("drainage_area_ratio_tolerance must be between 0 and 1")
        
        # Validate selected lake list
        if self.selected_lake_list:
            for lake_id in self.selected_lake_list:
                if not isinstance(lake_id, int) or lake_id <= 0:
                    errors.append(f"Invalid lake ID in selected_lake_list: {lake_id}")
        
        return errors


@dataclass
class WorkflowConfiguration:
    """Complete workflow configuration"""
    workspace_root: str  # Absolute path
    steps: Dict[str, StepConfiguration]
    thresholds: Dict[str, float]
    paths: Dict[str, str]  # All absolute paths
    metadata: Dict[str, Any]  # Additional metadata
    basinmaker: BasinMakerConfig = field(default_factory=BasinMakerConfig)  # BasinMaker configuration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with step serialization"""
        data = asdict(self)
        data['steps'] = {name: step.to_dict() for name, step in self.steps.items()}
        data['basinmaker'] = self.basinmaker.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowConfiguration':
        """Create from dictionary with step deserialization"""
        data = data.copy()
        data['steps'] = {
            name: StepConfiguration.from_dict(step_data) 
            for name, step_data in data['steps'].items()
        }
        # Handle BasinMaker config (with backward compatibility)
        if 'basinmaker' in data:
            data['basinmaker'] = BasinMakerConfig.from_dict(data['basinmaker'])
        else:
            data['basinmaker'] = BasinMakerConfig()
        return cls(**data)


class ConfigurationManager:
    """
    Parameterized configuration management for reproducible workflows.
    Handles loading, saving, and validation of workflow configurations with absolute paths.
    """
    
    def __init__(self, path_manager: AbsolutePathManager, config_format: str = 'yaml'):
        """
        Initialize configuration manager.
        
        Args:
            path_manager: AbsolutePathManager instance for path operations
            config_format: Format for configuration files ('yaml' or 'json')
        """
        self.path_manager = path_manager
        self.config_format = config_format.lower()
        
        if self.config_format not in ['yaml', 'json']:
            raise ValueError("config_format must be 'yaml' or 'json'")
        
        logger.info(f"Initialized ConfigurationManager with format: {self.config_format}")
    
    def create_default_config(self) -> WorkflowConfiguration:
        """
        Create a default workflow configuration with common parameters.
        
        Returns:
            Default WorkflowConfiguration object
        """
        default_config = WorkflowConfiguration(
            workspace_root=str(self.path_manager.workspace_root),
            steps={
                'step1_data_preparation': StepConfiguration(
                    enabled=True,
                    parameters={
                        'buffer_km': 2.0,
                        'resolution_m': 30,
                        'dem_source': 'SRTM',
                        'landcover_source': 'ESA_WorldCover',
                        'soil_source': 'SoilGrids'
                    },
                    inputs={},
                    outputs={
                        'dem': 'data/dem.tif',
                        'landcover': 'data/landcover.tif',
                        'soil': 'data/soil.tif'
                    }
                ),
                'step2_watershed_delineation': StepConfiguration(
                    enabled=True,
                    parameters={
                        'minimum_drainage_area_km2': 1.0,
                        'max_snap_distance_m': 5000,
                        'stream_threshold': 5000
                    },
                    inputs={
                        'dem': 'data/dem.tif'
                    },
                    outputs={
                        'watershed': 'outputs/watershed.geojson',
                        'streams': 'outputs/streams.geojson',
                        'subbasins': 'outputs/subbasins.geojson'
                    }
                ),
                'step3_lake_processing': StepConfiguration(
                    enabled=True,
                    parameters={
                        'lake_min_area_km2': 0.01,
                        'lake_buffer_m': 100,
                        'integration_method': 'snap_to_stream'
                    },
                    inputs={
                        'streams': 'outputs/streams.geojson',
                        'subbasins': 'outputs/subbasins.geojson'
                    },
                    outputs={
                        'catchment_without_merging_lakes': 'outputs/catchment_without_merging_lakes.shp',
                        'river_without_merging_lakes': 'outputs/river_without_merging_lakes.shp',
                        'lakes_all': 'outputs/lakes_all.shp',
                        'poi': 'outputs/poi.shp'
                    }
                ),
                'step3_5_lake_connectivity': StepConfiguration(
                    enabled=True,
                    parameters={},
                    inputs={
                        'lakes_all': 'outputs/lakes_all.shp',
                        'rivers': 'outputs/river_without_merging_lakes.shp'
                    },
                    outputs={
                        'sl_connected_lake': 'outputs/sl_connected_lake.shp',
                        'sl_non_connected_lake': 'outputs/sl_non_connected_lake.shp'
                    }
                ),
                'step4_basinmaker_integration': StepConfiguration(
                    enabled=True,
                    parameters={},  # Uses basinmaker config section
                    inputs={
                        'catchment_without_merging_lakes': 'outputs/catchment_without_merging_lakes.shp',
                        'river_without_merging_lakes': 'outputs/river_without_merging_lakes.shp',
                        'sl_connected_lake': 'outputs/sl_connected_lake.shp',
                        'sl_non_connected_lake': 'outputs/sl_non_connected_lake.shp',
                        'poi': 'outputs/poi.shp'
                    },
                    outputs={
                        'finalcat_info': 'outputs/finalcat_info.shp',
                        'finalcat_info_riv': 'outputs/finalcat_info_riv.shp',
                        'routing_manifest': 'outputs/routing_manifest.json'
                    }
                ),
                'step4_1_basinmaker_refinement': StepConfiguration(
                    enabled=False,  # Optional step
                    parameters={},  # Uses basinmaker config section
                    inputs={
                        'finalcat_info': 'outputs/finalcat_info.shp',
                        'finalcat_info_riv': 'outputs/finalcat_info_riv.shp'
                    },
                    outputs={
                        'refined_finalcat_info': 'outputs/refined_finalcat_info.shp',
                        'refined_finalcat_info_riv': 'outputs/refined_finalcat_info_riv.shp',
                        'routing_table': 'outputs/routing_table.csv'
                    }
                ),
                'step5_hru_generation': StepConfiguration(
                    enabled=True,
                    parameters={
                        'subbasin_min_area_km2': 0.05,
                        'hru_discretization_method': 'dominant_landcover',
                        'min_hru_area_km2': 0.001
                    },
                    inputs={
                        'finalcat_info': 'outputs/finalcat_info.shp',
                        'finalcat_info_riv': 'outputs/finalcat_info_riv.shp',
                        'landcover': 'data/landcover.tif',
                        'vegetation': 'data/vegetation.tif',
                        'soil': 'data/soil.tif',
                        'terrain': 'data/terrain.tif'
                    },
                    outputs={
                        'final_hrus': 'outputs/final_hrus.geojson',
                        'hru_attributes': 'outputs/hru_attributes.csv'
                    }
                ),
                'step6_climate_data': StepConfiguration(
                    enabled=True,
                    parameters={
                        'start_date': '2000-01-01',
                        'end_date': '2020-12-31',
                        'data_sources': ['ERA5', 'station_data']
                    },
                    inputs={
                        'final_hrus': 'outputs/final_hrus.geojson'
                    },
                    outputs={
                        'climate_forcing': 'data/climate_forcing.csv',
                        'observed_flow': 'data/observed_flow.csv'
                    }
                ),
                'step7_raven_model': StepConfiguration(
                    enabled=True,
                    parameters={
                        'simulation_start_date': '2000-01-01',
                        'simulation_end_date': '2020-12-31',
                        'time_step_hours': 24,
                        'routing_method': 'ROUTE_DIFFUSIVE_WAVE'
                    },
                    inputs={
                        'final_hrus': 'outputs/final_hrus.geojson',
                        'hru_attributes': 'outputs/hru_attributes.csv',
                        'climate_forcing': 'data/climate_forcing.csv',
                        'observed_flow': 'data/observed_flow.csv'
                    },
                    outputs={
                        'raven_rvh': 'model/model.rvh',
                        'raven_rvp': 'model/model.rvp',
                        'raven_rvi': 'model/model.rvi',
                        'raven_rvt': 'model/model.rvt',
                        'simulation_outputs': 'outputs/simulation_outputs'
                    }
                )
            },
            thresholds={
                'lake_min_area_km2': 0.01,
                'stream_threshold': 5000,
                'subbasin_min_area_km2': 0.05,
                'min_hru_area_km2': 0.001,
                'max_snap_distance_m': 5000,
                'minimum_drainage_area_km2': 1.0
            },
            paths={
                'external_data_dir': '/path/to/external/data',
                'raven_executable': '/path/to/Raven.exe',
                'climate_data_dir': '/path/to/climate/data'
            },
            metadata={
                'created_date': datetime.now().isoformat(),
                'created_by': 'ConfigurationManager',
                'version': '1.0',
                'description': 'Default workflow configuration for hydrological modeling'
            }
        )
        
        return default_config
    
    def load_config(self, config_file: Union[str, Path]) -> WorkflowConfiguration:
        """
        Load workflow configuration from file with absolute path resolution.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            WorkflowConfiguration object with resolved absolute paths
            
        Raises:
            PathResolutionError: If config file path cannot be resolved
            FileAccessError: If config file cannot be read
        """
        abs_config_path = self.path_manager.resolve_path(config_file)
        self.path_manager.validate_path(abs_config_path, must_exist=True, must_be_file=True)
        
        try:
            if self.config_format == 'yaml':
                with open(abs_config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
            else:
                with open(abs_config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
            
            # Create configuration object
            config = WorkflowConfiguration.from_dict(config_dict)
            
            # Resolve all paths to absolute paths
            resolved_config = self.resolve_config_paths(config)
            
            logger.info(f"Loaded configuration from: {abs_config_path}")
            return resolved_config
            
        except (OSError, IOError, yaml.YAMLError, json.JSONDecodeError, KeyError, ValueError) as e:
            raise FileAccessError(str(abs_config_path), "read configuration", str(e))
    
    def save_config(self, config: WorkflowConfiguration, config_file: Union[str, Path]) -> Path:
        """
        Save configuration to file with absolute path references.
        
        Args:
            config: WorkflowConfiguration object to save
            config_file: Path to output configuration file
            
        Returns:
            Absolute path to saved configuration file
            
        Raises:
            PathResolutionError: If config file path cannot be resolved
            FileAccessError: If config file cannot be written
        """
        abs_config_path = self.path_manager.ensure_file_writable(config_file)
        
        # Update metadata
        config.metadata['last_modified'] = datetime.now().isoformat()
        
        try:
            config_dict = config.to_dict()
            
            if self.config_format == 'yaml':
                with open(abs_config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                with open(abs_config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved configuration to: {abs_config_path}")
            return abs_config_path
            
        except (OSError, IOError, yaml.YAMLError, json.JSONEncodeError) as e:
            raise FileAccessError(str(abs_config_path), "write configuration", str(e))
    
    def resolve_config_paths(self, config: WorkflowConfiguration) -> WorkflowConfiguration:
        """
        Convert all paths in configuration to absolute paths.
        
        Args:
            config: WorkflowConfiguration with potentially relative paths
            
        Returns:
            WorkflowConfiguration with all absolute paths
            
        Raises:
            PathResolutionError: If any path cannot be resolved
        """
        # Resolve workspace root
        config.workspace_root = str(self.path_manager.resolve_path(config.workspace_root))
        
        # Resolve paths in global paths dict
        resolved_paths = {}
        for key, path in config.paths.items():
            if path and path != '/path/to/external/data':  # Skip placeholder paths
                try:
                    resolved_paths[key] = str(self.path_manager.resolve_path(path))
                except PathResolutionError:
                    # Keep original path if resolution fails (might be external)
                    resolved_paths[key] = path
                    logger.warning(f"Could not resolve path for {key}: {path}")
            else:
                resolved_paths[key] = path
        config.paths = resolved_paths
        
        # Resolve paths in step configurations
        for step_name, step_config in config.steps.items():
            # Resolve input paths
            resolved_inputs = {}
            for key, path in step_config.inputs.items():
                if path:
                    resolved_inputs[key] = str(self.path_manager.resolve_path(path))
                else:
                    resolved_inputs[key] = path
            step_config.inputs = resolved_inputs
            
            # Resolve output paths
            resolved_outputs = {}
            for key, path in step_config.outputs.items():
                if path:
                    resolved_outputs[key] = str(self.path_manager.resolve_path(path))
                else:
                    resolved_outputs[key] = path
            step_config.outputs = resolved_outputs
        
        return config
    
    def validate_config(self, config: WorkflowConfiguration) -> List[str]:
        """
        Validate configuration for completeness and consistency.
        
        Args:
            config: WorkflowConfiguration to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check workspace root exists
        try:
            workspace_path = Path(config.workspace_root)
            if not workspace_path.exists():
                errors.append(f"Workspace root does not exist: {config.workspace_root}")
        except Exception as e:
            errors.append(f"Invalid workspace root path: {config.workspace_root} ({e})")
        
        # Validate step configurations
        for step_name, step_config in config.steps.items():
            if not isinstance(step_config.enabled, bool):
                errors.append(f"Step {step_name}: 'enabled' must be boolean")
            
            if not isinstance(step_config.parameters, dict):
                errors.append(f"Step {step_name}: 'parameters' must be dictionary")
            
            if not isinstance(step_config.inputs, dict):
                errors.append(f"Step {step_name}: 'inputs' must be dictionary")
            
            if not isinstance(step_config.outputs, dict):
                errors.append(f"Step {step_name}: 'outputs' must be dictionary")
            
            # Check for empty output paths
            for output_key, output_path in step_config.outputs.items():
                if not output_path:
                    errors.append(f"Step {step_name}: output '{output_key}' has empty path")
        
        # Validate thresholds
        for threshold_name, threshold_value in config.thresholds.items():
            if not isinstance(threshold_value, (int, float)):
                errors.append(f"Threshold '{threshold_name}' must be numeric")
            
            # Check for reasonable threshold values
            if threshold_name.endswith('_km2') and threshold_value < 0:
                errors.append(f"Area threshold '{threshold_name}' cannot be negative")
            
            if threshold_name.endswith('_m') and threshold_value < 0:
                errors.append(f"Distance threshold '{threshold_name}' cannot be negative")
        
        # Validate BasinMaker configuration
        basinmaker_errors = config.basinmaker.validate()
        errors.extend([f"BasinMaker config: {error}" for error in basinmaker_errors])
        
        # Check for step dependencies (with optional steps)
        step_dependencies = {
            'step2_watershed_delineation': ['step1_data_preparation'],
            'step3_lake_processing': ['step1_data_preparation', 'step2_watershed_delineation'],
            'step3_5_lake_connectivity': ['step3_lake_processing'],
            'step4_basinmaker_integration': ['step3_lake_processing', 'step3_5_lake_connectivity'],
            'step4_1_basinmaker_refinement': ['step4_basinmaker_integration'],  # Optional
            'step5_hru_generation': ['step4_basinmaker_integration'],  # Can work without refinement
            'step6_climate_data': ['step5_hru_generation'],
            'step7_raven_model': ['step5_hru_generation', 'step6_climate_data']
        }
        
        enabled_steps = [name for name, step in config.steps.items() if step.enabled]
        
        for step_name in enabled_steps:
            if step_name in step_dependencies:
                required_steps = step_dependencies[step_name]
                for required_step in required_steps:
                    if required_step not in enabled_steps:
                        errors.append(f"Step {step_name} requires {required_step} to be enabled")
        
        return errors
    
    def get_step_config(self, config: WorkflowConfiguration, step_name: str) -> Optional[StepConfiguration]:
        """
        Get configuration for a specific step.
        
        Args:
            config: WorkflowConfiguration object
            step_name: Name of the step
            
        Returns:
            StepConfiguration object or None if not found
        """
        return config.steps.get(step_name)
    
    def update_step_config(self, config: WorkflowConfiguration, step_name: str, 
                          updates: Dict[str, Any]) -> WorkflowConfiguration:
        """
        Update configuration for a specific step.
        
        Args:
            config: WorkflowConfiguration object
            step_name: Name of the step to update
            updates: Dictionary of updates to apply
            
        Returns:
            Updated WorkflowConfiguration object
        """
        if step_name not in config.steps:
            raise ValueError(f"Step '{step_name}' not found in configuration")
        
        step_config = config.steps[step_name]
        
        # Apply updates
        for key, value in updates.items():
            if key == 'parameters':
                step_config.parameters.update(value)
            elif key == 'inputs':
                step_config.inputs.update(value)
            elif key == 'outputs':
                step_config.outputs.update(value)
            elif hasattr(step_config, key):
                setattr(step_config, key, value)
            else:
                logger.warning(f"Unknown step configuration key: {key}")
        
        return config
    
    def create_config_template(self, output_file: Union[str, Path]) -> Path:
        """
        Create a configuration template file with comments and examples.
        
        Args:
            output_file: Path to output template file
            
        Returns:
            Path to created template file
        """
        abs_output_path = self.path_manager.ensure_file_writable(output_file)
        
        template_content = """# Workflow Configuration Template
# This file defines the complete configuration for the hydrological modeling workflow

# Workspace root directory (absolute path)
workspace_root: "/absolute/path/to/workspace"

# Step configurations
steps:
  step1_data_preparation:
    enabled: true
    parameters:
      buffer_km: 2.0              # Buffer around outlet point (km)
      resolution_m: 30            # DEM resolution (meters)
      dem_source: "SRTM"          # DEM data source
      landcover_source: "ESA_WorldCover"  # Land cover data source
      soil_source: "SoilGrids"    # Soil data source
    inputs: {}
    outputs:
      dem: "data/dem.tif"
      landcover: "data/landcover.tif"
      soil: "data/soil.tif"

  step2_watershed_delineation:
    enabled: true
    parameters:
      minimum_drainage_area_km2: 1.0    # Minimum drainage area (km²)
      max_snap_distance_m: 5000         # Maximum snap distance (meters)
      stream_threshold: 5000             # Stream initiation threshold
    inputs:
      dem: "data/dem.tif"
    outputs:
      watershed: "outputs/watershed.geojson"
      streams: "outputs/streams.geojson"
      subbasins: "outputs/subbasins.geojson"

  step3_lake_processing:
    enabled: true
    parameters:
      lake_min_area_km2: 0.01           # Minimum lake area (km²)
      lake_buffer_m: 100                # Lake buffer distance (meters)
      integration_method: "snap_to_stream"  # Lake-stream integration method
    inputs:
      streams: "outputs/streams.geojson"
      subbasins: "outputs/subbasins.geojson"
    outputs:
      lakes: "outputs/lakes.geojson"
      integrated_streams: "outputs/streams_with_lakes.geojson"

  step4_hru_generation:
    enabled: true
    parameters:
      subbasin_min_area_km2: 0.05       # Minimum subbasin area (km²)
      hru_discretization_method: "dominant_landcover"  # HRU discretization method
      min_hru_area_km2: 0.001           # Minimum HRU area (km²)
    inputs:
      subbasins: "outputs/subbasins.geojson"
      landcover: "data/landcover.tif"
      soil: "data/soil.tif"
    outputs:
      hrus: "outputs/hrus.geojson"
      hru_attributes: "outputs/hru_attributes.csv"

  step5_raven_model:
    enabled: true
    parameters:
      simulation_start_date: "2000-01-01"    # Simulation start date
      simulation_end_date: "2020-12-31"      # Simulation end date
      time_step_hours: 24                     # Model time step (hours)
      routing_method: "ROUTE_DIFFUSIVE_WAVE"  # Routing method
    inputs:
      hrus: "outputs/hrus.geojson"
      hru_attributes: "outputs/hru_attributes.csv"
      streams: "outputs/streams_with_lakes.geojson"
    outputs:
      raven_rvh: "model/model.rvh"
      raven_rvp: "model/model.rvp"
      raven_rvi: "model/model.rvi"
      raven_rvt: "model/model.rvt"
      routing_table: "outputs/routing_table.csv"

# Global thresholds and parameters
thresholds:
  lake_min_area_km2: 0.01
  stream_threshold: 5000
  subbasin_min_area_km2: 0.05
  min_hru_area_km2: 0.001
  max_snap_distance_m: 5000
  minimum_drainage_area_km2: 1.0

# External paths (all should be absolute paths)
paths:
  external_data_dir: "/absolute/path/to/external/data"
  raven_executable: "/absolute/path/to/Raven.exe"
  climate_data_dir: "/absolute/path/to/climate/data"

# BasinMaker integration configuration
basinmaker:
  # Lake area thresholds (km²)
  thres_area_conn_lakes_km2: 0.5        # Connected lakes threshold
  thres_area_non_conn_lakes_km2: 1.0    # Non-connected lakes threshold
  
  # Selected lakes to force-keep (list of HyLakeId values)
  selected_lake_list: []
  
  # Stream processing parameters
  stream_initiation_threshold: 1.0       # Stream initiation threshold
  minimum_hru_area_km2: 0.001           # Minimum HRU area (km²)
  
  # Climate data parameters
  climate_station_search_radius_km: 50.0      # Climate station search radius (km)
  climate_record_completeness_threshold: 0.8  # Minimum data completeness (0-1)
  
  # Refinement options (Step 4.1)
  enable_refinement: false               # Enable optional refinement step
  merge_tiny_catchments: false          # Merge tiny subbasins
  update_stream_order: false            # Update stream order
  update_routing_attributes: false      # Update routing attributes
  generate_routing_table: false         # Generate routing table CSV
  
  # Validation tolerances
  area_conservation_tolerance: 0.01      # Area conservation tolerance (1%)
  drainage_area_ratio_tolerance: 0.2     # Drainage area ratio tolerance (20%)

# Metadata
metadata:
  created_date: "2025-01-01T00:00:00"
  created_by: "ConfigurationManager"
  version: "1.0"
  description: "Workflow configuration for hydrological modeling"
"""
        
        try:
            with open(abs_output_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
            
            logger.info(f"Created configuration template: {abs_output_path}")
            return abs_output_path
            
        except (OSError, IOError) as e:
            raise FileAccessError(str(abs_output_path), "write configuration template", str(e))