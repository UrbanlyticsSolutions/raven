#!/usr/bin/env python3
"""
RAVEN Executable Utility Class
==============================

A utility class for directly executing RAVEN models, including UBC Watershed Model,
bypassing RavenPy limitations and providing direct access to RAVEN's full capabilities.

This class is based on research findings that show RAVEN core supports UBCWM emulation
but RavenPy doesn't expose it as a high-level interface.

Author: Research-based implementation
Date: 2025-08-01
"""

import os
import sys
import subprocess
import tempfile
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RavenConfig:
    """Configuration class for RAVEN model setup."""
    model_name: str
    start_date: str
    end_date: str
    timestep: float = 1.0
    method: str = "ORDERED_SERIES"
    output_dir: str = "output"
    evaluation_metrics: List[str] = field(default_factory=lambda: ["NASH_SUTCLIFFE", "PCT_BIAS"])
    custom_outputs: List[Dict[str, str]] = field(default_factory=list)
    suppress_output: bool = False
    write_forcing_functions: bool = False


@dataclass
class RavenResults:
    """Results from a RAVEN simulation."""
    success: bool
    returncode: int
    stdout: str
    stderr: str
    workspace: str
    config_files: Dict[str, str]
    output_files: List[str]
    execution_time: float
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'success': self.success,
            'returncode': self.returncode,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'workspace': self.workspace,
            'config_files': self.config_files,
            'output_files': self.output_files,
            'execution_time': self.execution_time,
            'error_message': self.error_message
        }


class RavenExecutor:
    """
    A utility class for executing RAVEN models directly.
    
    This class provides a high-level interface for setting up and running
    RAVEN simulations, including specialized support for UBCWM and other
    emulated models.
    """
    
    def __init__(self, 
                 raven_exe_path: Optional[str] = None,
                 workspace_dir: Optional[str] = None,
                 cleanup_on_exit: bool = True):
        """
        Initialize the RAVEN executor.
        
        Args:
            raven_exe_path: Path to RAVEN executable. If None, tries to find it.
            workspace_dir: Directory for model files. If None, creates temporary directory.
            cleanup_on_exit: Whether to cleanup temporary files on exit.
        """
        self.raven_exe = raven_exe_path or self._find_raven_executable()
        self.workspace_dir = workspace_dir
        self.cleanup_on_exit = cleanup_on_exit
        self.temp_workspace = False
        
        # Validate RAVEN executable
        self._validate_raven_executable()
        
        # Model templates
        self.model_templates = self._load_model_templates()
        
        logger.info(f"RavenExecutor initialized with RAVEN: {self.raven_exe}")
    
    def _find_raven_executable(self) -> str:
        """Find RAVEN executable in common locations."""
        possible_paths = [
            # Absolute path to built executable
            r"E:\python\Raven\RavenHydroFramework\build\Release\Raven.exe",
            # Current workspace (fallback)
            r"E:\python\Raven\exe\Raven.exe",
            "exe/Raven.exe",
            "Raven.exe",
            # System PATH
            "raven",
            "Raven",
            # Common installation locations
            r"C:\Program Files\Raven\Raven.exe",
            "/usr/local/bin/raven",
            "/opt/raven/bin/raven"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
            
            # Check if it's in PATH
            try:
                result = subprocess.run(
                    ["where" if os.name == "nt" else "which", path], 
                    capture_output=True, 
                    text=True
                )
                if result.returncode == 0:
                    return path.strip() if path.strip() else path
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
                
        raise FileNotFoundError(
            "RAVEN executable not found. Please specify path explicitly or "
            "ensure RAVEN is installed and in your PATH."
        )
    
    def _validate_raven_executable(self) -> None:
        """Validate that the RAVEN executable works."""
        try:
            result = subprocess.run(
                [self.raven_exe, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                # Some versions don't support --version, try --help
                result = subprocess.run(
                    [self.raven_exe, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            
            if result.returncode != 0:
                raise RuntimeError(f"RAVEN executable failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("RAVEN executable validation timed out")
        except Exception as e:
            raise RuntimeError(f"Failed to validate RAVEN executable: {e}")
    
    def _load_model_templates(self) -> Dict[str, Dict[str, str]]:
        """Load predefined model templates."""
        return {
            "UBCWM": {
                "description": "UBC Watershed Model emulation",
                "processes": [
                    "PRECIP_RAVEN",
                    "FREEZE_DEGREE_DAY", 
                    "SNOWMELT_UBC",
                    "INF_UBC",
                    "BASE_LINEAR",
                    "PERC_LINEAR"
                ],
                "soil_profile": "UBC_SOIL_PROFILE",
                "compartments": ["SOIL:50.0", "SOIL:200.0"]
            },
            "GR4J": {
                "description": "GR4J model emulation",
                "processes": [
                    "PRECIP_RAVEN",
                    "INFILTRATION_GR4J",
                    "PERCOLATION_GR4J",
                    "BASEFLOW_GR4J"
                ]
            },
            "HMETS": {
                "description": "HMETS model emulation",
                "processes": [
                    "PRECIP_RAVEN",
                    "SNOWMELT_HMETS",
                    "INFILTRATION_HMETS"
                ]
            }
        }
    
    def setup_workspace(self, workspace_dir: Optional[str] = None) -> str:
        """
        Set up a workspace directory for RAVEN simulation.
        
        Args:
            workspace_dir: Directory to use. If None, creates temporary directory.
            
        Returns:
            Path to workspace directory
        """
        if workspace_dir is not None:
            self.workspace_dir = workspace_dir
            os.makedirs(self.workspace_dir, exist_ok=True)
            self.temp_workspace = False
        elif self.workspace_dir is None:
            self.workspace_dir = tempfile.mkdtemp(prefix="raven_simulation_")
            self.temp_workspace = True
        
        logger.info(f"Workspace set up at: {self.workspace_dir}")
        return self.workspace_dir
    
    def create_ubcwm_config(self, config: RavenConfig) -> Dict[str, str]:
        """
        Create UBC Watershed Model configuration files.
        
        Args:
            config: RavenConfig object with simulation parameters
            
        Returns:
            Dictionary with configuration file contents
        """
        # RVI file - Model configuration
        rvi_content = f"""# UBC Watershed Model Configuration
# Generated by RavenExecutor utility class
# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

:StartDate               {config.start_date} 00:00:00
:EndDate                 {config.end_date} 00:00:00
:TimeStep                {config.timestep}
:Method                  {config.method}

# UBC Watershed Model Process Configuration
:HydrologicProcesses
  :Precipitation           PRECIP_RAVEN                ATMOS_PRECIP    MULTIPLE
  :SnowRefreeze           FREEZE_DEGREE_DAY           SNOW_LIQ        SNOW
  :Snowmelt               SNOWMELT_UBC                SNOW            SNOW_LIQ
  :Infiltration           INF_UBC                     PONDED_WATER    MULTIPLE
  :Baseflow               BASE_LINEAR                 SOIL[0]         SURFACE_WATER
  :Percolation            PERC_LINEAR                 SOIL[0]         SOIL[1]
  :Baseflow               BASE_LINEAR                 SOIL[1]         SURFACE_WATER
:EndHydrologicProcesses

# Soil Profile Configuration
:SoilProfiles
  :name                   UBC_SOIL_PROFILE
    :Compartments         SOIL              50.0
    :Compartments         SOIL              200.0
  :EndSoilProfiles
:EndSoilProfiles

# Land Use Classes
:LandUseClasses
  :Attributes   IMP_PCT   FOREST_COV
  :Units        none      none
  FOREST        0.0       1.0
  URBAN         0.8       0.1
:EndLandUseClasses

# Vegetation Classes  
:VegetationClasses
  :Attributes   MAX_HT    MAX_LAI   MAX_LEAF_COND
  :Units        m         none      mm_per_s
  CONIFER       25.0      5.0       5.3
  DECIDUOUS     20.0      6.0       5.3
:EndVegetationClasses

# Evaluation Metrics
:EvaluationMetrics {' '.join(config.evaluation_metrics)}

# Output Configuration
{':WriteForcingFunctions' if config.write_forcing_functions else ''}
{':SuppressOutput' if config.suppress_output else ''}

# Custom Outputs
"""
        
        for output in config.custom_outputs:
            rvi_content += f":CustomOutput {output.get('frequency', 'DAILY')} {output.get('statistic', 'AVERAGE')} {output.get('variable', 'SOIL[0]')} {output.get('aggregation', 'BY_HRU')}\n"
        
        # RVP file - Parameters
        rvp_content = f"""# UBC Watershed Model Parameters
# Generated by RavenExecutor utility class

# Global Parameters
:GlobalParameter    RAIN_SNOW_FRACTION    0.0
:GlobalParameter    SNOW_SWI_MIN          0.05
:GlobalParameter    SNOW_SWI_MAX          0.15

# Soil Parameters
:SoilParameterList
  :Parameters         POROSITY    SAT_WILT    HBV_BETA    BASEFLOW_COEFF    PERC_COEFF
  :Units              none        none        none        1/d               1/d
  [DEFAULT]           0.4         0.1         2.0         0.05              0.1
:EndSoilParameterList

# Land Use Parameters
:LandUseParameterList
  :Parameters         UBC_ICEPT_FACTOR    UBC_INFIL_FACTOR    MELT_FACTOR
  :Units              none                none                mm/d/C
  [DEFAULT]           1.0                 1.0                 4.0
:EndLandUseParameterList

# Vegetation Parameters
:VegetationParameterList
  :Parameters         UBC_EVAP_FACTOR     PET_CORRECTION
  :Units              none                none
  [DEFAULT]           1.0                 1.0
:EndVegetationParameterList
"""

        # RVH file - Watershed discretization
        rvh_content = f"""# UBC Watershed Model HRU Definition
# Generated by RavenExecutor utility class

:SubBasins
  :Attributes   NAME    DOWNSTREAM_ID   PROFILE       REACH_LENGTH   GAUGED
  :Units        none    none           none          km             none
  1             Sub1    -1             RAVEN_DEFAULT  10.0          1
:EndSubBasins

:HRUs
  :Attributes    AREA    ELEVATION  LATITUDE    LONGITUDE    BASIN_ID   LAND_USE_CLASS   VEG_CLASS   SOIL_PROFILE
  :Units         km2     masl       deg         deg          none       none             none        none
  1              100.0   350.0      45.0        -75.0        1          FOREST           CONIFER     UBC_SOIL_PROFILE
:EndHRUs
"""

        # RVT file - Time series template
        rvt_content = f"""# UBC Watershed Model Time Series Data
# Generated by RavenExecutor utility class
# Note: Replace with actual forcing data

:MultiData
  :Attributes    TEMP_MAX    TEMP_MIN    PRECIP    PET
  :Units         degC        degC        mm/d      mm/d
  1              DATA_TEMP_MAX    DATA_TEMP_MIN    DATA_PRECIP    DATA_PET
:EndMultiData

# Template data structures (replace with real data)
:Data DATA_TEMP_MAX
{config.start_date} 00:00:00  1.0  daily
# Add your temperature maximum data here
:EndData

:Data DATA_TEMP_MIN
{config.start_date} 00:00:00  1.0  daily
# Add your temperature minimum data here
:EndData

:Data DATA_PRECIP
{config.start_date} 00:00:00  1.0  daily
# Add your precipitation data here
:EndData

:Data DATA_PET
{config.start_date} 00:00:00  1.0  daily
# Add your potential evapotranspiration data here
:EndData
"""

        return {
            'rvi': rvi_content,
            'rvp': rvp_content,
            'rvh': rvh_content,
            'rvt': rvt_content
        }
    
    def write_config_files(self, 
                          config_contents: Dict[str, str], 
                          model_name: str) -> Dict[str, str]:
        """
        Write RAVEN configuration files to workspace.
        
        Args:
            config_contents: Dictionary of file contents
            model_name: Base name for configuration files
            
        Returns:
            Dictionary mapping file types to full file paths
        """
        if self.workspace_dir is None:
            self.setup_workspace()
        
        file_paths = {}
        extensions = {'rvi': '.rvi', 'rvp': '.rvp', 'rvh': '.rvh', 'rvt': '.rvt'}
        
        for file_type, content in config_contents.items():
            if file_type in extensions:
                file_path = os.path.join(self.workspace_dir, f"{model_name}{extensions[file_type]}")
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    file_paths[file_type] = file_path
                    logger.info(f"Written {file_type.upper()} file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to write {file_type} file: {e}")
                    raise
        
        return file_paths
    
    def add_forcing_data(self, 
                        model_name: str, 
                        forcing_data: Dict[str, pd.DataFrame]) -> str:
        """
        Add forcing data to RVT file.
        
        Args:
            model_name: Name of the model
            forcing_data: Dictionary with DataFrames for each forcing variable
                         Keys should be: 'temp_max', 'temp_min', 'precip', 'pet'
            
        Returns:
            Path to updated RVT file
        """
        rvt_path = os.path.join(self.workspace_dir, f"{model_name}.rvt")
        
        if not os.path.exists(rvt_path):
            raise FileNotFoundError(f"RVT file not found: {rvt_path}")
        
        # Read existing RVT file
        with open(rvt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace data sections
        data_mapping = {
            'temp_max': 'DATA_TEMP_MAX',
            'temp_min': 'DATA_TEMP_MIN', 
            'precip': 'DATA_PRECIP',
            'pet': 'DATA_PET'
        }
        
        for var_name, data_key in data_mapping.items():
            if var_name in forcing_data:
                df = forcing_data[var_name]
                data_section = self._format_time_series_data(df, data_key)
                
                # Replace the data section in content
                start_marker = f":Data {data_key}"
                end_marker = ":EndData"
                
                start_idx = content.find(start_marker)
                if start_idx != -1:
                    end_idx = content.find(end_marker, start_idx) + len(end_marker)
                    content = content[:start_idx] + data_section + content[end_idx:]
        
        # Write updated content
        with open(rvt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Updated forcing data in: {rvt_path}")
        return rvt_path
    
    def _format_time_series_data(self, df: pd.DataFrame, data_key: str) -> str:
        """Format pandas DataFrame as RAVEN time series data."""
        if df.empty:
            return f":Data {data_key}\n# No data provided\n:EndData\n"
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Get the first column if multiple columns
        values = df.iloc[:, 0] if len(df.columns) > 0 else df
        
        # Format data
        start_date = df.index[0].strftime('%Y-%m-%d')
        data_lines = [f":Data {data_key}"]
        data_lines.append(f"{start_date} 00:00:00  1.0  daily")
        
        for date, value in zip(df.index, values):
            data_lines.append(f"{date.strftime('%Y-%m-%d')} {value:.3f}")
        
        data_lines.append(":EndData")
        return "\n".join(data_lines) + "\n"
    
    def run_simulation(self, 
                      model_name: str,
                      timeout: float = 300.0) -> RavenResults:
        """
        Run the RAVEN simulation.
        
        Args:
            model_name: Name of the model configuration files
            timeout: Simulation timeout in seconds
            
        Returns:
            RavenResults object with simulation results
        """
        if self.workspace_dir is None:
            raise ValueError("Workspace not set up. Call setup_workspace() first.")
        
        rvi_file = os.path.join(self.workspace_dir, f"{model_name}.rvi")
        if not os.path.exists(rvi_file):
            raise FileNotFoundError(f"RVI file not found: {rvi_file}")
        
        # Prepare command
        output_prefix = f"{model_name}_output"
        cmd = [self.raven_exe, f"{model_name}.rvi", "-o", output_prefix]
        
        # Change to workspace directory
        original_dir = os.getcwd()
        start_time = datetime.now()
        
        try:
            os.chdir(self.workspace_dir)
            
            logger.info(f"Running RAVEN command: {' '.join(cmd)}")
            logger.info(f"Working directory: {self.workspace_dir}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Find output files
            output_files = list(Path(self.workspace_dir).glob(f"{output_prefix}*"))
            output_file_paths = [str(f) for f in output_files]
            
            # Get config file paths
            config_files = {}
            for ext in ['rvi', 'rvp', 'rvh', 'rvt']:
                file_path = os.path.join(self.workspace_dir, f"{model_name}.{ext}")
                if os.path.exists(file_path):
                    config_files[ext] = file_path
            
            results = RavenResults(
                success=result.returncode == 0,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                workspace=self.workspace_dir,
                config_files=config_files,
                output_files=output_file_paths,
                execution_time=execution_time,
                error_message=result.stderr if result.returncode != 0 else None
            )
            
            if results.success:
                logger.info(f"Simulation completed successfully in {execution_time:.2f} seconds")
                logger.info(f"Output files: {len(output_file_paths)} files generated")
            else:
                logger.error(f"Simulation failed with return code {result.returncode}")
                logger.error(f"Error: {result.stderr}")
            
            return results
            
        except subprocess.TimeoutExpired:
            execution_time = (datetime.now() - start_time).total_seconds()
            return RavenResults(
                success=False,
                returncode=-1,
                stdout="",
                stderr="",
                workspace=self.workspace_dir,
                config_files={},
                output_files=[],
                execution_time=execution_time,
                error_message=f"Simulation timed out after {timeout} seconds"
            )
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return RavenResults(
                success=False,
                returncode=-1,
                stdout="",
                stderr="",
                workspace=self.workspace_dir,
                config_files={},
                output_files=[],
                execution_time=execution_time,
                error_message=f"Execution error: {str(e)}"
            )
        finally:
            os.chdir(original_dir)
    
    def read_output_file(self, 
                        output_file_path: str, 
                        file_type: str = "auto") -> pd.DataFrame:
        """
        Read RAVEN output file into pandas DataFrame.
        
        Args:
            output_file_path: Path to output file
            file_type: Type of file ('hydrographs', 'storage', 'custom', 'auto')
            
        Returns:
            DataFrame with output data
        """
        if not os.path.exists(output_file_path):
            raise FileNotFoundError(f"Output file not found: {output_file_path}")
        
        try:
            # Auto-detect file type if not specified
            if file_type == "auto":
                if "Hydrographs" in output_file_path:
                    file_type = "hydrographs"
                elif "Storage" in output_file_path:
                    file_type = "storage"
                else:
                    file_type = "custom"
            
            # Read the file
            if file_type in ["hydrographs", "storage", "custom"]:
                # Standard RAVEN output format
                df = pd.read_csv(
                    output_file_path,
                    delimiter=r'\s+',
                    comment='#',
                    parse_dates=['date'],
                    index_col='date'
                )
            else:
                # Generic CSV read
                df = pd.read_csv(output_file_path)
            
            logger.info(f"Successfully read output file: {output_file_path}")
            logger.info(f"Data shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to read output file {output_file_path}: {e}")
            raise
    
    def cleanup(self) -> None:
        """Clean up temporary workspace if created."""
        if self.temp_workspace and self.cleanup_on_exit and self.workspace_dir:
            try:
                shutil.rmtree(self.workspace_dir)
                logger.info(f"Cleaned up temporary workspace: {self.workspace_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup workspace: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def get_model_templates(self) -> Dict[str, Dict[str, str]]:
        """Get available model templates."""
        return self.model_templates.copy()
    
    def validate_config(self, config: RavenConfig) -> List[str]:
        """
        Validate RAVEN configuration.
        
        Args:
            config: RavenConfig to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate dates
        try:
            start = datetime.strptime(config.start_date, '%Y-%m-%d')
            end = datetime.strptime(config.end_date, '%Y-%m-%d')
            if start >= end:
                errors.append("End date must be after start date")
        except ValueError as e:
            errors.append(f"Invalid date format: {e}")
        
        # Validate timestep
        if config.timestep <= 0:
            errors.append("Timestep must be positive")
        
        # Validate method
        valid_methods = ["ORDERED_SERIES", "EULER", "RK4"]
        if config.method not in valid_methods:
            errors.append(f"Method must be one of {valid_methods}")
        
        return errors


# Convenience function for quick UBCWM setup
def create_ubcwm_simulation(model_name: str,
                           start_date: str,
                           end_date: str,
                           forcing_data: Optional[Dict[str, pd.DataFrame]] = None,
                           workspace_dir: Optional[str] = None,
                           raven_exe: Optional[str] = None) -> Tuple[RavenExecutor, RavenConfig]:
    """
    Convenience function to quickly set up a UBCWM simulation.
    
    Args:
        model_name: Name for the model
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        forcing_data: Optional forcing data dictionary
        workspace_dir: Optional workspace directory
        raven_exe: Optional path to RAVEN executable
        
    Returns:
        Tuple of (RavenExecutor, RavenConfig)
    """
    # Create executor
    executor = RavenExecutor(raven_exe_path=raven_exe, workspace_dir=workspace_dir)
    
    # Create config
    config = RavenConfig(
        model_name=model_name,
        start_date=start_date,
        end_date=end_date,
        custom_outputs=[
            {"frequency": "DAILY", "statistic": "AVERAGE", "variable": "SNOW", "aggregation": "BY_HRU"},
            {"frequency": "DAILY", "statistic": "AVERAGE", "variable": "SOIL[0]", "aggregation": "BY_HRU"},
            {"frequency": "DAILY", "statistic": "AVERAGE", "variable": "SOIL[1]", "aggregation": "BY_HRU"}
        ]
    )
    
    # Validate config
    errors = executor.validate_config(config)
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    # Set up workspace
    executor.setup_workspace()
    
    # Create and write config files
    config_contents = executor.create_ubcwm_config(config)
    executor.write_config_files(config_contents, model_name)
    
    # Add forcing data if provided
    if forcing_data:
        executor.add_forcing_data(model_name, forcing_data)
    
    return executor, config


if __name__ == "__main__":
    # Example usage
    print("RAVEN Executor Utility Class")
    print("=" * 50)
    
    try:
        # Create a simple UBCWM simulation
        with RavenExecutor() as executor:
            print(f"RAVEN executable: {executor.raven_exe}")
            
            # Set up configuration
            config = RavenConfig(
                model_name="test_ubcwm",
                start_date="2020-01-01",
                end_date="2020-01-31"
            )
            
            # Validate
            errors = executor.validate_config(config)
            if errors:
                print(f"Configuration errors: {errors}")
            else:
                print("Configuration is valid!")
            
            # Create files
            executor.setup_workspace()
            config_contents = executor.create_ubcwm_config(config)
            file_paths = executor.write_config_files(config_contents, "test_ubcwm")
            
            print(f"Workspace: {executor.workspace_dir}")
            print(f"Config files: {list(file_paths.keys())}")
            print("\nReady for simulation with real forcing data!")
            
    except Exception as e:
        print(f"Error: {e}")
