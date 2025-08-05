"""
RavenPy-based model builder - uses Emulator and emulators to generate Raven files
"""

from pathlib import Path
import tempfile
from typing import Dict, List, Union

import xarray as xr
import pandas as pd

from ravenpy import Emulator, ravenpy as raven_cli
from ravenpy.config import emulators as rp_emulators
from ravenpy import OutputReader


class BuildResult:
    """Result of model build operation"""
    
    def __init__(self, success: bool, output_dir: str = None, files_created: List[str] = None, 
                 errors: List[str] = None, outputs_path: str = None):
        self.success = success
        self.output_dir = output_dir
        self.files_created = files_created or []
        self.errors = errors or []
        self.outputs_path = outputs_path  # Path to model outputs if run


class RavenPyModelBuilder:
    """Builds a complete Raven model using RavenPy"""
    
    # Mapping from user model_type to RavenPy emulator class
    EMULATOR_MAP = {
        'GR4JCN': rp_emulators.GR4JCN,
        'HMETS': rp_emulators.HMETS,
        'HBVEC': rp_emulators.HBVEC,
        'MOHYSE': rp_emulators.Mohyse
    }
    
    def build_model(self, configuration, 
                    output_directory: Union[str, Path] = None,
                    run_model: bool = False) -> BuildResult:
        """
        Build Raven model using RavenPy Emulator
        
        Parameters
        ----------
        configuration : ModelConfiguration
            Prepared model configuration
        output_directory : str or Path, optional
            Directory where model files will be written
        run_model : bool, default False
            Whether to run Raven after writing configuration files
        """
        errors: List[str] = []
        
        try:
            # Select emulator class
            model_type = configuration.metadata.get('model_type')
            emulator_cls = self.EMULATOR_MAP.get(model_type)
            if emulator_cls is None:
                return BuildResult(False, errors=[f"Unsupported model_type for RavenPy: {model_type}"])
            
            # Instantiate emulator config
            cfg = emulator_cls()
            
            # Override start/end dates
            cfg.start_date = configuration.rvi_config['StartDate']
            cfg.end_date = configuration.rvi_config['EndDate']
            
            # Create Emulator instance
            m = Emulator(config=cfg)
            
            # Retrieve climate dataset
            climate_ds: xr.Dataset = configuration.rvt_config['ClimateData']
            if not isinstance(climate_ds, xr.Dataset):
                return BuildResult(False, errors=["Climate data must be xarray Dataset for RavenPy builder"])
            
            # Determine variable mapping and data_type list
            var_mapping: Dict[str, str] = configuration.rvt_config.get('VariableMapping', {})
            ds_vars = list(climate_ds.data_vars.keys())
            
            data_type = []
            alt_names = {}
            
            # Temperature
            if 'tas' in ds_vars:
                data_type.append('TEMP_AVE')
            elif 'tasmin' in ds_vars and 'tasmax' in ds_vars:
                data_type.extend(['TEMP_MIN', 'TEMP_MAX'])
            else:
                errors.append("Temperature variable (tas or tasmin/tasmax) not found in climate data")
            
            # Precipitation
            if 'pr' in ds_vars:
                data_type.append('PRECIP')
            else:
                errors.append("Precipitation variable 'pr' not found in climate data")
            
            if errors:
                return BuildResult(False, errors=errors)
            
            # Provide dataset to emulator
            m(ts=climate_ds, data_type=data_type, alt_names=alt_names)
            
            # Determine workdir
            if output_directory is None:
                workdir = Path(tempfile.mkdtemp(prefix="RavenPyModel_"))
            else:
                workdir = Path(output_directory)
                workdir.mkdir(parents=True, exist_ok=True)
            
            # Write RV files
            m.write_rv(workdir=workdir)
            
            # Collect generated files list
            files_created = [str(p) for p in workdir.glob('*') if p.suffix.startswith('.rv')]
            
            outputs_path = None
            if run_model:
                # Run Raven with ravenpy.run
                outputs_path = raven_cli.run(modelname=cfg.name, configdir=workdir, overwrite=True)
            
            return BuildResult(True, output_dir=str(workdir), files_created=files_created, outputs_path=str(outputs_path) if outputs_path else None)
            
        except Exception as e:
            errors.append(str(e))
            return BuildResult(False, errors=errors) 