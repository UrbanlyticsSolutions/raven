#!/usr/bin/env python3
"""
Centralized Configuration Manager with Fail-Fast Validation

This module provides strict configuration management that FAILS FAST if required 
parameters are not found in the centralized configuration files.

NO SILENT FALLBACKS. NO DEFAULT VALUES. EXPLICIT ERROR REPORTING.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Union, Optional


class ConfigurationError(Exception):
    """Raised when configuration is missing or invalid"""
    pass


class CentralizedConfigManager:
    """
    Centralized configuration manager with fail-fast validation.
    
    Features:
    - Loads and validates all centralized config files
    - FAILS FAST if required config is missing
    - Provides clear error messages with exact paths
    - NO silent fallbacks to defaults
    """
    
    def __init__(self, config_dir: Union[str, Path]):
        """Initialize with config directory path"""
        self.config_dir = Path(config_dir)
        self.configs = {}
        
        # Validate config directory exists
        if not self.config_dir.exists():
            raise ConfigurationError(f"Config directory not found: {self.config_dir}")
            
        # Load all required config files
        self._load_all_configs()
        
    def _load_all_configs(self):
        """Load ESSENTIAL configuration files from centralized location with fail-fast validation"""
        print("\n=== LOADING ESSENTIAL CONFIGS FROM CENTRALIZED LOCATION ===")
        
        # Define ONLY the 4 essential configuration files
        essential_configs = {
            # Core RAVEN parameter and classification files
            'raven_class_definitions': 'raven_class_definitions.json',
            'raven_lookup_database': 'raven_lookup_database.json', 
            'raven_complete_parameter_table': 'raven_complete_parameter_table.json',
            
            # Unified workflow and model configuration
            'raven_config': 'raven_config.json'
        }
        
        # FAIL FAST: Load every config file or fail completely
        for config_name, filename in essential_configs.items():
            config_path = self.config_dir / filename
            
            # FAIL FAST: Config file MUST exist
            if not config_path.exists():
                raise ConfigurationError(
                    f"CRITICAL CONFIG MISSING: '{config_name}' not found at {config_path}"
                )
                
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    
                # FAIL FAST: Config MUST not be empty
                if not loaded_config:
                    raise ConfigurationError(
                        f"CRITICAL CONFIG EMPTY: '{config_name}' at {config_path} is empty"
                    )
                    
                self.configs[config_name] = loaded_config
                print(f"[OK] LOADED: {config_name} -> {len(str(loaded_config))} chars")
                
            except json.JSONDecodeError as e:
                raise ConfigurationError(
                    f"CRITICAL JSON ERROR: '{config_name}' at {config_path} has invalid JSON: {e}"
                )
            except Exception as e:
                raise ConfigurationError(
                    f"CRITICAL LOAD FAILURE: '{config_name}' at {config_path} failed to load: {e}"
                )
                
        print(f"[SUCCESS] ALL {len(essential_configs)} ESSENTIAL CONFIGS LOADED FROM CENTRALIZED LOCATION")
        
        # Validate required sections exist in core configs
        self._validate_config_structure()
        
    def _validate_config_structure(self):
        """Validate that all required config sections exist with FAIL-FAST approach"""
        print("\n=== VALIDATING ALL CONFIG STRUCTURES ===")
        
        # Essential RAVEN configuration validation
        required_sections = {
            'raven_class_definitions': [
                'vegetation_classes', 'landuse_classes', 'soil_profiles', 
                'soil_classes', 'default_soil_parameters'
            ],
            'raven_lookup_database': [
                'soil_classification', 'landcover_classification', 
                'vegetation_classification'  
            ],
            'raven_config': [
                'global_parameters', 'watershed_delineation', 'hru_generation',
                'routing_configuration', 'hydrological_processes'
            ]
        }
        
        # FAIL FAST: Validate core configuration structures
        for config_name, sections in required_sections.items():
            if config_name not in self.configs:
                raise ConfigurationError(
                    f"CRITICAL CORE CONFIG MISSING: '{config_name}' not loaded"
                )
                
            config = self.configs[config_name]
            for section in sections:
                if section not in config:
                    raise ConfigurationError(
                        f"CRITICAL SECTION MISSING: '{section}' not found in {config_name}"
                    )
                    
                # FAIL FAST: Section must not be empty
                if not config[section]:
                    raise ConfigurationError(
                        f"CRITICAL SECTION EMPTY: '{section}' in {config_name} is empty"
                    )
            print(f"[OK] VALIDATED: {config_name} structure")
        
        # Validate unified config format
        if 'raven_config' in self.configs:
            config = self.configs['raven_config']
            if not isinstance(config, dict):
                raise ConfigurationError(
                    "CRITICAL CONFIG FORMAT: 'raven_config' must be a dictionary"
                )
            print("[OK] VALIDATED: raven_config format")
        
        # Validate all configs are dictionaries (except arrays if expected)
        for config_name, config_data in self.configs.items():
            if not isinstance(config_data, (dict, list)):
                raise ConfigurationError(
                    f"CRITICAL CONFIG TYPE: '{config_name}' must be dict or list, got {type(config_data)}"
                )
        
        print("[SUCCESS] ALL CONFIG STRUCTURES VALIDATED")
        print("===================================================\n")
                    
    def get_vegetation_class_params(self, vegetation_class: str) -> Dict[str, Any]:
        """
        Get vegetation class parameters with FAIL-FAST validation
        
        Args:
            vegetation_class: Name of vegetation class (e.g., 'CONIFEROUS')
            
        Returns:
            Dictionary with vegetation parameters
            
        Raises:
            ConfigurationError: If vegetation class not found
        """
        veg_classes = self.configs['raven_class_definitions']['vegetation_classes']
        
        if vegetation_class not in veg_classes:
            available = list(veg_classes.keys())
            raise ConfigurationError(
                f"CRITICAL: Vegetation class '{vegetation_class}' not found in config. "
                f"Available classes: {available}"
            )
            
        return veg_classes[vegetation_class]
        
    def get_landuse_class_params(self, landuse_class: str) -> Dict[str, Any]:
        """
        Get landuse class parameters with FAIL-FAST validation
        
        Args:
            landuse_class: Name of landuse class (e.g., 'FOREST')
            
        Returns:
            Dictionary with landuse parameters
            
        Raises:
            ConfigurationError: If landuse class not found
        """
        landuse_classes = self.configs['raven_class_definitions']['landuse_classes']
        
        if landuse_class not in landuse_classes:
            available = list(landuse_classes.keys())
            raise ConfigurationError(
                f"CRITICAL: Landuse class '{landuse_class}' not found in config. "
                f"Available classes: {available}"
            )
            
        return landuse_classes[landuse_class]
        
    def get_soil_profile_params(self, soil_profile: str) -> Dict[str, Any]:
        """
        Get soil profile parameters with FAIL-FAST validation
        
        Args:
            soil_profile: Name of soil profile (e.g., 'LOAM')
            
        Returns:
            Dictionary with soil profile parameters
            
        Raises:
            ConfigurationError: If soil profile not found
        """
        soil_profiles = self.configs['raven_class_definitions']['soil_profiles']
        
        if soil_profile not in soil_profiles:
            available = list(soil_profiles.keys())
            raise ConfigurationError(
                f"CRITICAL: Soil profile '{soil_profile}' not found in config. "
                f"Available profiles: {available}"
            )
            
        return soil_profiles[soil_profile]
        
    def get_soil_class_params(self, soil_class: str) -> Dict[str, Any]:
        """
        Get soil class parameters with FAIL-FAST validation
        
        Args:
            soil_class: Name of soil class (e.g., 'LOAM_TOP')
            
        Returns:
            Dictionary with soil class parameters
            
        Raises:
            ConfigurationError: If soil class not found
        """
        soil_classes = self.configs['raven_class_definitions']['soil_classes']
        
        if soil_class not in soil_classes:
            available = list(soil_classes.keys())
            raise ConfigurationError(
                f"CRITICAL: Soil class '{soil_class}' not found in config. "
                f"Available classes: {available}"
            )
            
        return soil_classes[soil_class]
        
    def get_required_parameter(self, config_name: str, *path: str) -> Any:
        """
        Get a required parameter from config with FAIL-FAST validation
        
        Args:
            config_name: Name of config file ('raven_class_definitions', etc.)
            *path: Path to parameter (e.g., 'vegetation_classes', 'CONIFEROUS', 'MAX_HT')
            
        Returns:
            Parameter value
            
        Raises:
            ConfigurationError: If parameter not found at path
        """
        if config_name not in self.configs:
            available = list(self.configs.keys())
            raise ConfigurationError(
                f"CRITICAL: Config '{config_name}' not loaded. Available: {available}"
            )
            
        current = self.configs[config_name]
        path_str = " -> ".join(path)
        
        try:
            for key in path:
                current = current[key]
            return current
        except (KeyError, TypeError):
            raise ConfigurationError(
                f"CRITICAL: Parameter not found at path: {config_name} -> {path_str}"
            )
            
    def validate_all_dynamic_classes(self, dynamic_classes: Dict[str, set]) -> None:
        """
        Validate that all dynamic classes exist in centralized config
        
        Args:
            dynamic_classes: Dictionary with sets of class names
            
        Raises:
            ConfigurationError: If any required class is missing
        """
        print("\n=== VALIDATING ALL DYNAMIC CLASSES ===")
        
        # Validate vegetation classes
        if 'vegetation' in dynamic_classes:
            veg_classes = self.configs['raven_class_definitions']['vegetation_classes']
            for veg_class in dynamic_classes['vegetation']:
                if veg_class not in veg_classes:
                    available = list(veg_classes.keys())
                    raise ConfigurationError(
                        f"CRITICAL: Vegetation class '{veg_class}' not in config. "
                        f"Available: {available}"
                    )
            print(f"[OK] All vegetation classes validated: {sorted(dynamic_classes['vegetation'])}")
            
        # Validate landuse classes  
        if 'landuse' in dynamic_classes:
            landuse_classes = self.configs['raven_class_definitions']['landuse_classes']
            for landuse_class in dynamic_classes['landuse']:
                if landuse_class not in landuse_classes:
                    available = list(landuse_classes.keys())
                    raise ConfigurationError(
                        f"CRITICAL: Landuse class '{landuse_class}' not in config. "
                        f"Available: {available}"
                    )
            print(f"[OK] All landuse classes validated: {sorted(dynamic_classes['landuse'])}")
            
        # Validate soil profiles
        if 'soil' in dynamic_classes:
            soil_profiles = self.configs['raven_class_definitions']['soil_profiles']
            for soil_profile in dynamic_classes['soil']:
                if soil_profile not in soil_profiles:
                    available = list(soil_profiles.keys())
                    raise ConfigurationError(
                        f"CRITICAL: Soil profile '{soil_profile}' not in config. "
                        f"Available: {available}"
                    )
            print(f"[OK] All soil profiles validated: {sorted(dynamic_classes['soil'])}")
            
        print("=== ALL DYNAMIC CLASSES VALIDATION PASSED ===\n")
        
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all loaded configurations"""
        return self.configs.copy()
        
    def get_raven_config(self) -> Dict[str, Any]:
        """
        Get unified RAVEN configuration with FAIL-FAST validation
        
        Returns:
            Dictionary with unified RAVEN configuration
            
        Raises:
            ConfigurationError: If RAVEN config not found
        """
        if 'raven_config' not in self.configs:
            raise ConfigurationError(
                "CRITICAL: Unified RAVEN config 'raven_config' not found"
            )
            
        return self.configs['raven_config']
        
    def get_workflow_config(self, config_name: str = 'raven_config') -> Dict[str, Any]:
        """
        Get workflow configuration with FAIL-FAST validation
        
        Args:
            config_name: Name of workflow config ('workflow_config', 'complete_workflow_config')
            
        Returns:
            Dictionary with workflow configuration
            
        Raises:
            ConfigurationError: If workflow config not found
        """
        if config_name not in self.configs:
            available = list(self.configs.keys())
            raise ConfigurationError(
                f"CRITICAL: Config '{config_name}' not found. Available: {available}"
            )
            
        return self.configs[config_name]
        
        
    def get_all_config_names(self) -> List[str]:
        """Get list of all loaded configuration names"""
        return list(self.configs.keys())
        
    def get_config_by_name(self, config_name: str) -> Dict[str, Any]:
        """
        Get any configuration by name with FAIL-FAST validation
        
        Args:
            config_name: Name of configuration to retrieve
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationError: If config not found
        """
        if config_name not in self.configs:
            available = list(self.configs.keys())
            raise ConfigurationError(
                f"CRITICAL: Config '{config_name}' not found. Available: {available}"
            )
            
        return self.configs[config_name]
        
    def print_config_summary(self):
        """Print comprehensive summary of ALL loaded configurations"""
        print("\n" + "="*60)
        print("CENTRALIZED CONFIG MANAGER - COMPLETE SUMMARY")
        print("="*60)
        
        # Show essential configs
        config_groups = [
            ("ESSENTIAL CONFIGS", list(self.configs.keys()))
        ]
        
        total_configs = 0
        for group_name, config_names in config_groups:
            if config_names:
                print(f"\n{group_name}:")
                print("-" * len(group_name))
                for config_name in sorted(config_names):
                    config_data = self.configs[config_name]
                    if isinstance(config_data, dict):
                        sections = len(config_data)
                        total_items = sum(len(v) if isinstance(v, (dict, list)) else 1 
                                        for v in config_data.values())
                        print(f"  [OK] {config_name:<45} {sections:2d} sections, {total_items:4d} items")
                    else:
                        print(f"  [OK] {config_name:<45} {type(config_data).__name__}")
                    total_configs += 1
        
        print(f"\n{'='*60}")
        print(f"TOTAL ESSENTIAL CONFIGS LOADED: {total_configs}")
        print(f"STREAMLINED CONFIG SYSTEM - REDUNDANT FILES REMOVED")
        print(f"ALL CONFIGS LOADED FROM: {self.config_dir}")
        print("FAIL-FAST VALIDATION: ENABLED")
        print("="*60 + "\n")


def create_config_manager(config_dir: Union[str, Path] = None) -> CentralizedConfigManager:
    """
    Factory function to create centralized config manager
    
    Args:
        config_dir: Path to config directory (defaults to project root/config)
        
    Returns:
        Initialized CentralizedConfigManager
    """
    if config_dir is None:
        # Default to project root config directory
        project_root = Path(__file__).parent.parent
        config_dir = project_root / "config"
        
    return CentralizedConfigManager(config_dir)


if __name__ == "__main__":
    """Test the centralized config manager"""
    try:
        # Test config manager
        config_manager = create_config_manager()
        config_manager.print_config_summary()
        
        # Test parameter extraction
        print("Testing parameter extraction...")
        
        # Test vegetation class
        coniferous_params = config_manager.get_vegetation_class_params('CONIFEROUS')
        print(f"CONIFEROUS params: {coniferous_params}")
        
        # Test landuse class
        forest_params = config_manager.get_landuse_class_params('FOREST')
        print(f"FOREST params: {forest_params}")
        
        # Test soil profile
        loam_params = config_manager.get_soil_profile_params('LOAM')
        print(f"LOAM params: {loam_params}")
        
        print("[SUCCESS] All tests passed!")
        
    except ConfigurationError as e:
        print(f"[ERROR] Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected Error: {e}")
        sys.exit(1)