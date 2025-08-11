#!/usr/bin/env python3
"""
FAIL-FAST Configuration Validation System
Prevents broken file system by validating ALL inputs before execution
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

class ValidationLevel(Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR" 
    WARNING = "WARNING"
    INFO = "INFO"

@dataclass
class ValidationResult:
    level: ValidationLevel
    message: str
    file_path: Optional[str] = None
    missing_keys: Optional[List[str]] = None
    step: Optional[str] = None

class FailFastValidator:
    """
    FAIL-FAST validator that checks ALL configuration and input files
    before allowing any step to execute
    """
    
    def __init__(self, workspace_dir: str, project_root: str = None):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.results: List[ValidationResult] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def validate_all(self, target_step: int = 5) -> bool:
        """
        Validate ALL inputs required for target step
        
        Parameters:
        -----------
        target_step : int
            Target step to validate for (1-5)
            
        Returns:
        --------
        bool
            True if all validations pass, False if any CRITICAL/ERROR found
        """
        self.results.clear()
        
        print("="*60)
        print("🔍 FAIL-FAST CONFIGURATION VALIDATION")
        print("="*60)
        
        # 1. Validate essential config files
        self._validate_essential_configs()
        
        # 2. Validate workspace structure
        self._validate_workspace_structure()
        
        # 3. Validate step dependencies based on target
        for step in range(1, target_step + 1):
            self._validate_step_dependencies(step)
        
        # 4. Validate file accessibility
        self._validate_file_accessibility()
        
        # 5. Print validation summary
        return self._print_validation_summary()
    
    def _validate_essential_configs(self):
        """Validate all essential configuration files"""
        print("\n🔧 VALIDATING ESSENTIAL CONFIGURATIONS...")
        
        essential_configs = {
            'raven_config.json': ['model_types', 'routing_methods', 'default_parameters'],
            'raven_class_definitions.json': ['soil_classes', 'landuse_classes', 'vegetation_classes'],
            'raven_lookup_database.json': ['soil_info', 'landuse_info', 'vegetation_info'],
            'raven_complete_parameter_table.json': ['soil_parameters', 'landuse_parameters']
        }
        
        for config_file, required_keys in essential_configs.items():
            config_path = self.config_dir / config_file
            
            if not config_path.exists():
                self.results.append(ValidationResult(
                    level=ValidationLevel.CRITICAL,
                    message=f"Essential config file missing: {config_file}",
                    file_path=str(config_path)
                ))
                continue
                
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                missing_keys = [key for key in required_keys if key not in config_data]
                if missing_keys:
                    self.results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        message=f"Missing required keys in {config_file}",
                        file_path=str(config_path),
                        missing_keys=missing_keys
                    ))
                else:
                    print(f"  ✅ {config_file}: All required keys present")
                    
            except json.JSONDecodeError as e:
                self.results.append(ValidationResult(
                    level=ValidationLevel.CRITICAL,
                    message=f"Invalid JSON in {config_file}: {str(e)}",
                    file_path=str(config_path)
                ))
    
    def _validate_workspace_structure(self):
        """Validate workspace directory structure"""
        print("\n📁 VALIDATING WORKSPACE STRUCTURE...")
        
        if not self.workspace_dir.exists():
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Workspace directory does not exist: {self.workspace_dir}"
            ))
            return
        
        # Check write permissions
        test_file = self.workspace_dir / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
            print(f"  ✅ Workspace writable: {self.workspace_dir}")
        except Exception as e:
            self.results.append(ValidationResult(
                level=ValidationLevel.CRITICAL,
                message=f"Workspace not writable: {str(e)}",
                file_path=str(self.workspace_dir)
            ))
    
    def _validate_step_dependencies(self, step: int):
        """Validate dependencies for a specific step"""
        print(f"\n🔗 VALIDATING STEP {step} DEPENDENCIES...")
        
        step_requirements = {
            1: {
                'files': [],
                'previous_steps': []
            },
            2: {
                'files': ['step1_results.json'],
                'previous_steps': [1]
            },
            3: {
                'files': ['step1_results.json', 'step2_results.json'],
                'previous_steps': [1, 2],
                'required_data': ['dem.tif', 'watershed.geojson', 'streams.geojson']
            },
            4: {
                'files': ['step1_results.json', 'step2_results.json', 'step3_results.json'],
                'previous_steps': [1, 2, 3],
                'required_data': ['lakes_with_routing_ids.shp', 'subbasins_with_lakes.shp']
            },
            5: {
                'files': ['step1_results.json', 'step2_results.json', 'step3_results.json', 'step4_results.json'],
                'previous_steps': [1, 2, 3, 4],
                'required_data': ['hrus.geojson', 'magpie_hydraulic_parameters.csv']
            }
        }
        
        if step not in step_requirements:
            return
            
        requirements = step_requirements[step]
        
        # Check previous step results
        for required_file in requirements['files']:
            file_path = self.workspace_dir / required_file
            if not file_path.exists():
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Step {step} missing required file: {required_file}",
                    file_path=str(file_path),
                    step=f"step{step}"
                ))
            else:
                # Validate JSON structure
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if not data.get('success', False):
                        self.results.append(ValidationResult(
                            level=ValidationLevel.ERROR,
                            message=f"Previous step failed: {required_file}",
                            file_path=str(file_path),
                            step=f"step{step}"
                        ))
                    else:
                        print(f"  ✅ {required_file}: Valid and successful")
                except Exception as e:
                    self.results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        message=f"Cannot read {required_file}: {str(e)}",
                        file_path=str(file_path),
                        step=f"step{step}"
                    ))
        
        # Check required data files
        if 'required_data' in requirements:
            data_dir = self.workspace_dir / "data"
            for data_file in requirements['required_data']:
                file_path = data_dir / data_file
                if not file_path.exists():
                    self.results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        message=f"Step {step} missing required data: {data_file}",
                        file_path=str(file_path),
                        step=f"step{step}"
                    ))
                else:
                    print(f"  ✅ {data_file}: Available")
    
    def _validate_file_accessibility(self):
        """Validate that all required files are accessible"""
        print("\n🔐 VALIDATING FILE ACCESSIBILITY...")
        
        critical_paths = [
            self.workspace_dir / "data",
            self.config_dir,
            self.project_root / "processors",
            self.project_root / "workflows"
        ]
        
        for path in critical_paths:
            if not path.exists():
                self.results.append(ValidationResult(
                    level=ValidationLevel.CRITICAL,
                    message=f"Critical path missing: {path.name}",
                    file_path=str(path)
                ))
            elif not path.is_dir():
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Expected directory but found file: {path.name}",
                    file_path=str(path)
                ))
            else:
                print(f"  ✅ {path.name}: Accessible")
    
    def _print_validation_summary(self) -> bool:
        """Print validation summary and return success status"""
        print("\n" + "="*60)
        print("📊 VALIDATION SUMMARY")
        print("="*60)
        
        # Count by level
        counts = {level: 0 for level in ValidationLevel}
        for result in self.results:
            counts[result.level] += 1
        
        # Print counts
        print(f"🔴 CRITICAL: {counts[ValidationLevel.CRITICAL]}")
        print(f"🟠 ERROR: {counts[ValidationLevel.ERROR]}")
        print(f"🟡 WARNING: {counts[ValidationLevel.WARNING]}")
        print(f"🔵 INFO: {counts[ValidationLevel.INFO]}")
        
        # Print detailed issues
        if counts[ValidationLevel.CRITICAL] > 0 or counts[ValidationLevel.ERROR] > 0:
            print("\n❌ CRITICAL ISSUES:")
            for result in self.results:
                if result.level in [ValidationLevel.CRITICAL, ValidationLevel.ERROR]:
                    print(f"  {result.level.value}: {result.message}")
                    if result.file_path:
                        print(f"    File: {result.file_path}")
                    if result.missing_keys:
                        print(f"    Missing: {', '.join(result.missing_keys)}")
        
        # Final verdict
        has_critical_errors = counts[ValidationLevel.CRITICAL] > 0 or counts[ValidationLevel.ERROR] > 0
        
        if has_critical_errors:
            print("\n🚨 VALIDATION FAILED - CANNOT PROCEED")
            print("Fix all CRITICAL and ERROR issues before running workflow")
            return False
        else:
            print("\n✅ VALIDATION PASSED - READY TO PROCEED")
            if counts[ValidationLevel.WARNING] > 0:
                print("⚠️  Consider addressing warnings for optimal performance")
            return True
    
    def get_file_locations_summary(self) -> Dict[str, List[str]]:
        """Get summary of where all files are located"""
        locations = {
            'config_files': [],
            'step_results': [],
            'data_files': [],
            'model_files': []
        }
        
        # Config files
        for config_file in self.config_dir.glob("*.json"):
            locations['config_files'].append(str(config_file))
        
        # Step results
        for step_file in self.workspace_dir.glob("step*_results.json"):
            locations['step_results'].append(str(step_file))
        
        # Data files
        data_dir = self.workspace_dir / "data"
        if data_dir.exists():
            for data_file in data_dir.glob("*"):
                if data_file.is_file():
                    locations['data_files'].append(str(data_file))
        
        # Model files
        models_dir = self.workspace_dir / "models"
        if models_dir.exists():
            for model_file in models_dir.rglob("*"):
                if model_file.is_file():
                    locations['model_files'].append(str(model_file))
        
        return locations

def main():
    """Command line interface for validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FAIL-FAST Configuration Validator")
    parser.add_argument("workspace_dir", help="Workspace directory to validate")
    parser.add_argument("--target-step", type=int, default=5, choices=[1,2,3,4,5],
                       help="Target step to validate for")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--show-files", action="store_true", 
                       help="Show file location summary")
    
    args = parser.parse_args()
    
    validator = FailFastValidator(args.workspace_dir, args.project_root)
    
    if args.show_files:
        print("\n📂 FILE LOCATION SUMMARY")
        print("="*40)
        locations = validator.get_file_locations_summary()
        for category, files in locations.items():
            print(f"\n{category.upper()}:")
            for file_path in files:
                print(f"  {file_path}")
    
    success = validator.validate_all(args.target_step)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
