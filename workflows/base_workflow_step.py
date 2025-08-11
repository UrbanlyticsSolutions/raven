"""
Base class for all workflow steps with absolute path support and comprehensive tracking.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import logging

from infrastructure.path_manager import AbsolutePathManager, PathResolutionError, FileAccessError
from infrastructure.metadata_tracker import MetadataTracker, ProcessingStep
from infrastructure.qaqc_validator import QAQCValidator, ValidationResult
from infrastructure.configuration_manager import ConfigurationManager, WorkflowConfiguration

logger = logging.getLogger(__name__)


class FileSystemOperations:
    """Centralized file system operations with absolute path enforcement"""
    
    def __init__(self, path_manager: AbsolutePathManager, metadata_tracker: MetadataTracker):
        self.path_manager = path_manager
        self.metadata_tracker = metadata_tracker
    
    def read_file(self, file_path: Union[str, Path], must_exist: bool = True) -> Path:
        """
        Read file with absolute path resolution and validation.
        
        Args:
            file_path: Path to file to read
            must_exist: Whether file must exist
            
        Returns:
            Absolute path to validated file
            
        Raises:
            PathResolutionError: If path cannot be resolved
            FileAccessError: If file cannot be accessed
        """
        abs_path = self.path_manager.resolve_path(file_path)
        self.path_manager.validate_path(abs_path, must_exist=must_exist, must_be_file=True)
        
        logger.info(f"Reading file: {abs_path}")
        return abs_path
    
    def write_file(self, file_path: Union[str, Path], source_info: Dict[str, Any], 
                   processing_step: Optional[Dict[str, Any]] = None) -> Path:
        """
        Prepare file for writing with automatic metadata tracking.
        
        Args:
            file_path: Path to file to write
            source_info: Information about data source
            processing_step: Processing step information
            
        Returns:
            Absolute path ready for writing
            
        Raises:
            PathResolutionError: If path cannot be resolved
            FileAccessError: If file cannot be prepared for writing
        """
        abs_path = self.path_manager.ensure_file_writable(file_path)
        
        # Track metadata after file is written (caller responsibility to call track_output)
        logger.info(f"Prepared file for writing: {abs_path}")
        return abs_path
    
    def track_output(self, file_path: Union[str, Path], source_info: Dict[str, Any],
                    processing_step: Optional[Dict[str, Any]] = None,
                    coordinate_system: Optional[str] = None,
                    spatial_extent: Optional[Dict[str, float]] = None) -> Path:
        """
        Track output file with metadata.
        
        Args:
            file_path: Path to output file
            source_info: Information about data source
            processing_step: Processing step information
            coordinate_system: Coordinate system information
            spatial_extent: Spatial extent information
            
        Returns:
            Path to metadata file
        """
        abs_path = self.path_manager.resolve_path(file_path)
        
        processing_steps = []
        if processing_step:
            processing_steps.append(processing_step)
        
        metadata_path = self.metadata_tracker.track_dataset(
            file_path=abs_path,
            source_info=source_info,
            processing_steps=processing_steps,
            coordinate_system=coordinate_system,
            spatial_extent=spatial_extent
        )
        
        logger.info(f"Tracked output metadata: {metadata_path}")
        return metadata_path
    
    def copy_file(self, source: Union[str, Path], destination: Union[str, Path],
                  preserve_metadata: bool = True) -> Path:
        """
        Copy file with metadata preservation.
        
        Args:
            source: Source file path
            destination: Destination file path
            preserve_metadata: Whether to preserve metadata
            
        Returns:
            Absolute path to destination file
        """
        abs_source = self.path_manager.resolve_path(source)
        abs_dest = self.path_manager.ensure_file_writable(destination)
        
        self.path_manager.validate_path(abs_source, must_exist=True, must_be_file=True)
        
        # Copy file (implementation would use shutil.copy2)
        import shutil
        shutil.copy2(abs_source, abs_dest)
        
        # Copy metadata if requested
        if preserve_metadata:
            source_metadata = self.metadata_tracker.get_metadata(abs_source)
            if source_metadata:
                # Create new metadata for destination
                self.metadata_tracker.track_dataset(
                    file_path=abs_dest,
                    source_info={'description': f'Copied from {abs_source}'},
                    processing_steps=[{
                        'step_name': 'file_copy',
                        'timestamp': datetime.now(),
                        'parameters': {'source': str(abs_source)},
                        'input_files': [str(abs_source)],
                        'output_files': [str(abs_dest)],
                        'processing_time_seconds': 0.0,
                        'software_version': 'FileSystemOperations'
                    }],
                    coordinate_system=source_metadata.coordinate_system,
                    spatial_extent=source_metadata.spatial_extent
                )
        
        logger.info(f"Copied file: {abs_source} -> {abs_dest}")
        return abs_dest


class BaseWorkflowStep:
    """
    Base class for all workflow steps with absolute path support.
    Provides common infrastructure for path management, metadata tracking, and validation.
    """
    
    def __init__(self, workspace_dir: Union[str, Path], config: Optional[WorkflowConfiguration] = None,
                 step_name: Optional[str] = None):
        """
        Initialize base workflow step.
        
        Args:
            workspace_dir: Workspace directory for this step
            config: Optional workflow configuration
            step_name: Name of this step (for configuration lookup)
            
        Raises:
            PathResolutionError: If workspace directory cannot be resolved
        """
        self.step_name = step_name or self.__class__.__name__
        
        # Initialize path management
        self.path_manager = AbsolutePathManager(workspace_dir)
        
        # Initialize metadata tracking
        self.metadata_tracker = MetadataTracker(self.path_manager)
        
        # Initialize QA/QC validation
        self.validator = QAQCValidator(self.path_manager)
        
        # Initialize configuration management
        self.config_manager = ConfigurationManager(self.path_manager)
        
        # Initialize file operations
        self.file_ops = FileSystemOperations(self.path_manager, self.metadata_tracker)
        
        # Store configuration
        self.config = config
        self.step_config = None
        if config and step_name:
            self.step_config = self.config_manager.get_step_config(config, step_name)
        
        logger.info(f"Initialized {self.step_name} with workspace: {self.path_manager.workspace_root}")
    
    def validate_inputs(self, inputs: Dict[str, Union[str, Path]]) -> List[ValidationResult]:
        """
        Validate input files and parameters with explicit error reporting.
        
        Args:
            inputs: Dictionary of input file paths
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        for input_name, input_path in inputs.items():
            if not input_path:
                continue
                
            try:
                abs_path = self.path_manager.resolve_path(input_path)
                
                # Check if file exists
                if not abs_path.exists():
                    result = ValidationResult(
                        is_valid=False,
                        validation_type="input_file_existence",
                        file_path=str(abs_path),
                        errors=[f"Input file does not exist: {input_name}"],
                        warnings=[],
                        metrics={},
                        timestamp=datetime.now()
                    )
                    results.append(result)
                    continue
                
                # Validate file based on type
                file_ext = abs_path.suffix.lower()
                
                if file_ext in ['.shp', '.geojson', '.gpkg']:
                    # Spatial data validation
                    result = self.validator.validate_spatial_data_integrity(abs_path)
                    results.append(result)
                elif file_ext == '.csv' and 'hru' in input_name.lower():
                    # HRU data validation
                    result = self.validator.validate_hru_fields(abs_path)
                    results.append(result)
                elif file_ext == '.csv' and 'routing' in input_name.lower():
                    # Routing table validation
                    result = self.validator.validate_routing_connectivity(abs_path)
                    results.append(result)
                else:
                    # Generic file validation
                    result = ValidationResult(
                        is_valid=True,
                        validation_type="generic_file",
                        file_path=str(abs_path),
                        errors=[],
                        warnings=[],
                        metrics={'file_size_bytes': abs_path.stat().st_size},
                        timestamp=datetime.now()
                    )
                    results.append(result)
                    
            except (PathResolutionError, FileAccessError) as e:
                result = ValidationResult(
                    is_valid=False,
                    validation_type="input_validation_error",
                    file_path=str(input_path),
                    errors=[f"Input validation failed for {input_name}: {str(e)}"],
                    warnings=[],
                    metrics={},
                    timestamp=datetime.now()
                )
                results.append(result)
        
        return results
    
    def validate_outputs(self, outputs: Dict[str, Union[str, Path]]) -> List[ValidationResult]:
        """
        Validate output files and data quality with explicit failure reporting.
        
        Args:
            outputs: Dictionary of output file paths
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        for output_name, output_path in outputs.items():
            if not output_path:
                continue
                
            try:
                abs_path = self.path_manager.resolve_path(output_path)
                
                # Check if file was created
                if not abs_path.exists():
                    result = ValidationResult(
                        is_valid=False,
                        validation_type="output_file_creation",
                        file_path=str(abs_path),
                        errors=[f"Output file was not created: {output_name}"],
                        warnings=[],
                        metrics={},
                        timestamp=datetime.now()
                    )
                    results.append(result)
                    continue
                
                # Validate output based on type
                file_ext = abs_path.suffix.lower()
                
                if file_ext in ['.shp', '.geojson', '.gpkg']:
                    # Spatial data validation
                    result = self.validator.validate_spatial_data_integrity(abs_path)
                    results.append(result)
                elif file_ext == '.csv' and 'hru' in output_name.lower():
                    # HRU data validation
                    result = self.validator.validate_hru_fields(abs_path)
                    results.append(result)
                elif file_ext == '.csv' and 'routing' in output_name.lower():
                    # Routing table validation
                    result = self.validator.validate_routing_connectivity(abs_path)
                    results.append(result)
                else:
                    # Generic output validation
                    result = ValidationResult(
                        is_valid=True,
                        validation_type="generic_output",
                        file_path=str(abs_path),
                        errors=[],
                        warnings=[],
                        metrics={'file_size_bytes': abs_path.stat().st_size},
                        timestamp=datetime.now()
                    )
                    results.append(result)
                    
            except (PathResolutionError, FileAccessError) as e:
                result = ValidationResult(
                    is_valid=False,
                    validation_type="output_validation_error",
                    file_path=str(output_path),
                    errors=[f"Output validation failed for {output_name}: {str(e)}"],
                    warnings=[],
                    metrics={},
                    timestamp=datetime.now()
                )
                results.append(result)
        
        return results
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute step with automatic validation and metadata tracking.
        This is the main execution pattern for all steps.
        
        Args:
            **kwargs: Step-specific parameters
            
        Returns:
            Dictionary with execution results
        """
        start_time = time.time()
        
        logger.info(f"Starting execution of {self.step_name}")
        
        try:
            # Validate inputs if provided
            inputs = kwargs.get('inputs', {})
            if inputs:
                input_validation_results = self.validate_inputs(inputs)
                failed_validations = [r for r in input_validation_results if not r.is_valid]
                
                if failed_validations:
                    error_messages = []
                    for result in failed_validations:
                        error_messages.extend(result.errors)
                    
                    return {
                        'success': False,
                        'error': f"Input validation failed: {'; '.join(error_messages)}",
                        'validation_results': [r.to_dict() for r in input_validation_results]
                    }
            
            # Execute step-specific logic (implemented by subclasses)
            step_result = self._execute_step(**kwargs)
            
            if not step_result.get('success', False):
                return step_result
            
            # Validate outputs
            outputs = step_result.get('files', {})
            if outputs:
                output_validation_results = self.validate_outputs(outputs)
                failed_validations = [r for r in output_validation_results if not r.is_valid]
                
                if failed_validations:
                    error_messages = []
                    for result in failed_validations:
                        error_messages.extend(result.errors)
                    
                    logger.warning(f"Output validation warnings: {'; '.join(error_messages)}")
                    step_result['validation_warnings'] = error_messages
                
                step_result['validation_results'] = [r.to_dict() for r in output_validation_results]
            
            # Calculate processing time
            processing_time = time.time() - start_time
            step_result['processing_time_seconds'] = processing_time
            
            logger.info(f"Completed {self.step_name} in {processing_time:.2f} seconds")
            
            return step_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Step {self.step_name} failed after {processing_time:.2f} seconds: {str(e)}"
            logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'processing_time_seconds': processing_time
            }
    
    def _execute_step(self, **kwargs) -> Dict[str, Any]:
        """
        Step-specific execution logic to be implemented by subclasses.
        
        Args:
            **kwargs: Step-specific parameters
            
        Returns:
            Dictionary with step execution results
        """
        raise NotImplementedError("Subclasses must implement _execute_step method")
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> Path:
        """
        Save step results to JSON file with metadata tracking.
        
        Args:
            results: Results dictionary to save
            filename: Optional filename (defaults to step_name_results.json)
            
        Returns:
            Path to saved results file
        """
        if filename is None:
            filename = f"{self.step_name.lower()}_results.json"
        
        results_file = self.path_manager.workspace_root / filename
        results_file = self.file_ops.write_file(
            results_file,
            source_info={'description': f'Results from {self.step_name}'}
        )
        
        # Save results
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Track metadata
        self.file_ops.track_output(
            results_file,
            source_info={'description': f'Results from {self.step_name}'},
            processing_step={
                'step_name': self.step_name,
                'timestamp': datetime.now(),
                'parameters': {},
                'input_files': [],
                'output_files': [str(results_file)],
                'processing_time_seconds': results.get('processing_time_seconds', 0.0),
                'software_version': 'BaseWorkflowStep'
            }
        )
        
        logger.info(f"Saved results to: {results_file}")
        return results_file
    
    def load_previous_results(self, step_name: str, filename: str = None) -> Optional[Dict[str, Any]]:
        """
        Load results from a previous step with explicit error handling.
        
        Args:
            step_name: Name of the previous step
            filename: Optional filename (defaults to step_name_results.json)
            
        Returns:
            Results dictionary or None if not found
            
        Raises:
            FileAccessError: If results file cannot be read
        """
        if filename is None:
            filename = f"{step_name.lower()}_results.json"
        
        # Look for results in multiple possible locations using proper pathlib
        possible_locations = [
            # Direct in workspace root
            self.path_manager.workspace_root / filename,
            # In step-specific subdirectories (common pattern)
            self.path_manager.workspace_root / f"{step_name}_data" / filename,
            self.path_manager.workspace_root / f"{step_name}_{step_name.replace('step', '')}" / filename,
            self.path_manager.workspace_root / step_name / filename,
            # Parent directory locations
            self.path_manager.workspace_root.parent / filename,
            self.path_manager.workspace_root.parent / f"{step_name}_data" / filename,
            self.path_manager.workspace_root.parent / step_name / filename,
            # Data directory locations
            self.path_manager.workspace_root.parent.parent / "data" / "spatial" / filename,
            # Step-specific patterns for our project structure
            self.path_manager.workspace_root / "step2_watershed" / filename,
            self.path_manager.workspace_root / "step3_lakes" / filename,
            # Step3 results are often in outlet-specific subdirectories
            *list(self.path_manager.workspace_root.glob("step2_watershed/outlet_*/step3_results.json")),
            *list(self.path_manager.workspace_root.glob("step3_lakes/outlet_*/step3_results.json")),
        ]
        
        for results_file in possible_locations:
            try:
                abs_path = self.path_manager.resolve_path(results_file)
                if abs_path.exists():
                    self.path_manager.validate_path(abs_path, must_exist=True, must_be_file=True)
                    
                    import json
                    with open(abs_path, 'r') as f:
                        results = json.load(f)
                    
                    logger.info(f"Loaded {step_name} results from: {abs_path}")
                    return results
                    
            except (PathResolutionError, FileAccessError, json.JSONDecodeError) as e:
                logger.debug(f"Could not load results from {results_file}: {e}")
                continue
        
        # No results found in any location
        raise FileAccessError(
            str(possible_locations[0]),
            "load previous results",
            f"Results from {step_name} not found in any expected location"
        )