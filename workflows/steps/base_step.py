"""
Base class for all workflow steps

This module defines the common interface and functionality that all workflow steps
must implement. It provides logging, error handling, and context management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path
import logging
from datetime import datetime

class WorkflowStep(ABC):
    """
    Base class for all workflow steps
    
    All workflow steps must inherit from this class and implement the execute() method.
    This ensures consistent interface and behavior across all steps.
    """
    
    def __init__(self, step_name: str, step_category: str, description: str = ""):
        """
        Initialize workflow step
        
        Parameters:
        -----------
        step_name : str
            Unique name for this step
        step_category : str
            Category this step belongs to (e.g., 'validation', 'processing')
        description : str, optional
            Human-readable description of what this step does
        """
        self.step_name = step_name
        self.step_category = step_category
        self.description = description
        self.context_manager = None
        self.logger = self._setup_logging()
        
        # Execution metadata
        self.start_time = None
        self.end_time = None
        self.execution_time_seconds = None
        
    def set_context_manager(self, context_manager):
        """
        Set context manager for workflow state tracking
        
        Parameters:
        -----------
        context_manager : ContextManager
            Context manager instance for tracking workflow state
        """
        self.context_manager = context_manager
        
    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the workflow step
        
        This method must be implemented by all subclasses. It should:
        1. Validate inputs
        2. Perform the step's processing
        3. Return outputs in a consistent format
        
        Parameters:
        -----------
        inputs : Dict[str, Any]
            Input data for this step
            
        Returns:
        --------
        Dict[str, Any]
            Output data from this step, must include 'success' key
        """
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any], required_keys: List[str]) -> None:
        """
        Validate that required input keys are present
        
        Parameters:
        -----------
        inputs : Dict[str, Any]
            Input dictionary to validate
        required_keys : List[str]
            List of required keys
            
        Raises:
        -------
        ValueError
            If any required keys are missing
        """
        missing_keys = [key for key in required_keys if key not in inputs]
        if missing_keys:
            raise ValueError(f"Missing required inputs: {missing_keys}")
    
    def validate_file_exists(self, file_path: str) -> Path:
        """
        Validate that a file exists and return Path object
        
        Parameters:
        -----------
        file_path : str
            Path to file to validate
            
        Returns:
        --------
        Path
            Validated Path object
            
        Raises:
        -------
        FileNotFoundError
            If file does not exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
        return path
    
    def create_output_path(self, workspace_dir: str, filename: str) -> Path:
        """
        Create output file path in workspace directory
        
        Parameters:
        -----------
        workspace_dir : str
            Workspace directory path
        filename : str
            Output filename
            
        Returns:
        --------
        Path
            Full path for output file
        """
        workspace = Path(workspace_dir)
        workspace.mkdir(exist_ok=True, parents=True)
        return workspace / filename
    
    def _log_step_start(self):
        """Log step start and record start time"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting step: {self.step_name}")
        
        if self.context_manager:
            self.context_manager.mark_step_started(self.step_name)
    
    def _log_step_complete(self, outputs: List[str] = None):
        """
        Log step completion and record execution time
        
        Parameters:
        -----------
        outputs : List[str], optional
            List of output file paths created by this step
        """
        self.end_time = datetime.now()
        if self.start_time:
            self.execution_time_seconds = (self.end_time - self.start_time).total_seconds()
            self.logger.info(f"Completed step: {self.step_name} ({self.execution_time_seconds:.1f}s)")
        else:
            self.logger.info(f"Completed step: {self.step_name}")
        
        if outputs:
            for output in outputs:
                self.logger.info(f"   Created: {output}")
        
        if self.context_manager:
            self.context_manager.mark_step_completed(self.step_name, outputs or [])
    
    def _log_step_failed(self, error_msg: str):
        """
        Log step failure
        
        Parameters:
        -----------
        error_msg : str
            Error message describing the failure
        """
        self.end_time = datetime.now()
        if self.start_time:
            self.execution_time_seconds = (self.end_time - self.start_time).total_seconds()
            self.logger.error(f"Failed step: {self.step_name} ({self.execution_time_seconds:.1f}s)")
        else:
            self.logger.error(f"Failed step: {self.step_name}")
        
        self.logger.error(f"   Error: {error_msg}")
        
        if self.context_manager:
            self.context_manager.mark_step_failed(self.step_name, error_msg)
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging for this step
        
        Returns:
        --------
        logging.Logger
            Configured logger instance
        """
        logger = logging.getLogger(f"WorkflowStep.{self.step_name}")
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def get_execution_metadata(self) -> Dict[str, Any]:
        """
        Get execution metadata for this step
        
        Returns:
        --------
        Dict[str, Any]
            Metadata about step execution
        """
        return {
            'step_name': self.step_name,
            'step_category': self.step_category,
            'description': self.description,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'execution_time_seconds': self.execution_time_seconds
        }
    
    def __str__(self) -> str:
        """String representation of the step"""
        return f"{self.step_category}.{self.step_name}"
    
    def __repr__(self) -> str:
        """Detailed string representation of the step"""
        return f"WorkflowStep(name='{self.step_name}', category='{self.step_category}')"