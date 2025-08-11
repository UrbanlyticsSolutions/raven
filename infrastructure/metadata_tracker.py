"""
MetadataTracker for comprehensive dataset metadata tracking with YAML/JSON output.
"""

import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
from dataclasses import dataclass, asdict
import hashlib
import logging

from .path_manager import AbsolutePathManager, PathResolutionError, FileAccessError

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStep:
    """Represents a single processing step in the workflow"""
    step_name: str
    timestamp: datetime
    parameters: Dict[str, Any]
    input_files: List[str]  # Absolute paths
    output_files: List[str]  # Absolute paths
    processing_time_seconds: float
    software_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO timestamp"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingStep':
        """Create from dictionary with ISO timestamp parsing"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class DatasetMetadata:
    """Complete metadata for a dataset"""
    file_path: str  # Absolute path
    source: str
    creation_date: datetime
    processing_steps: List[ProcessingStep]
    file_size_bytes: int
    checksum: str
    coordinate_system: Optional[str] = None
    spatial_extent: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO timestamps"""
        data = asdict(self)
        data['creation_date'] = self.creation_date.isoformat()
        data['processing_steps'] = [step.to_dict() for step in self.processing_steps]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetMetadata':
        """Create from dictionary with ISO timestamp parsing"""
        data = data.copy()
        data['creation_date'] = datetime.fromisoformat(data['creation_date'])
        data['processing_steps'] = [ProcessingStep.from_dict(step) for step in data['processing_steps']]
        return cls(**data)


class MetadataTracker:
    """
    Comprehensive metadata tracking for all datasets with YAML/JSON output.
    Generates metadata files with source, date, and processing steps for every dataset.
    """
    
    def __init__(self, path_manager: AbsolutePathManager, metadata_format: str = 'yaml'):
        """
        Initialize metadata tracker.
        
        Args:
            path_manager: AbsolutePathManager instance for path operations
            metadata_format: Format for metadata files ('yaml' or 'json')
        """
        self.path_manager = path_manager
        self.metadata_format = metadata_format.lower()
        
        if self.metadata_format not in ['yaml', 'json']:
            raise ValueError("metadata_format must be 'yaml' or 'json'")
        
        logger.info(f"Initialized MetadataTracker with format: {self.metadata_format}")
    
    def _get_metadata_path(self, file_path: Path) -> Path:
        """Get the metadata file path for a given dataset file"""
        if self.metadata_format == 'yaml':
            return file_path.with_suffix(file_path.suffix + '.metadata.yaml')
        else:
            return file_path.with_suffix(file_path.suffix + '.metadata.json')
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file"""
        if not file_path.exists():
            return ""
        
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except (OSError, IOError) as e:
            logger.warning(f"Could not calculate checksum for {file_path}: {e}")
            return ""
    
    def _get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes"""
        try:
            return file_path.stat().st_size if file_path.exists() else 0
        except (OSError, IOError):
            return 0
    
    def _save_metadata(self, metadata: DatasetMetadata, metadata_path: Path) -> None:
        """Save metadata to file"""
        try:
            metadata_path = self.path_manager.ensure_file_writable(metadata_path)
            
            metadata_dict = metadata.to_dict()
            
            if self.metadata_format == 'yaml':
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    yaml.dump(metadata_dict, f, default_flow_style=False, indent=2)
            else:
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved metadata to: {metadata_path}")
            
        except (OSError, IOError, yaml.YAMLError, json.JSONEncodeError) as e:
            raise FileAccessError(str(metadata_path), "write metadata", str(e))
    
    def _load_metadata(self, metadata_path: Path) -> Optional[DatasetMetadata]:
        """Load metadata from file"""
        if not metadata_path.exists():
            return None
        
        try:
            self.path_manager.validate_path(metadata_path, must_exist=True, must_be_file=True)
            
            if self.metadata_format == 'yaml':
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata_dict = yaml.safe_load(f)
            else:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)
            
            return DatasetMetadata.from_dict(metadata_dict)
            
        except (OSError, IOError, yaml.YAMLError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Could not load metadata from {metadata_path}: {e}")
            return None
    
    def track_dataset(self, file_path: Union[str, Path], source_info: Dict[str, Any], 
                     processing_steps: Optional[List[Dict[str, Any]]] = None,
                     coordinate_system: Optional[str] = None,
                     spatial_extent: Optional[Dict[str, float]] = None) -> Path:
        """
        Create metadata file for dataset with comprehensive tracking information.
        
        Args:
            file_path: Path to the dataset file
            source_info: Information about data source (url, description, etc.)
            processing_steps: List of processing steps applied
            coordinate_system: Coordinate system information
            spatial_extent: Spatial extent as dict with keys: min_x, max_x, min_y, max_y
            
        Returns:
            Path to created metadata file
            
        Raises:
            PathResolutionError: If file path cannot be resolved
            FileAccessError: If metadata cannot be written
        """
        abs_file_path = self.path_manager.resolve_path(file_path)
        metadata_path = self._get_metadata_path(abs_file_path)
        
        # Convert processing steps to ProcessingStep objects
        steps = []
        if processing_steps:
            for step_dict in processing_steps:
                # Ensure required fields are present
                step_dict.setdefault('timestamp', datetime.now())
                step_dict.setdefault('parameters', {})
                step_dict.setdefault('input_files', [])
                step_dict.setdefault('output_files', [str(abs_file_path)])
                step_dict.setdefault('processing_time_seconds', 0.0)
                step_dict.setdefault('software_version', 'unknown')
                
                if isinstance(step_dict['timestamp'], str):
                    step_dict['timestamp'] = datetime.fromisoformat(step_dict['timestamp'])
                elif not isinstance(step_dict['timestamp'], datetime):
                    step_dict['timestamp'] = datetime.now()
                
                steps.append(ProcessingStep(**step_dict))
        
        # Create metadata object
        metadata = DatasetMetadata(
            file_path=str(abs_file_path),
            source=source_info.get('description', 'Unknown source'),
            creation_date=datetime.now(),
            processing_steps=steps,
            file_size_bytes=self._get_file_size(abs_file_path),
            checksum=self._calculate_checksum(abs_file_path),
            coordinate_system=coordinate_system,
            spatial_extent=spatial_extent
        )
        
        # Save metadata
        self._save_metadata(metadata, metadata_path)
        
        return metadata_path
    
    def update_metadata(self, file_path: Union[str, Path], processing_step: Dict[str, Any]) -> None:
        """
        Update existing metadata with new processing step.
        
        Args:
            file_path: Path to the dataset file
            processing_step: New processing step to add
            
        Raises:
            PathResolutionError: If file path cannot be resolved
            FileAccessError: If metadata cannot be updated
        """
        abs_file_path = self.path_manager.resolve_path(file_path)
        metadata_path = self._get_metadata_path(abs_file_path)
        
        # Load existing metadata
        metadata = self._load_metadata(metadata_path)
        if metadata is None:
            raise FileAccessError(
                str(metadata_path),
                "update metadata",
                "Metadata file does not exist or is corrupted"
            )
        
        # Prepare processing step
        processing_step.setdefault('timestamp', datetime.now())
        processing_step.setdefault('parameters', {})
        processing_step.setdefault('input_files', [])
        processing_step.setdefault('output_files', [str(abs_file_path)])
        processing_step.setdefault('processing_time_seconds', 0.0)
        processing_step.setdefault('software_version', 'unknown')
        
        if isinstance(processing_step['timestamp'], str):
            processing_step['timestamp'] = datetime.fromisoformat(processing_step['timestamp'])
        elif not isinstance(processing_step['timestamp'], datetime):
            processing_step['timestamp'] = datetime.now()
        
        # Add new processing step
        new_step = ProcessingStep(**processing_step)
        metadata.processing_steps.append(new_step)
        
        # Update file metadata
        metadata.file_size_bytes = self._get_file_size(abs_file_path)
        metadata.checksum = self._calculate_checksum(abs_file_path)
        
        # Save updated metadata
        self._save_metadata(metadata, metadata_path)
        
        logger.info(f"Updated metadata for {abs_file_path} with step: {new_step.step_name}")
    
    def get_metadata(self, file_path: Union[str, Path]) -> Optional[DatasetMetadata]:
        """
        Retrieve metadata for a dataset.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            DatasetMetadata object or None if not found
            
        Raises:
            PathResolutionError: If file path cannot be resolved
        """
        abs_file_path = self.path_manager.resolve_path(file_path)
        metadata_path = self._get_metadata_path(abs_file_path)
        
        return self._load_metadata(metadata_path)
    
    def verify_file_integrity(self, file_path: Union[str, Path]) -> bool:
        """
        Verify file integrity by comparing current checksum with stored metadata.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            True if file integrity is verified, False otherwise
        """
        metadata = self.get_metadata(file_path)
        if metadata is None:
            logger.warning(f"No metadata found for {file_path}")
            return False
        
        abs_file_path = self.path_manager.resolve_path(file_path)
        current_checksum = self._calculate_checksum(abs_file_path)
        
        if current_checksum != metadata.checksum:
            logger.error(f"File integrity check failed for {abs_file_path}")
            logger.error(f"Expected: {metadata.checksum}, Got: {current_checksum}")
            return False
        
        return True
    
    def list_datasets_with_metadata(self, directory: Union[str, Path]) -> List[Path]:
        """
        List all datasets in a directory that have metadata files.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of dataset file paths that have metadata
        """
        abs_dir = self.path_manager.resolve_path(directory)
        self.path_manager.validate_path(abs_dir, must_exist=True, must_be_dir=True)
        
        datasets = []
        suffix = '.metadata.yaml' if self.metadata_format == 'yaml' else '.metadata.json'
        
        for metadata_file in abs_dir.rglob(f'*{suffix}'):
            # Get the original dataset file path
            dataset_file = metadata_file.with_suffix('').with_suffix('')
            if dataset_file.exists():
                datasets.append(dataset_file)
        
        return datasets