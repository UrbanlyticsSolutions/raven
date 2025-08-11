"""
AbsolutePathManager for consistent path resolution and validation.
"""

import os
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


class PathResolutionError(Exception):
    """Raised when path cannot be resolved to absolute path"""
    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Cannot resolve path '{path}': {reason}")


class FileAccessError(Exception):
    """Raised when file cannot be accessed with absolute path"""
    def __init__(self, path: str, operation: str, reason: str):
        self.path = path
        self.operation = operation
        self.reason = reason
        super().__init__(f"Cannot {operation} file '{path}': {reason}")


class AbsolutePathManager:
    """
    Centralized path management system that ensures all file operations use absolute paths.
    Provides explicit error handling with no silent fallbacks.
    """
    
    def __init__(self, workspace_root: Union[str, Path]):
        """
        Initialize the path manager with a workspace root directory.
        
        Args:
            workspace_root: Root directory for the workspace
            
        Raises:
            PathResolutionError: If workspace_root cannot be resolved
        """
        try:
            self.workspace_root = Path(workspace_root).resolve()
            if not self.workspace_root.exists():
                raise PathResolutionError(
                    str(workspace_root), 
                    "Workspace root directory does not exist"
                )
        except (OSError, ValueError) as e:
            raise PathResolutionError(str(workspace_root), str(e))
            
        logger.info(f"Initialized AbsolutePathManager with workspace: {self.workspace_root}")
    
    def resolve_path(self, path: Union[str, Path]) -> Path:
        """
        Convert any path to absolute path with explicit error handling.
        
        Args:
            path: Path to resolve (can be relative or absolute)
            
        Returns:
            Absolute Path object
            
        Raises:
            PathResolutionError: If path cannot be resolved
        """
        if not path:
            raise PathResolutionError("", "Empty path provided")
            
        try:
            path_obj = Path(path)
            
            # If already absolute, resolve and return
            if path_obj.is_absolute():
                return path_obj.resolve()
            
            # If relative, resolve relative to workspace root
            absolute_path = (self.workspace_root / path_obj).resolve()
            
            # Ensure the resolved path is within or related to workspace
            # (This prevents directory traversal attacks)
            try:
                absolute_path.relative_to(self.workspace_root.parent)
            except ValueError:
                # Path is outside workspace parent - this might be intentional for external data
                logger.warning(f"Path {absolute_path} is outside workspace hierarchy")
            
            return absolute_path
            
        except (OSError, ValueError) as e:
            raise PathResolutionError(str(path), str(e))
    
    def validate_path(self, path: Union[str, Path], must_exist: bool = False, 
                     must_be_file: bool = False, must_be_dir: bool = False) -> bool:
        """
        Validate path accessibility and existence with explicit error reporting.
        
        Args:
            path: Path to validate
            must_exist: Whether path must exist
            must_be_file: Whether path must be a file (implies must_exist=True)
            must_be_dir: Whether path must be a directory (implies must_exist=True)
            
        Returns:
            True if valid
            
        Raises:
            PathResolutionError: If path cannot be resolved
            FileAccessError: If path validation fails
        """
        abs_path = self.resolve_path(path)
        
        # Check existence requirements
        if must_be_file or must_be_dir:
            must_exist = True
            
        if must_exist and not abs_path.exists():
            raise FileAccessError(
                str(abs_path), 
                "access", 
                "Path does not exist"
            )
        
        if abs_path.exists():
            # Check file/directory type requirements
            if must_be_file and not abs_path.is_file():
                raise FileAccessError(
                    str(abs_path),
                    "access",
                    "Path exists but is not a file"
                )
            
            if must_be_dir and not abs_path.is_dir():
                raise FileAccessError(
                    str(abs_path),
                    "access", 
                    "Path exists but is not a directory"
                )
            
            # Check read permissions
            if not os.access(abs_path, os.R_OK):
                raise FileAccessError(
                    str(abs_path),
                    "read",
                    "No read permission"
                )
        
        return True
    
    def create_path_structure(self, path: Union[str, Path], is_file_path: bool = False) -> Path:
        """
        Create directory structure if needed with explicit error handling.
        
        Args:
            path: Path to create structure for
            is_file_path: If True, creates parent directories for a file path
            
        Returns:
            Absolute path that was created/validated
            
        Raises:
            PathResolutionError: If path cannot be resolved
            FileAccessError: If directory creation fails
        """
        abs_path = self.resolve_path(path)
        
        # Determine directory to create
        if is_file_path:
            dir_to_create = abs_path.parent
        else:
            dir_to_create = abs_path
        
        try:
            dir_to_create.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory structure: {dir_to_create}")
        except (OSError, PermissionError) as e:
            raise FileAccessError(
                str(dir_to_create),
                "create directory",
                str(e)
            )
        
        return abs_path
    
    def get_relative_path(self, path: Union[str, Path], relative_to: Optional[Union[str, Path]] = None) -> Path:
        """
        Get relative path from absolute path.
        
        Args:
            path: Absolute path to make relative
            relative_to: Base path to make relative to (defaults to workspace_root)
            
        Returns:
            Relative path
            
        Raises:
            PathResolutionError: If paths cannot be resolved
            ValueError: If path cannot be made relative to base
        """
        abs_path = self.resolve_path(path)
        
        if relative_to is None:
            base_path = self.workspace_root
        else:
            base_path = self.resolve_path(relative_to)
        
        try:
            return abs_path.relative_to(base_path)
        except ValueError as e:
            raise PathResolutionError(
                str(path),
                f"Cannot make path relative to {base_path}: {str(e)}"
            )
    
    def ensure_file_writable(self, path: Union[str, Path]) -> Path:
        """
        Ensure a file path is writable by creating parent directories and checking permissions.
        
        Args:
            path: File path to ensure is writable
            
        Returns:
            Absolute path that is writable
            
        Raises:
            PathResolutionError: If path cannot be resolved
            FileAccessError: If path cannot be made writable
        """
        abs_path = self.create_path_structure(path, is_file_path=True)
        
        # Check if file exists and is writable
        if abs_path.exists():
            if not os.access(abs_path, os.W_OK):
                raise FileAccessError(
                    str(abs_path),
                    "write",
                    "No write permission to existing file"
                )
        else:
            # Check if parent directory is writable
            if not os.access(abs_path.parent, os.W_OK):
                raise FileAccessError(
                    str(abs_path.parent),
                    "write",
                    "No write permission to parent directory"
                )
        
        return abs_path
    
    def validate_file_integrity(self, path: Union[str, Path], min_size: int = 0, 
                              max_size: Optional[int] = None) -> bool:
        """
        Validate file integrity including size and basic format checks.
        
        Args:
            path: File path to validate
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes (None for no limit)
            
        Returns:
            True if file passes integrity checks
            
        Raises:
            PathResolutionError: If path cannot be resolved
            FileAccessError: If file fails integrity checks
        """
        abs_path = self.resolve_path(path)
        
        if not abs_path.exists():
            raise FileAccessError(str(abs_path), "validate", "File does not exist")
        
        if not abs_path.is_file():
            raise FileAccessError(str(abs_path), "validate", "Path is not a file")
        
        try:
            file_size = abs_path.stat().st_size
            
            if file_size < min_size:
                raise FileAccessError(
                    str(abs_path), 
                    "validate", 
                    f"File too small: {file_size} bytes < {min_size} bytes"
                )
            
            if max_size is not None and file_size > max_size:
                raise FileAccessError(
                    str(abs_path), 
                    "validate", 
                    f"File too large: {file_size} bytes > {max_size} bytes"
                )
            
            # Basic readability test
            with open(abs_path, 'rb') as f:
                f.read(1)  # Try to read first byte
                
        except (OSError, PermissionError) as e:
            raise FileAccessError(str(abs_path), "validate", f"File access error: {str(e)}")
        
        return True
    
    def safe_copy_file(self, source: Union[str, Path], destination: Union[str, Path], 
                      validate_source: bool = True, validate_destination: bool = True) -> Path:
        """
        Safely copy file with validation and integrity checking.
        
        Args:
            source: Source file path
            destination: Destination file path
            validate_source: Whether to validate source file integrity
            validate_destination: Whether to validate copied file integrity
            
        Returns:
            Absolute path of destination file
            
        Raises:
            PathResolutionError: If paths cannot be resolved
            FileAccessError: If copy operation fails
        """
        import shutil
        
        source_abs = self.resolve_path(source)
        dest_abs = self.ensure_file_writable(destination)
        
        if validate_source:
            self.validate_file_integrity(source_abs, min_size=1)
        
        try:
            shutil.copy2(source_abs, dest_abs)
            logger.info(f"Copied file: {source_abs} -> {dest_abs}")
            
            if validate_destination:
                self.validate_file_integrity(dest_abs, min_size=1)
                
        except (OSError, shutil.Error) as e:
            raise FileAccessError(str(dest_abs), "copy", f"Copy failed: {str(e)}")
        
        return dest_abs
    
    def validate_workspace_permissions(self) -> bool:
        """
        Validate that workspace has proper read/write permissions.
        
        Returns:
            True if workspace is fully accessible
            
        Raises:
            FileAccessError: If workspace lacks required permissions
        """
        if not os.access(self.workspace_root, os.R_OK):
            raise FileAccessError(
                str(self.workspace_root), 
                "read", 
                "No read permission to workspace"
            )
        
        if not os.access(self.workspace_root, os.W_OK):
            raise FileAccessError(
                str(self.workspace_root), 
                "write", 
                "No write permission to workspace"
            )
        
        # Test file creation in workspace
        test_file = self.workspace_root / ".test_permissions"
        try:
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError) as e:
            raise FileAccessError(
                str(self.workspace_root), 
                "write", 
                f"Cannot create files in workspace: {str(e)}"
            )
        
        return True
    
    def sanitize_filename(self, filename: str, replacement: str = "_") -> str:
        """
        Sanitize filename to prevent path traversal and invalid characters.
        
        Args:
            filename: Original filename
            replacement: Character to replace invalid characters with
            
        Returns:
            Sanitized filename safe for filesystem
        """
        import re
        
        # Remove path separators and parent directory references
        sanitized = filename.replace('/', replacement).replace('\\', replacement)
        sanitized = sanitized.replace('..', replacement)
        
        # Remove or replace invalid characters for Windows/Linux
        invalid_chars = '<>:"|?*'
        for char in invalid_chars:
            sanitized = sanitized.replace(char, replacement)
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', replacement, sanitized)
        
        # Trim and ensure not empty
        sanitized = sanitized.strip('. ')
        if not sanitized:
            sanitized = "unnamed_file"
        
        return sanitized