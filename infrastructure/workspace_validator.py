"""
Comprehensive workspace validation and integrity checking system.
Validates workspace structure, permissions, file integrity, and disk space.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set
import logging
import psutil
import hashlib
from .path_manager import AbsolutePathManager, PathResolutionError, FileAccessError
from .file_operations import SecureFileOperations, FileValidationError

logger = logging.getLogger(__name__)


class WorkspaceValidationError(Exception):
    """Raised when workspace validation fails"""
    def __init__(self, validation_type: str, details: str):
        self.validation_type = validation_type
        self.details = details
        super().__init__(f"Workspace validation failed ({validation_type}): {details}")


class WorkspaceValidator:
    """
    Comprehensive workspace validation system with integrity checking.
    """
    
    def __init__(self, path_manager: AbsolutePathManager):
        """
        Initialize validator with path manager.
        
        Args:
            path_manager: Configured AbsolutePathManager instance
        """
        self.path_manager = path_manager
        self.file_ops = SecureFileOperations(path_manager)
        self.workspace_root = path_manager.workspace_root
        
    def validate_workspace_complete(self, 
                                  required_dirs: Optional[List[str]] = None,
                                  required_files: Optional[List[str]] = None,
                                  min_free_space_gb: float = 1.0,
                                  check_file_integrity: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive workspace validation.
        
        Args:
            required_dirs: List of required directory paths (relative to workspace)
            required_files: List of required file paths (relative to workspace)
            min_free_space_gb: Minimum free disk space required in GB
            check_file_integrity: Whether to perform file integrity checks
            
        Returns:
            Dictionary with validation results and metrics
            
        Raises:
            WorkspaceValidationError: If validation fails
        """
        validation_results = {
            'workspace_root': str(self.workspace_root),
            'validation_timestamp': self._get_timestamp(),
            'checks_performed': [],
            'warnings': [],
            'errors': [],
            'metrics': {}
        }
        
        try:
            # 1. Basic workspace access validation
            print("Validating workspace access permissions...")
            self._validate_workspace_access()
            validation_results['checks_performed'].append('workspace_access')
            
            # 2. Disk space validation
            print(f"Checking disk space (minimum: {min_free_space_gb} GB)...")
            space_metrics = self._validate_disk_space(min_free_space_gb)
            validation_results['metrics'].update(space_metrics)
            validation_results['checks_performed'].append('disk_space')
            
            # 3. Directory structure validation
            if required_dirs:
                print("Validating required directory structure...")
                dir_results = self._validate_directory_structure(required_dirs)
                validation_results['metrics'].update(dir_results)
                validation_results['checks_performed'].append('directory_structure')
            
            # 4. Required files validation
            if required_files:
                print("Validating required files...")
                file_results = self._validate_required_files(required_files, check_file_integrity)
                validation_results['metrics'].update(file_results)
                validation_results['checks_performed'].append('required_files')
            
            # 5. File system health check
            print("Performing file system health check...")
            health_results = self._validate_filesystem_health()
            validation_results['metrics'].update(health_results)
            validation_results['checks_performed'].append('filesystem_health')
            
            # 6. Security validation
            print("Validating workspace security...")
            security_results = self._validate_workspace_security()
            validation_results['metrics'].update(security_results)
            validation_results['checks_performed'].append('workspace_security')
            
            validation_results['overall_status'] = 'PASSED'
            validation_results['validation_summary'] = self._generate_validation_summary(validation_results)
            
            print("Workspace validation PASSED")
            return validation_results
            
        except WorkspaceValidationError as e:
            validation_results['errors'].append(str(e))
            validation_results['overall_status'] = 'FAILED'
            validation_results['failure_reason'] = e.details
            print(f"Workspace validation FAILED: {e}")
            raise
        except Exception as e:
            validation_results['errors'].append(f"Unexpected error: {str(e)}")
            validation_results['overall_status'] = 'ERROR'
            print(f"Workspace validation ERROR: {e}")
            raise WorkspaceValidationError("unexpected_error", f"Validation failed: {str(e)}")
    
    def _validate_workspace_access(self) -> None:
        """Validate basic workspace access permissions."""
        try:
            self.path_manager.validate_workspace_permissions()
        except FileAccessError as e:
            raise WorkspaceValidationError("workspace_access", str(e))
    
    def _validate_disk_space(self, min_free_space_gb: float) -> Dict[str, Any]:
        """
        Validate available disk space.
        
        Args:
            min_free_space_gb: Minimum required free space in GB
            
        Returns:
            Dictionary with disk space metrics
        """
        try:
            disk_usage = shutil.disk_usage(self.workspace_root)
            free_space_bytes = disk_usage.free
            total_space_bytes = disk_usage.total
            used_space_bytes = disk_usage.used
            
            free_space_gb = free_space_bytes / (1024**3)
            total_space_gb = total_space_bytes / (1024**3)
            used_space_gb = used_space_bytes / (1024**3)
            used_percent = (used_space_bytes / total_space_bytes) * 100
            
            metrics = {
                'disk_free_gb': round(free_space_gb, 2),
                'disk_total_gb': round(total_space_gb, 2),
                'disk_used_gb': round(used_space_gb, 2),
                'disk_used_percent': round(used_percent, 1),
                'disk_space_check': 'PASSED' if free_space_gb >= min_free_space_gb else 'FAILED'
            }
            
            if free_space_gb < min_free_space_gb:
                raise WorkspaceValidationError(
                    "disk_space",
                    f"Insufficient disk space: {free_space_gb:.2f} GB available, {min_free_space_gb} GB required"
                )
            
            # Warning for low disk space (less than 5GB or 90% full)
            if free_space_gb < 5.0 or used_percent > 90:
                logger.warning(f"Low disk space warning: {free_space_gb:.2f} GB free ({used_percent:.1f}% used)")
            
            return metrics
            
        except Exception as e:
            raise WorkspaceValidationError("disk_space", f"Disk space check failed: {str(e)}")
    
    def _validate_directory_structure(self, required_dirs: List[str]) -> Dict[str, Any]:
        """
        Validate required directory structure.
        
        Args:
            required_dirs: List of required directory paths (relative to workspace)
            
        Returns:
            Dictionary with directory validation metrics
        """
        missing_dirs = []
        invalid_dirs = []
        valid_dirs = []
        
        for dir_path in required_dirs:
            try:
                abs_dir = self.path_manager.resolve_path(dir_path)
                self.path_manager.validate_path(abs_dir, must_exist=True, must_be_dir=True)
                valid_dirs.append(str(abs_dir))
            except PathResolutionError as e:
                invalid_dirs.append(f"{dir_path}: {str(e)}")
            except FileAccessError as e:
                if "does not exist" in str(e):
                    missing_dirs.append(dir_path)
                else:
                    invalid_dirs.append(f"{dir_path}: {str(e)}")
        
        metrics = {
            'required_directories_count': len(required_dirs),
            'valid_directories_count': len(valid_dirs),
            'missing_directories_count': len(missing_dirs),
            'invalid_directories_count': len(invalid_dirs),
            'missing_directories': missing_dirs,
            'invalid_directories': invalid_dirs,
            'directory_structure_check': 'PASSED' if not missing_dirs and not invalid_dirs else 'FAILED'
        }
        
        if missing_dirs or invalid_dirs:
            raise WorkspaceValidationError(
                "directory_structure",
                f"Directory validation failed. Missing: {missing_dirs}, Invalid: {invalid_dirs}"
            )
        
        return metrics
    
    def _validate_required_files(self, required_files: List[str], check_integrity: bool) -> Dict[str, Any]:
        """
        Validate required files exist and optionally check integrity.
        
        Args:
            required_files: List of required file paths (relative to workspace)
            check_integrity: Whether to perform file integrity checks
            
        Returns:
            Dictionary with file validation metrics
        """
        missing_files = []
        invalid_files = []
        valid_files = []
        corrupted_files = []
        
        for file_path in required_files:
            try:
                abs_file = self.path_manager.resolve_path(file_path)
                self.path_manager.validate_path(abs_file, must_exist=True, must_be_file=True)
                
                if check_integrity:
                    try:
                        self.path_manager.validate_file_integrity(abs_file, min_size=1)
                        valid_files.append(str(abs_file))
                    except FileAccessError as e:
                        corrupted_files.append(f"{file_path}: {str(e)}")
                else:
                    valid_files.append(str(abs_file))
                    
            except PathResolutionError as e:
                invalid_files.append(f"{file_path}: {str(e)}")
            except FileAccessError as e:
                if "does not exist" in str(e):
                    missing_files.append(file_path)
                else:
                    invalid_files.append(f"{file_path}: {str(e)}")
        
        metrics = {
            'required_files_count': len(required_files),
            'valid_files_count': len(valid_files),
            'missing_files_count': len(missing_files),
            'invalid_files_count': len(invalid_files),
            'corrupted_files_count': len(corrupted_files),
            'missing_files': missing_files,
            'invalid_files': invalid_files,
            'corrupted_files': corrupted_files,
            'required_files_check': 'PASSED' if not missing_files and not invalid_files and not corrupted_files else 'FAILED'
        }
        
        if missing_files or invalid_files or corrupted_files:
            raise WorkspaceValidationError(
                "required_files",
                f"File validation failed. Missing: {missing_files}, Invalid: {invalid_files}, Corrupted: {corrupted_files}"
            )
        
        return metrics
    
    def _validate_filesystem_health(self) -> Dict[str, Any]:
        """
        Perform basic file system health checks.
        
        Returns:
            Dictionary with filesystem health metrics
        """
        try:
            # Test file creation and deletion
            test_file = self.workspace_root / ".health_check_test"
            test_content = "filesystem health check test"
            
            # Test write
            test_file.write_text(test_content)
            
            # Test read
            read_content = test_file.read_text()
            if read_content != test_content:
                raise WorkspaceValidationError("filesystem_health", "File write/read integrity check failed")
            
            # Test delete
            test_file.unlink()
            
            if test_file.exists():
                raise WorkspaceValidationError("filesystem_health", "File deletion failed")
            
            # Count files and directories in workspace
            file_count = 0
            dir_count = 0
            total_size = 0
            
            for item in self.workspace_root.rglob('*'):
                if item.is_file():
                    file_count += 1
                    try:
                        total_size += item.stat().st_size
                    except (OSError, PermissionError):
                        pass  # Skip files we can't access
                elif item.is_dir():
                    dir_count += 1
            
            return {
                'filesystem_health_check': 'PASSED',
                'workspace_file_count': file_count,
                'workspace_dir_count': dir_count,
                'workspace_total_size_mb': round(total_size / (1024**2), 2)
            }
            
        except Exception as e:
            raise WorkspaceValidationError("filesystem_health", f"Filesystem health check failed: {str(e)}")
    
    def _validate_workspace_security(self) -> Dict[str, Any]:
        """
        Perform basic workspace security validation.
        
        Returns:
            Dictionary with security validation metrics
        """
        try:
            # Check for suspicious files or patterns
            suspicious_extensions = {'.exe', '.bat', '.cmd', '.scr', '.vbs', '.js', '.jar'}
            suspicious_files = []
            
            # Check for files with suspicious extensions (in top level only)
            for item in self.workspace_root.iterdir():
                if item.is_file() and item.suffix.lower() in suspicious_extensions:
                    suspicious_files.append(str(item.relative_to(self.workspace_root)))
            
            # Check permissions
            permissions_secure = True
            permission_issues = []
            
            try:
                stat_info = self.workspace_root.stat()
                # On Unix systems, check if workspace is world-writable
                if hasattr(stat_info, 'st_mode'):
                    mode = stat_info.st_mode
                    if mode & 0o002:  # World writable
                        permissions_secure = False
                        permission_issues.append("Workspace is world-writable")
            except (AttributeError, OSError):
                # Not a Unix system or can't check permissions
                pass
            
            metrics = {
                'security_check': 'PASSED' if not suspicious_files and permissions_secure else 'WARNING',
                'suspicious_files_count': len(suspicious_files),
                'suspicious_files': suspicious_files,
                'permissions_secure': permissions_secure,
                'permission_issues': permission_issues
            }
            
            # Security issues are warnings, not failures
            if suspicious_files:
                logger.warning(f"Found suspicious files in workspace: {suspicious_files}")
            if not permissions_secure:
                logger.warning(f"Workspace security issues: {permission_issues}")
            
            return metrics
            
        except Exception as e:
            raise WorkspaceValidationError("workspace_security", f"Security validation failed: {str(e)}")
    
    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable validation summary."""
        checks = validation_results.get('checks_performed', [])
        metrics = validation_results.get('metrics', {})
        
        summary_lines = [
            f"Workspace: {validation_results['workspace_root']}",
            f"Validation Status: {validation_results['overall_status']}",
            f"Checks Performed: {', '.join(checks)}",
            f"Timestamp: {validation_results['validation_timestamp']}"
        ]
        
        if metrics.get('disk_free_gb'):
            summary_lines.append(f"Free Disk Space: {metrics['disk_free_gb']} GB")
        
        if metrics.get('workspace_file_count'):
            summary_lines.append(f"Files: {metrics['workspace_file_count']}, Directories: {metrics['workspace_dir_count']}")
        
        if validation_results.get('warnings'):
            summary_lines.append(f"Warnings: {len(validation_results['warnings'])}")
        
        if validation_results.get('errors'):
            summary_lines.append(f"Errors: {len(validation_results['errors'])}")
        
        return "\n".join(summary_lines)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def generate_validation_report(self, validation_results: Dict[str, Any], 
                                 output_file: Optional[Union[str, Path]] = None) -> str:
        """
        Generate detailed validation report.
        
        Args:
            validation_results: Results from validate_workspace_complete()
            output_file: Optional file path to write report
            
        Returns:
            Report content as string
        """
        report_lines = [
            "=" * 60,
            "WORKSPACE VALIDATION REPORT",
            "=" * 60,
            "",
            validation_results.get('validation_summary', ''),
            "",
            "DETAILED METRICS:",
            "-" * 20
        ]
        
        # Add detailed metrics
        metrics = validation_results.get('metrics', {})
        for key, value in metrics.items():
            if isinstance(value, (list, dict)):
                report_lines.append(f"{key}: {len(value) if isinstance(value, list) else 'complex'}")
                if isinstance(value, list) and value:
                    for item in value[:5]:  # Show first 5 items
                        report_lines.append(f"  - {item}")
                    if len(value) > 5:
                        report_lines.append(f"  ... and {len(value) - 5} more")
            else:
                report_lines.append(f"{key}: {value}")
        
        # Add warnings and errors
        if validation_results.get('warnings'):
            report_lines.extend(["", "WARNINGS:", "-" * 10])
            for warning in validation_results['warnings']:
                report_lines.append(f"  ! {warning}")
        
        if validation_results.get('errors'):
            report_lines.extend(["", "ERRORS:", "-" * 8])
            for error in validation_results['errors']:
                report_lines.append(f"  ✗ {error}")
        
        report_lines.extend(["", "=" * 60])
        
        report_content = "\n".join(report_lines)
        
        # Write to file if requested
        if output_file:
            try:
                report_path = self.file_ops.safe_write_text(output_file, report_content)
                logger.info(f"Validation report written to: {report_path}")
            except Exception as e:
                logger.error(f"Failed to write validation report: {e}")
        
        return report_content