"""
Project Management Step for RAVEN Workflows

This module provides comprehensive project and output file management
for hydrological modeling workflows.
"""

import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from workflows.steps.base_step import WorkflowStep


class ProjectManagementStep(WorkflowStep):
    """
    Project structure creation and management workflow step
    
    Creates standardized project folder structure in root directory
    based on project name argument from the workflow.
    """
    
    def __init__(self, workspace_dir: Path = None):
        super().__init__(
            step_name="project_management",
            step_category="initialization", 
            description="Create and manage project folder structure"
        )
        
        # Use current working directory as default workspace
        self.workspace_dir = workspace_dir or Path.cwd()
        self.project_structure = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def initialize_project_structure(self, base_dir: Path) -> Dict[str, Any]:
        """
        Initialize comprehensive project directory structure
        
        Parameters:
        -----------
        base_dir : Path
            Base directory for project creation
            
        Returns:
        --------
        Dict with project structure information
        """
        try:
            project_root = base_dir / self.project_name
            
            # Define project structure
            structure = {
                'root': project_root,
                'data': {
                    'root': project_root / 'data',
                    'stations': project_root / 'data' / 'stations',
                    'spatial': project_root / 'data' / 'spatial',
                    'metadata': project_root / 'data' / 'metadata'
                },
                'processing': {
                    'root': project_root / 'processing',
                    'dem': project_root / 'processing' / 'dem',
                    'watersheds': project_root / 'processing' / 'watersheds',
                    'streams': project_root / 'processing' / 'streams',
                    'lakes': project_root / 'processing' / 'lakes'
                },
                'outputs': {
                    'root': project_root / 'outputs',
                    'shapefiles': project_root / 'outputs' / 'shapefiles',
                    'rasters': project_root / 'outputs' / 'rasters',
                    'models': project_root / 'outputs' / 'models',
                    'validation': project_root / 'outputs' / 'validation'
                },
                'reports': {
                    'root': project_root / 'reports',
                    'maps': project_root / 'reports' / 'maps',
                    'analysis': project_root / 'reports' / 'analysis'
                },
                'logs': {
                    'root': project_root / 'logs',
                    'steps': project_root / 'logs' / 'steps'
                }
            }
            
            # Create only root directories initially (lazy creation for subdirs)
            essential_dirs = [
                structure['root'],
                structure['data']['root'],
                structure['processing']['root'],
                structure['outputs']['root'],
                structure['reports']['root'],
                structure['logs']['root']
            ]
            
            for dir_path in essential_dirs:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            self.project_structure = structure
            
            # Create project manifest
            manifest = {
                'project_name': self.project_name,
                'created': datetime.now().isoformat(),
                'structure': {k: str(v) for k, v in structure.items() if isinstance(v, Path)},
                'version': '1.0',
                'workflow_type': 'raven_hydrological_modeling'
            }
            
            manifest_file = project_root / 'project_manifest.json'
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            
            return {
                'success': True,
                'project_root': str(project_root),
                'structure': structure,
                'manifest_file': str(manifest_file)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def ensure_directory(self, category: str, subcategory: str = None) -> Path:
        """
        Ensure directory exists and return path
        
        Parameters:
        -----------
        category : str
            Main category (data, processing, outputs, reports, logs)
        subcategory : str, optional
            Subcategory within main category
            
        Returns:
        --------
        Path to the directory
        """
        if not self.project_structure:
            raise ValueError("Project structure not initialized")
        
        if subcategory:
            dir_path = self.project_structure[category][subcategory]
        else:
            dir_path = self.project_structure[category]['root']
        
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def generate_filename(self, file_type: str, station_id: str = None, 
                         component: str = None, scope: str = 'single',
                         extension: str = None) -> str:
        """
        Generate standardized filename based on naming conventions
        
        Parameters:
        -----------
        file_type : str
            Type of file (watershed, streams, lakes, etc.)
        station_id : str, optional
            Station identifier
        component : str, optional
            Component name for the file
        scope : str
            Scope of the file (single, multi, all)
        extension : str, optional
            File extension (auto-detected from pattern if not provided)
            
        Returns:
        --------
        Standardized filename
        """
        if file_type not in self.naming_patterns:
            raise ValueError(f"Unknown file type: {file_type}")
        
        pattern = self.naming_patterns[file_type]
        
        # Extract extension from pattern if not provided
        if not extension and '.' in pattern:
            extension = pattern.split('.')[-1]
        
        # Build filename components
        filename_parts = {
            'project': self.project_name,
            'component': component or file_type,
            'station': station_id or 'multi',
            'timestamp': self.timestamp,
            'region': component or 'region',
            'scope': scope,
            'type': file_type
        }
        
        try:
            filename = pattern.format(**filename_parts)
        except KeyError as e:
            error_msg = f"Filename pattern error: {e}"
            raise ValueError(error_msg)
        
        return filename
    
    def organize_file(self, source_path: Path, file_type: str, 
                     station_id: str = None, component: str = None,
                     target_category: str = 'outputs') -> Dict[str, Any]:
        """
        Organize file into proper project structure with standardized naming
        
        Parameters:
        -----------
        source_path : Path
            Source file path
        file_type : str
            Type of file for organization
        station_id : str, optional
            Station identifier
        component : str, optional
            Component identifier
        target_category : str
            Target category (outputs, processing, etc.)
            
        Returns:
        --------
        Dict with organization results
        """
        try:
            if not source_path.exists():
                return {'success': False, 'error': f'Source file not found: {source_path}'}
            
            # Determine target directory based on file type
            subcategory_map = {
                'watershed': 'shapefiles',
                'streams': 'shapefiles', 
                'lakes': 'shapefiles',
                'outlets': 'shapefiles',
                'dem': 'rasters',
                'landcover': 'rasters',
                'soil': 'rasters',
                'summary': 'analysis',
                'report': 'analysis',
                'map': 'maps'
            }
            
            subcategory = subcategory_map.get(file_type, 'root')
            target_dir = self.ensure_directory(target_category, subcategory)
            
            # Generate standardized filename
            extension = source_path.suffix.lstrip('.')
            new_filename = self.generate_filename(
                file_type, station_id, component, extension=extension
            )
            
            target_path = target_dir / new_filename
            
            # Copy/move file based on category
            if target_category == 'outputs':
                shutil.copy2(source_path, target_path)
                operation = 'copied'
            else:
                shutil.move(str(source_path), str(target_path))
                operation = 'moved'
            
            # Handle shapefile components
            organized_files = [target_path]
            if extension == 'shp':
                for shp_ext in ['.shx', '.dbf', '.prj', '.cpg']:
                    component_source = source_path.with_suffix(shp_ext)
                    if component_source.exists():
                        component_target = target_path.with_suffix(shp_ext)
                        if target_category == 'outputs':
                            shutil.copy2(component_source, component_target)
                        else:
                            shutil.move(str(component_source), str(component_target))
                        organized_files.append(component_target)
            
            # Update file catalog
            catalog_entry = {
                'original_path': str(source_path),
                'organized_path': str(target_path),
                'file_type': file_type,
                'station_id': station_id,
                'component': component,
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'related_files': [str(f) for f in organized_files]
            }
            
            catalog_key = f"{file_type}_{station_id or 'multi'}_{component or 'default'}"
            self.file_catalog[catalog_key] = catalog_entry
            
            return {
                'success': True,
                'target_path': str(target_path),
                'organized_files': [str(f) for f in organized_files],
                'operation': operation
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_project_summary(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive project summary report
        
        Parameters:
        -----------
        workflow_results : dict
            Results from workflow execution
            
        Returns:
        --------
        Dict with summary creation results
        """
        try:
            summary_dir = self.ensure_directory('reports', 'analysis')
            
            # Generate comprehensive summary
            summary = {
                'project_info': {
                    'name': self.project_name,
                    'created': self.timestamp,
                    'completed': datetime.now().isoformat()
                },
                'workflow_results': workflow_results,
                'file_catalog': self.file_catalog,
                'project_structure': {
                    k: str(v) for k, v in self.project_structure.items() 
                    if isinstance(v, Path)
                },
                'statistics': self._calculate_project_statistics(),
                'quality_metrics': self._assess_quality_metrics()
            }
            
            # Save JSON summary
            summary_filename = self.generate_filename('summary', scope='complete')
            summary_path = summary_dir / summary_filename
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Create markdown report
            markdown_report = self._generate_markdown_report(summary)
            report_filename = self.generate_filename('report', scope='complete', extension='md')
            report_path = summary_dir / report_filename
            
            with open(report_path, 'w') as f:
                f.write(markdown_report)
            
            return {
                'success': True,
                'summary_path': str(summary_path),
                'report_path': str(report_path),
                'statistics': summary['statistics']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_project_statistics(self) -> Dict[str, Any]:
        """Calculate project statistics"""
        stats = {
            'total_files': len(self.file_catalog),
            'file_types': {},
            'stations_processed': set(),
            'total_size_mb': 0
        }
        
        for entry in self.file_catalog.values():
            file_type = entry['file_type']
            stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            
            if entry['station_id']:
                stats['stations_processed'].add(entry['station_id'])
            
            # Calculate file size
            try:
                file_path = Path(entry['organized_path'])
                if file_path.exists():
                    stats['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
            except:
                pass
        
        stats['stations_processed'] = len(stats['stations_processed'])
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        
        return stats
    
    def _assess_quality_metrics(self) -> Dict[str, Any]:
        """Assess project quality metrics"""
        return {
            'structure_completeness': self._check_structure_completeness(),
            'naming_consistency': self._check_naming_consistency(),
            'file_integrity': self._check_file_integrity()
        }
    
    def _check_structure_completeness(self) -> float:
        """Check if project structure is complete"""
        expected_dirs = ['data', 'processing', 'outputs', 'reports', 'logs']
        existing_dirs = [d for d in expected_dirs if d in self.project_structure]
        return len(existing_dirs) / len(expected_dirs)
    
    def _check_naming_consistency(self) -> float:
        """Check naming convention consistency"""
        consistent_files = 0
        total_files = len(self.file_catalog)
        
        for entry in self.file_catalog.values():
            path = Path(entry['organized_path'])
            if self.project_name in path.name:
                consistent_files += 1
        
        return consistent_files / total_files if total_files > 0 else 1.0
    
    def _check_file_integrity(self) -> float:
        """Check file integrity (existence and accessibility)"""
        accessible_files = 0
        total_files = len(self.file_catalog)
        
        for entry in self.file_catalog.values():
            path = Path(entry['organized_path'])
            if path.exists() and path.is_file():
                accessible_files += 1
        
        return accessible_files / total_files if total_files > 0 else 1.0
    
    def _generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        """Generate markdown report"""
        report = f"""# {self.project_name} - Project Report

## Project Information
- **Name**: {self.project_name}
- **Created**: {summary['project_info']['created']}
- **Completed**: {summary['project_info']['completed']}

## Statistics
- **Total Files**: {summary['statistics']['total_files']}
- **Stations Processed**: {summary['statistics']['stations_processed']}
- **Total Size**: {summary['statistics']['total_size_mb']} MB

## File Types
"""
        
        for file_type, count in summary['statistics']['file_types'].items():
            report += f"- **{file_type.title()}**: {count} files\n"
        
        report += f"""
## Quality Metrics
- **Structure Completeness**: {summary['quality_metrics']['structure_completeness']:.1f}
- **Naming Consistency**: {summary['quality_metrics']['naming_consistency']:.1f}
- **File Integrity**: {summary['quality_metrics']['file_integrity']:.1f}

## Project Structure
```
{self.project_name}/
|-- data/           # Input data and metadata
|-- processing/     # Intermediate processing files
|-- outputs/        # Final organized outputs
|-- reports/        # Analysis reports and maps
+-- logs/          # Execution logs
```

*Report generated automatically by RAVEN Workflow System*
"""
        
        return report
    
    def execute(self, project_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute project structure creation
        
        Parameters:
        -----------
        project_name : str
            Name of the project for folder creation
        **kwargs : dict
            Additional parameters (currently unused)
            
        Returns:
        --------
        Dict with project creation results and folder paths
        """
        try:
            # Create project folder in workspace directory 
            project_root = self.workspace_dir / project_name
            project_root.mkdir(parents=True, exist_ok=True)
            
            # Define standardized 4-folder structure
            structure = {
                'input_data': project_root / 'input_data',
                'processing_files': project_root / 'processing_files', 
                'analysis_results': project_root / 'analysis_results',
                'raven_inputs': project_root / 'raven_inputs'
            }
            
            # Create main directories only (single layer)
            for folder_name, folder_path in structure.items():
                folder_path.mkdir(parents=True, exist_ok=True)
            
            self.project_structure = structure
            
            result = {
                'success': True,
                'project_name': project_name,
                'project_root': str(project_root),
                'workspace_dir': str(self.workspace_dir),
                'folder_structure': {name: str(path) for name, path in structure.items()}
            }
            
            # Log success
            logger = logging.getLogger(self.__class__.__name__)
            logger.info(f"Project '{project_name}' created at {project_root}")
            logger.info(f"Created folders: {list(structure.keys())}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to create project structure: {str(e)}"
            logger = logging.getLogger(self.__class__.__name__)
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}