#!/usr/bin/env python3
"""
RAVEN Project Manager

Manages RAVEN hydrological modeling projects with organized directory structures,
metadata tracking, and workflow orchestration.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

class RAVENProjectManager:
    """
    RAVEN Project Manager for organizing and managing hydrological modeling projects
    """
    
    def __init__(self, workspace_dir: str = None):
        """
        Initialize project manager
        
        Parameters:
        -----------
        workspace_dir : str, optional
            Main workspace directory for all projects
        """
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "raven_projects"
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Project registry file
        self.registry_file = self.workspace_dir / "project_registry.json"
        self.registry = self._load_registry()
        
    def _setup_logging(self):
        """Setup logging for project management"""
        logger = logging.getLogger("RAVENProjectManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_file = self.workspace_dir / "project_manager.log"
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
            
        return logger
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load project registry"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load registry: {e}")
        
        return {
            "projects": {},
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_registry(self):
        """Save project registry"""
        self.registry["last_updated"] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def create_project(self, project_name: str, description: str = "",
                      project_type: str = "single_outlet") -> Dict[str, Any]:
        """
        Create a new RAVEN project with organized directory structure
        
        Parameters:
        -----------
        project_name : str
            Unique name for the project
        description : str, optional
            Project description
        project_type : str, optional
            Type of project ('single_outlet', 'multi_gauge', 'routing_product')
            
        Returns:
        --------
        Dict with project information and directory structure
        """
        if project_name in self.registry["projects"]:
            return {
                'success': False,
                'error': f'Project {project_name} already exists'
            }
        
        # Create project directory structure
        project_dir = self.workspace_dir / project_name
        
        # Simplified RAVEN project structure
        subdirs = [
            'data',               # All input and processed data
            'model',              # RAVEN model files (.rvh, .rvp, .rvi, .rvt, .rvc)
            'results',            # All outputs (shapefiles, maps, logs)
        ]
        
        for subdir in subdirs:
            (project_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Create project metadata
        project_metadata = {
            'name': project_name,
            'description': description,
            'project_type': project_type,
            'created': datetime.now().isoformat(),
            'status': 'created',
            'directory': str(project_dir),
            'workflow_steps': [],
            'data_sources': {},
            'model_configuration': {},
            'last_modified': datetime.now().isoformat()
        }
        
        # Save project metadata file
        metadata_file = project_dir / 'project_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(project_metadata, f, indent=2, default=str)
        
        # Create project README
        readme_content = f"""# {project_name}

## Project Overview
{description}

**Project Type**: {project_type}  
**Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Directory Structure
- `data/` - All input data (DEM, landcover, soil) and processed files
- `model/` - RAVEN model files (.rvh, .rvp, .rvi, .rvt, .rvc)
- `results/` - All outputs (shapefiles, maps, logs, analysis)

## Workflow Status
Status: {project_metadata['status']}

## Model Configuration
To be populated during workflow execution.

## Data Sources
To be populated during data acquisition.
"""
        
        readme_file = project_dir / 'README.md'
        readme_file.write_text(readme_content)
        
        # Add to registry
        self.registry["projects"][project_name] = project_metadata
        self._save_registry()
        
        self.logger.info(f"Created project: {project_name}")
        
        return {
            'success': True,
            'project_name': project_name,
            'project_dir': str(project_dir),
            'metadata': project_metadata,
            'directories_created': len(subdirs)
        }
    
    def get_project(self, project_name: str) -> Dict[str, Any]:
        """Get project information"""
        if project_name not in self.registry["projects"]:
            return {
                'success': False,
                'error': f'Project {project_name} not found'
            }
        
        project_info = self.registry["projects"][project_name].copy()
        
        # Check if project directory exists
        project_dir = Path(project_info['directory'])
        if not project_dir.exists():
            project_info['directory_exists'] = False
            project_info['status'] = 'directory_missing'
        else:
            project_info['directory_exists'] = True
        
        return {
            'success': True,
            'project': project_info
        }
    
    def list_projects(self) -> Dict[str, Any]:
        """List all projects"""
        projects = []
        for name, info in self.registry["projects"].items():
            project_summary = {
                'name': name,
                'description': info.get('description', ''),
                'project_type': info.get('project_type', 'unknown'),
                'status': info.get('status', 'unknown'),
                'created': info.get('created', ''),
                'last_modified': info.get('last_modified', ''),
                'directory': info.get('directory', '')
            }
            projects.append(project_summary)
        
        return {
            'success': True,
            'project_count': len(projects),
            'projects': projects
        }
    
    def update_project_status(self, project_name: str, status: str, 
                            workflow_step: str = None, metadata_updates: Dict = None) -> Dict[str, Any]:
        """Update project status and metadata"""
        if project_name not in self.registry["projects"]:
            return {
                'success': False,
                'error': f'Project {project_name} not found'
            }
        
        project = self.registry["projects"][project_name]
        project['status'] = status
        project['last_modified'] = datetime.now().isoformat()
        
        if workflow_step:
            if 'workflow_steps' not in project:
                project['workflow_steps'] = []
            
            step_info = {
                'step': workflow_step,
                'completed': datetime.now().isoformat(),
                'status': 'completed'
            }
            project['workflow_steps'].append(step_info)
        
        if metadata_updates:
            project.update(metadata_updates)
        
        # Update project metadata file
        project_dir = Path(project['directory'])
        if project_dir.exists():
            metadata_file = project_dir / 'project_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(project, f, indent=2, default=str)
        
        self._save_registry()
        
        return {
            'success': True,
            'project_name': project_name,
            'new_status': status
        }
    
    def delete_project(self, project_name: str, remove_files: bool = False) -> Dict[str, Any]:
        """Delete project from registry and optionally remove files"""
        if project_name not in self.registry["projects"]:
            return {
                'success': False,
                'error': f'Project {project_name} not found'
            }
        
        project = self.registry["projects"][project_name]
        
        if remove_files:
            project_dir = Path(project['directory'])
            if project_dir.exists():
                import shutil
                shutil.rmtree(project_dir)
                self.logger.info(f"Removed project directory: {project_dir}")
        
        del self.registry["projects"][project_name]
        self._save_registry()
        
        self.logger.info(f"Deleted project: {project_name}")
        
        return {
            'success': True,
            'project_name': project_name,
            'files_removed': remove_files
        }
    
    def get_project_directory(self, project_name: str, subdir: str = None) -> Optional[Path]:
        """Get project directory path"""
        if project_name not in self.registry["projects"]:
            return None
        
        project_dir = Path(self.registry["projects"][project_name]['directory'])
        
        if subdir:
            return project_dir / subdir
        
        return project_dir


def main():
    """CLI interface for project manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAVEN Project Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create project
    create_parser = subparsers.add_parser('create', help='Create new project')
    create_parser.add_argument('name', help='Project name')
    create_parser.add_argument('--description', default='', help='Project description')
    create_parser.add_argument('--type', default='single_outlet', 
                              choices=['single_outlet', 'multi_gauge', 'routing_product'],
                              help='Project type')
    
    # List projects
    list_parser = subparsers.add_parser('list', help='List all projects')
    
    # Get project info
    info_parser = subparsers.add_parser('info', help='Get project information')
    info_parser.add_argument('name', help='Project name')
    
    # Delete project
    delete_parser = subparsers.add_parser('delete', help='Delete project')
    delete_parser.add_argument('name', help='Project name')
    delete_parser.add_argument('--remove-files', action='store_true', 
                              help='Also remove project files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize project manager
    pm = RAVENProjectManager()
    
    if args.command == 'create':
        result = pm.create_project(args.name, args.description, args.type)
        if result['success']:
            print(f"✓ Created project: {args.name}")
            print(f"  Directory: {result['project_dir']}")
            print(f"  Directories created: {result['directories_created']}")
        else:
            print(f"✗ Failed to create project: {result['error']}")
    
    elif args.command == 'list':
        result = pm.list_projects()
        if result['project_count'] == 0:
            print("No projects found.")
        else:
            print(f"Found {result['project_count']} projects:")
            for project in result['projects']:
                print(f"  {project['name']} ({project['project_type']}) - {project['status']}")
    
    elif args.command == 'info':
        result = pm.get_project(args.name)
        if result['success']:
            project = result['project']
            print(f"Project: {project['name']}")
            print(f"Description: {project.get('description', 'N/A')}")
            print(f"Type: {project.get('project_type', 'N/A')}")
            print(f"Status: {project.get('status', 'N/A')}")
            print(f"Created: {project.get('created', 'N/A')}")
            print(f"Directory: {project.get('directory', 'N/A')}")
            print(f"Directory exists: {project.get('directory_exists', 'N/A')}")
        else:
            print(f"✗ {result['error']}")
    
    elif args.command == 'delete':
        result = pm.delete_project(args.name, args.remove_files)
        if result['success']:
            print(f"✓ Deleted project: {args.name}")
            if result['files_removed']:
                print("  Project files removed")
        else:
            print(f"✗ Failed to delete project: {result['error']}")


if __name__ == "__main__":
    main()