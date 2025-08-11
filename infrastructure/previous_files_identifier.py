#!/usr/bin/env python3
"""
Previous Files Location Identifier
Maps and validates where all workflow step outputs are saved
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FileLocation:
    path: str
    step: str
    file_type: str
    exists: bool
    size_mb: Optional[float] = None
    modified: Optional[str] = None

class PreviousFilesIdentifier:
    """
    Identifies and maps all previous workflow step outputs
    """
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.file_map: Dict[str, List[FileLocation]] = {}
        
    def scan_all_previous_files(self) -> Dict[str, Any]:
        """
        Scan and identify ALL previous workflow files
        
        Returns:
        --------
        Dict containing complete file mapping
        """
        print("🔍 SCANNING ALL PREVIOUS WORKFLOW FILES...")
        print("="*50)
        
        file_summary = {
            'workspace_root': str(self.workspace_dir),
            'scan_timestamp': datetime.now().isoformat(),
            'step_results': {},
            'data_files': {},
            'model_files': {},
            'config_files': {},
            'summary': {
                'total_files': 0,
                'total_size_mb': 0,
                'missing_critical_files': []
            }
        }
        
        # 1. Scan step result files
        file_summary['step_results'] = self._scan_step_results()
        
        # 2. Scan data directory
        file_summary['data_files'] = self._scan_data_directory()
        
        # 3. Scan model files
        file_summary['model_files'] = self._scan_model_files()
        
        # 4. Scan config files
        file_summary['config_files'] = self._scan_config_files()
        
        # 5. Calculate summary
        file_summary['summary'] = self._calculate_summary(file_summary)
        
        # 6. Print detailed report
        self._print_detailed_report(file_summary)
        
        return file_summary
    
    def _scan_step_results(self) -> Dict[str, Any]:
        """Scan workflow step result files"""
        print("\n📊 STEP RESULT FILES:")
        
        step_results = {}
        
        for step in range(1, 6):
            step_file = self.workspace_dir / f"step{step}_results.json"
            
            if step_file.exists():
                try:
                    with open(step_file, 'r') as f:
                        data = json.load(f)
                    
                    file_info = {
                        'path': str(step_file),
                        'exists': True,
                        'success': data.get('success', False),
                        'size_mb': step_file.stat().st_size / (1024*1024),
                        'modified': datetime.fromtimestamp(step_file.stat().st_mtime).isoformat(),
                        'output_files': self._extract_output_files_from_step(data)
                    }
                    
                    status = "✅ SUCCESS" if data.get('success', False) else "❌ FAILED"
                    size_str = f"{file_info['size_mb']:.2f}MB"
                    print(f"  Step {step}: {status} ({size_str})")
                    
                except Exception as e:
                    file_info = {
                        'path': str(step_file),
                        'exists': True,
                        'error': f"Cannot read: {str(e)}",
                        'size_mb': step_file.stat().st_size / (1024*1024)
                    }
                    print(f"  Step {step}: ⚠️  ERROR reading file")
            else:
                file_info = {
                    'path': str(step_file),
                    'exists': False
                }
                print(f"  Step {step}: ❌ MISSING")
            
            step_results[f'step{step}'] = file_info
        
        return step_results
    
    def _scan_data_directory(self) -> Dict[str, Any]:
        """Scan data directory for workflow outputs"""
        print("\n📁 DATA DIRECTORY FILES:")
        
        data_dir = self.workspace_dir / "data"
        data_files = {
            'directory': str(data_dir),
            'exists': data_dir.exists(),
            'files': {}
        }
        
        if not data_dir.exists():
            print("  ❌ Data directory does not exist")
            return data_files
        
        # Categorize data files
        file_categories = {
            'geospatial': ['.shp', '.geojson', '.dbf', '.prj', '.shx'],
            'raster': ['.tif', '.tiff'],
            'tables': ['.csv', '.json'],
            'images': ['.png', '.jpg', '.jpeg'],
            'other': []
        }
        
        for file_path in data_dir.rglob("*"):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                category = 'other'
                
                for cat, extensions in file_categories.items():
                    if suffix in extensions:
                        category = cat
                        break
                
                if category not in data_files['files']:
                    data_files['files'][category] = []
                
                file_info = {
                    'name': file_path.name,
                    'path': str(file_path),
                    'size_mb': file_path.stat().st_size / (1024*1024),
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                
                data_files['files'][category].append(file_info)
        
        # Print summary by category
        for category, files in data_files['files'].items():
            if files:
                count = len(files)
                total_size = sum(f['size_mb'] for f in files)
                print(f"  {category.title()}: {count} files ({total_size:.1f}MB)")
        
        return data_files
    
    def _scan_model_files(self) -> Dict[str, Any]:
        """Scan model output files"""
        print("\n🎯 MODEL FILES:")
        
        models_dir = self.workspace_dir / "models"
        model_files = {
            'directory': str(models_dir),
            'exists': models_dir.exists(),
            'raven_files': {},
            'other_files': []
        }
        
        if not models_dir.exists():
            print("  ❌ Models directory does not exist")
            return model_files
        
        # Look for RAVEN model files
        raven_extensions = ['.rvh', '.rvp', '.rvi', '.rvt', '.rvc']
        
        for file_path in models_dir.rglob("*"):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                
                if suffix in raven_extensions:
                    file_type = suffix[1:]  # Remove dot
                    if file_type not in model_files['raven_files']:
                        model_files['raven_files'][file_type] = []
                    
                    file_info = {
                        'name': file_path.name,
                        'path': str(file_path),
                        'size_mb': file_path.stat().st_size / (1024*1024),
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }
                    model_files['raven_files'][file_type].append(file_info)
                else:
                    file_info = {
                        'name': file_path.name,
                        'path': str(file_path),
                        'size_mb': file_path.stat().st_size / (1024*1024)
                    }
                    model_files['other_files'].append(file_info)
        
        # Print RAVEN file summary
        raven_file_order = ['rvh', 'rvp', 'rvi', 'rvt', 'rvc']
        for file_type in raven_file_order:
            files = model_files['raven_files'].get(file_type, [])
            if files:
                print(f"  .{file_type}: {len(files)} files")
            else:
                print(f"  .{file_type}: ❌ MISSING")
        
        if model_files['other_files']:
            print(f"  Other: {len(model_files['other_files'])} files")
        
        return model_files
    
    def _scan_config_files(self) -> Dict[str, Any]:
        """Scan configuration files"""
        print("\n⚙️ CONFIGURATION FILES:")
        
        project_root = self.workspace_dir.parent if (self.workspace_dir / "config").exists() else Path(__file__).parent.parent
        config_dir = project_root / "config"
        
        config_files = {
            'directory': str(config_dir),
            'exists': config_dir.exists(),
            'files': {}
        }
        
        if not config_dir.exists():
            print("  ❌ Config directory does not exist")
            return config_files
        
        essential_configs = [
            'raven_config.json',
            'raven_class_definitions.json',
            'raven_lookup_database.json',
            'raven_complete_parameter_table.json'
        ]
        
        for config_name in essential_configs:
            config_path = config_dir / config_name
            
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        data = json.load(f)
                    
                    file_info = {
                        'path': str(config_path),
                        'exists': True,
                        'valid_json': True,
                        'size_mb': config_path.stat().st_size / (1024*1024),
                        'sections': len(data) if isinstance(data, dict) else 'N/A'
                    }
                    print(f"  {config_name}: ✅ Valid")
                    
                except json.JSONDecodeError:
                    file_info = {
                        'path': str(config_path),
                        'exists': True,
                        'valid_json': False,
                        'error': 'Invalid JSON'
                    }
                    print(f"  {config_name}: ❌ Invalid JSON")
            else:
                file_info = {
                    'path': str(config_path),
                    'exists': False
                }
                print(f"  {config_name}: ❌ MISSING")
            
            config_files['files'][config_name] = file_info
        
        return config_files
    
    def _extract_output_files_from_step(self, step_data: Dict) -> List[str]:
        """Extract output file paths from step result data"""
        output_files = []
        
        # Common keys that contain file paths
        file_keys = ['files', 'files_created', 'output_files', 'exported_raven_files']
        
        for key in file_keys:
            if key in step_data:
                files = step_data[key]
                if isinstance(files, list):
                    output_files.extend(files)
                elif isinstance(files, dict):
                    output_files.extend(files.values())
        
        # Also check specific keys
        specific_keys = ['workspace', 'model_files', 'routing_table_file']
        for key in specific_keys:
            if key in step_data and isinstance(step_data[key], str):
                output_files.append(step_data[key])
        
        return [f for f in output_files if isinstance(f, str)]
    
    def _calculate_summary(self, file_summary: Dict) -> Dict[str, Any]:
        """Calculate overall summary statistics"""
        total_files = 0
        total_size_mb = 0
        missing_critical = []
        
        # Count files and sizes
        for section in ['step_results', 'data_files', 'model_files', 'config_files']:
            if section in file_summary:
                section_data = file_summary[section]
                
                if section == 'step_results':
                    for step, info in section_data.items():
                        if info.get('exists', False):
                            total_files += 1
                            total_size_mb += info.get('size_mb', 0)
                        else:
                            missing_critical.append(f"{step}_results.json")
                
                elif section == 'data_files' and section_data.get('exists', False):
                    for category, files in section_data.get('files', {}).items():
                        total_files += len(files)
                        total_size_mb += sum(f.get('size_mb', 0) for f in files)
                
                elif section == 'model_files' and section_data.get('exists', False):
                    for file_type, files in section_data.get('raven_files', {}).items():
                        if files:
                            total_files += len(files)
                            total_size_mb += sum(f.get('size_mb', 0) for f in files)
                        else:
                            missing_critical.append(f"RAVEN .{file_type} files")
                    
                    total_files += len(section_data.get('other_files', []))
                
                elif section == 'config_files' and section_data.get('exists', False):
                    for config_name, info in section_data.get('files', {}).items():
                        if info.get('exists', False):
                            total_files += 1
                            total_size_mb += info.get('size_mb', 0)
                        else:
                            missing_critical.append(config_name)
        
        return {
            'total_files': total_files,
            'total_size_mb': round(total_size_mb, 2),
            'missing_critical_files': missing_critical,
            'workspace_status': 'COMPLETE' if not missing_critical else 'INCOMPLETE'
        }
    
    def _print_detailed_report(self, file_summary: Dict):
        """Print detailed file location report"""
        print("\n" + "="*60)
        print("📊 COMPLETE FILE LOCATION SUMMARY")
        print("="*60)
        
        summary = file_summary['summary']
        
        print(f"📂 Workspace Root: {file_summary['workspace_root']}")
        print(f"📈 Total Files: {summary['total_files']}")
        print(f"💾 Total Size: {summary['total_size_mb']} MB")
        print(f"🎯 Status: {summary['workspace_status']}")
        
        if summary['missing_critical_files']:
            print("\n❌ MISSING CRITICAL FILES:")
            for missing_file in summary['missing_critical_files']:
                print(f"  - {missing_file}")
        
        print(f"\n⏰ Scan completed: {file_summary['scan_timestamp']}")

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Previous Files Location Identifier")
    parser.add_argument("workspace_dir", help="Workspace directory to scan")
    parser.add_argument("--save-report", help="Save detailed report to JSON file")
    
    args = parser.parse_args()
    
    identifier = PreviousFilesIdentifier(args.workspace_dir)
    file_summary = identifier.scan_all_previous_files()
    
    if args.save_report:
        report_path = Path(args.save_report)
        with open(report_path, 'w') as f:
            json.dump(file_summary, f, indent=2)
        print(f"\n💾 Detailed report saved: {report_path}")

if __name__ == "__main__":
    main()
