"""
Watershed Mapping Step

Creates comprehensive maps and visualizations of watershed delineation results.
Integrates with the full delineation workflow to provide visual outputs.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from processors.watershed_mapper import WatershedMapper

class WatershedMappingStep:
    """
    Workflow step for creating watershed maps and visualizations
    """
    
    def __init__(self, workspace_dir: Path = None):
        """
        Initialize watershed mapping step
        
        Parameters:
        -----------
        workspace_dir : Path, optional
            Working directory for processing
        """
        self.workspace_dir = workspace_dir or Path.cwd() / "mapping"
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize mapper
        self.mapper = WatershedMapper(workspace_dir=self.workspace_dir)
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the step"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(self.workspace_dir / "watershed_mapping.log")
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def execute(self, watershed_results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute watershed mapping step
        
        Parameters:
        -----------
        watershed_results : dict
            Results from watershed delineation containing file paths
        **kwargs : additional parameters
            outlet_name : str, optional - Name for the watershed
            create_summary : bool, optional - Create summary statistics plot
            
        Returns:
        --------
        Dict[str, Any]
            Mapping results with file paths and metadata
        """
        
        outlet_name = kwargs.get('outlet_name', 'Watershed')
        create_summary = kwargs.get('create_summary', True)
        
        self.logger.info(f"Starting watershed mapping for: {outlet_name}")
        
        try:
            # Extract required files from watershed results
            watershed_file = watershed_results.get('watershed_file')
            streams_file = watershed_results.get('original_stream_network')
            outlet_coords = watershed_results.get('outlet_coordinates', (0, 0))
            
            # Optional files
            dem_file = kwargs.get('dem_file')
            lakes_file = watershed_results.get('all_lakes_file')
            connected_lakes_file = watershed_results.get('connected_lakes_file')
            non_connected_lakes_file = watershed_results.get('non_connected_lakes_file')
            subbasins_file = kwargs.get('subbasins_file')
            
            # Validate required files
            if not watershed_file or not Path(watershed_file).exists():
                return {
                    'success': False,
                    'error': f'Watershed file not found: {watershed_file}'
                }
            
            if not streams_file or not Path(streams_file).exists():
                return {
                    'success': False,
                    'error': f'Streams file not found: {streams_file}'
                }
            
            # Create main watershed map
            map_title = f"{outlet_name} - Watershed Delineation Results"
            main_map_file = str(self.workspace_dir / f"{outlet_name.lower().replace(' ', '_')}_watershed_map.png")
            
            main_map_result = self.mapper.create_comprehensive_map(
                watershed_file=watershed_file,
                streams_file=streams_file,
                outlet_coords=outlet_coords,
                dem_file=dem_file,
                lakes_file=lakes_file,
                connected_lakes_file=connected_lakes_file,
                non_connected_lakes_file=non_connected_lakes_file,
                subbasins_file=subbasins_file,
                output_file=main_map_file,
                title=map_title
            )
            
            if not main_map_result['success']:
                return main_map_result
            
            results = {
                'success': True,
                'step_type': 'watershed_mapping',
                'outlet_name': outlet_name,
                'main_map_file': main_map_result['map_file'],
                'map_title': map_title,
                'components_mapped': main_map_result['components_mapped'],
                'files_created': [main_map_result['map_file']]
            }
            
            # Create summary statistics plot if requested
            if create_summary:
                summary_file = str(self.workspace_dir / f"{outlet_name.lower().replace(' ', '_')}_summary.png")
                
                summary_result = self.mapper.create_summary_plot(
                    results_dict=watershed_results,
                    output_file=summary_file
                )
                
                if summary_result['success']:
                    results['summary_plot_file'] = summary_result['summary_plot']
                    results['statistics'] = summary_result['statistics']
                    results['files_created'].append(summary_result['summary_plot'])
                else:
                    self.logger.warning(f"Summary plot creation failed: {summary_result.get('error')}")
            
            # Add mapping metadata
            results.update({
                'watershed_area_km2': main_map_result['watershed_area_km2'],
                'stream_length_km': main_map_result['stream_length_km'],
                'mapping_workspace': str(self.workspace_dir),
                'files_count': len(results['files_created'])
            })
            
            self.logger.info(f"Watershed mapping completed successfully for {outlet_name}")
            self.logger.info(f"Files created: {len(results['files_created'])}")
            self.logger.info(f"Components mapped: {', '.join(results['components_mapped'])}")
            
            return results
            
        except Exception as e:
            error_msg = f"Watershed mapping failed for {outlet_name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'step_type': 'watershed_mapping'
            }

if __name__ == "__main__":
    # Test the mapping step
    step = WatershedMappingStep()
    
    # Example watershed results (would come from actual delineation)
    test_results = {
        'watershed_file': '/path/to/watershed.shp',
        'original_stream_network': '/path/to/streams.geojson',
        'outlet_coordinates': (-118.93, 49.73),
        'watershed_area_km2': 125.5,
        'stream_length_km': 45.2,
        'connected_lake_count': 2,
        'non_connected_lake_count': 5,
        'total_lake_area_km2': 3.4,
        'max_stream_order': 4
    }
    
    result = step.execute(
        watershed_results=test_results,
        outlet_name="Test Watershed",
        create_summary=True
    )
    
    if result['success']:
        print(f"✅ Mapping successful!")
        print(f"Main map: {result['main_map_file']}")
        if 'summary_plot_file' in result:
            print(f"Summary plot: {result['summary_plot_file']}")
    else:
        print(f"❌ Mapping failed: {result['error']}")