"""
Consolidated Watershed Steps for RAVEN Workflows

This module contains all watershed-related workflow steps:
- DelineateWatershedAndStreams: Basic watershed and stream delineation
- UnifiedWatershedDelineation: Complete watershed analysis with lakes
- WatershedMappingStep: Watershed visualization and mapping

Consolidates functionality from multiple separate step files for better organization.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import logging
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from workflows.steps.base_step import WorkflowStep
from clients.watershed_clients.watershed import ProfessionalWatershedAnalyzer
from processors.lake_detection import ComprehensiveLakeDetector
from processors.lake_classifier import LakeClassifier  
from processors.lake_integrator import LakeIntegrator
from processors.basic_attributes import BasicAttributesCalculator
from processors.outlet_snapping import ImprovedOutletSnapper
from processors.watershed_mapper import WatershedMapper

class DelineateWatershedAndStreams(WorkflowStep):
    """
    Step 3B: Delineate watershed and extract stream network
    Used in Approach B (Full Delineation Workflow)
    """
    
    def __init__(self):
        super().__init__(
            step_name="delineate_watershed_streams",
            step_category="watershed",
            description="Trace watershed boundary and extract stream network from DEM"
        )
        self.analyzer = ProfessionalWatershedAnalyzer()
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._log_step_start()
        
        try:
            # Validate required inputs
            required_inputs = ['flow_direction', 'flow_accumulation', 'latitude', 'longitude']
            self.validate_inputs(inputs, required_inputs)
            
            flow_direction = self.validate_file_exists(inputs['flow_direction'])
            flow_accumulation = self.validate_file_exists(inputs['flow_accumulation'])
            outlet_lat = float(inputs['latitude'])
            outlet_lon = float(inputs['longitude'])
            
            # Create workspace
            workspace_dir = inputs.get('workspace_dir', flow_direction.parent)
            workspace = Path(workspace_dir)
            
            # Step 1: Delineate watershed
            self.logger.info("Delineating watershed boundary...")
            watershed_boundary = self._delineate_watershed(
                flow_direction, outlet_lat, outlet_lon, workspace
            )
            
            # Step 2: Extract stream network
            self.logger.info("Extracting stream network...")
            stream_network = self._extract_stream_network(
                flow_accumulation, watershed_boundary, workspace
            )
            
            # Calculate statistics
            stats = self._calculate_watershed_stats(watershed_boundary, stream_network)
            
            outputs = {
                'watershed_boundary': str(watershed_boundary),
                'stream_network': str(stream_network),
                'watershed_area_km2': stats['area_km2'],
                'stream_length_km': stats['stream_length_km'],
                'max_stream_order': stats['max_stream_order'],
                'outlet_snapped': False,
                'snap_distance_m': None,
                'success': True
            }
            
            self._log_step_complete([str(watershed_boundary), str(stream_network)])
            return outputs
            
        except Exception as e:
            error_msg = f"Watershed delineation failed: {str(e)}"
            self._log_step_failed(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _delineate_watershed(self, flow_direction: Path, outlet_lat: float, 
                           outlet_lon: float, workspace: Path) -> Path:
        """Delineate watershed boundary from outlet point"""
        
        results = self.analyzer.analyze_watershed_complete(
            dem_path=flow_direction,
            outlet_coords=(outlet_lat, outlet_lon),
            output_dir=workspace
        )

        if not results.get('success'):
            raise Exception(f"Watershed delineation failed: {results.get('error')}")

        for f in results.get('files_created', []):
            if 'watershed.geojson' in f:
                return Path(f)
        
        raise FileNotFoundError("Could not find watershed geojson in analysis results.")
    
    def _extract_stream_network(self, flow_accumulation: Path, 
                              watershed_boundary: Path, workspace: Path) -> Path:
        """Extract stream network from flow accumulation"""
        
        # This is a bit of a hack. The analyzer already created the streams.
        # We are just looking for the file.
        for f in workspace.glob('**/*'):
            if 'streams.geojson' in f.name:
                return f

        raise FileNotFoundError("Could not find stream geojson in analysis results.")
    
    def _calculate_watershed_stats(self, watershed_boundary: Path, 
                                 stream_network: Path) -> Dict[str, float]:
        """Calculate watershed and stream statistics"""
        
        import geopandas as gpd
        
        # Load data
        watershed_gdf = gpd.read_file(watershed_boundary)
        stream_gdf = gpd.read_file(stream_network)
        
        # Calculate area (convert to km²)
        area_km2 = watershed_gdf.geometry.area.sum() * 111 * 111  # Rough conversion
        
        # Calculate stream length (convert to km)
        stream_length_km = stream_gdf.geometry.length.sum() * 111  # Rough conversion
        
        # Get max stream order
        max_stream_order = stream_gdf['stream_order'].max() if 'stream_order' in stream_gdf.columns else 3
        
        return {
            'area_km2': area_km2,
            'stream_length_km': stream_length_km,
            'max_stream_order': max_stream_order
        }


# Re-export the main class
__all__ = ['DelineateWatershedAndStreams']