"""
Routing Product Steps for RAVEN Workflows

This module contains steps for working with existing BasinMaker routing products.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import geopandas as gpd
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from workflows.steps.base_step import WorkflowStep
from processors.subregion_extractor import SubregionExtractor
from workflows.steps.hydrosheds_adapter import HydroShedsAdapter

class ExtractSubregionFromRoutingProduct(WorkflowStep):
    """
    Step 2A: Extract subregion from routing product
    Used in Approach A (Routing Product Workflow)
    """
    
    def __init__(self):
        super().__init__(
            step_name="extract_subregion_routing",
            step_category="routing_product",
            description="Extract upstream watershed network from routing product"
        )
        
        self.extractor = SubregionExtractor()
        self.hydrosheds_adapter = HydroShedsAdapter()
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._log_step_start()
        
        try:
            # Validate required inputs
            required_inputs = ['routing_product_path', 'target_subbasin_id']
            self.validate_inputs(inputs, required_inputs)
            
            routing_product_path = Path(inputs['routing_product_path'])
            target_subbasin_id = int(inputs['target_subbasin_id'])
            
            # Validate routing product exists
            if not routing_product_path.exists():
                raise FileNotFoundError(f"Routing product not found: {routing_product_path}")
            
            # Create output directory
            workspace_dir = inputs.get('workspace_dir', routing_product_path.parent / 'extracted_subregion')
            output_folder = Path(workspace_dir)
            output_folder.mkdir(exist_ok=True, parents=True)
            
            # Extract subregion using appropriate method
            self.logger.info(f"Extracting subregion for SubId: {target_subbasin_id}")
            
            # Check if this is a HydroSHEDS-based routing product
            routing_product_version = inputs.get('routing_product_version', '')
            if 'hydrosheds' in routing_product_version:
                # Use HydroSHEDS adapter
                extraction_result = self.hydrosheds_adapter.extract_upstream_network(
                    target_subbasin_id, output_folder
                )
            else:
                # Use traditional BasinMaker extractor
                extraction_result = self.extractor.extract_subregion_from_routing_product(
                    routing_product_folder=routing_product_path,
                    most_downstream_subbasin_id=target_subbasin_id,
                    most_upstream_subbasin_id=-1,  # All upstream
                    output_folder=output_folder
                )
            
            if not extraction_result.get('success', False):
                raise RuntimeError(f"Subregion extraction failed: {extraction_result.get('error', 'Unknown error')}")
            
            # Validate extracted files
            required_files = ['extracted_catchments', 'extracted_rivers']
            extracted_files = {}
            
            for file_key in required_files:
                if file_key in extraction_result:
                    file_path = self.validate_file_exists(extraction_result[file_key])
                    extracted_files[file_key] = str(file_path)
            
            # Optional files
            optional_files = ['extracted_lakes', 'extracted_gauges']
            for file_key in optional_files:
                if file_key in extraction_result and extraction_result[file_key] is not None:
                    if Path(extraction_result[file_key]).exists():
                        extracted_files[file_key] = extraction_result[file_key]
            
            # Load and analyze extracted data
            catchments_gdf = gpd.read_file(extracted_files['extracted_catchments'])
            rivers_gdf = gpd.read_file(extracted_files['extracted_rivers'])
            
            # Calculate summary statistics
            total_area_km2 = catchments_gdf.geometry.area.sum() / 1e6  # Convert to km²
            subbasin_count = len(catchments_gdf)
            total_stream_length_km = rivers_gdf.geometry.length.sum() / 1000  # Convert to km
            
            outputs = {
                'extracted_catchments': extracted_files['extracted_catchments'],
                'extracted_rivers': extracted_files['extracted_rivers'],
                'extracted_lakes': extracted_files.get('extracted_lakes'),
                'extracted_gauges': extracted_files.get('extracted_gauges'),
                'subbasin_count': subbasin_count,
                'total_area_km2': total_area_km2,
                'total_stream_length_km': total_stream_length_km,
                'target_subbasin_id': target_subbasin_id,
                'extraction_successful': True,
                'success': True
            }
            
            created_files = [f for f in extracted_files.values() if f is not None]
            self._log_step_complete(created_files)
            
            return outputs
            
        except Exception as e:
            error_msg = f"Routing product subregion extraction failed: {str(e)}"
            self._log_step_failed(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _validate_routing_product_structure(self, routing_product_path: Path) -> bool:
        """Validate that routing product has required files"""
        
        required_patterns = [
            "*catchment*.shp",
            "*finalcat*.shp",
            "*river*.shp",
            "*riv*.shp"
        ]
        
        for pattern in required_patterns:
            if list(routing_product_path.glob(pattern)):
                return True
        
        return False