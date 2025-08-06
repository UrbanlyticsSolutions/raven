"""
RAVEN Workflow Steps Library

Refactored to remove redundant/overlapping functions and consolidate into unified steps.

Consolidated Step Categories:
- validation_steps: Coordinate and model validation
- routing_product_steps: Routing product operations  
- dem_clipping_step: DEM download and clipping (replaces dem_processing_steps)
- watershed_delineation_complete: Unified watershed + lake processing (replaces watershed_steps + lake_processing_steps)
- landcover_extraction_step: Landcover data extraction (replaces part of landcover_soil_integration)
- soil_extraction_step: Soil data extraction (replaces part of landcover_soil_integration)
- hru_generation_steps: HRU creation and attributes
- raven_generation_steps: RAVEN model file generation
"""

from .validation_steps import (
    ValidateCoordinatesAndFindRoutingProduct,
    ValidateCoordinatesAndSetDEMArea,
    ValidateCompleteModel
)

from .routing_product_steps import (
    ExtractSubregionFromRoutingProduct
)

# Consolidated DEM processing
from .dem_clipping_step import (
    DEMClippingStep
)

# Consolidated watershed + lake processing
from .watershed_steps import (
    DelineateWatershedAndStreams
)

# Separate landcover and soil extraction
from .landcover_extraction_step import (
    LandcoverExtractionStep
)

from .soil_extraction_step import (
    SoilExtractionStep
)

from .hru_generation_steps import (
    GenerateHRUsFromRoutingProduct,
    CreateSubBasinsAndHRUs
)

from .raven_generation_steps import (
    GenerateRAVENModelFiles,
    SelectModelAndGenerateStructure,
    GenerateModelInstructions
)

# Project management system
from .project_management_step import (
    ProjectManagementStep
)

# Consolidated step registry (removed overlapping functions)
STEP_REGISTRY = {
    # Validation steps
    'validate_coordinates_find_routing': ValidateCoordinatesAndFindRoutingProduct,
    'validate_coordinates_set_dem': ValidateCoordinatesAndSetDEMArea,
    'validate_complete_model': ValidateCompleteModel,
    
    # Routing product steps
    'extract_subregion_routing': ExtractSubregionFromRoutingProduct,
    
    # Consolidated DEM processing (replaces download_prepare_dem)
    'clip_dem': DEMClippingStep,
    
    # Consolidated watershed + lake processing (replaces delineate_watershed_streams + detect_classify_lakes)
    'watershed_delineation_complete': DelineateWatershedAndStreams,
    
    # Separate data extraction steps (replaces integrate_landcover_soil)
    'extract_landcover': LandcoverExtractionStep,
    'extract_soil': SoilExtractionStep,
    
    # HRU generation steps
    'generate_hrus_routing': GenerateHRUsFromRoutingProduct,
    'create_subbasins_hrus': CreateSubBasinsAndHRUs,
    
    # RAVEN generation steps
    'generate_raven_files': GenerateRAVENModelFiles,
    'select_model_structure': SelectModelAndGenerateStructure,
    'generate_model_instructions': GenerateModelInstructions,
    
    # Project management
    'project_manager': ProjectManagementStep
}

# Consolidated workflow approach step combinations
APPROACH_A_STEPS = [
    'validate_coordinates_find_routing',
    'extract_subregion_routing', 
    'generate_hrus_routing',
    'extract_landcover',  # Replaced integrate_landcover_soil
    'extract_soil',       # Replaced integrate_landcover_soil
    'generate_raven_files',
    'validate_complete_model'
]

APPROACH_B_STEPS = [
    'validate_coordinates_set_dem',
    'clip_dem',                        # Replaced download_prepare_dem
    'watershed_delineation_complete',  # Replaced delineate_watershed_streams + detect_classify_lakes
    'create_subbasins_hrus',
    'extract_landcover',               # Replaced integrate_landcover_soil
    'extract_soil',                    # Replaced integrate_landcover_soil  
    'select_model_structure',
    'generate_model_instructions',
    'validate_complete_model'
]

def get_step(step_name: str):
    """
    Get step instance by name
    
    Parameters:
    -----------
    step_name : str
        Name of the step from STEP_REGISTRY
        
    Returns:
    --------
    WorkflowStep instance
    """
    if step_name not in STEP_REGISTRY:
        available_steps = list(STEP_REGISTRY.keys())
        raise ValueError(f"Unknown step: {step_name}. Available steps: {available_steps}")
    
    return STEP_REGISTRY[step_name]()

def list_available_steps() -> list:
    """List all available step names"""
    return list(STEP_REGISTRY.keys())

def get_approach_steps(approach: str) -> list:
    """
    Get step names for a specific approach
    
    Parameters:
    -----------
    approach : str
        'routing_product' or 'full_delineation'
        
    Returns:
    --------
    List of step names for the approach
    """
    if approach == 'routing_product' or approach == 'a':
        return APPROACH_A_STEPS.copy()
    elif approach == 'full_delineation' or approach == 'b':
        return APPROACH_B_STEPS.copy()
    else:
        raise ValueError(f"Unknown approach: {approach}. Use 'routing_product' or 'full_delineation'")

def create_workflow_from_steps(step_names: list):
    """
    Create a workflow from a list of step names
    
    Parameters:
    -----------
    step_names : list
        List of step names to execute in order
        
    Returns:
    --------
    List of WorkflowStep instances
    """
    return [get_step(name) for name in step_names]

__all__ = [
    # Consolidated step classes (removed overlapping ones)
    'ValidateCoordinatesAndFindRoutingProduct',
    'ValidateCoordinatesAndSetDEMArea', 
    'ValidateCompleteModel',
    'ExtractSubregionFromRoutingProduct',
    'DEMClippingStep',                    # Replaces DownloadAndPrepareDEM
    'UnifiedWatershedDelineation',        # Replaces DelineateWatershedAndStreams + DetectAndClassifyLakes
    'LandcoverExtractionStep',           # Replaces part of LandcoverSoilIntegrator
    'SoilExtractionStep',                # Replaces part of LandcoverSoilIntegrator
    'GenerateHRUsFromRoutingProduct',
    'CreateSubBasinsAndHRUs',
    'GenerateRAVENModelFiles',
    'SelectModelAndGenerateStructure',
    'GenerateModelInstructions',
    'ProjectManager',                    # Project and file management
    
    # Utility functions
    'get_step',
    'list_available_steps',
    'get_approach_steps',
    'create_workflow_from_steps',
    
    # Constants
    'STEP_REGISTRY',
    'APPROACH_A_STEPS',
    'APPROACH_B_STEPS'
]