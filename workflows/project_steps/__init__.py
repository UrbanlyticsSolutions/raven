"""
RAVEN Workflow Project Steps
============================

Organized collection of RAVEN hydrological modeling workflow steps.

Project Structure:
- Step 1: Data Preparation (DEM, routing products, spatial data)
- Step 2: Watershed Delineation (catchment boundaries, stream networks)
- Step 3: Lake Processing (lake detection, classification, routing)
- Step 4: HRU Generation (Hydrologic Response Units)
- Step 5: RAVEN Model Generation (model files, parameters)
- Step 6: Model Validation and Execution
- Climate/Hydrometric Data: External data acquisition and processing

Each step is self-contained with its own module and can be executed independently
or as part of the complete workflow orchestration.
"""

# Import all step modules for easy access
from .step1_data_preparation.step1_data_preparation import Step1DataPreparation
from .step2_watershed_delineation.step2_watershed_delineation import Step2WatershedDelineation
from .step3_lake_processing.step3_lake_processing import Step3LakeProcessing
# Step4 import - make optional since step4_hru_generation_clean.py is standalone
try:
    from .step4_hru_generation.step4_hru_generation import Step4HRUGeneration
except ImportError:
    print("Warning: Step4HRUGeneration not available - use step4_hru_generation_clean.py directly")
    Step4HRUGeneration = None
# Step 5 imports - optional due to RavenPy dependency
try:
    from .step5_raven_model.step5_raven_model import Step5RAVENModel, CompleteStep5RAVENModel
except ImportError as e:
    print(f"WARNING: Step 5 imports failed: {e}")
    Step5RAVENModel = None
    CompleteStep5RAVENModel = None
# from .step6_validate_run_model.step6_validate_run_model import Step6ValidateRunModel  # Temporarily disabled due to null bytes
from .climate_hydrometric_data.step_climate_hydrometric_data import ClimateHydrometricDataProcessor

__all__ = [
    'Step1DataPreparation',
    'Step2WatershedDelineation', 
    'Step3LakeProcessing',
    'Step4HRUGeneration',
    'Step5RAVENModel',
    'CompleteStep5RAVENModel',
    # 'Step6ValidateRunModel',  # Temporarily disabled
    'ClimateHydrometricDataProcessor'
]

# Step registry for dynamic access
STEP_REGISTRY = {
    'step1': Step1DataPreparation,
    'step2': Step2WatershedDelineation,
    'step3': Step3LakeProcessing,
    'step4': Step4HRUGeneration,
    'step5': Step5RAVENModel,
    'step5_complete': CompleteStep5RAVENModel,
    # 'step6': Step6ValidateRunModel,  # Temporarily disabled
    'climate_hydrometric': ClimateHydrometricDataProcessor
}

def get_step(step_name: str):
    """Get a step class by name."""
    return STEP_REGISTRY.get(step_name)

def list_available_steps():
    """List all available step names."""
    return list(STEP_REGISTRY.keys())
