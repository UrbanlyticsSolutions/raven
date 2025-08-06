# Workflow Steps Library

## Overview

The RAVEN workflow system uses a **modular step library** where individual processing steps can be combined into different workflow approaches. This design allows maximum flexibility and reusability.

## Step Organization

### **Step Categories**
```
workflows/steps/
├── validation_steps.py          # Coordinate and model validation
├── routing_product_steps.py     # Routing product operations
├── dem_processing_steps.py      # DEM download and conditioning
├── watershed_steps.py           # Watershed delineation
├── lake_processing_steps.py     # Lake detection and classification
├── hru_generation_steps.py      # HRU creation and attributes
├── raven_generation_steps.py    # RAVEN model file generation
└── __init__.py                  # Step registry and imports
```

## Step Usage in Workflows

### **Approach A: Routing Product (5 steps)**
```python
from workflows.steps.validation_steps import ValidateCoordinatesAndFindRoutingProduct
from workflows.steps.routing_product_steps import ExtractSubregionFromRoutingProduct
from workflows.steps.hru_generation_steps import GenerateHRUsFromRoutingProduct
from workflows.steps.raven_generation_steps import GenerateRAVENModelFiles
from workflows.steps.validation_steps import ValidateCompleteModel

steps = [
    ValidateCoordinatesAndFindRoutingProduct(),
    ExtractSubregionFromRoutingProduct(),
    GenerateHRUsFromRoutingProduct(),
    GenerateRAVENModelFiles(),
    ValidateCompleteModel()
]
```

### **Approach B: Full Delineation (8 steps)**
```python
from workflows.steps.validation_steps import ValidateCoordinatesAndSetDEMArea
from workflows.steps.dem_processing_steps import DownloadAndPrepareDEM
from workflows.steps.watershed_steps import DelineateWatershedAndStreams
from workflows.steps.lake_processing_steps import DetectAndClassifyLakes
from workflows.steps.hru_generation_steps import CreateSubBasinsAndHRUs
from workflows.steps.raven_generation_steps import SelectModelAndGenerateStructure
from workflows.steps.raven_generation_steps import GenerateModelInstructions
from workflows.steps.validation_steps import ValidateCompleteModel

steps = [
    ValidateCoordinatesAndSetDEMArea(),
    DownloadAndPrepareDEM(),
    DelineateWatershedAndStreams(),
    DetectAndClassifyLakes(),
    CreateSubBasinsAndHRUs(),
    SelectModelAndGenerateStructure(),
    GenerateModelInstructions(),
    ValidateCompleteModel()
]
```

## Available Steps

### **Validation Steps**
| Step | Purpose | Used In |
|------|---------|---------|
| `ValidateCoordinatesAndFindRoutingProduct` | Find routing product for coordinates | Approach A |
| `ValidateCoordinatesAndSetDEMArea` | Set DEM download area | Approach B |
| `ValidateCompleteModel` | Final model validation | Both |

### **Routing Product Steps**
| Step | Purpose | Used In |
|------|---------|---------|
| `ExtractSubregionFromRoutingProduct` | Extract watershed from routing product | Approach A |

### **DEM Processing Steps**
| Step | Purpose | Used In |
|------|---------|---------|
| `DownloadAndPrepareDEM` | Download and condition DEM | Approach B |

### **Watershed Steps**
| Step | Purpose | Used In |
|------|---------|---------|
| `DelineateWatershedAndStreams` | Watershed and stream delineation | Approach B |

### **Lake Processing Steps**
| Step | Purpose | Used In |
|------|---------|---------|
| `DetectAndClassifyLakes` | Lake detection and classification | Approach B |

### **HRU Generation Steps**
| Step | Purpose | Used In |
|------|---------|---------|
| `GenerateHRUsFromRoutingProduct` | HRUs from routing product | Approach A |
| `CreateSubBasinsAndHRUs` | Sub-basins and HRUs from scratch | Approach B |

### **RAVEN Generation Steps**
| Step | Purpose | Used In |
|------|---------|---------|
| `GenerateRAVENModelFiles` | Complete model generation | Approach A |
| `SelectModelAndGenerateStructure` | Model selection + RVH/RVP | Approach B |
| `GenerateModelInstructions` | RVI/RVT/RVC generation | Approach B |

## Step Base Class

All steps inherit from a common base class:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path

class WorkflowStep(ABC):
    """Base class for all workflow steps"""
    
    def __init__(self, step_name: str, step_category: str):
        self.step_name = step_name
        self.step_category = step_category
        self.context_manager = None
        
    def set_context_manager(self, context_manager):
        """Set context manager for workflow state tracking"""
        self.context_manager = context_manager
        
    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow step"""
        pass
        
    def _log_step_start(self):
        """Log step start"""
        if self.context_manager:
            self.context_manager.mark_step_started(self.step_name)
    
    def _log_step_complete(self, outputs: list):
        """Log step completion"""
        if self.context_manager:
            self.context_manager.mark_step_completed(self.step_name, outputs)
    
    def _log_step_failed(self, error_msg: str):
        """Log step failure"""
        if self.context_manager:
            self.context_manager.mark_step_failed(self.step_name, error_msg)
```

## Step Implementation Pattern

Each step follows a consistent pattern:

```python
class ExampleStep(WorkflowStep):
    """Example workflow step implementation"""
    
    def __init__(self):
        super().__init__("example_step", "example_category")
        
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._log_step_start()
        
        try:
            # Validate inputs
            required_inputs = ['input1', 'input2']
            for inp in required_inputs:
                if inp not in inputs:
                    raise ValueError(f"Missing required input: {inp}")
            
            # Process data
            result = self._process_data(inputs)
            
            # Generate outputs
            outputs = {
                'output1': result['data'],
                'output2': result['metadata'],
                'success': True
            }
            
            self._log_step_complete([outputs['output1']])
            return outputs
            
        except Exception as e:
            error_msg = f"Step failed: {str(e)}"
            self._log_step_failed(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _process_data(self, inputs):
        """Step-specific processing logic"""
        # Implementation here
        pass
```

## Step Registry

The step registry provides centralized access to all available steps:

```python
# workflows/steps/__init__.py
from .validation_steps import *
from .routing_product_steps import *
from .dem_processing_steps import *
from .watershed_steps import *
from .lake_processing_steps import *
from .hru_generation_steps import *
from .raven_generation_steps import *

# Step registry for dynamic loading
STEP_REGISTRY = {
    # Validation steps
    'validate_coordinates_find_routing': ValidateCoordinatesAndFindRoutingProduct,
    'validate_coordinates_set_dem': ValidateCoordinatesAndSetDEMArea,
    'validate_complete_model': ValidateCompleteModel,
    
    # Routing product steps
    'extract_subregion_routing': ExtractSubregionFromRoutingProduct,
    
    # DEM processing steps
    'download_prepare_dem': DownloadAndPrepareDEM,
    
    # Watershed steps
    'delineate_watershed_streams': DelineateWatershedAndStreams,
    
    # Lake processing steps
    'detect_classify_lakes': DetectAndClassifyLakes,
    
    # HRU generation steps
    'generate_hrus_routing': GenerateHRUsFromRoutingProduct,
    'create_subbasins_hrus': CreateSubBasinsAndHRUs,
    
    # RAVEN generation steps
    'generate_raven_files': GenerateRAVENModelFiles,
    'select_model_structure': SelectModelAndGenerateStructure,
    'generate_model_instructions': GenerateModelInstructions
}

def get_step(step_name: str) -> WorkflowStep:
    """Get step instance by name"""
    if step_name not in STEP_REGISTRY:
        raise ValueError(f"Unknown step: {step_name}")
    return STEP_REGISTRY[step_name]()

def list_available_steps() -> list:
    """List all available step names"""
    return list(STEP_REGISTRY.keys())
```

## Custom Workflow Creation

Users can create custom workflows by combining steps:

```python
from workflows.steps import get_step

class CustomWorkflow:
    def __init__(self, step_names: list):
        self.steps = [get_step(name) for name in step_names]
    
    def execute(self, initial_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute custom workflow"""
        current_inputs = initial_inputs.copy()
        
        for step in self.steps:
            result = step.execute(current_inputs)
            
            if not result.get('success', False):
                return result
            
            # Pass outputs to next step
            current_inputs.update(result)
        
        return current_inputs

# Example: Custom hybrid workflow
custom_workflow = CustomWorkflow([
    'validate_coordinates_set_dem',
    'download_prepare_dem',
    'extract_subregion_routing',  # Try routing product for HRUs
    'generate_raven_files'
])
```

## Step Dependencies

### **Input/Output Flow**
```
ValidateCoordinates → coordinates, bounds
    ↓
DownloadDEM → dem_files
    ↓
DelineateWatershed → watershed_boundary, streams
    ↓
DetectLakes → classified_lakes
    ↓
CreateHRUs → final_hrus
    ↓
GenerateRAVEN → model_files
    ↓
ValidateModel → validated_model
```

### **Dependency Matrix**
| Step | Requires | Produces |
|------|----------|----------|
| ValidateCoordinates | lat, lon | coordinates, bounds |
| ExtractRouting | routing_product, subbasin_id | catchments, rivers, lakes |
| DownloadDEM | bounds | dem_files |
| DelineateWatershed | dem_files, coordinates | watershed, streams |
| DetectLakes | watershed, streams | classified_lakes |
| CreateHRUs | watershed, lakes | final_hrus |
| GenerateRAVEN | final_hrus | model_files |
| ValidateModel | model_files | validated_model |

## Testing Steps

Each step includes comprehensive testing:

```python
def test_example_step():
    """Test example step functionality"""
    step = ExampleStep()
    
    # Test valid inputs
    inputs = {
        'input1': 'test_value1',
        'input2': 'test_value2'
    }
    
    result = step.execute(inputs)
    
    assert result['success'] == True
    assert 'output1' in result
    assert 'output2' in result
    
    # Test missing inputs
    invalid_inputs = {'input1': 'test_value1'}
    result = step.execute(invalid_inputs)
    
    assert result['success'] == False
    assert 'error' in result
```

## Related Documentation

- [Approach A Implementation](../approach-a-routing-product.md)
- [Approach B Implementation](../approach-b-full-delineation.md)
- [Step Implementation Guide](./implementation-guide.md)
- [Custom Workflow Creation](./custom-workflows.md)

---

**The modular step library enables flexible workflow composition while maintaining consistency and reusability.**