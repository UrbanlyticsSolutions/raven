# RAVEN Project Steps Organization

## Overview
This document describes the organized structure for RAVEN workflow project steps 1-6, providing a clear, modular architecture for hydrological modeling workflows.

## Folder Structure

```
workflows/
├── project_steps/                           # Main project steps organization
│   ├── __init__.py                         # Master module with step registry
│   │
│   ├── step1_data_preparation/             # Step 1: Data Preparation
│   │   ├── __init__.py                     # Step 1 module initialization
│   │   └── step1_data_preparation.py       # DEM, routing products, spatial data
│   │
│   ├── step2_watershed_delineation/        # Step 2: Watershed Delineation
│   │   ├── __init__.py                     # Step 2 module initialization
│   │   └── step2_watershed_delineation.py  # Flow direction, streams, boundaries
│   │
│   ├── step3_lake_processing/              # Step 3: Lake Processing
│   │   ├── __init__.py                     # Step 3 module initialization
│   │   └── step3_lake_processing.py        # Lake detection, classification, routing
│   │
│   ├── step4_hru_generation/               # Step 4: HRU Generation
│   │   ├── __init__.py                     # Step 4 module initialization
│   │   └── step4_hru_generation.py         # Land cover, soil, HRU discretization
│   │
│   ├── step5_raven_model/                  # Step 5: RAVEN Model Generation
│   │   ├── __init__.py                     # Step 5 module initialization
│   │   └── step5_raven_model.py            # Model files (.rvh, .rvp, .rvi, .rvt, .rvc)
│   │
│   ├── step6_validate_run_model/           # Step 6: Model Validation & Execution
│   │   ├── __init__.py                     # Step 6 module initialization
│   │   └── step6_validate_run_model.py     # Validation, execution, analysis
│   │
│   └── climate_hydrometric_data/           # Climate & Hydrometric Data Processing
│       ├── __init__.py                     # Data module initialization
│       └── step_climate_hydrometric_data.py # ECCC data, station data, processing
│
├── single_outlet_orchestrator.py           # Main workflow orchestrator (updated imports)
├── steps/                                  # Legacy modular steps library
└── approaches/                             # Complete workflow implementations
```

## Step Descriptions

### Step 1: Data Preparation
- **Purpose**: Initial data acquisition and spatial setup
- **Outputs**: DEM data, routing products, coordinate systems
- **Key Functions**: `prepare_spatial_data()`, `validate_coordinates()`

### Step 2: Watershed Delineation  
- **Purpose**: Hydrological network extraction
- **Outputs**: Stream networks, watershed boundaries, subbasins
- **Key Functions**: `delineate_watershed()`, `extract_streams()`

### Step 3: Lake Processing
- **Purpose**: Lake integration and routing
- **Outputs**: Classified lakes, lake-stream connectivity
- **Key Functions**: `detect_lakes()`, `integrate_lake_routing()`

### Step 4: HRU Generation
- **Purpose**: Hydrologic Response Unit creation
- **Outputs**: HRUs with land cover and soil properties
- **Key Functions**: `generate_hrus()`, `assign_parameters()`

### Step 5: RAVEN Model Generation
- **Purpose**: Complete RAVEN model file creation
- **Outputs**: All 5 RAVEN model files (.rvh, .rvp, .rvi, .rvt, .rvc)
- **Key Functions**: `generate_model_files()`, `validate_model()`

### Step 6: Model Validation & Execution
- **Purpose**: Model execution and result analysis
- **Outputs**: Simulation results, performance metrics
- **Key Functions**: `validate_model()`, `run_simulation()`

### Climate & Hydrometric Data
- **Purpose**: External data acquisition and processing
- **Outputs**: Climate forcing data, validation datasets
- **Key Functions**: `download_climate_data()`, `process_hydrometric_data()`

## Usage Examples

### Import Individual Steps
```python
from workflows.project_steps import Step1DataPreparation, Step5RAVENModel
from workflows.project_steps.step2_watershed_delineation import Step2WatershedDelineation
```

### Use Step Registry
```python
from workflows.project_steps import get_step, list_available_steps

# Get step by name
Step1 = get_step('step1')
CompleteStep5 = get_step('step5_complete')

# List all available steps
available_steps = list_available_steps()
```

### Complete Workflow
```python
from workflows.single_outlet_orchestrator import SingleOutletOrchestrator

orchestrator = SingleOutletOrchestrator(
    lat=49.7313, 
    lon=-118.9439, 
    project_name="BigWhite"
)
results = orchestrator.execute_complete_workflow()
```

## Benefits of This Organization

1. **Modular Design**: Each step is self-contained with clear interfaces
2. **Easy Maintenance**: Related functionality grouped together
3. **Clear Documentation**: Each module has specific purpose and responsibilities
4. **Flexible Execution**: Steps can be run individually or as complete workflow
5. **Import Simplicity**: Centralized step registry for dynamic access
6. **Version Control**: Better organization for tracking changes by step

## Migration Notes

- Updated `single_outlet_orchestrator.py` imports to use new structure
- All step functionality preserved - only organization changed
- Backward compatibility maintained through step registry
- Documentation updated to reflect new structure

## Future Enhancements

- Step-specific configuration files
- Enhanced logging per step
- Step dependency validation
- Performance monitoring per step
- Step-specific test suites
