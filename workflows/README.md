# RAVEN Workflows - Outlet-Driven Model Generation

## Overview

Complete implementation of the 22-step outlet-driven workflow for generating RAVEN hydrological models from outlet coordinates.

## Usage

```python
from workflows import OutletDrivenWorkflow

# Initialize workflow
workflow = OutletDrivenWorkflow(workspace_dir="my_project")

# Execute complete workflow (coordinates near Montreal)
result = workflow.execute_complete_workflow(
    latitude=45.5017,
    longitude=-73.5673,
    outlet_name="Montreal_Test_Outlet"
)

if result['success']:
    print(f"RAVEN model generated successfully!")
    print(f"Model files: {result['raven_model_files']}")
else:
    print(f"Workflow failed: {result['error']}")
```

## Workflow Steps

### Phase 1: Outlet Point Processing (Steps 1-4)
1. **Validate Coordinates** - Check lat/lon ranges and supported region
2. **Find Nearest River** - Search 986,463 river database 
3. **Check Lake Outlet** - Search 1.4M lake database for proximity
4. **Set Study Area** - Calculate DEM download boundary

### Phase 2: Watershed Delineation (Steps 5-8)
5. **Download DEM** - USGS 3DEP 30m elevation (**NETWORK REQUIRED**)
6. **Prepare DEM** - Fill depressions, calculate flow direction/accumulation
7. **Delineate Watershed** - Trace boundary from outlet point
8. **Extract Streams** - Generate stream network from flow accumulation

### Phase 3: Lake Integration (Steps 9-12)
9. **Extract Watershed Lakes** - Find lakes within watershed boundary
10. **Analyze Lake Connections** - Connected vs isolated classification
11. **Filter Lakes by Importance** - Size and depth thresholds
12. **Integrate Lakes with Streams** - Modify network for proper routing

### Phase 4: HRU Generation (Steps 13-16)
13. **Create Sub-basins** - Divide at confluences and lake outlets
14. **Assign HRU Attributes** - Use BasinMaker lookup tables
15. **Calculate Hydraulic Parameters** - Manning's n, slope, elevation
16. **Generate Final HRUs** - BasinMaker overlay logic (lake + land HRUs)

### Phase 5: RAVEN Model Generation (Steps 17-22)
17. **Select Model Type** - GR4JCN, HMETS, or HBVEC based on watershed
18. **Create RVH File** - Spatial structure (sub-basins, HRUs, lakes)
19. **Create RVP File** - Parameters for land use/soil classes
20. **Create RVI File** - Model execution instructions
21. **Create RVT File** - Climate data template
22. **Validate Complete Model** - Check formatting and connectivity

## Features

### Context Management Integration
- Cross-session workflow state persistence
- Intelligent step recommendations
- Progress tracking and validation
- Error handling and recovery

### Real Data Integration
- **Local Databases**: 584 Canadian lakes, 986,463 rivers, HydroLAKES
- **Network Data**: USGS 3DEP DEM (location-specific, cannot pre-cache)
- **BasinMaker Integration**: Real lookup tables and parameter conversion

### Modular Design
- Each step is independent and testable
- Failed steps can be retried individually
- Workflow state preserved across interruptions
- Easy to extend with additional steps

## Network Dependencies

- **REQUIRED**: USGS 3DEP DEM download (Step 5)
- **OPTIONAL**: Climate data APIs (for model execution)
- **OPTIONAL**: Streamflow data (for calibration)

All other operations use local databases.

## Output

Complete 5-file RAVEN model ready for simulation:
- **model.rvh** - Watershed spatial structure
- **model.rvp** - Model parameters
- **model.rvi** - Execution instructions  
- **model.rvt** - Climate data template
- **model.rvc** - Initial conditions (optional)

## Implementation Status

### Completed
- Workflow framework and orchestrator
- Context management integration
- Steps 1-6 (outlet processing + DEM download/prep)
- Error handling and logging

### In Progress
- Steps 7-22 implementation
- BasinMaker processor integration
- RAVEN file generation

### Planned
- Complete workflow testing
- Performance optimization
- Additional model templates