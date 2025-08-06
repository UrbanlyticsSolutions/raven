# Single Outlet Delineation Workflow - Technical Documentation

## Overview

The Single Outlet Delineation Workflow creates complete RAVEN watershed models from a single outlet coordinate using comprehensive hydrological analysis including lake detection, classification, and integration.

## Workflow Steps

### **Step 1: Watershed Delineation**
- **Purpose**: Trace watershed boundary and extract stream network from DEM
- **Input**: DEM file, outlet coordinates, stream threshold
- **Process**: 
  - Fill depressions using professional watershed analyzer
  - Calculate flow direction and accumulation
  - Snap outlet to stream network
  - Trace upstream watershed boundary
  - Extract stream network using flow accumulation threshold
- **Output**: Watershed boundary shapefile, stream network shapefile

### **Step 2: Lake Detection** 
- **Purpose**: Identify water bodies and depressions within watershed
- **Input**: DEM file, watershed boundary
- **Process**:
  - Analyze DEM for depressions within watershed bounds
  - Apply minimum area thresholds (0.01 km² default)
  - Validate lake geometries and remove invalid polygons
- **Output**: Detected lakes shapefile, lake statistics

### **Step 3: Lake Classification**
- **Purpose**: Classify lakes as connected vs non-connected to stream network
- **Input**: Detected lakes, stream network
- **Process**:
  - Test each lake for spatial intersection with stream network
  - Connected lakes: intersect with streams (routing lakes)
  - Non-connected lakes: isolated water bodies (storage lakes)
  - Apply different area thresholds for each type
- **Output**: Connected lakes shapefile, non-connected lakes shapefile

### **Step 4: Lake-Stream Integration**
- **Purpose**: Integrate connected lakes into stream network routing
- **Input**: Stream network, connected lakes, watershed boundary
- **Process**:
  - Identify lake inlet and outlet points
  - Modify stream network to route through lakes
  - Create lake-stream connection topology
  - Update subbasin boundaries for lake inclusion
- **Output**: Integrated stream network, lake routing table

### **Step 5: Real Data Acquisition**
- **Landcover Data**: 
  - **Process**: Attempt to download real landcover data from APIs
  - **Failure**: Return error - NO synthetic fallback provided
- **Soil Data**:
  - **Process**: Attempt to download real soil data from APIs  
  - **Failure**: Return error - NO synthetic fallback provided

### **Step 6: HRU Generation**
- **Purpose**: Create hydrological response units including lakes
- **Input**: Integrated watershed, landcover, soil data
- **Process**:
  - Generate land HRUs from subbasins minus lake areas
  - Create lake HRUs with lake-specific parameters
  - Calculate area-weighted hydraulic properties
  - Assign routing connectivity between HRUs
- **Output**: Final HRUs shapefile with complete attributes

### **Step 7: Model Structure Generation**
- **Purpose**: Generate RAVEN spatial structure files (RVH, RVP)
- **Input**: HRUs, subbasins, lakes, watershed characteristics
- **Process**:
  - Select optimal RAVEN model based on watershed characteristics
  - Generate RVH file (spatial structure)
  - Generate RVP file (parameters)
- **Output**: RAVEN structure and parameter files

### **Step 8: Model Instructions Generation**
- **Purpose**: Generate RAVEN execution files (RVI, RVT, RVC)
- **Input**: Selected model, HRUs, watershed area
- **Process**:
  - Generate RVI file (model instructions)
  - Generate RVT file (climate template)
  - Generate RVC file (initial conditions)
- **Output**: RAVEN execution files

### **Step 9: Model Validation**
- **Purpose**: Comprehensive validation of complete RAVEN model
- **Input**: All 5 RAVEN files (RVH, RVP, RVI, RVT, RVC)
- **Process**:
  - Validate file formats and syntax
  - Check cross-file consistency
  - Verify physical reasonableness of parameters
  - Assess model readiness for simulation
- **Output**: Validation report, quality score

## Key Features

### **Complete Lake Processing**
- Depression-based lake detection from DEM
- Hydrologically-informed classification
- Stream network integration for proper routing
- Lake HRUs with storage parameters

### **Real Data Only**
- **NO synthetic data fallbacks**
- Clear error reporting when real data unavailable
- Honest failure when data sources not implemented
- Forces proper data client implementation

### **Comprehensive HRUs**
- Land HRUs with landcover/soil attributes
- Lake HRUs with storage/routing parameters
- Complete connectivity for RAVEN routing
- BasinMaker-compatible structure

## Error Handling

### **Fail-Fast Approach**
- Immediate failure when real data unavailable
- No silent fallbacks to synthetic data
- Clear error messages indicating missing implementations
- Workflow stops at first critical failure

### **Lake Processing Fallbacks**
- If lake detection fails: Continue without lakes
- If lake classification fails: Treat all as non-connected
- If lake integration fails: Use original stream network
- Log warnings but continue workflow

## File Organization

### **Workspace Structure**
```
outlet_workspace/
├── spatial/           # Spatial data files
│   ├── watershed_boundary.shp
│   ├── stream_network.shp
│   ├── final_hrus.shp
│   └── subbasins.shp
├── lakes/            # Lake processing files
│   ├── detected_lakes.shp
│   ├── connected_lakes.shp
│   ├── non_connected_lakes.shp
│   └── lake_routing.csv
├── model/            # RAVEN model files
│   ├── model.rvh
│   ├── model.rvp
│   ├── model.rvi
│   ├── model.rvt
│   └── model.rvc
└── data/            # Input data files
    ├── outlet_dem.tif
    ├── landcover.tif
    └── soil.tif
```

## Usage Example

```python
from workflows.single_outlet_delineation import SingleOutletDelineation

# Initialize workflow
workflow = SingleOutletDelineation(workspace_dir="watershed_analysis")

# Execute for specific outlet
results = workflow.execute_single_delineation(
    latitude=50.6667,
    longitude=-120.3333,
    outlet_name="Test_Outlet",
    stream_threshold=1000,
    buffer_km=2.0
)

if results['success']:
    print(f"Watershed area: {results['watershed_area_km2']:.1f} km²")
    print(f"Connected lakes: {results['summary']['connected_lake_count']}")
    print(f"Total HRUs: {results['total_hru_count']}")
    print(f"Files created: {len(results['files_created'])}")
else:
    print(f"[FAILED]: {results['error']}")
```

## Command Line Usage

```bash
python workflows/single_outlet_delineation.py 50.6667 -120.3333 \
    --outlet-name "Kamloops_Test" \
    --workspace-dir "/path/to/workspace" \
    --stream-threshold 1000 \
    --buffer-km 2.0
```

This workflow provides complete watershed analysis with proper lake hydrology using only real data sources.