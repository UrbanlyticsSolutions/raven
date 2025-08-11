# RAVEN Hydrological Modeling Workflow Analysis

## Overview

This document provides a comprehensive analysis of the RAVEN hydrological modeling workflow, detailing each step, data sources, processors, and integration points.

## Workflow Architecture

The workflow follows a sequential 6-step process with an additional climate/hydrometric data acquisition step:

```
Step 1 → Climate/Hydro → Step 2 → Step 3 → Step 4 → Step 5 → Step 6
```

## Detailed Step Analysis

### Step 1: Data Preparation
**File**: `workflows/project_steps/step1_data_preparation/step1_data_preparation.py`

**Purpose**: Download and prepare spatial data for watershed analysis

**Key Components**:
- **DEM Download**: USGS 3DEP 30m elevation data
- **Landcover Data**: ESA WorldCover or similar sources
- **Soil Data**: SoilGrids or regional soil databases
- **Coordinate System Standardization**: BasinMaker-style CRS processing

**Data Sources**:
- USGS 3DEP for elevation data
- ESA WorldCover for land cover
- SoilGrids for soil properties
- MGHydro API for watershed boundaries

**Processors Used**:
- `CoordinateSystemProcessor`: CRS standardization and reprojection
- `DEMClippingStep`: DEM download and clipping
- `LandcoverExtractionStep`: Land cover data extraction
- `SoilExtractionStep`: Soil data extraction

**Outputs**:
- `dem.tif`: Digital elevation model
- `landcover.tif`: Land cover raster
- `soil.tif`: Soil properties raster
- `study_area.geojson`: Watershed boundary
- Coordinate system metadata

**Integration Points**:
- ✅ **BasicAttributesCalculator**: Not integrated (could be added for initial terrain analysis)
- ✅ **CoordinateSystemProcessor**: Already integrated

---

### Climate & Hydrometric Data Acquisition
**File**: `workflows/project_steps/climate_hydrometric_data/step_climate_hydrometric_data.py`

**Purpose**: Acquire climate forcing and hydrometric calibration data

**Key Components**:
- **Climate Data**: Temperature, precipitation, humidity
- **Hydrometric Data**: Streamflow observations for calibration
- **Gap Filling**: IDW interpolation and vicinity station data
- **Quality Control**: Data validation and filtering

**Data Sources**:
- Environment and Climate Change Canada
- USGS water data
- Regional climate networks

**Search Parameters**:
- Climate stations: 50km search radius
- Hydrometric stations: 50km search radius
- Minimum coverage: 30+ years for climate, 10+ years for streamflow

**Outputs**:
- `climate/climate_forcing.csv`: Climate forcing data
- `hydrometric/observed_streamflow.csv`: Streamflow observations
- Station metadata and quality metrics

**Integration Points**:
- ✅ **ObservationPointIntegrator**: Not integrated (could use hydrometric station locations)

---

### Step 2: Watershed Delineation
**File**: `workflows/project_steps/step2_watershed_delineation/step2_watershed_delineation.py`

**Purpose**: Delineate watershed boundary and extract stream network

**Key Components**:
- **Flow Direction**: D8 or D-infinity algorithms
- **Flow Accumulation**: Upstream area calculation
- **Stream Network**: Stream extraction using threshold
- **Outlet Snapping**: Snap outlet to stream network
- **Subbasin Delineation**: Create subbasins for routing
- **Hydrological Correction**: BasinMaker-style subbasin merging

**Processors Used**:
- `ImprovedOutletSnapper`: Outlet location optimization
- `BasicAttributesCalculator`: ✅ **INTEGRATED** - Calculates geometric and topographic attributes
- `DelineateWatershedAndStreams`: Core delineation logic

**Outputs**:
- `subbasins.shp`: Subbasin polygons
- `streams.shp`: Stream network
- `watershed_boundary.shp`: Watershed outline
- `subbasins_with_attributes.shp`: ✅ **NEW** - Enhanced with basic attributes
- Flow direction and accumulation rasters

**Basic Attributes Calculated**:
- `Area_m`: Subbasin area in square meters
- `Perimeter_m`: Subbasin perimeter in meters
- `RivLength`: River length within subbasin
- `RivSlope`: Average river slope
- `MeanElev`, `MinElev`, `MaxElev`: Elevation statistics
- `BasArea`: Drainage area (upstream accumulation)

**Integration Points**:
- ✅ **BasicAttributesCalculator**: **INTEGRATED** - Added after watershed delineation
- ❌ **ObservationPointIntegrator**: Not integrated (could be added for early gauge integration)

---

### Step 3: Lake Processing
**File**: `workflows/project_steps/step3_lake_processing/step3_lake_processing.py`

**Purpose**: Integrate lakes and reservoirs into watershed routing

**Key Components**:
- **Lake Detection**: Identify water bodies from various sources
- **Lake Classification**: Connected vs. non-connected lakes
- **Size Filtering**: BasinMaker-style area thresholds
- **Routing Integration**: Link lakes to stream network
- **Hydraulic Routing**: Magpie-based lake routing

**Data Sources**:
- HydroLAKES global database
- Local lake inventories
- Landsat-derived water bodies

**Filtering Criteria**:
- Connected lakes: ≥0.01 km² (10,000 m²)
- Non-connected lakes: ≥0.1 km² (100,000 m²)
- Depth threshold: ≥0.01 m

**Processors Used**:
- `ComprehensiveLakeDetector`: Multi-source lake detection
- `LakeClassifier`: Stream connectivity analysis
- `LakeProcessor`: Lake-river integration
- `MagpieHydraulicRouting`: Advanced routing calculations

**Outputs**:
- `lakes_with_routing_ids.shp`: Filtered lakes with routing information
- `watershed_routing_routing.rvh`: RAVEN routing file with lakes
- Lake-stream connectivity tables

**Integration Points**:
- ❌ **BasicAttributesCalculator**: Not needed (lakes handled separately)
- ❌ **ObservationPointIntegrator**: Not integrated (could add lake gauges)

---

### Step 4: HRU Generation
**File**: `workflows/project_steps/step4_hru_generation/step4_hru_generation_clean.py`

**Purpose**: Generate Hydrological Response Units for RAVEN modeling

**Key Components**:
- **Thematic Overlay**: Combine subbasins, landcover, soil, lakes
- **HRU Creation**: Generate unique combinations
- **Attribute Assignment**: Calculate HRU properties
- **Class Mapping**: Map to RAVEN parameter classes
- **Quality Control**: Validate HRU attributes

**Processors Used**:
- `HRUGenerator`: Core HRU generation logic
- `HRUAttributesCalculator`: HRU-specific attribute calculation
- `RAVENLookupTableGenerator`: Parameter lookup tables
- `HRUClassMapper`: Class mapping and validation

**Thematic Data Integration**:
- Subbasins from Step 2
- Landcover from Step 1
- Soil data from Step 1
- Lakes from Step 3

**Outputs**:
- `hrus.geojson`: HRU polygons with attributes
- `finalcat_hru_info.shp`: BasinMaker-compatible HRU file
- Parameter lookup tables
- HRU statistics and validation reports

**Integration Points**:
- ✅ **BasicAttributesCalculator**: Could replace/supplement `HRUAttributesCalculator`
- ❌ **ObservationPointIntegrator**: Not integrated (could add gauge information to HRUs)

---

### Step 5: RAVEN Model Generation
**File**: `workflows/project_steps/step5_raven_model/step5_raven_model.py`

**Purpose**: Generate RAVEN model files (.rvh, .rvi, .rvp, .rvt)

**Key Components**:
- **RVH Generation**: Watershed structure and HRU definitions
- **RVI Generation**: Model configuration and parameters
- **RVP Generation**: Parameter values and lookup tables
- **RVT Generation**: Time series data integration
- **Model Execution**: Run RAVEN simulation

**Processors Used**:
- `RVHGenerator`: Watershed structure file
- `RAVENGenerator`: Model configuration
- `RVPGenerator`: Parameter file
- `RVTGenerator`: Time series file
- `ObservationPointIntegrator`: ✅ **INTEGRATED** - Gauge station integration
- `RAVENParameterExtractor`: Dynamic parameter extraction
- `RAVENLookupTableGenerator`: Comprehensive lookup tables

**Model Files Generated**:
- `.rvh`: Watershed structure (subbasins, HRUs, routing)
- `.rvi`: Model configuration (processes, classes, parameters)
- `.rvp`: Parameter values (soil, vegetation, channel properties)
- `.rvt`: Time series data (climate forcing, observations)

**Key Features**:
- Dynamic class generation from actual HRU data
- JSON-based parameter database
- BasinMaker-compatible attribute calculation
- Automatic gauge station integration

**Integration Points**:
- ✅ **ObservationPointIntegrator**: **INTEGRATED** - Sets GAUGED attribute correctly
- ❌ **BasicAttributesCalculator**: Not directly used (attributes from Step 2)

**GAUGED Attribute Logic**:
```python
# Before integration (always 0):
gauged = int(sub_hrus.iloc[0].get('GAUGED', 0))  # Always 0

# After integration (actual gauge locations):
# ObservationPointIntegrator updates catchments with Has_POI=1
# GAUGED attribute reflects actual gauge station presence
```

---

### Step 6: Model Validation & Calibration
**File**: `workflows/project_steps/step6_validate_run_model/step6_validate_run_model.py`

**Purpose**: Validate, calibrate, and run RAVEN model

**Key Components**:
- **Model Validation**: Check model files and parameters
- **Calibration**: Parameter optimization using observed data
- **Simulation**: Run RAVEN model
- **Performance Metrics**: Calculate model performance statistics
- **Visualization**: Generate plots and reports

**Integration Points**:
- Uses hydrometric data from Climate/Hydro step
- Validates against observed streamflow
- Could integrate additional gauge stations via `ObservationPointIntegrator`

---

## Data Flow Summary

```
Step 1: Spatial Data
├── DEM, Landcover, Soil
├── Coordinate System Standardization
└── Study Area Definition

Climate/Hydro: External Data
├── Climate Forcing (50km search)
├── Streamflow Observations (50km search)
└── Gap Filling & Quality Control

Step 2: Watershed Delineation
├── Flow Direction/Accumulation
├── Stream Network Extraction
├── Subbasin Creation
└── ✅ Basic Attributes Calculation (INTEGRATED)

Step 3: Lake Processing
├── Lake Detection & Classification
├── Size/Connectivity Filtering
└── Routing Integration

Step 4: HRU Generation
├── Thematic Overlay
├── HRU Creation & Attribution
└── Parameter Class Mapping

Step 5: RAVEN Model Generation
├── Model File Generation (.rvh, .rvi, .rvp, .rvt)
├── ✅ Observation Point Integration (INTEGRATED)
└── Model Execution

Step 6: Validation & Calibration
├── Model Validation
├── Parameter Calibration
└── Performance Assessment
```

## Processor Integration Status

### ✅ BasicAttributesCalculator - INTEGRATED
**Location**: Step 2 (Watershed Delineation)
**Function**: Calculates geometric and topographic attributes for catchments
**Integration**: Added after watershed delineation, before saving results
**Benefits**: 
- Provides BasinMaker-compatible attributes
- Calculates drainage areas with upstream accumulation
- Adds elevation statistics from DEM
- Validates attribute quality

### ✅ ObservationPointIntegrator - INTEGRATED  
**Location**: Step 5 (RAVEN Model Generation)
**Function**: Integrates gauge stations into watershed routing
**Integration**: Added before RVH file generation
**Benefits**:
- Sets GAUGED attribute correctly (0/1 instead of always 0)
- Validates drainage areas against observations
- Links gauge stations to subbasins
- Improves model calibration potential

**Current Gauge Discovery Method**:
The workflow currently discovers gauge stations through the **Climate/Hydrometric Data step**, which:
1. Searches for hydrometric stations within 50km of the outlet
2. Downloads streamflow observations for calibration
3. Stores station metadata and locations
4. The `ObservationPointIntegrator` can use this data to set GAUGED attributes

## File Structure

```
project_workspace/
├── data/
│   ├── dem.tif
│   ├── landcover.tif
│   ├── soil.tif
│   ├── study_area.geojson
│   ├── subbasins.shp
│   ├── subbasins_with_attributes.shp  # ✅ NEW
│   ├── streams.shp
│   ├── watershed_boundary.shp
│   ├── lakes_with_routing_ids.shp
│   └── hrus.geojson
├── climate/
│   └── climate_forcing.csv
├── hydrometric/
│   └── observed_streamflow.csv
├── models/files/outlet_XX.XXXX_YY.YYYY/
│   ├── outlet_XX.XXXX_YY.YYYY.rvh
│   ├── outlet_XX.XXXX_YY.YYYY.rvi
│   ├── outlet_XX.XXXX_YY.YYYY.rvp
│   └── outlet_XX.XXXX_YY.YYYY.rvt
└── results/
    ├── step1_results.json
    ├── step2_results.json
    ├── step3_results.json
    ├── step4_results.json
    └── step5_results.json
```

## Key Improvements Made

1. **Basic Attributes Integration**: Added comprehensive geometric and topographic attribute calculation to Step 2
2. **Observation Point Integration**: Added gauge station integration to Step 5 for proper GAUGED attribute setting
3. **Coordinate System Standardization**: Enhanced Step 1 with BasinMaker-style CRS processing
4. **No Duplication**: Integrated processors at optimal points without duplicating existing functionality

## Workflow Execution

The workflow can be executed through several entry points:
- `simple_workflow.py`: Complete workflow execution
- `run_steps_2_to_5.py`: Skip Step 1 and climate data
- `workflows/single_outlet_orchestrator.py`: Full orchestration with project management

Each step is self-contained and can be executed independently, with results saved as JSON files for inter-step communication.