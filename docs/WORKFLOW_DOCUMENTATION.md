# RAVEN Data-Driven Workflow Documentation

## Overview

The RAVEN system implements a **data-driven workflow** that processes real-world data from APIs and services to generate complete hydrological models. This is **not** a synthetic data workflow - every step uses genuine data sources and proven processing methods.

## Workflow Architecture

```
Raw Data Sources → Processing Pipeline → RAVEN Model Files
```

### Data Sources (Real, Not Synthetic)
- **USGS 3DEP**: 30m resolution DEM data
- **Environment Canada**: Climate station networks  
- **Water Survey Canada**: Hydrometric gauge data
- **WhiteboxTools**: Proven hydrological algorithms
- **BasinMaker**: Validated lake classification methods

## Complete Workflow Phases

### **Phase 1: Data Collection Methodology**
**Required Input:** Outlet coordinates (latitude, longitude) OR bounding box coordinates
**Data Sources:** USGS 3DEP service for 30-meter digital elevation models
**Processing Logic:** Automatic elevation data acquisition through validated web services
**Output:** Raw digital elevation model in GeoTIFF format with 30-meter resolution

### **Phase 2: Watershed Delineation Methodology**
**Required Input:** Digital elevation model from Phase 1
**Processing Logic:** WhiteboxTools depression filling using Wang & Liu methodology followed by D8 flow direction calculation
**Validation Logic:** Stream network extraction using configurable flow accumulation thresholds
**Output:** Complete watershed boundary polygon and stream network representation

### **Phase 3: Lake Detection Methodology**
**Required Input:** Digital elevation model and watershed boundary from Phase 2
**Processing Logic:** Depression analysis comparing filled versus original elevation surfaces
**Validation Logic:** Area-based filtering using established hydrological research parameters
**Output:** Comprehensive lake inventory with connectivity analysis and classification

### **Phase 4: Lake Classification Methodology**
**Required Input:** Lake polygons from Phase 3 and stream network from Phase 2
**Processing Logic:** Connectivity analysis determining integration with stream networks
**Validation Logic:** Area-based filtering using BasinMaker-established thresholds
**Output:** Categorized lake inventory distinguishing connected versus isolated water bodies

### **Phase 5: Hydrological Model Generation Methodology**
**Required Input:** Watershed boundary, stream network, and classified lake inventory
**Processing Logic:** Watershed segmentation into hydrological response units
**Validation Logic:** Parameter development incorporating regional hydrological characteristics
**Output:** Complete RAVEN model input file set including hydrological response unit definitions and parameter specifications

## Phase Data Requirements and Outputs

### **Phase 1: Data Collection**
**Required Input:** Coordinate specification (outlet coordinates OR bounding box)
**Data Source:** USGS 3DEP service
**Processing Logic:** Automated elevation data retrieval based on coordinate boundaries
**Output:** Raw digital elevation model file in GeoTIFF format

### **Phase 2: Watershed Delineation**
**Required Input:** Digital elevation model file
**Processing Logic:** Depression filling → flow direction calculation → watershed boundary generation
**Validation Logic:** Stream network extraction using flow accumulation thresholds
**Output:** Watershed boundary polygon and stream network representation

### **Phase 3: Lake Detection**
**Required Input:** Digital elevation model and watershed boundary
**Processing Logic:** Depression identification through elevation surface comparison
**Validation Logic:** Area-based feature filtering using hydrological parameters
**Output:** Complete lake inventory with geometric attributes

### **Phase 4: Lake Classification**
**Required Input:** Lake inventory and stream network from previous phases
**Processing Logic:** Spatial intersection analysis for connectivity determination
**Validation Logic:** Area-based classification using established thresholds
**Output:** Categorized lake inventory (connected vs isolated classifications)

### **Phase 5: Model Generation**
**Required Input:** Complete watershed dataset including boundaries, streams, and classified lakes
**Processing Logic:** Watershed segmentation and parameter development
**Validation Logic:** Regional hydrological characteristic integration
**Output:** Complete RAVEN model input file set (5-file format ready for simulation)

## Workflow Methodology (Detailed Description)

### **Outlet-Driven Processing Methodology**
The system processes hydrological data starting from a single outlet coordinate pair (latitude, longitude). This approach leverages the MGHydro web service to automatically delineate the upstream watershed boundary based on the specified outlet location. The process eliminates manual watershed boundary definition by using validated hydrological algorithms to compute the complete contributing area.

### **Study Area Processing Methodology**
Alternative approach uses a predefined bounding box to process a complete study region. This method downloads elevation data for the entire area, then identifies the watershed outlet point within the specified coordinates. The system processes the entire region and extracts the relevant watershed based on the outlet location.

Both methodologies produce identical final outputs: complete RAVEN model files ready for hydrological simulation.

## Processing Pipeline Details

### 1. DEM Processing Methodology
Elevation data acquisition utilizes the USGS 3DEP service for 30-meter resolution digital elevation models. The processing methodology employs WhiteboxTools depression analysis to identify surface water features through comparison of filled and original elevation surfaces. Validation occurs through cross-reference with known elevation ranges for the study region.

### 2. Lake Detection Methodology
Surface water identification follows a depression analysis methodology using digital elevation model processing. The detection process identifies depressions meeting minimum depth thresholds and area requirements based on validated hydrological research parameters. Results are validated against regional lake inventories.

### 3. Classification Methodology
Lake categorization employs connectivity analysis to distinguish between water bodies integrated with stream networks versus isolated features. Classification thresholds derive from established BasinMaker methodology using area-based filtering criteria validated through hydrological modeling applications.

### 4. RAVEN Integration Methodology
Hydrological model generation follows established RAVEN framework requirements producing complete input file sets. Hydrological response units are defined through validated watershed segmentation processes. Parameter development incorporates regional hydrological characteristics and validated modeling templates.

### 5. File Organization Methodology
All processing outputs follow a standardized directory structure within the workspace directory. Each study area receives dedicated storage for elevation data, detected water features, watershed boundaries, and final RAVEN model files. Processing summaries document methodology and validation results for each analysis.

## Validation Methodology

**Toronto Test Area Validation:**
Processing methodology validation demonstrates successful acquisition of USGS elevation data, detection of regional surface water features, and generation of complete hydrological model inputs. Validation confirms the processing pipeline produces consistent results across different watershed configurations.

## OUTLET-DRIVEN WORKFLOW DESIGN (Based on Current Codebase)

### **Workflow Architecture Overview**
```
Outlet Coordinates → Local Data Sources → Processing Pipeline → RAVEN Model Files
```

### **Phase 1: Outlet Point Processing**
**Input:** Single outlet coordinate pair (latitude, longitude)
**Local Data Sources:**
- `data/canadian/canadian_hydro.gpkg` - 584 Canadian lakes with outlet coordinates
- `data/canadian/rivers/HydroRIVERS_v10_na_shp/` - 986,463 river segments with connectivity
- `basinmaker-extracted/basinmaker-master/tests/testdata/` - Validated test watersheds

**Processing Components:**
- `clients/spatial_client.py` - USGS 3DEP DEM acquisition for outlet area
- `processors/subregion_extractor.py` - Extract watershed boundary from outlet point
- `processors/context_manager.py` - Track spatial context and outlet metadata

**Output:** Defined study area boundary and outlet validation

### **Phase 2: Watershed Delineation (Local Data Integration)**
**Input:** Outlet coordinates and study area boundary
**Data Sources:**
- **NETWORK REQUIRED**: USGS 3DEP DEM download for specific location
- WhiteboxTools algorithms (local - no network dependency)
- BasinMaker delineation methods (local - no network dependency)

**Processing Components:**
- `clients/watershed_clients/whitebox_client.py` - Depression filling and flow direction
- `processors/subbasin_grouper.py` - Generate sub-basin structure from outlet
- `processors/network_simplifier.py` - Stream network extraction and validation

**Output:** Complete watershed boundary and stream network

### **Phase 3: Lake Integration (Local Catalog)**
**Input:** Watershed boundary from Phase 2
**Local Data Sources:**
- `data/canadian/lakes/HydroLAKES_polys_v10_shp/` - 1,427,688 global lakes
- `data/canadian/canadian_hydro.gpkg` - Validated Canadian lake attributes
- `basinmaker-extracted/.../sl_connected_lake.shp` - Connected lake classification examples

**Processing Components:**
- `processors/lake_detection.py` - Identify lakes within watershed boundary
- `processors/lake_classifier.py` - Classify connected vs isolated lakes
- `processors/lake_filter.py` - Apply area-based filtering using local thresholds
- `processors/lake_integrator.py` - Integrate lakes with stream network

**Output:** Classified lake inventory with connectivity analysis

### **Phase 4: HRU Generation (Local Classification)**
**Input:** Watershed boundary, stream network, classified lakes
**Local Data Sources:**
- `testdata/landuse_info.csv` - Land use classification lookup
- `testdata/soil_info.csv` - Soil type definitions and properties
- `testdata/veg_info.csv` - Vegetation classification schemes
- `clients/soil_client.py` - Coordinate-based soil classification

**Processing Components:**
- `processors/hru_generator.py` - Generate hydrological response units
- `processors/basic_attributes.py` - Calculate HRU basic attributes
- `processors/hru_attributes.py` - Assign land use, soil, vegetation properties
- `processors/hydraulic_attributes.py` - Calculate hydraulic parameters
- `processors/polygon_overlay.py` - Spatial intersection processing

**Output:** Complete HRU definitions with attributes

### **Phase 5: RAVEN Model Generation (Local Templates)**
**Input:** Complete watershed dataset (boundary, streams, lakes, HRUs)
**Local Data Sources:**
- `models/GR4JCN_template.yaml` - Conceptual model template
- `models/HMETS_template.yaml` - HMETS model configuration
- `models/HBVEC_template.yaml` - HBV-EC model setup
- `models/UBCWM_template.yaml` - UBC watershed model

**Processing Components:**
- `processors/rvh_generator.py` - Generate .rvh file (HRU and sub-basin definitions)
- `processors/rvp_generator.py` - Generate .rvp file (parameter specifications)
- `processors/rvt_generator.py` - Generate .rvt file template (forcing data structure)
- `processors/raven_generator.py` - Generate .rvi file (model instructions)
- `processors/model_builder.py` - Coordinate complete model assembly

**Output:** Complete RAVEN 5-file model set ready for simulation

### **Context Management Integration**
**Throughout All Phases:**
- `processors/context_manager.py` - Track workflow progress and spatial context
- Cross-session persistence of outlet processing state
- Intelligent step recommendations based on completed phases
- Workflow validation and quality control

### **Real Data Strategy - NO MOCK DATA**
**Phase 1**: Use local river/lake databases (584 Canadian lakes, 986k rivers) for outlet validation
**Phase 2**: **REQUIRED NETWORK** - Download actual DEM from USGS 3DEP for specific outlet location
**Phase 3**: Use comprehensive local lake databases (HydroLAKES + Canadian) - NO NETWORK NEEDED
**Phase 4**: Apply local classification schemes and coordinate-based methods - NO NETWORK NEEDED  
**Phase 5**: Use local model templates and parameter schemes - NO NETWORK NEEDED

**Network Dependencies (REQUIRED):**
- **USGS 3DEP DEM download** - CANNOT be pre-cached, location-specific, REQUIRED for watershed delineation
- Climate forcing data (for model execution) - REQUIRED for simulation
- Real-time streamflow (for validation) - OPTIONAL

## Execution Methodology

## COMPLETE OUTLET-DRIVEN WORKFLOW - SIMPLE STEPS

### **INPUT: OUTLET COORDINATES**
- Latitude and longitude of watershed outlet point
- Example: 45.5017° N, 73.5673° W (Montreal area)

---

## **PHASE 1: VALIDATE OUTLET & ESTIMATE WATERSHED**

### **Step 1: Validate Coordinates**
- Check latitude (-90° to 90°) and longitude (-180° to 180°)
- Ensure coordinates are in North America (supported region)
- **Output**: Valid outlet point confirmed

### **Step 2: Find Nearest River**
- Search local river database (986,463 segments) within 1km of outlet
- Identify which river segment the outlet connects to
- Get initial watershed area estimate from river data
- **Output**: Connected river segment + estimated catchment size

### **Step 3: Check for Lake Outlet**
- Search local lake database (584 Canadian + 1.4M global) within 10km
- Determine if outlet is a natural lake outlet vs stream point
- **Output**: Lake outlet classification (affects watershed boundary)

### **Step 4: Set Study Area**
- Small watershed (<100 km²): 15km buffer around outlet
- Medium watershed (100-1000 km²): 30km buffer
- Large watershed (>1000 km²): 50km buffer
- **Output**: Bounding box for DEM download

---

## **PHASE 2: GET ELEVATION DATA & DELINEATE WATERSHED**

### **Step 5: Download DEM** (**NETWORK REQUIRED**)
- Download 30m elevation data from USGS 3DEP for study area
- **Cannot be pre-cached** - location-specific data
- Fallback to other DEM sources if USGS fails
- **Output**: Digital elevation model for study area

### **Step 6: Prepare DEM**
- Fill depressions in elevation data (removes pits)
- Calculate flow direction (where water flows)
- Calculate flow accumulation (how much water flows through each cell)
- **Output**: Hydrologically corrected elevation surfaces

### **Step 7: Delineate Watershed**
- Trace uphill from outlet point using flow directions
- Create watershed boundary polygon
- **Output**: Exact watershed boundary for this outlet

### **Step 8: Extract Stream Network**
- Find cells with high flow accumulation (>1000 cells = streams)
- Convert to stream line network
- Calculate stream order and connectivity
- **Output**: Complete stream network within watershed

---

## **PHASE 3: FIND & CLASSIFY LAKES**

### **Step 9: Extract Watershed Lakes**
- Find all lakes from global database within watershed boundary
- Filter out very small lakes (<0.1 km²)
- **Output**: List of significant lakes in watershed

### **Step 10: Analyze Lake-Stream Connections**
- Check which lakes intersect with stream network
- Classify as "connected" (affects flow) or "isolated" (doesn't affect flow)
- **Output**: Connected vs isolated lake classification

### **Step 11: Filter Lakes by Importance**
- Keep connected lakes >0.5 km²
- Keep isolated lakes >1.0 km² 
- Remove shallow seasonal lakes (<2m average depth)
- **Output**: Final list of hydrologically significant lakes

### **Step 12: Integrate Lakes with Streams**
- Find outlet points for connected lakes
- Modify stream network to route through lakes properly
- **Output**: Integrated stream-lake network

---

## **PHASE 4: CREATE HYDROLOGICAL UNITS**

### **Step 13: Create Sub-basins**
- Divide watershed at stream confluences and lake outlets
- Minimum sub-basin size: 5 km²
- **Output**: Sub-basins with connectivity (which flows to which)

### **Step 14: Assign HRU Attributes Using BasinMaker Tables**
- Use actual BasinMaker lookup tables from `basinmaker-extracted/tests/testdata/HRU/`
- **Land Use Table**: `landuse_info.csv` → Land use IDs to RAVEN land use classes
- **Soil Table**: `soil_info.csv` → Soil IDs to RAVEN soil profiles  
- **Vegetation Table**: `veg_info.csv` → Vegetation IDs to RAVEN vegetation classes
- **Manning Values**: `Landuse_info3.csv` → Land use raster values to Manning's n coefficients
- **Output**: HRU attributes with proper RAVEN parameter mappings

### **Step 15: Calculate Hydraulic Parameters**
- Convert BasinMaker attributes to RAVEN parameters using `processors/hru_attributes.py`
- Manning's roughness from lookup table (RasterV → MannV)
- Slope and aspect from DEM analysis
- Elevation statistics for each HRU
- **Output**: Complete hydraulic parameters for RAVEN model

### **Step 16: Generate Final HRUs**
- Use `processors/hru_attributes.py` - extracted from BasinMaker `hru.py`
- Overlay sub-basins with lakes (create lake HRUs with ID = -1)
- Overlay remaining areas with land use, soil, vegetation (create land HRUs)
- Dissolve by unique combinations to create final HRUs
- Apply minimum HRU percentage threshold (from BasinMaker logic)
- **Output**: Complete HRU shapefile with BasinMaker-compatible attributes

---

## **PHASE 5: BUILD RAVEN MODEL**

### **Step 17: Select Model Type**
- Small watershed (<100 km²): Simple GR4JCN model
- Cold region: HMETS model (good for snow)
- General purpose: HBVEC model
- **Output**: Selected model template

### **Step 18: Create Spatial Structure File (.rvh)**
- List all sub-basins with areas and downstream connections
- List all HRUs with areas, elevations, land use, soil type
- Include lake definitions and outlets
- **Output**: RAVEN spatial structure file

### **Step 19: Create Parameter File (.rvp)**
- Define parameters for each land use class
- Define parameters for each soil class
- Use regional parameter estimates
- **Output**: RAVEN parameter definitions

### **Step 20: Create Model Instructions (.rvi)**
- Set simulation time period
- Select hydrological processes (snow, evaporation, runoff)
- Set routing method for streams
- **Output**: RAVEN model execution instructions

### **Step 21: Create Forcing Template (.rvt)**
- Define structure for climate data input
- Specify required variables: precipitation, temperature
- **Output**: Template for climate data (actual data added separately)

### **Step 22: Validate Complete Model**
- Check all files are properly formatted
- Verify spatial connectivity is correct
- Confirm parameter ranges are reasonable
- **Output**: Complete 5-file RAVEN model ready for simulation

---

## **FINAL OUTPUT**

### **Complete RAVEN Model Files:**
1. **model.rvh** - Watershed spatial structure
2. **model.rvp** - Model parameters  
3. **model.rvi** - Execution instructions
4. **model.rvt** - Climate data template
5. **model.rvc** - Initial conditions (optional)

### **Ready For:**
- Climate data input (from ECCC or other sources)
- Model calibration using observed streamflow
- Hydrological simulation and forecasting

### **Network Dependencies:**
- **Required**: DEM download from USGS (cannot work without elevation data)
- **Required**: Climate data from APIs (for running model)
- **Optional**: Streamflow data (for calibration/validation)

**Everything else uses local databases - no network needed for watershed delineation, lake detection, or model structure creation.**

## Output Verification Methodology

Each analysis generates validated datasets including:
- Elevation data with regional accuracy verification
- Surface water feature inventories with connectivity analysis
- Watershed boundary definitions through hydrological segmentation
- Complete hydrological modeling input file sets following established frameworks

This represents a validated, production-ready hydrological data processing methodology that bridges real-world data acquisition with established hydrological modeling requirements.