# Approach B: Full Delineation Workflow

## 🎯 Overview

The **Full Delineation Workflow** creates complete watershed models from scratch using DEM analysis and hydrological processing. This approach provides maximum flexibility and global coverage at the cost of longer processing time.

## 🌍 Key Advantages

- **Universal Coverage**: Works anywhere with DEM data availability
- **Fully Customizable**: Complete control over resolution and methodology
- **Current Data**: Uses latest elevation and spatial datasets
- **Research Ready**: Suitable for method development and validation
- **Transparent**: Full visibility into all processing steps
- **Flexible**: Handles any watershed size or complexity

## 📊 Workflow Steps (8 Total)

### **Step 1: Validate Coordinates and Set DEM Area**
```python
from workflows.steps.validation_steps import ValidateCoordinatesAndSetDEMArea

step = ValidateCoordinatesAndSetDEMArea()
result = step.execute({
    'latitude': 45.5017,
    'longitude': -73.5673
})
```

**Purpose**: Validate outlet coordinates and calculate optimal DEM download area
**Processing**:
- Validate coordinate ranges and format
- Calculate intelligent DEM bounds based on geographic context
- Estimate watershed size using topographic analysis
- Optimize DEM resolution for processing efficiency

**Intelligent Sizing Logic**:
```python
if 49.0 <= latitude <= 60.0:  # Canadian Prairies/Boreal
    default_buffer_km = 25    # Typically larger watersheds
elif 45.0 <= latitude <= 49.0:  # Southern Canada
    default_buffer_km = 20    # Mixed watershed sizes
elif latitude >= 60.0:       # Arctic/Subarctic
    default_buffer_km = 35    # Very large watersheds
else:
    default_buffer_km = 20    # Default
```

**Output**:
```python
{
    'latitude': 45.5017,
    'longitude': -73.5673,
    'dem_bounds': [-74.0, 45.0, -73.0, 46.0],
    'buffer_km': 20,
    'dem_resolution': '30m',
    'estimated_size_mb': 45.2,
    'processing_method': 'geographic_defaults',
    'success': True
}
```

---

### **Step 2: Download and Prepare DEM** ⚠️ **NETWORK REQUIRED**
```python
from workflows.steps.dem_processing_steps import DownloadAndPrepareDEM

step = DownloadAndPrepareDEM()
result = step.execute({
    'dem_bounds': [-74.0, 45.0, -73.0, 46.0],
    'dem_resolution': '30m'
})
```

**Purpose**: Acquire and hydrologically condition elevation data
**Processing**:
- Download 30m USGS 3DEP elevation data
- Mosaic multiple tiles if needed
- Fill depressions using WhiteboxTools Wang & Liu algorithm
- Calculate D8 flow direction and accumulation
- Generate hydrologically-corrected surfaces

**WhiteboxTools Integration**:
```python
# Depression filling
wbt.fill_depressions_wang_and_liu(dem_file, filled_dem)

# Flow direction calculation  
wbt.d8_pointer(filled_dem, flow_dir)

# Flow accumulation
wbt.d8_flow_accumulation(flow_dir, flow_accum)
```

**Output**:
```python
{
    'original_dem': '/workspace/dem_original.tif',
    'filled_dem': '/workspace/dem_filled.tif',
    'flow_direction': '/workspace/flow_dir.tif',
    'flow_accumulation': '/workspace/flow_accum.tif',
    'dem_resolution_m': 30,
    'depressions_filled': 1247,
    'processing_time_seconds': 145.2,
    'success': True
}
```

**Error Conditions**:
- Network connectivity issues
- USGS service unavailable
- Insufficient disk space
- WhiteboxTools processing failures

---

### **Step 3: Delineate Watershed and Streams**
```python
from workflows.steps.watershed_steps import DelineateWatershedAndStreams

step = DelineateWatershedAndStreams()
result = step.execute({
    'flow_direction': '/workspace/flow_dir.tif',
    'flow_accumulation': '/workspace/flow_accum.tif',
    'outlet_latitude': 45.5017,
    'outlet_longitude': -73.5673
})
```

**Purpose**: Trace watershed boundary and extract stream network
**Processing**:
- Snap outlet to nearest high-flow accumulation cell
- Trace upstream watershed boundary using flow directions
- Extract stream network using flow accumulation threshold
- Calculate stream order using Strahler ordering
- Generate watershed and stream vector data

**Stream Threshold Calculation**:
```python
# Adaptive threshold based on watershed characteristics
base_threshold = 1000  # cells
area_factor = max(1.0, watershed_area_km2 / 100.0)
stream_threshold = int(base_threshold * area_factor)
```

**Output**:
```python
{
    'watershed_boundary': '/workspace/watershed.shp',
    'stream_network': '/workspace/streams.shp',
    'watershed_area_km2': 245.8,
    'stream_length_km': 156.7,
    'max_stream_order': 4,
    'outlet_snapped': True,
    'snap_distance_m': 45.0,
    'success': True
}
```

**Error Conditions**:
- Outlet outside DEM bounds
- No upstream area (outlet at divide)
- Stream threshold too restrictive
- Vectorization failures

---

### **Step 4: Detect and Classify Lakes**
```python
from workflows.steps.lake_processing_steps import DetectAndClassifyLakes

step = DetectAndClassifyLakes()
result = step.execute({
    'watershed_boundary': '/workspace/watershed.shp',
    'stream_network': '/workspace/streams.shp',
    'filled_dem': '/workspace/dem_filled.tif'
})
```

**Purpose**: Identify, classify, and integrate lakes within watershed
**Processing**:
- Find lakes within watershed using spatial intersection
- Classify lakes as connected vs isolated using stream network
- Apply BasinMaker significance thresholds
- Integrate connected lakes with stream network
- Generate lake outlet points and routing

**BasinMaker Lake Classification**:
```python
# Connected lake thresholds (BasinMaker standards)
connected_area_threshold = 0.5  # km²
connected_depth_threshold = 2.0  # m

# Isolated lake thresholds
isolated_area_threshold = 1.0   # km²
isolated_depth_threshold = 3.0  # m
```

**Output**:
```python
{
    'significant_connected_lakes': '/workspace/connected_lakes.shp',
    'significant_isolated_lakes': '/workspace/isolated_lakes.shp',
    'integrated_stream_network': '/workspace/integrated_streams.shp',
    'lake_outlets': '/workspace/lake_outlets.shp',
    'connected_lake_count': 5,
    'isolated_lake_count': 3,
    'total_lake_area_km2': 12.3,
    'success': True
}
```

**Error Conditions**:
- Lake database access failures
- Stream-lake intersection errors
- Outlet detection failures
- Network integration issues

---

### **Step 5: Create Sub-basins and HRUs**
```python
from workflows.steps.hru_generation_steps import CreateSubBasinsAndHRUs

step = CreateSubBasinsAndHRUs()
result = step.execute({
    'watershed_boundary': '/workspace/watershed.shp',
    'integrated_stream_network': '/workspace/integrated_streams.shp',
    'significant_connected_lakes': '/workspace/connected_lakes.shp',
    'filled_dem': '/workspace/dem_filled.tif'
})
```

**Purpose**: Generate sub-basins and complete HRUs with BasinMaker attributes
**Processing**:
- Create sub-basins at stream confluences and lake outlets
- Apply BasinMaker lookup tables for land use, soil, vegetation
- Calculate hydraulic parameters from DEM analysis
- Generate lake HRUs (ID = -1) and land HRUs
- Apply minimum area thresholds (1% of watershed)

**BasinMaker HRU Generation**:
```python
# Create lake HRUs (BasinMaker standard)
for lake in connected_lakes:
    lake_hru = {
        'hru_id': f"LAKE_{lake.id}",
        'hru_type': 'LAKE',
        'landuse_class': 'WATER',
        'soil_class': 'WATER',
        'vegetation_class': 'WATER',
        'mannings_n': 0.03,
        'geometry': lake.geometry
    }

# Create land HRUs from remaining areas
for subbasin in subbasins:
    remaining_area = subbasin.geometry.difference(lake_union)
    if remaining_area.area > min_area_threshold:
        land_hru = create_land_hru(remaining_area, attributes)
```

**Output**:
```python
{
    'sub_basins': '/workspace/subbasins.shp',
    'final_hrus': '/workspace/final_hrus.shp',
    'hydraulic_parameters': '/workspace/hydraulic_params.shp',
    'subbasin_count': 15,
    'lake_hru_count': 5,
    'land_hru_count': 18,
    'total_hru_count': 23,
    'total_area_km2': 245.8,
    'success': True
}
```

**Error Conditions**:
- Sub-basin delineation failures
- Lookup table access errors
- Hydraulic parameter calculation issues
- Minimum area threshold violations

---

### **Step 6: Select Model and Generate Structure**
```python
from workflows.steps.raven_generation_steps import SelectModelAndGenerateStructure

step = SelectModelAndGenerateStructure()
result = step.execute({
    'final_hrus': '/workspace/final_hrus.shp',
    'sub_basins': '/workspace/subbasins.shp',
    'significant_connected_lakes': '/workspace/connected_lakes.shp'
})
```

**Purpose**: Select RAVEN model type and generate spatial structure files
**Processing**:
- Analyze watershed characteristics for model selection
- Generate RVH file (spatial structure) with HRUs and connectivity
- Generate RVP file (parameters) using BasinMaker lookup tables
- Apply model-specific parameter sets and configurations

**Model Selection Logic**:
```python
if watershed_area_km2 < 100:
    selected_model = "GR4JCN"  # Simple conceptual
elif is_cold_region and has_snow:
    selected_model = "HMETS"   # Cold region optimized
elif has_significant_lakes:
    selected_model = "HBVEC"   # Lake routing capable
else:
    selected_model = "UBCWM"   # General purpose
```

**Output**:
```python
{
    'selected_model': 'HMETS',
    'rvh_file': '/workspace/model.rvh',
    'rvp_file': '/workspace/model.rvp',
    'model_description': 'HMETS model optimized for cold regions with snow',
    'hru_count': 23,
    'subbasin_count': 15,
    'parameter_count': 45,
    'success': True
}
```

---

### **Step 7: Generate Model Instructions**
```python
from workflows.steps.raven_generation_steps import GenerateModelInstructions

step = GenerateModelInstructions()
result = step.execute({
    'selected_model': 'HMETS',
    'sub_basins': '/workspace/subbasins.shp',
    'final_hrus': '/workspace/final_hrus.shp'
})
```

**Purpose**: Generate RAVEN model execution and climate data files
**Processing**:
- Generate RVI file (model instructions) with process selection
- Generate RVT file (climate template) for forcing data structure
- Generate RVC file (initial conditions) with reasonable defaults
- Configure model-specific processes and routing methods

**HMETS Process Configuration**:
```python
processes = [
    ":SnowBalance SNOBAL_SIMPLE_MELT SNOW PONDED_WATER",
    ":Precipitation PRECIP_RAVEN ATMOS_PRECIP MULTIPLE", 
    ":Infiltration INF_GREEN_AMPT PONDED_WATER SOIL[0]",
    ":Baseflow BASE_LINEAR SOIL[2] SURFACE_WATER",
    ":Evaporation PET_PENMAN_MONTEITH SOIL[0] ATMOSPHERE"
]
```

**Output**:
```python
{
    'rvi_file': '/workspace/model.rvi',
    'rvt_file': '/workspace/model.rvt',
    'rvc_file': '/workspace/model.rvc',
    'simulation_period': '2020-01-01 to 2022-12-31',
    'time_step': '1.0 day',
    'climate_stations': 15,
    'process_count': 12,
    'success': True
}
```

---

### **Step 8: Validate Complete Model**
```python
from workflows.steps.validation_steps import ValidateCompleteModel

step = ValidateCompleteModel()
result = step.execute({
    'rvh_file': '/workspace/model.rvh',
    'rvp_file': '/workspace/model.rvp',
    'rvi_file': '/workspace/model.rvi',
    'rvt_file': '/workspace/model.rvt',
    'rvc_file': '/workspace/model.rvc'
})
```

**Purpose**: Comprehensive validation of complete RAVEN model
**Processing**:
- Validate all 5 RAVEN files for format and syntax
- Check cross-file references and consistency
- Validate parameter ranges and physical reasonableness
- Generate comprehensive model summary and metadata

**Validation Checks**:
- **File Format**: Proper RAVEN syntax and structure
- **Cross-References**: HRU IDs, land use classes, soil profiles consistent
- **Parameter Ranges**: Manning's n, hydraulic conductivity, etc. within limits
- **Mass Balance**: Area calculations and connectivity preserved

**Output**:
```python
{
    'model_valid': True,
    'validation_results': {
        'rvh_valid': True,
        'rvp_valid': True,
        'rvi_valid': True,
        'rvt_valid': True,
        'rvc_valid': True,
        'cross_references_valid': True,
        'parameter_ranges_valid': True
    },
    'model_summary': '/workspace/model_summary.json',
    'model_ready_for_simulation': True,
    'success': True
}
```

## 🎯 Complete Workflow Execution

### **Python Implementation**
```python
from workflows.approaches.full_delineation_workflow import FullDelineationWorkflow

# Initialize workflow
workflow = FullDelineationWorkflow(workspace_dir="montreal_watershed")

# Execute complete workflow
result = workflow.execute_complete_workflow(
    latitude=45.5017,
    longitude=-73.5673,
    outlet_name="Montreal_Full_Delineation"
)

if result['success']:
    print(f"✅ RAVEN model generated in {result['execution_time']:.1f} minutes")
    print(f"📁 Model files: {len(result['model_files'])} files created")
    print(f"🎯 HRUs: {result['total_hru_count']} hydrological response units")
    print(f"🏞️ Area: {result['total_area_km2']:.1f} km²")
    print(f"💧 Lakes: {result['lake_count']} significant lakes integrated")
else:
    print(f"❌ Workflow failed: {result['error']}")
```

### **Command Line Usage**
```bash
# Execute full delineation workflow
python -m workflows.full_delineation --lat 45.5017 --lon -73.5673 --name Montreal_Test

# With custom DEM resolution
python -m workflows.full_delineation --lat 45.5017 --lon -73.5673 --dem-res 10m

# With custom buffer size
python -m workflows.full_delineation --lat 45.5017 --lon -73.5673 --buffer 30
```

## 📈 Performance Characteristics

### **Execution Time Breakdown**
- **Step 1**: Coordinate validation and DEM area (10-20 seconds)
- **Step 2**: DEM download and processing (5-15 minutes) ⚠️
- **Step 3**: Watershed and stream delineation (2-5 minutes)
- **Step 4**: Lake detection and classification (1-3 minutes)
- **Step 5**: Sub-basin and HRU generation (3-7 minutes)
- **Step 6**: Model structure generation (30-60 seconds)
- **Step 7**: Model instructions generation (15-30 seconds)
- **Step 8**: Model validation (10-20 seconds)
- **Total**: 15-30 minutes average

### **Resource Requirements**
- **Memory**: 1-4 GB (DEM size dependent)
- **Storage**: 200-1000 MB for intermediate files
- **CPU**: Multi-core beneficial for DEM processing
- **Network**: 50-500 MB for DEM download

### **Scalability Factors**
- **DEM Size**: Quadratic scaling with area
- **Watershed Complexity**: Linear scaling with stream network
- **Lake Count**: Linear scaling with lake processing

## 🛠️ Customization Options

### **DEM Processing Parameters**
```python
dem_config = {
    'resolution': '10m',        # 10m, 30m, or 90m
    'fill_method': 'wang_liu',  # Depression filling algorithm
    'flow_algorithm': 'd8',     # Flow direction method
    'stream_threshold': 1000    # Flow accumulation threshold
}
```

### **Lake Detection Parameters**
```python
lake_config = {
    'connected_area_min': 0.5,    # km² minimum for connected lakes
    'isolated_area_min': 1.0,     # km² minimum for isolated lakes
    'depth_threshold': 2.0,       # m minimum depth
    'buffer_distance': 30         # m for stream-lake intersection
}
```

### **HRU Generation Parameters**
```python
hru_config = {
    'min_hru_percent': 1.0,       # Minimum HRU as % of watershed
    'min_subbasin_area': 5.0,     # km² minimum sub-basin size
    'landuse_source': 'coordinate', # Attribute assignment method
    'soil_source': 'coordinate',
    'veg_source': 'coordinate'
}
```

## 🚨 Limitations and Considerations

### **Network Dependencies**
- **Critical**: Step 2 requires stable internet for DEM download
- **Fallback**: Alternative DEM sources (SRTM, ASTER) if USGS unavailable
- **Caching**: Downloaded DEMs can be reused for nearby outlets

### **Processing Complexity**
- **Memory Intensive**: Large watersheds require significant RAM
- **Time Consuming**: 15-30 minutes typical execution
- **Error Prone**: More steps mean more potential failure points

### **Data Quality Considerations**
- **DEM Resolution**: 30m may miss small features
- **Temporal Currency**: DEM data may not reflect recent changes
- **Validation Required**: Results need quality control review

## ✅ Best Practices

### **Pre-Execution Planning**
1. Verify adequate disk space (1-2 GB recommended)
2. Ensure stable network connection for DEM download
3. Check coordinate accuracy and projection
4. Review expected watershed size and complexity

### **Quality Control**
1. Visual inspection of watershed boundary
2. Verification of stream network density and connectivity
3. Review of lake detection and classification results
4. Validation of HRU count and area distribution

### **Performance Optimization**
1. Use appropriate DEM resolution for watershed size
2. Adjust stream threshold for realistic network density
3. Monitor memory usage for large watersheds
4. Consider parallel processing for batch operations

### **Error Recovery**
1. Retry DEM download with different sources if initial fails
2. Adjust stream threshold if no streams detected
3. Review outlet coordinates if watershed delineation fails
4. Check intermediate files if processing stops unexpectedly

## 🔗 Related Documentation

- [DEM Data Sources and Processing](../data-sources/elevation-data.md)
- [WhiteboxTools Integration](../integration/whitebox-tools.md)
- [Lake Detection Methodology](../methodology/lake-detection.md)
- [HRU Generation with BasinMaker](../integration/basinmaker-hru.md)
- [RAVEN Model Templates](../models/raven-templates.md)

---

**Approach B provides complete flexibility and global coverage for watershed modeling when routing products are not available.**