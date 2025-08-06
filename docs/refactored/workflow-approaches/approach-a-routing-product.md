# Approach A: Routing Product Workflow

## Overview

The **Routing Product Workflow** leverages existing BasinMaker routing products to rapidly generate RAVEN models. This approach uses pre-processed, professionally validated watershed data to deliver results in 2-3 minutes.

## Key Advantages

- **Ultra-Fast**: 2-3 minutes total execution time
- **No Network Required**: Uses local routing product data
- **High Quality**: Pre-validated by BasinMaker methodology
- **Proven**: Used in operational hydrological models
- **Consistent**: Same methodology across all watersheds
- **Resource Efficient**: Low CPU and memory requirements

## Workflow Steps (5 Total)

### **Step 1: Validate Coordinates and Find Routing Product**
```python
from workflows.steps.validation_steps import ValidateCoordinatesAndFindRoutingProduct

step = ValidateCoordinatesAndFindRoutingProduct()
result = step.execute({
    'latitude': 45.5017,
    'longitude': -73.5673
})
```

**Purpose**: Validate outlet coordinates and locate applicable routing product
**Processing**:
- Validate coordinate ranges and format
- Search available routing products for coverage
- Identify target subbasin ID within routing product
- Verify routing product data integrity

**Output**:
```python
{
    'latitude': 45.5017,
    'longitude': -73.5673,
    'routing_product_path': '/data/canadian/routing_product_v2.1/',
    'target_subbasin_id': 12345,
    'routing_product_version': 'v2.1',
    'coverage_confirmed': True,
    'success': True
}
```

**Error Conditions**:
- Invalid coordinates
- No routing product coverage for location
- Corrupted routing product data

---

### **Step 2: Extract Subregion from Routing Product**
```python
from workflows.steps.routing_product_steps import ExtractSubregionFromRoutingProduct

step = ExtractSubregionFromRoutingProduct()
result = step.execute({
    'routing_product_path': '/data/canadian/routing_product_v2.1/',
    'target_subbasin_id': 12345
})
```

**Purpose**: Extract upstream watershed network from routing product
**Processing**:
- Load routing product files (catchments, rivers, lakes, gauges)
- Identify all upstream subbasins using connectivity matrix
- Extract spatial data for watershed region
- Maintain BasinMaker topology and attributes

**BasinMaker Integration**:
Uses `SubregionExtractor.extract_subregion_from_routing_product()` - direct implementation of BasinMaker's `Select_Routing_product_based_SubId_purepy()` function.

**Output**:
```python
{
    'extracted_catchments': '/workspace/catchments.shp',
    'extracted_rivers': '/workspace/rivers.shp', 
    'extracted_lakes': '/workspace/lakes.shp',
    'extracted_gauges': '/workspace/gauges.shp',
    'subbasin_count': 45,
    'total_area_km2': 1247.8,
    'success': True
}
```

**Error Conditions**:
- Missing routing product files
- Invalid subbasin ID
- Topology errors in routing product

---

### **Step 3: Generate HRUs from Routing Product**
```python
from workflows.steps.hru_generation_steps import GenerateHRUsFromRoutingProduct

step = GenerateHRUsFromRoutingProduct()
result = step.execute({
    'extracted_catchments': '/workspace/catchments.shp',
    'extracted_lakes': '/workspace/lakes.shp'
})
```

**Purpose**: Create Hydrological Response Units from routing product data
**Processing**:
- Use existing catchment polygons as sub-basins
- Create lake HRUs with ID = -1 (BasinMaker standard)
- Create land HRUs from remaining catchment areas
- Apply BasinMaker lookup tables for land use, soil, vegetation
- Calculate hydraulic parameters from routing product attributes

**BasinMaker Integration**:
- Uses authentic BasinMaker lookup tables from `basinmaker-extracted/tests/testdata/HRU/`
- Implements BasinMaker HRU generation logic from `hru.py`
- Maintains BasinMaker attribute structure and naming

**Output**:
```python
{
    'final_hrus': '/workspace/final_hrus.shp',
    'lake_hru_count': 8,
    'land_hru_count': 37,
    'total_hru_count': 45,
    'total_area_km2': 1247.8,
    'attribute_completeness': 100.0,
    'success': True
}
```

**Error Conditions**:
- Missing attribute data in routing product
- Lookup table access failures
- Geometric processing errors

---

### **Step 4: Generate RAVEN Model Files**
```python
from workflows.steps.raven_generation_steps import GenerateRAVENModelFiles

step = GenerateRAVENModelFiles()
result = step.execute({
    'final_hrus': '/workspace/final_hrus.shp',
    'extracted_rivers': '/workspace/rivers.shp',
    'extracted_lakes': '/workspace/lakes.shp'
})
```

**Purpose**: Generate complete 5-file RAVEN model from routing product data
**Processing**:
- Select appropriate RAVEN model type based on watershed characteristics
- Generate RVH file (spatial structure) from HRUs and connectivity
- Generate RVP file (parameters) from BasinMaker lookup tables
- Generate RVI file (instructions) using model template
- Generate RVT file (climate template) for forcing data
- Generate RVC file (initial conditions) with defaults

**Model Selection Logic**:
```python
if watershed_area_km2 < 100:
    model_type = "GR4JCN"  # Simple conceptual
elif has_lakes and is_cold_region:
    model_type = "HMETS"   # Cold region with lakes
elif has_significant_lakes:
    model_type = "HBVEC"   # Lake routing capable
else:
    model_type = "UBCWM"   # General purpose
```

**Output**:
```python
{
    'rvh_file': '/workspace/model.rvh',
    'rvp_file': '/workspace/model.rvp',
    'rvi_file': '/workspace/model.rvi',
    'rvt_file': '/workspace/model.rvt',
    'rvc_file': '/workspace/model.rvc',
    'selected_model': 'HMETS',
    'model_files_count': 5,
    'success': True
}
```

**Error Conditions**:
- Template loading failures
- Parameter validation errors
- File generation errors

---

### **Step 5: Validate Complete Model**
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
- Validate file existence and format
- Check RAVEN syntax compliance
- Verify cross-file references and consistency
- Validate parameter ranges and values
- Generate model summary and metadata

**Validation Checks**:
- **Format Validation**: Proper RAVEN file structure
- **Consistency Validation**: Cross-file reference integrity
- **Parameter Validation**: Values within acceptable ranges
- **Completeness Validation**: All required sections present

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
        'cross_references_valid': True
    },
    'model_summary': '/workspace/model_summary.json',
    'hru_count': 45,
    'subbasin_count': 45,
    'model_ready_for_simulation': True,
    'success': True
}
```

## Complete Workflow Execution

### **Python Implementation**
```python
from workflows.approaches.routing_product_workflow import RoutingProductWorkflow

# Initialize workflow
workflow = RoutingProductWorkflow(workspace_dir="montreal_watershed")

# Execute complete workflow
result = workflow.execute_complete_workflow(
    latitude=45.5017,
    longitude=-73.5673,
    outlet_name="Montreal_Routing_Product"
)

if result['success']:
    print(f"RAVEN model generated in {result['execution_time']:.1f} minutes")
    print(f"Model files: {len(result['model_files'])} files created")
    print(f"HRUs: {result['total_hru_count']} hydrological response units")
    print(f"Area: {result['total_area_km2']:.1f} km²")
else:
    print(f"Workflow failed: {result['error']}")
```

### **Command Line Usage**
```bash
# Execute routing product workflow
python -m workflows.routing_product --lat 45.5017 --lon -73.5673 --name Montreal_Test

# With specific routing product version
python -m workflows.routing_product --lat 45.5017 --lon -73.5673 --routing-version v2.1
```

## Performance Characteristics

### **Execution Time Breakdown**
- **Step 1**: Coordinate validation (5-10 seconds)
- **Step 2**: Subregion extraction (30-60 seconds)
- **Step 3**: HRU generation (45-90 seconds)
- **Step 4**: RAVEN file generation (15-30 seconds)
- **Step 5**: Model validation (10-20 seconds)
- **Total**: 2-3 minutes average

### **Resource Requirements**
- **Memory**: 200-500 MB typical
- **Storage**: 50-200 MB for model files
- **CPU**: Single-core sufficient
- **Network**: Not required (local data only)

### **Scalability**
- **Linear scaling** with watershed size
- **Excellent performance** for operational use
- **Batch processing** capable for multiple outlets

## Data Requirements

### **Required Routing Product Files**
```
routing_product_folder/
├── catchment_polygon.shp      # Sub-basin boundaries
├── river_polyline.shp         # Stream network
├── lake_polygon.shp           # Lake polygons
├── gauge_point.shp            # Observation points
├── connectivity_matrix.csv    # Upstream-downstream relationships
└── attributes/                # Lookup tables and metadata
    ├── landuse_info.csv
    ├── soil_info.csv
    └── veg_info.csv
```

### **Routing Product Coverage**
- **Canada**: Complete coverage with BasinMaker v2.1
- **North America**: Partial coverage available
- **Global**: Limited to specific regions

## Limitations and Considerations

### **Geographic Limitations**
- **Coverage Dependent**: Only works where routing products exist
- **Resolution Fixed**: Cannot customize spatial resolution
- **Methodology Fixed**: Uses BasinMaker standard approach

### **Data Currency**
- **Static Snapshots**: May not reflect recent landscape changes
- **Update Cycle**: Depends on routing product maintenance schedule
- **Validation Date**: Check routing product creation date

### **Customization Limits**
- **Standard Parameters**: Limited ability to modify processing parameters
- **Fixed Thresholds**: Cannot adjust lake/stream detection criteria
- **Template Based**: Model configuration follows standard templates

## Best Practices

### **Pre-Execution Checks**
1. Verify routing product availability for target coordinates
2. Check routing product version and currency
3. Confirm workspace permissions and storage space
4. Validate coordinate accuracy and projection

### **Quality Assurance**
1. Review extracted watershed boundary for reasonableness
2. Verify HRU count and area calculations
3. Check model file completeness and format
4. Validate against known watershed characteristics

### **Error Recovery**
1. Check routing product data integrity if extraction fails
2. Verify coordinate accuracy if no coverage found
3. Review workspace permissions if file operations fail
4. Consult routing product documentation for troubleshooting

## Related Documentation

- [Routing Product Data Sources](../data-sources/routing-products.md)
- [BasinMaker Integration Guide](../integration/basinmaker.md)
- [RAVEN Model Validation](../validation/raven-models.md)
- [Performance Optimization](../performance/routing-product.md)

---

**Approach A provides the fastest path from outlet coordinates to complete RAVEN models when routing products are available.**