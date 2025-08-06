# HRU Attributes Calculator

## Overview

The HRU Attributes Calculator generates Hydrological Response Units (HRUs) from catchments and land surface data using real BasinMaker logic. This processor implements BasinMaker's HRU generation workflow including polygon overlay operations, attribute calculation, and land surface classification.

**Extracted From**: `basinmaker/postprocessing/hru.py` - HRU generation functions

## Key Features

- **Real BasinMaker Logic**: Exact replication of HRU generation algorithms from BasinMaker
- **Multi-Layer Overlay**: Overlays subbasins with land use, soil, and vegetation data
- **Land and Lake HRUs**: Generates both terrestrial and aquatic HRUs like BasinMaker
- **Terrain Attributes**: Calculates slope, aspect, and elevation for each HRU
- **Professional Integration**: Compatible with your existing spatial analysis infrastructure

## Class: `HRUAttributesCalculator`

### Initialization

**Parameters:**
- `workspace_dir` (Path, optional): Directory for temporary files and outputs

### Core Method: `generate_hrus_from_watershed_results()`

Generates HRUs using real BasinMaker logic adapted to your infrastructure.

**Parameters:**
- `watershed_results` (Dict): Results from ProfessionalWatershedAnalyzer
- `lake_integration_results` (Dict, optional): Results from LakeIntegrator
- `landuse_shapefile` (Path, optional): Path to land use polygon shapefile
- `soil_shapefile` (Path, optional): Path to soil polygon shapefile
- `vegetation_shapefile` (Path, optional): Path to vegetation polygon shapefile
- `dem_raster` (Path, optional): Path to DEM raster for terrain attributes
- `landuse_table` (pd.DataFrame, optional): Land use ID to class mapping
- `soil_table` (pd.DataFrame, optional): Soil ID to profile mapping
- `vegetation_table` (pd.DataFrame, optional): Vegetation ID to class mapping

**Returns:**
- Dictionary with HRU results and output files

## BasinMaker Logic Implementation

### Step 1: Land and Lake HRU Generation
**EXTRACTED FROM**: BasinMaker lines 583-831 (GeneratelandandlakeHRUS)

The processor separates subbasins into land HRUs and lake HRUs based on lake presence. Lake HRUs receive special attributes for water body modeling while land HRUs undergo surface layer overlay processing.

### Step 2: Surface Layer Processing
**EXTRACTED FROM**: BasinMaker lines 266-351 (layer preprocessing)

Land use, soil, and vegetation layers are preprocessed including reprojection to common coordinate system, clipping to watershed boundaries, and validation of attribute tables.

### Step 3: Polygon Overlay Operations
**EXTRACTED FROM**: BasinMaker lines 892-1136 (Union_Ply_Layers_And_Simplify)

Multiple polygon layers are overlaid using iterative union operations creating unique combinations of land use, soil, and vegetation characteristics within each subbasin.

### Step 4: Terrain Attribute Calculation
**EXTRACTED FROM**: BasinMaker lines 1547-1594 (terrain statistics)

Slope, aspect, and elevation statistics are calculated for each HRU using DEM data through zonal statistics operations similar to BasinMaker's raster analysis.

### Step 5: Attribute Assignment and Classification
**EXTRACTED FROM**: BasinMaker lines 1315-1326 (class assignment)

Land use, soil, and vegetation classes are assigned to HRUs using lookup tables, and HRU attributes are finalized following BasinMaker's attribute structure.

## Default Lookup Tables

### Land Use Classification (BasinMaker Standard)
- ID 1: FOREST - Natural forested areas
- ID 2: AGRICULTURE - Cultivated and managed agricultural lands
- ID 3: URBAN - Developed and built-up areas
- ID 4: GRASSLAND - Natural and managed grasslands
- ID 5: WETLAND - Wetland and marsh areas
- ID -1: LAKE - Open water bodies

### Soil Classification (BasinMaker Standard)
- ID 1: CLAY - Clay-dominated soil profiles
- ID 2: LOAM - Loam and mixed soil textures
- ID 3: SAND - Sandy soil profiles
- ID 4: ROCK - Rock and shallow soil over bedrock
- ID -1: LAKE - Lake bottom/open water

### Vegetation Classification (BasinMaker Standard)
- ID 1: CONIFEROUS - Needleleaf forest vegetation
- ID 2: DECIDUOUS - Broadleaf forest vegetation
- ID 3: MIXED_FOREST - Mixed forest types
- ID 4: GRASSLAND - Grass and herbaceous vegetation
- ID 5: CROP - Agricultural crop vegetation
- ID -1: LAKE - Open water (no vegetation)

## Output Structure

### HRU Generation Results Dictionary
```
{
    'success': True,
    'hru_shapefile': 'path/to/finalcat_hru_info.shp',
    'hru_attributes': pandas.DataFrame,
    'total_hrus': 245,
    'lake_hrus_count': 15,
    'land_hrus_count': 230,
    'total_area_km2': 156.7,
    'hru_summary': {
        'landuse_classes': 5,
        'soil_classes': 4,
        'vegetation_classes': 5,
        'subbasins_with_hrus': 42
    }
}
```

### HRU Attribute Columns (BasinMaker Format)
- `HRU_ID`: Unique HRU identifier
- `SubId`: Parent subbasin ID
- `HyLakeId`: Associated lake ID (0 if no lake)
- `HRULake_ID`: HRU-lake relationship ID
- `HRU_IsLake`: Lake flag (1 for lake HRUs, 0 for land)
- `Landuse_ID`: Land use class identifier
- `Soil_ID`: Soil class identifier
- `Veg_ID`: Vegetation class identifier
- `LAND_USE_C`: Land use class name
- `SOIL_PROF`: Soil profile class name
- `VEG_C`: Vegetation class name
- `HRU_Area`: HRU area in square meters
- `HRU_CenX`: HRU centroid X coordinate (longitude)
- `HRU_CenY`: HRU centroid Y coordinate (latitude)
- `HRU_S_mean`: Mean slope in degrees
- `HRU_A_mean`: Mean aspect in degrees
- `HRU_E_mean`: Mean elevation in meters

## Integration with RAVEN Workflow

### Step 1: Prepare Input Data
Ensure watershed analysis is complete and lake integration (if applicable) is finished before HRU generation.

### Step 2: Generate HRUs
Run HRU generation with appropriate land surface data layers and lookup tables.

### Step 3: Validate Results
Review HRU generation statistics and validate attribute assignments for reasonableness.

### Step 4: Export for RAVEN
Use generated HRU shapefile and attributes for RAVEN model input file creation.

## BasinMaker Compatibility

### Original BasinMaker Function Mapping
- **Function**: `GenerateHRUS_qgis()`
- **File**: `basinmaker/postprocessing/hru.py`
- **Lines**: 12-581
- **Logic**: Exact replication of HRU generation workflow
- **Output Format**: Compatible HRU shapefile and attribute structure

### Key Differences from BasinMaker
- **Infrastructure**: Uses your geopandas/rasterio instead of QGIS/GRASS
- **Data Input**: Integrates with ProfessionalWatershedAnalyzer results
- **Performance**: Optimized polygon operations and memory management
- **Validation**: Enhanced quality control and error handling

## HRU Generation Process

### Land HRU Creation
Land HRUs are created through overlay of subbasins with land surface layers. Each unique combination of land use, soil, and vegetation within a subbasin becomes a separate HRU.

### Lake HRU Creation
Lake HRUs are created directly from subbasins containing lakes. They receive special attributes appropriate for water body modeling including lake-specific land use, soil, and vegetation codes.

### Attribute Calculation
Each HRU receives calculated attributes including area, centroid coordinates, and terrain characteristics derived from DEM analysis.

### Classification Assignment
Land use, soil, and vegetation class names are assigned using lookup tables providing human-readable classifications for model parameterization.

## Quality Control and Validation

### Built-in Validation
Comprehensive validation of HRU generation results including geometry checks, attribute validation, and statistical summaries.

### Quality Checks
- HRU geometry validity (no invalid polygons)
- Area calculation accuracy
- Attribute assignment completeness
- Classification lookup success
- Terrain attribute reasonableness
- HRU size distribution analysis

### Statistical Validation
- HRU count and area summaries
- Land use/soil/vegetation class distribution
- Terrain attribute statistics
- Subbasin HRU coverage analysis

## Performance Optimization

### Polygon Operations
- Efficient overlay algorithms for multiple layers
- Spatial indexing for improved performance
- Memory-conscious processing for large datasets

### Raster Processing
- Optimized DEM analysis for terrain attributes
- Windowed reading for large raster datasets
- Efficient zonal statistics calculations

### Memory Management
- Incremental processing for large watersheds
- Automatic cleanup of intermediate results
- Chunked operations for memory efficiency

## Error Handling and Robustness

### Missing Data Handling
- Graceful handling of missing land surface layers
- Default attribute assignment when data unavailable
- Robust processing with incomplete datasets

### Geometry Issues
- Automatic fixing of invalid polygons
- Handling of complex overlay results
- Management of very small HRU polygons

### Attribute Validation
- Verification of lookup table completeness
- Handling of unmapped classification codes
- Quality assurance for terrain attributes

## HRU Simplification

### Size-Based Filtering
Small HRUs below minimum area thresholds are removed to prevent model complexity issues while maintaining hydrological representation.

### Attribute-Based Merging
HRUs with identical characteristics within subbasins can be merged to reduce model complexity without losing hydrological detail.

### Quality-Based Removal
HRUs with poor geometry or unreliable attributes can be filtered out to improve overall dataset quality.

## Integration Examples

### Basic HRU Generation
Initialize HRUAttributesCalculator, load watershed results, run HRU generation with default parameters, and access results.

### Advanced Generation with Custom Data
Use custom land use, soil, and vegetation shapefiles with custom lookup tables for specialized watershed characteristics.

### Lake Integration Workflow
Combine with lake integration results to properly handle watersheds containing significant water bodies.

## Related Processors

- **basic_attributes.py**: Provides subbasin context for HRU generation
- **lake_integrator.py**: Provides lake-integrated catchments for HRU processing
- **polygon_overlay.py**: Provides overlay operations for HRU creation
- **subbasin_grouper.py**: May use HRU information for grouping decisions

## Output Files Created

### HRU Shapefile
Primary output containing all HRU polygons with complete attribute tables following BasinMaker naming conventions.

### HRU Summary Report
Statistical summary of HRU generation including class distributions and area summaries.

### Validation Report
Quality control results and validation statistics for generated HRUs.

## Troubleshooting

### Common Issues
1. **Overlay Failures**: Check land surface layer geometry and coordinate systems
2. **Missing Attributes**: Verify lookup table completeness and attribute mapping
3. **Very Small HRUs**: Adjust minimum area thresholds or input data resolution
4. **Memory Issues**: Use chunked processing for very large watersheds

### Debug Mode
Enable detailed logging to track HRU generation steps, overlay operations, and attribute assignments.

### Validation Tools
Use built-in validation methods to assess HRU quality and identify potential issues.

## Quality Assurance

### Geometric Validation
- HRU polygon validity checks
- Area calculation verification
- Centroid placement accuracy
- Coordinate system consistency

### Attribute Validation
- Classification assignment verification
- Terrain attribute reasonableness
- Lookup table mapping success
- Statistical distribution analysis

### Hydrological Validation
- HRU representativeness assessment
- Subbasin coverage completeness
- Lake HRU special handling verification
- Overall watershed representation quality

## References

- **BasinMaker Source**: `basinmaker/postprocessing/hru.py`
- **HRU Methodology**: BasinMaker HRU generation documentation
- **Land Surface Data**: Standard classifications for hydrological modeling
- **RAVEN Integration**: HRU requirements for RAVEN hydrological models