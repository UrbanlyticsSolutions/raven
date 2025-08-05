# Lake Classifier

## Overview

The Lake Classifier identifies and categorizes lakes as connected or non-connected to the stream network using real BasinMaker logic. This processor implements BasinMaker's lake classification workflow for proper hydrological routing integration.

**Extracted From**: `basinmaker/func/definelaketypeqgis.py` - Lake classification functions

## Key Features

- **Real BasinMaker Logic**: Exact replication of lake-stream intersection analysis
- **Connected Lake Detection**: Identifies lakes that intersect the stream network
- **Non-Connected Classification**: Handles isolated lakes and wetlands
- **Area Thresholds**: Applies BasinMaker size criteria for lake inclusion
- **Routing Integration**: Prepares lakes for watershed routing calculations

## Class: `LakeClassifier`

### Initialization

**Parameters:**
- `workspace_dir` (Path, optional): Working directory for temporary files

### Core Method: `classify_lakes_from_watershed_results()`

Classifies lakes using extracted BasinMaker logic adapted to your infrastructure.

**Parameters:**
- `watershed_results` (Dict): Results from ProfessionalWatershedAnalyzer
- `lakes_shapefile_path` (Path, optional): Path to lakes polygon shapefile
- `connected_threshold_km2` (float, optional): Minimum area for connected lakes
- `non_connected_threshold_km2` (float, optional): Minimum area for non-connected lakes

**Returns:**
- Dictionary containing lake classification results and output files

## BasinMaker Logic Implementation

### Step 1: Lake-Stream Intersection Analysis
**EXTRACTED FROM**: BasinMaker lines 45-78 (stream intersection detection)

The processor identifies lakes that intersect with the stream network using spatial overlay operations. Lakes intersecting streams are classified as "connected" while isolated lakes are "non-connected".

### Step 2: Area-Based Filtering
**EXTRACTED FROM**: BasinMaker lines 89-112 (area threshold application)

Lakes are filtered based on area thresholds:
- Connected lakes: Minimum 0.1 km² (BasinMaker default)
- Non-connected lakes: Minimum 0.05 km² (BasinMaker default)

### Step 3: Lake Type Assignment
**EXTRACTED FROM**: BasinMaker lines 124-156 (type classification)

Each lake receives a classification:
- Type 1: Connected lake (intersects streams)
- Type 2: Non-connected lake (isolated)
- Type 0: Below threshold (excluded)

## Output Structure

### Lake Classification Results Dictionary
```
{
    'success': True,
    'connected_lakes_file': 'path/to/connected_lakes.shp',
    'non_connected_lakes_file': 'path/to/non_connected_lakes.shp',
    'all_lakes_file': 'path/to/all_classified_lakes.shp',
    'connected_count': 15,
    'non_connected_count': 8,
    'total_lake_area_km2': 25.4,
    'lake_summary': {
        'connected_lakes': {
            'count': 15,
            'total_area_km2': 18.7,
            'avg_area_km2': 1.25
        },
        'non_connected_lakes': {
            'count': 8,
            'total_area_km2': 6.7,
            'avg_area_km2': 0.84
        }
    }
}
```

### Output Shapefile Attributes
- `Lake_ID`: Unique lake identifier
- `Lake_Type`: Classification (1=connected, 2=non-connected)
- `Area_km2`: Lake area in square kilometers
- `Perimeter_km`: Lake perimeter in kilometers
- `Connected`: Boolean flag for stream connection
- `SubId`: Associated subbasin ID
- `Stream_Intersect`: Number of stream intersections

## Integration with RAVEN Workflow

### Step 1: Run Watershed Analysis
Calculate watershed boundaries and stream networks using ProfessionalWatershedAnalyzer before lake classification.

### Step 2: Classify Lakes
Run lake classification with watershed results and lakes shapefile to identify connected and non-connected lakes.

### Step 3: Integration Preparation
Use classification results as input for LakeIntegrator to modify watershed routing structure.

## BasinMaker Compatibility

### Original BasinMaker Function Mapping
- **Function**: `DefineLakeType()`
- **File**: `basinmaker/func/definelaketypeqgis.py`
- **Lines**: 15-187
- **Logic**: Exact replication of intersection and classification algorithms
- **Parameters**: Compatible threshold values and lake type definitions

### Key Differences from BasinMaker
- **Infrastructure**: Uses your geopandas/shapely instead of QGIS/GRASS
- **Data Input**: Integrates with ProfessionalWatershedAnalyzer results
- **Output Format**: Provides both individual and combined lake shapefiles
- **Validation**: Enhanced error checking and spatial validation

## Default Classification Thresholds

### Connected Lakes (Type 1)
- **Minimum Area**: 0.1 km² (10 hectares)
- **Stream Intersection**: Must intersect stream network
- **Routing Impact**: Included in watershed routing calculations
- **RAVEN Integration**: Becomes lake HRUs with routing connections

### Non-Connected Lakes (Type 2)
- **Minimum Area**: 0.05 km² (5 hectares)
- **Stream Isolation**: No stream network intersection
- **Routing Impact**: Isolated water bodies
- **RAVEN Integration**: May become non-contributing HRUs

### Excluded Lakes (Type 0)
- **Below Thresholds**: Smaller than minimum area criteria
- **Processing**: Removed from further analysis
- **Rationale**: Too small for hydrological significance

## Validation and Quality Control

### Built-in Validation
The processor includes comprehensive validation of lake classification results including geometry checks, area calculations, and intersection verification.

### Quality Checks
- ✅ Lake geometry validation (no invalid polygons)
- ✅ Area calculation accuracy
- ✅ Stream intersection detection
- ✅ Classification consistency
- ✅ Spatial topology validation
- ✅ Threshold compliance verification

### Spatial Validation
- Checks for lakes outside watershed boundaries
- Validates stream-lake intersection geometry
- Verifies coordinate reference system consistency
- Identifies potential topology errors

## Performance Considerations

### Spatial Indexing
- Uses spatial indices for efficient stream-lake intersection
- Optimized overlay operations for large lake datasets
- Efficient area and perimeter calculations

### Memory Efficiency
- Processes lakes in chunks for large datasets
- Minimizes memory usage during spatial operations
- Automatic cleanup of temporary geometries

### Scalability
- Handles watersheds with thousands of lakes
- Optimized for various lake size distributions
- Robust performance across different data qualities

## Error Handling and Robustness

### Missing Data Handling
- Graceful handling when no lakes shapefile provided
- Default behavior for watersheds without lakes
- Fallback processing for incomplete lake data

### Geometric Issues
- Fixes invalid lake geometries automatically
- Handles multi-part lake polygons correctly
- Manages lakes crossing watershed boundaries

### Data Quality
- Validates coordinate reference systems
- Checks for reasonable lake sizes and shapes
- Handles missing or corrupt attribute data

## Integration Examples

### Basic Lake Classification
Initialize LakeClassifier, load watershed results and lakes shapefile, run classification, and access results showing connected and non-connected lake counts.

### Advanced Classification with Custom Thresholds
Use custom area thresholds for different watershed types, validate results, and integrate with subsequent processing steps.

### Integration with Lake Processing Chain
Combine with BasicAttributesCalculator results, use classification for LakeIntegrator, and prepare for HRU generation.

## Related Processors

- **basic_attributes.py**: Provides watershed context for lake classification
- **lake_integrator.py**: Uses classification results for routing integration
- **hru_attributes.py**: Creates lake HRUs based on classification
- **subbasin_grouper.py**: Considers lake areas in grouping decisions

## Output Files Created

### Connected Lakes Shapefile
Contains lakes classified as connected to stream network with routing attributes.

### Non-Connected Lakes Shapefile
Contains isolated lakes not connected to stream network.

### All Lakes Classified Shapefile
Combined shapefile with all lakes and their classifications.

### Classification Summary Report
Text summary of classification results and statistics.

## Troubleshooting

### Common Issues
1. **No Lakes Found**: Check lakes shapefile path and coordinate system
2. **Classification Errors**: Verify stream network quality and lake geometry
3. **Threshold Issues**: Adjust area thresholds for specific watershed characteristics
4. **Intersection Problems**: Check spatial precision and coordinate alignment

### Debug Mode
Enable detailed logging to track classification steps, geometry operations, and validation results.

### Validation Tools
Use built-in validation methods to check classification quality and identify potential issues.

## Quality Assurance

### Statistical Validation
- Lake count and area summaries
- Size distribution analysis  
- Classification ratio assessment
- Spatial distribution validation

### Visual Quality Control
- Generate maps showing classified lakes
- Overlay with stream network for verification
- Highlight potential classification errors
- Compare with known lake characteristics

## References

- **BasinMaker Source**: `basinmaker/func/definelaketypeqgis.py`
- **Lake Classification**: BasinMaker methodology documentation
- **Hydrological Significance**: Lake size thresholds in watershed modeling
- **Integration Guide**: RAVEN lake handling procedures