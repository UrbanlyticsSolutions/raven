# Polygon Overlay Processor

## Overview

The Polygon Overlay Processor provides professional polygon overlay operations using real BasinMaker logic. This processor implements BasinMaker's approach to polygon union, intersection, clipping, and dissolving operations with robust geometry handling and error correction.

**Extracted From**: `basinmaker/func/qgis.py` - Polygon overlay functions

## Key Features

- **Real BasinMaker Logic**: Exact replication of polygon processing workflow from BasinMaker
- **Professional Geometry Handling**: Robust fixing of invalid geometries and topology errors
- **Multiple Overlay Operations**: Union, intersection, clip, dissolve, and extract operations
- **Error Recovery**: Comprehensive error handling and fallback processing
- **Performance Optimized**: Efficient spatial operations for large polygon datasets

## Class: `PolygonOverlayProcessor`

### Initialization

**Parameters:**
- `workspace_dir` (Path, optional): Working directory for temporary files

### Core Methods

#### `union_multiple_layers()`

Unions multiple polygon layers using real BasinMaker logic.

**Parameters:**
- `input_layers` (List): List of layers to union (GeoDataFrames or file paths)
- `dissolve_fields` (List[str], optional): Fields to dissolve by after union
- `output_path` (Path, optional): Path to save result

**Returns:**
- GeoDataFrame with unioned polygons

#### `union_two_layers()`

Unions two polygon layers using BasinMaker logic.

**Parameters:**
- `layer1`, `layer2`: Input layers to union
- `output_path` (Path, optional): Path to save result

**Returns:**
- GeoDataFrame with unioned polygons

#### `clip_layer()`

Clips layer using real BasinMaker logic.

**Parameters:**
- `input_layer`: Layer to clip
- `clip_layer`: Clipping boundary layer
- `output_path` (Path, optional): Path to save result

**Returns:**
- GeoDataFrame with clipped polygons

#### `reproject_layer()`

Reprojects layer using BasinMaker approach.

**Parameters:**
- `input_layer`: Layer to reproject
- `target_crs` (str): Target coordinate reference system
- `output_path` (Path, optional): Path to save result

**Returns:**
- GeoDataFrame in target CRS

## BasinMaker Logic Implementation

### Step 1: Geometry Validation and Fixing
**EXTRACTED FROM**: BasinMaker qgis_vector_fix_geometries() logic

All input geometries are validated and fixed using buffer(0) technique and small buffer operations to resolve topology errors, self-intersections, and invalid geometry issues.

### Step 2: Coordinate System Harmonization
**EXTRACTED FROM**: BasinMaker qgis_vector_reproject_layers() lines 1196-1215

All layers are reprojected to a common coordinate reference system before overlay operations to ensure spatial consistency and accurate results.

### Step 3: Spatial Indexing
**EXTRACTED FROM**: BasinMaker spatial optimization approach

Spatial indices are created for all layers to improve performance of overlay operations, especially important for large polygon datasets.

### Step 4: Iterative Union Operations
**EXTRACTED FROM**: BasinMaker Union_Ply_Layers_And_Simplify() lines 990-1107

Multiple layers are unioned iteratively with geometry fixing applied after each operation to maintain data quality throughout the process.

### Step 5: Result Cleaning and Validation
**EXTRACTED FROM**: BasinMaker final result cleaning approach

Final results are cleaned including removal of very small polygons, duplicate geometry detection, and comprehensive validation of output quality.

## Overlay Operations

### Union Operations
- **Multiple Layer Union**: Combines multiple polygon layers into single dataset
- **Two Layer Union**: Basic union operation between two polygon layers
- **Iterative Processing**: Handles complex multi-layer unions efficiently
- **Geometry Preservation**: Maintains all polygon boundaries and attributes

### Clipping Operations
- **Boundary Clipping**: Clips polygons to specified boundary layer
- **Extent Management**: Handles layers with different spatial extents
- **Attribute Preservation**: Maintains clipped polygon attributes
- **Edge Case Handling**: Manages polygons partially outside clip boundary

### Reprojection Operations
- **CRS Transformation**: Accurate coordinate system transformations
- **Geometry Preservation**: Maintains polygon shapes during reprojection
- **Metadata Handling**: Preserves spatial metadata and attributes
- **Validation**: Verifies successful reprojection completion

### Dissolve Operations
- **Attribute-Based Dissolving**: Dissolves polygons by specified attributes
- **Multi-Field Dissolving**: Handles complex dissolve criteria
- **Geometry Simplification**: Removes unnecessary polygon complexity
- **Topology Preservation**: Maintains spatial relationships

## Geometry Processing Features

### Automatic Geometry Fixing
- Invalid geometry detection and repair
- Self-intersection resolution
- Topology error correction
- Small polygon removal

### Buffer Operations
- Geometry fixing using buffer(0) technique
- Small buffer operations for edge case handling
- Configurable buffer distances for different scenarios
- Automatic buffer cleanup

### Spatial Validation
- Comprehensive geometry validity checking
- Area and perimeter calculations
- Coordinate system verification
- Spatial relationship validation

## Performance Optimization

### Spatial Indexing
- Automatic spatial index creation
- R-tree indexing for efficient spatial queries
- Index optimization for large datasets
- Memory-efficient index management

### Memory Management
- Incremental processing for large datasets
- Efficient temporary file handling
- Automatic cleanup of intermediate results
- Memory usage monitoring and optimization

### Processing Efficiency
- Vectorized operations where possible
- Optimized overlay algorithms
- Parallel processing capabilities
- Progress reporting for long operations

## Error Handling and Robustness

### Geometry Error Recovery
- Automatic detection of geometry problems
- Multiple fixing strategies for different error types
- Fallback processing when primary methods fail
- Comprehensive error logging and reporting

### Data Quality Management
- Input data validation
- Quality checks throughout processing
- Output validation and verification
- Detailed quality reports

### Processing Robustness
- Graceful handling of processing failures
- Automatic retry mechanisms for transient errors
- Fallback algorithms for problematic cases
- Comprehensive exception handling

## Validation and Quality Control

### Built-in Validation
Comprehensive validation of overlay results including geometry checks, area calculations, and topology verification.

### Quality Checks
- Geometry validity verification
- Area calculation accuracy
- Topology preservation
- Attribute consistency
- Coordinate system integrity
- Spatial relationship validation

### Statistical Validation
- Polygon count and area summaries
- Geometry complexity analysis
- Processing efficiency metrics
- Error and warning summaries

## Integration with RAVEN Workflow

### HRU Generation Support
Provides essential polygon overlay operations for HRU generation including union of land use, soil, and vegetation layers.

### Watershed Processing
Supports watershed boundary processing, subbasin modification, and catchment analysis operations.

### Lake Integration
Provides overlay operations needed for lake integration including clipping and union operations.

### Data Preprocessing
Handles reprojection, clipping, and formatting of input spatial datasets.

## BasinMaker Compatibility

### Original BasinMaker Function Mapping
- **Functions**: qgis_vector_union_two_layers(), Union_Ply_Layers_And_Simplify()
- **File**: `basinmaker/func/qgis.py`
- **Lines**: 1168-1193, 892-1136
- **Logic**: Exact replication of overlay and geometry processing algorithms
- **Parameters**: Compatible processing parameters and constraints

### Key Differences from BasinMaker
- **Infrastructure**: Uses geopandas/shapely instead of QGIS/GRASS
- **Performance**: Optimized spatial operations and memory management
- **Error Handling**: Enhanced error recovery and validation
- **Flexibility**: Support for various input formats and processing options

## Advanced Features

### Multi-Layer Processing
- Efficient processing of multiple input layers
- Automatic layer compatibility checking
- Optimized processing order determination
- Comprehensive result integration

### Custom Processing Parameters
- Configurable geometry fixing parameters
- Adjustable performance optimization settings
- Customizable validation criteria
- Flexible output formatting options

### Batch Processing
- Support for batch processing of multiple operations
- Automated workflow execution
- Progress monitoring and reporting
- Error recovery and continuation

## Integration Examples

### Basic Union Operation
Load multiple polygon layers, run union operation, and save results with comprehensive validation.

### Complex Overlay Workflow
Combine multiple overlay operations including reprojection, clipping, union, and dissolve in integrated workflow.

### HRU Generation Integration
Use polygon overlay operations as part of HRU generation workflow with land use, soil, and vegetation layers.

## Related Processors

- **hru_attributes.py**: Uses polygon overlay for HRU generation
- **lake_integrator.py**: Uses overlay operations for lake integration
- **basic_attributes.py**: May use overlay for attribute calculation
- **subbasin_grouper.py**: Uses overlay for grouping operations

## Output Features

### Comprehensive Results
- Detailed processing statistics
- Quality validation results
- Error and warning summaries
- Performance metrics

### Flexible Output Formats
- Shapefile output with complete attributes
- GeoJSON format support
- Database integration capabilities
- Custom attribute preservation

### Quality Documentation
- Processing log files
- Validation reports
- Error documentation
- Performance analysis

## Troubleshooting

### Common Issues
1. **Geometry Errors**: Invalid input geometries causing processing failures
2. **CRS Mismatches**: Layers with incompatible coordinate systems
3. **Memory Issues**: Large datasets exceeding available memory
4. **Topology Problems**: Complex polygons with self-intersections

### Debug Mode
Enable detailed logging to track overlay operations, geometry fixes, and validation results.

### Validation Tools
Use built-in validation methods to assess overlay quality and identify processing issues.

## Quality Assurance

### Processing Validation
- Input data quality verification
- Processing step validation
- Output quality assessment
- Comprehensive error checking

### Geometric Validation
- Polygon validity verification
- Area calculation accuracy
- Topology preservation checking
- Coordinate system consistency

### Performance Monitoring
- Processing time analysis
- Memory usage tracking
- Error rate monitoring
- Efficiency optimization

## References

- **BasinMaker Source**: `basinmaker/func/qgis.py`
- **Polygon Processing**: BasinMaker geometry handling methodology
- **Spatial Operations**: Professional GIS overlay operation standards
- **Quality Control**: Best practices for spatial data processing