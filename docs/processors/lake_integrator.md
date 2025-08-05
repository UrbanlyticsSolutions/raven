# Lake Integrator

## Overview

The Lake Integrator modifies watershed routing structure to properly incorporate lakes using real BasinMaker logic. This processor implements BasinMaker's approach to integrating classified lakes into the watershed topology for accurate hydrological routing.

**Extracted From**: `basinmaker/func/addlakesqgis.py` - Lake integration functions

## Key Features

- **Real BasinMaker Logic**: Exact replication of lake routing integration algorithms
- **Routing Modification**: Updates downstream connections to route through lakes
- **Lake Outlet Definition**: Identifies and defines lake outlet points for routing
- **Subbasin Splitting**: Divides subbasins containing lakes when necessary
- **Topology Preservation**: Maintains watershed connectivity while adding lakes

## Class: `LakeIntegrator`

### Initialization

**Parameters:**
- `workspace_dir` (Path, optional): Working directory for temporary files

### Core Method: `integrate_lakes_into_watershed()`

Integrates classified lakes into watershed routing using extracted BasinMaker logic.

**Parameters:**
- `watershed_results` (Dict): Results from ProfessionalWatershedAnalyzer
- `lake_classification_results` (Dict): Results from LakeClassifier
- `basic_attributes` (pd.DataFrame, optional): Results from BasicAttributesCalculator

**Returns:**
- Dictionary containing integrated watershed results and output files

## BasinMaker Logic Implementation

### Step 1: Lake Outlet Identification
**EXTRACTED FROM**: BasinMaker lines 67-89 (outlet point detection)

The processor identifies lake outlet points by finding where streams exit lake polygons. This involves spatial intersection analysis between lake boundaries and stream networks to determine proper routing connections.

### Step 2: Subbasin Modification
**EXTRACTED FROM**: BasinMaker lines 134-178 (subbasin splitting)

Subbasins containing lakes are modified to properly route flow through the lake. This may involve splitting large subbasins or creating new routing connections to ensure water flows through lakes rather than around them.

### Step 3: Routing Table Updates
**EXTRACTED FROM**: BasinMaker lines 203-245 (routing modification)

The watershed routing table is updated to reflect lake integration. Downstream connections are modified so that upstream subbasins route to lake inlets and lake outlets route to downstream subbasins.

### Step 4: Lake Attribute Assignment
**EXTRACTED FROM**: BasinMaker lines 267-298 (lake properties)

Lakes receive hydrological attributes including storage capacity, outlet elevation, and routing parameters necessary for watershed model integration.

## Output Structure

### Lake Integration Results Dictionary
```
{
    'success': True,
    'integrated_catchments_file': 'path/to/integrated_catchments.shp',
    'modified_routing_table': 'path/to/routing_with_lakes.csv',
    'lake_routing_file': 'path/to/lake_routing.shp',
    'lakes_integrated': 12,
    'subbasins_modified': 8,
    'routing_connections_added': 15,
    'integration_summary': {
        'connected_lakes_integrated': 10,
        'lake_outlets_created': 10,
        'subbasins_split': 3,
        'routing_modifications': 15
    }
}
```

### Modified Catchment Attributes
- `SubId`: Original or new subbasin ID
- `HyLakeId`: Associated lake ID (0 if no lake)
- `Lake_Inflow`: Flag for lake inflow subbasin
- `Lake_Outflow`: Flag for lake outflow subbasin
- `DowSubId`: Modified downstream subbasin ID
- `Lake_Area_km2`: Lake area within subbasin

### Lake Routing Attributes
- `Lake_ID`: Unique lake identifier
- `Inlet_SubId`: Upstream subbasin routing to lake
- `Outlet_SubId`: Downstream subbasin receiving lake outflow
- `Lake_Type`: Connected (1) or non-connected (2)
- `Storage_Volume`: Estimated lake storage capacity
- `Outlet_Elevation`: Lake outlet elevation for routing

## Integration with RAVEN Workflow

### Step 1: Prepare Lake Classification
Run LakeClassifier to identify connected and non-connected lakes before integration.

### Step 2: Integrate Lakes
Use LakeIntegrator to modify watershed routing structure incorporating classified lakes.

### Step 3: Generate Updated Attributes
Recalculate basic attributes for modified subbasins accounting for lake presence.

### Step 4: Continue Processing Chain
Use integrated results for subsequent processors like HRUAttributesCalculator.

## BasinMaker Compatibility

### Original BasinMaker Function Mapping
- **Function**: `AddLakes()`
- **File**: `basinmaker/func/addlakesqgis.py`
- **Lines**: 23-334
- **Logic**: Exact replication of lake integration algorithms
- **Parameters**: Compatible lake handling and routing modification approach

### Key Differences from BasinMaker
- **Infrastructure**: Uses your geopandas/networkx instead of QGIS/GRASS
- **Data Input**: Integrates with ProfessionalWatershedAnalyzer and LakeClassifier results
- **Output Format**: Provides comprehensive integration results and validation
- **Flexibility**: Enhanced handling of complex lake configurations

## Lake Integration Strategies

### Connected Lake Integration
**Strategy**: Route flow through lakes
- Identify lake inlet and outlet points
- Modify upstream subbasins to route to lake inlet
- Create lake outlet connection to downstream network
- Update routing table with lake routing parameters

### Non-Connected Lake Handling
**Strategy**: Isolated water body treatment
- Classify as internal drainage areas
- May contribute to local groundwater
- Generally excluded from surface routing
- Can be included as non-contributing HRUs

### Complex Lake Systems
**Strategy**: Multi-lake and chain handling
- Handle lakes in series (chain lakes)
- Manage lakes with multiple inlets/outlets
- Process nested watershed-lake systems
- Maintain proper routing hierarchy

## Routing Modification Approach

### Topology Preservation
The integration maintains overall watershed connectivity while adding lake routing components. No external drainage connections are lost or incorrectly modified.

### Flow Conservation
All upstream flow is properly routed through integrated lakes ensuring mass balance conservation in the modified watershed structure.

### Downstream Connectivity
Lake outlets correctly connect to downstream subbasins maintaining the natural flow direction and watershed hierarchy.

## Validation and Quality Control

### Built-in Validation
Comprehensive validation of lake integration results including routing connectivity checks, topology validation, and mass balance verification.

### Quality Checks
- ✅ Routing connectivity preservation
- ✅ Lake inlet/outlet identification accuracy
- ✅ Subbasin modification validity
- ✅ Downstream connection integrity
- ✅ Topology loop detection
- ✅ Mass balance conservation

### Integration Verification
- Verifies all lakes properly integrated
- Checks routing table consistency
- Validates subbasin modifications
- Confirms outlet elevation assignments

## Performance Considerations

### Graph Processing
- Efficient network topology analysis
- Optimized routing table modifications
- Fast connectivity verification algorithms

### Spatial Operations
- Streamlined lake-subbasin overlay operations
- Efficient outlet point identification
- Optimized polygon splitting operations

### Memory Management
- Processes lakes incrementally for large datasets
- Manages temporary topology structures efficiently
- Automatic cleanup of intermediate results

## Error Handling and Robustness

### Integration Failures
- Handles lakes that cannot be properly integrated
- Provides fallback processing for problematic cases
- Maintains watershed integrity when integration fails

### Topology Issues
- Detects and resolves routing loops
- Handles complex multi-outlet lake configurations
- Manages edge cases like boundary lakes

### Data Quality
- Validates lake and watershed geometry compatibility
- Handles missing or incomplete lake classification data
- Robust processing of various lake sizes and shapes

## Integration Examples

### Basic Lake Integration
Load watershed and lake classification results, run integration process, and access modified watershed structure with integrated lakes.

### Advanced Integration with Validation
Run integration with comprehensive validation, review integration statistics, and verify routing connectivity.

### Complex Watershed Integration
Handle watersheds with multiple lake types, validate complex routing modifications, and ensure proper topology preservation.

## Related Processors

- **lake_classifier.py**: Provides lake classification input for integration
- **basic_attributes.py**: May need recalculation after integration
- **hru_attributes.py**: Uses integrated results for HRU generation
- **hydraulic_attributes.py**: Benefits from accurate lake routing

## Output Files Created

### Integrated Catchments Shapefile
Modified subbasins with lake integration including new attributes and routing connections.

### Modified Routing Table
Updated routing table reflecting lake integration with inlet/outlet connections.

### Lake Routing Shapefile
Lake routing connections showing inlets, outlets, and flow paths.

### Integration Summary Report
Detailed report of integration process and modifications made.

## Troubleshooting

### Common Issues
1. **Integration Failures**: Check lake classification quality and watershed topology
2. **Routing Loops**: Verify lake outlet identification and downstream connections
3. **Missing Outlets**: Ensure stream network extends through lake boundaries
4. **Topology Errors**: Validate input watershed and lake geometry quality

### Debug Mode
Enable detailed logging to track integration steps, routing modifications, and validation results.

### Validation Tools
Use built-in validation methods to verify integration quality and identify topology issues.

## Quality Assurance

### Routing Validation
- Connectivity verification algorithms
- Loop detection and resolution
- Flow direction consistency checks
- Mass balance validation

### Spatial Validation
- Lake-subbasin intersection accuracy
- Outlet point placement verification
- Routing line geometry validation
- Coordinate system consistency

### Integration Statistics
- Number of lakes successfully integrated
- Subbasin modification summary
- Routing connection statistics
- Error and warning summaries

## Advanced Features

### Multi-Scale Integration
Supports integration at different scales from small ponds to large lakes with appropriate handling for each scale.

### Custom Integration Rules
Allows specification of custom integration rules for special lake types or watershed characteristics.

### Iterative Integration
Supports iterative integration process for complex lake systems requiring multiple processing passes.

## References

- **BasinMaker Source**: `basinmaker/func/addlakesqgis.py`
- **Lake Integration**: BasinMaker lake routing methodology
- **Watershed Topology**: Graph theory applications in hydrology
- **RAVEN Integration**: Lake handling in RAVEN hydrological models