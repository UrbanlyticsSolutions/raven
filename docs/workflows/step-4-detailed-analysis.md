# Step 4: Detailed Watershed Analysis with Lake Detection/Integration

## Overview

Step 4 is the most complex step in the Full Delineation Workflow. It takes the high-quality clipped DEM from Step 3 and performs comprehensive watershed analysis including lake detection, classification, and integration using the **UnifiedWatershedDelineation** step.

## Why This Step is Needed

Even though we got the watershed boundary from MyHydro in Step 1, Step 4 is essential because:

1. **MyHydro provides basic geometry** - Simple watershed boundary polygon
2. **Step 4 provides detailed analysis** - Stream networks, lake integration, subbasin topology, hydraulic parameters
3. **RAVEN needs detailed data** - HRU generation requires streams, lakes, routing, and attributes

## Input Data

```python
{
    'dem_file': '/workspace/dem/workflow_dem.tif',           # High-quality 30m DEM from Step 3
    'outlet_latitude': 45.5017,                            # Original outlet coordinates
    'outlet_longitude': -73.5673,
    'stream_threshold': 1000                                # Flow accumulation threshold
}
```

## Detailed Sub-Steps

### **Sub-Step 4.1: Basic Watershed Delineation**
**Duration**: 2-4 minutes  
**Processor Used**: `ProfessionalWatershedAnalyzer` (via UnifiedWatershedDelineation)

#### **4.1.1 Hydrological Conditioning**
```python
# Fill depressions using Wang & Liu algorithm
wbt.fill_depressions_wang_and_liu(dem_file, filled_dem)

# Calculate D8 flow direction
wbt.d8_pointer(filled_dem, flow_dir)

# Calculate flow accumulation  
wbt.d8_flow_accumulation(flow_dir, flow_accum)
```

**What happens**:
- Removes sinks and depressions in DEM that would trap water flow
- Creates flow direction raster (8-directional)
- Calculates flow accumulation (how much water flows through each cell)

#### **4.1.2 Outlet Snapping and Watershed Tracing**
```python
# Snap outlet to nearest high-flow cell (within 500m radius)
snapped_outlet = snap_outlet_to_stream(outlet_coords, flow_accum, max_distance=500)

# Trace watershed boundary upstream from snapped outlet
watershed_boundary = trace_watershed_boundary(flow_dir, snapped_outlet)
```

**What happens**:
- Finds the nearest high-flow accumulation cell to the provided coordinates
- Traces all cells that drain to this outlet point
- Creates detailed watershed boundary polygon

#### **4.1.3 Stream Network Extraction**
```python
# Extract streams using flow accumulation threshold
stream_cells = flow_accum >= stream_threshold  # Default: 1000 cells

# Vectorize stream network
stream_network = vectorize_streams(stream_cells, flow_dir)

# Calculate stream order using Strahler ordering
stream_network = calculate_stream_order(stream_network)
```

**What happens**:
- Identifies cells with enough upstream flow to be considered streams
- Converts raster stream cells to vector line network
- Assigns stream orders (1st order = headwaters, higher = main stems)

**Output from Sub-Step 4.1**:
```python
{
    'watershed_boundary': '/workspace/watershed/detailed_watershed.geojson',
    'stream_network': '/workspace/watershed/detailed_streams.geojson',
    'watershed_area_km2': 245.8,
    'stream_length_km': 156.7,
    'max_stream_order': 4,
    'outlet_snapped': True,
    'snap_distance_m': 45.0
}
```

---

### **Sub-Step 4.2: Lake Detection**
**Duration**: 1-2 minutes  
**Processor Used**: `LakeDetector`

#### **4.2.1 Spatial Query for Lakes**
```python
# Get watershed bounds for spatial query
watershed_bounds = watershed_gdf.total_bounds  # [minx, miny, maxx, maxy]

# Query Canadian Hydro Database
lakes_result = self.lakes_client.get_lakes_for_watershed(
    watershed_bounds=watershed_bounds,
    watershed_geometry=watershed_gdf.geometry.iloc[0]
)
```

**What happens**:
- Queries the Canadian Hydro Database (HydroLAKES) for all lakes within watershed bounds
- Performs spatial intersection to get only lakes inside the watershed boundary
- Filters out very small water bodies (< 0.01 km²)

#### **4.2.2 Lake Geometry Validation**
```python
# Fix invalid geometries
lakes_gdf = lakes_gdf[lakes_gdf.geometry.is_valid]

# Remove multi-part lakes that cross watershed boundaries
lakes_gdf = fix_multipart_geometries(lakes_gdf, watershed_boundary)

# Calculate accurate lake areas in km²
lakes_gdf['area_km2'] = calculate_accurate_areas(lakes_gdf)
```

**What happens**:
- Validates and fixes any invalid lake polygon geometries
- Handles complex multi-part lakes that may cross boundaries
- Calculates precise areas using proper projections

**Output from Sub-Step 4.2**:
```python
{
    'lakes_detected': 8,
    'total_lake_area_km2': 12.3,
    'lakes_shapefile': '/workspace/watershed/detected_lakes.geojson',
    'largest_lake_area_km2': 4.2,
    'smallest_lake_area_km2': 0.05
}
```

---

### **Sub-Step 4.3: Lake Classification**
**Duration**: 30-60 seconds  
**Processor Used**: `LakeClassifier`

#### **4.3.1 Stream-Lake Intersection Analysis**
```python
# Create spatial index for efficient intersection testing
streams_sindex = streams_gdf.sindex

for idx, lake in lakes_gdf.iterrows():
    # Find potential stream intersections using spatial index
    possible_matches_idx = list(streams_sindex.intersection(lake.geometry.bounds))
    possible_matches = streams_gdf.iloc[possible_matches_idx]
    
    # Check for actual geometric intersections
    intersects = possible_matches.geometry.intersects(lake.geometry)
    intersecting_streams = possible_matches[intersects]
    
    if len(intersecting_streams) > 0:
        lake_type = 'connected'
        stream_intersections = len(intersecting_streams)
    else:
        lake_type = 'non_connected'  
        stream_intersections = 0
```

**What happens**:
- Uses spatial indexing for fast intersection testing (R-tree algorithm)
- Tests each lake polygon against the stream network
- Counts how many stream segments intersect each lake

#### **4.3.2 BasinMaker Threshold Application**
```python
# Apply BasinMaker classification thresholds
connected_lakes = classified_lakes[
    (classified_lakes['lake_type'] == 'connected') &
    (classified_lakes['area_km2'] >= 0.5) &        # Connected threshold: 0.5 km²
    (classified_lakes['estimated_depth_m'] >= 2.0)  # Depth threshold: 2.0 m
]

non_connected_lakes = classified_lakes[
    (classified_lakes['lake_type'] == 'non_connected') &
    (classified_lakes['area_km2'] >= 1.0) &        # Non-connected threshold: 1.0 km²
    (classified_lakes['estimated_depth_m'] >= 3.0)  # Depth threshold: 3.0 m
]
```

**What happens**:
- Applies BasinMaker's proven thresholds for hydrological significance
- Connected lakes have lower area threshold (they affect routing more)
- Non-connected lakes need larger area to be hydrologically significant

**Output from Sub-Step 4.3**:
```python
{
    'connected_lakes_file': '/workspace/watershed/connected_lakes.geojson',
    'non_connected_lakes_file': '/workspace/watershed/non_connected_lakes.geojson',
    'all_lakes_file': '/workspace/watershed/all_classified_lakes.geojson',
    'connected_count': 5,
    'non_connected_count': 3,
    'lake_classification_summary': {
        'connected_lakes': {
            'count': 5,
            'total_area_km2': 8.7,
            'avg_area_km2': 1.74,
            'stream_connections': 12
        },
        'non_connected_lakes': {
            'count': 3, 
            'total_area_km2': 3.6,
            'avg_area_km2': 1.20,
            'stream_connections': 0
        }
    }
}
```

---

### **Sub-Step 4.4: Lake Integration**
**Duration**: 1-3 minutes  
**Processor Used**: `LakeIntegrator`

#### **4.4.1 Lake Outlet Detection**
```python
for idx, lake in connected_lakes.iterrows():
    # Find streams that intersect lake boundary
    intersecting_streams = streams_gdf[streams_gdf.geometry.intersects(lake.geometry)]
    
    for stream_idx, stream in intersecting_streams.iterrows():
        # Find intersection points between stream and lake boundary
        intersection = stream.geometry.intersection(lake.geometry.boundary)
        
        # Determine flow direction to identify outlets vs inlets
        flow_direction = determine_flow_direction(stream, lake, flow_dir_raster)
        
        if flow_direction == 'outflow':
            lake_outlets.append({
                'lake_id': idx,
                'outlet_point': intersection,
                'stream_id': stream_idx,
                'outlet_elevation': get_elevation(intersection, dem_raster)
            })
```

**What happens**:
- Identifies where streams enter and exit each connected lake
- Uses flow direction raster to determine inlet vs outlet points
- Records outlet elevations for routing calculations

#### **4.4.2 Routing Network Modification**
```python
# Create new routing connections through lakes
for lake_outlet in lake_outlets:
    # Find upstream subbasins that should route to lake inlet
    upstream_subbasins = find_upstream_subbasins(lake_outlet['lake_id'], stream_network)
    
    # Modify downstream connections to route through lake
    for subbasin_id in upstream_subbasins:
        # Change: subbasin → downstream_subbasin
        # To: subbasin → lake_inlet → lake_outlet → downstream_subbasin
        modify_routing_connection(subbasin_id, lake_outlet)
    
    # Update routing table with lake parameters
    add_lake_routing_parameters(lake_outlet)
```

**What happens**:
- Modifies the watershed routing topology to flow through lakes
- Creates lake inlet and outlet routing connections
- Preserves mass balance and flow conservation

#### **4.4.3 Subbasin Modification**
```python
# Split subbasins that contain lakes
for lake in connected_lakes:
    containing_subbasins = find_containing_subbasins(lake, subbasins_gdf)
    
    for subbasin in containing_subbasins:
        if lake.area / subbasin.area > 0.1:  # Lake covers >10% of subbasin
            # Split subbasin into lake and land components
            lake_subbasin = create_lake_subbasin(lake, subbasin)
            land_subbasin = create_land_subbasin(subbasin.geometry.difference(lake.geometry))
            
            # Update routing connections
            update_subbasin_routing(lake_subbasin, land_subbasin, subbasin)
```

**What happens**:
- Identifies subbasins that contain significant lakes
- Splits these subbasins into separate lake and land components
- Updates routing connections to maintain proper flow paths

**Output from Sub-Step 4.4**:
```python
{
    'integrated_catchments_file': '/workspace/watershed/integrated_catchments.geojson',
    'modified_routing_table': '/workspace/watershed/routing_with_lakes.csv',
    'lake_routing_file': '/workspace/watershed/lake_routing.geojson',
    'lakes_integrated': 5,
    'subbasins_modified': 3,
    'routing_connections_added': 12,
    'lake_outlets_created': 8,
    'integration_summary': {
        'total_lake_inlets': 7,
        'total_lake_outlets': 8, 
        'subbasins_split': 3,
        'new_routing_connections': 12
    }
}
```

---

### **Sub-Step 4.5: Watershed Attributes Calculation**
**Duration**: 30-90 seconds  
**Processor Used**: `BasicAttributesCalculator`

#### **4.5.1 Geometric Attributes**
```python
# Calculate subbasin geometric properties
for subbasin in integrated_catchments:
    subbasin['area_km2'] = calculate_area(subbasin.geometry)
    subbasin['perimeter_km'] = calculate_perimeter(subbasin.geometry)
    subbasin['avg_elevation_m'] = calculate_average_elevation(subbasin.geometry, dem_raster)
    subbasin['avg_slope'] = calculate_average_slope(subbasin.geometry, dem_raster)
    subbasin['aspect'] = calculate_dominant_aspect(subbasin.geometry, dem_raster)
```

#### **4.5.2 Hydrological Attributes**
```python
# Calculate stream and drainage properties
for subbasin in integrated_catchments:
    # Stream density
    stream_length_in_subbasin = calculate_stream_length(subbasin, streams_gdf)
    subbasin['stream_density'] = stream_length_in_subbasin / subbasin['area_km2']
    
    # Drainage characteristics
    subbasin['time_of_concentration'] = calculate_tc(subbasin, streams_gdf, dem_raster)
    subbasin['channel_slope'] = calculate_channel_slope(subbasin, streams_gdf, dem_raster)
    
    # Lake characteristics (if present)
    if subbasin['has_lake']:
        subbasin['lake_area_pct'] = (subbasin['lake_area_km2'] / subbasin['area_km2']) * 100
        subbasin['lake_storage_m3'] = estimate_lake_storage(subbasin['lake_geometry'])
```

#### **4.5.3 Hydraulic Parameters**
```python
# Calculate channel hydraulic properties using empirical relationships
for subbasin in integrated_catchments:
    drainage_area = subbasin['area_km2']
    
    # Channel width (m) - Regional regression equation
    subbasin['channel_width_m'] = 2.3 * (drainage_area ** 0.5)
    
    # Channel depth (m) - Manning's equation application
    subbasin['channel_depth_m'] = 0.6 * (drainage_area ** 0.3)
    
    # Manning's roughness - Based on channel characteristics
    if subbasin['has_lake']:
        subbasin['mannings_n'] = 0.035  # Lake outlet channels
    elif subbasin['forest_pct'] > 70:
        subbasin['mannings_n'] = 0.050  # Forested channels
    else:
        subbasin['mannings_n'] = 0.040  # Mixed channels
```

**Output from Sub-Step 4.5**:
```python
{
    'attributes_file': '/workspace/watershed/detailed_attributes.csv',
    'summary_statistics': {
        'avg_subbasin_area_km2': 16.4,
        'total_stream_length_km': 156.7,
        'avg_stream_density': 0.64,  # km/km²
        'avg_slope': 0.045,          # 4.5%
        'avg_elevation_m': 385.2,
        'lake_coverage_pct': 5.0,
        'time_of_concentration_hours': 4.8
    },
    'hydraulic_parameters': {
        'avg_channel_width_m': 3.2,
        'avg_channel_depth_m': 0.8,
        'avg_mannings_n': 0.042
    }
}
```

---

## Complete Step 4 Output Structure

```python
{
    'success': True,
    'step_type': 'unified_watershed_delineation',
    'outlet_coordinates': (45.5017, -73.5673),
    'stream_threshold': 1000,
    'processing_time_minutes': 8.3,
    
    # Basic watershed results (Sub-step 4.1)
    'watershed_boundary': '/workspace/watershed/detailed_watershed.geojson',
    'original_stream_network': '/workspace/watershed/detailed_streams.geojson', 
    'watershed_area_km2': 245.8,
    'stream_length_km': 156.7,
    'max_stream_order': 4,
    'outlet_snapped': True,
    'snap_distance_m': 45.0,
    
    # Lake detection results (Sub-step 4.2)
    'lakes_detected_file': '/workspace/watershed/detected_lakes.geojson',
    'lakes_detected_count': 8,
    'total_lakes_area_km2': 12.3,
    
    # Lake classification results (Sub-step 4.3)
    'connected_lakes_file': '/workspace/watershed/connected_lakes.geojson',
    'non_connected_lakes_file': '/workspace/watershed/non_connected_lakes.geojson', 
    'all_lakes_file': '/workspace/watershed/all_classified_lakes.geojson',
    'connected_lake_count': 5,
    'non_connected_lake_count': 3,
    'total_lake_area_km2': 12.3,
    
    # Lake integration results (Sub-step 4.4)
    'integrated_catchments_file': '/workspace/watershed/integrated_catchments.geojson',
    'modified_routing_table': '/workspace/watershed/routing_with_lakes.csv',
    'lake_routing_file': '/workspace/watershed/lake_routing.geojson',
    'lakes_integrated': 5,
    'subbasins_modified': 3,
    'routing_connections_added': 12,
    
    # Attributes results (Sub-step 4.5)
    'attributes_file': '/workspace/watershed/detailed_attributes.csv',
    'summary_statistics': {
        'avg_subbasin_area_km2': 16.4,
        'total_stream_length_km': 156.7,
        'avg_stream_density': 0.64,
        'avg_slope': 0.045,
        'avg_elevation_m': 385.2,
        'lake_coverage_pct': 5.0,
        'time_of_concentration_hours': 4.8
    },
    
    # All files created
    'files_created': [
        '/workspace/watershed/detailed_watershed.geojson',
        '/workspace/watershed/detailed_streams.geojson',
        '/workspace/watershed/detected_lakes.geojson',
        '/workspace/watershed/connected_lakes.geojson',
        '/workspace/watershed/non_connected_lakes.geojson',
        '/workspace/watershed/all_classified_lakes.geojson',
        '/workspace/watershed/integrated_catchments.geojson',
        '/workspace/watershed/routing_with_lakes.csv',
        '/workspace/watershed/lake_routing.geojson',
        '/workspace/watershed/detailed_attributes.csv'
    ],
    'files_count': 10
}
```

## Performance and Resource Usage

### **Sub-Step Performance Breakdown**:
- **4.1 Basic Delineation**: 2-4 minutes (DEM processing intensive)
- **4.2 Lake Detection**: 1-2 minutes (database query + spatial operations)
- **4.3 Lake Classification**: 30-60 seconds (spatial indexing)
- **4.4 Lake Integration**: 1-3 minutes (routing topology modification)
- **4.5 Attributes Calculation**: 30-90 seconds (statistics calculation)
- **Total Step 4**: 5-12 minutes average

### **Resource Requirements**:
- **Memory**: 2-8 GB (depends on DEM size and lake count)
- **CPU**: Multi-core beneficial for DEM processing
- **Storage**: 50-200 MB for intermediate files
- **Database**: Fast spatial queries to Canadian Hydro Database

### **Scalability Factors**:
- **Watershed Size**: Linear scaling for most operations
- **DEM Resolution**: Quadratic scaling for DEM processing
- **Lake Count**: Linear scaling for lake processing
- **Stream Complexity**: Linear scaling for routing modifications

## Error Handling

### **Common Issues and Solutions**:
1. **No streams detected**: Lower stream threshold or check DEM quality
2. **Lake detection fails**: Check database connectivity and spatial bounds
3. **Integration errors**: Validate lake geometries and stream topology
4. **Attribute calculation fails**: Check CRS consistency and data validity

### **Quality Control Checks**:
- Watershed area matches MyHydro result (±5%)
- All lake outlets have downstream connections
- Routing table has no circular references
- Stream network is properly connected
- Attribute values are within reasonable ranges

This detailed breakdown shows how Step 4 transforms a simple DEM and outlet coordinates into a complete, RAVEN-ready watershed model with proper stream networks, classified lakes, and integrated routing topology.