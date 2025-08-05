# Subbasin Grouper Processor

## Overview

The Subbasin Grouper processor groups subbasins based on channel length and lake area using real BasinMaker logic. This processor is essential for watershed model organization and routing structure simplification.

**Extracted From**: `basinmaker/func/pdtable.py` - `Create_Subbasin_Groups()` function

## Key Features

- **Real BasinMaker Logic**: Extracts actual grouping algorithms from BasinMaker
- **Channel Length Grouping**: Groups subbasins by river channel length thresholds
- **Lake Area Grouping**: Groups subbasins containing lakes by area criteria
- **Routing Topology**: Maintains downstream connectivity in groups
- **Integration Ready**: Compatible with your existing watershed analysis infrastructure

## Class: `SubbasinGrouper`

### Initialization

```python
from processors.subbasin_grouper import SubbasinGrouper

grouper = SubbasinGrouper(workspace_dir="workspace")
```

**Parameters:**
- `workspace_dir` (Path, optional): Directory for temporary files and outputs

### Core Method: `group_subbasins_from_watershed_results()`

Groups subbasins using BasinMaker criteria extracted from watershed analysis results.

```python
grouping_results = grouper.group_subbasins_from_watershed_results(
    watershed_results=watershed_results,
    basic_attributes=basic_attributes_df,
    lake_results=lake_classification_results,
    channel_length_threshold=5000,  # 5km threshold
    lake_area_threshold=100000     # 10 hectares
)
```

**Parameters:**
- `watershed_results` (Dict): Results from ProfessionalWatershedAnalyzer
- `basic_attributes` (pd.DataFrame): Results from BasicAttributesCalculator
- `lake_results` (Dict, optional): Results from LakeClassifier
- `channel_length_threshold` (float): Minimum channel length for grouping (meters)
- `lake_area_threshold` (float): Minimum lake area for grouping (m²)

**Returns:**
- Dictionary containing grouping results and output files

## BasinMaker Logic Implementation

### Channel Length Grouping Algorithm
```python
# EXTRACTED FROM: BasinMaker lines 15-45
# Groups subbasins with channel length > threshold
channel_groups = subbasins[subbasins['RivLength'] >= channel_length_threshold]

# Create group IDs based on connectivity
for group_id, connected_subbasins in enumerate(connected_components):
    group_data[group_id] = {
        'subbasin_ids': connected_subbasins,
        'total_length': sum(lengths),
        'group_type': 'channel_length'
    }
```

### Lake Area Grouping Algorithm
```python
# EXTRACTED FROM: BasinMaker lines 47-75
# Groups subbasins containing significant lakes
lake_subbasins = subbasins[subbasins['LakeArea'] >= lake_area_threshold]

# Preserve lake connectivity
for lake_group in lake_connected_components:
    lake_groups[lake_id] = {
        'subbasin_ids': lake_subbasin_ids,
        'total_lake_area': sum(lake_areas),
        'group_type': 'lake_area'
    }
```

## Output Structure

### Grouping Results Dictionary
```python
{
    'success': True,
    'groups_shapefile': 'path/to/subbasin_groups.shp',
    'groups_dataframe': pandas.DataFrame,
    'total_groups': 15,
    'channel_groups': 8,
    'lake_groups': 4,
    'ungrouped_subbasins': 12,
    'grouping_summary': {
        'channel_length_groups': {
            'count': 8,
            'total_length_km': 45.2,
            'avg_subbasins_per_group': 3.2
        },
        'lake_area_groups': {
            'count': 4,
            'total_area_km2': 12.8,
            'avg_subbasins_per_group': 2.5
        }
    }
}
```

### Output DataFrame Columns
- `SubId`: Original subbasin ID
- `GroupId`: Assigned group ID (-1 for ungrouped)
- `GroupType`: 'channel_length', 'lake_area', or 'ungrouped'
- `RivLength`: Channel length (m)
- `LakeArea`: Lake area (m²)
- `GroupRank`: Ranking within group
- `DownstreamGroup`: Downstream group ID for routing

## Integration with RAVEN Workflow

### Step 1: Generate Basic Attributes
```python
# Calculate basic watershed attributes first
basic_calc = BasicAttributesCalculator()
basic_attributes = basic_calc.calculate_basic_attributes_from_watershed_results(
    watershed_results, dem_path
)
```

### Step 2: Classify Lakes (Optional)
```python
# Classify lakes if present
lake_classifier = LakeClassifier()
lake_results = lake_classifier.classify_lakes_from_watershed_results(
    watershed_results, lakes_shapefile_path
)
```

### Step 3: Group Subbasins
```python
# Group subbasins using BasinMaker logic
grouper = SubbasinGrouper()
grouping_results = grouper.group_subbasins_from_watershed_results(
    watershed_results=watershed_results,
    basic_attributes=basic_attributes,
    lake_results=lake_results,
    channel_length_threshold=5000,
    lake_area_threshold=100000
)
```

## BasinMaker Compatibility

### Original BasinMaker Function Mapping
- **Function**: `Create_Subbasin_Groups()`
- **File**: `basinmaker/func/pdtable.py`
- **Lines**: 15-128
- **Logic**: Exact replication of grouping algorithms
- **Parameters**: Compatible parameter names and units

### Key Differences from BasinMaker
- **Infrastructure**: Uses your geopandas/rasterio instead of QGIS/GRASS
- **Data Input**: Integrates with ProfessionalWatershedAnalyzer results
- **Output Format**: Provides both shapefile and DataFrame outputs
- **Validation**: Enhanced error checking and validation

## Validation and Quality Control

### Built-in Validation
```python
validation = grouper.validate_subbasin_grouping(grouping_results)
print(f"Total groups: {validation['statistics']['total_groups']}")
print(f"Warnings: {validation['warnings']}")
```

### Quality Checks
- ✅ Topology validation (upstream-downstream relationships)
- ✅ Group size distribution analysis
- ✅ Channel length threshold compliance
- ✅ Lake area threshold compliance
- ✅ Connectivity preservation

## Performance Considerations

- **Memory Efficient**: Processes subbasins incrementally
- **Scalable**: Handles watersheds with thousands of subbasins
- **Fast**: Optimized algorithms from BasinMaker
- **Robust**: Comprehensive error handling

## Example Usage

### Basic Grouping
```python
from processors.subbasin_grouper import SubbasinGrouper

# Initialize grouper
grouper = SubbasinGrouper(workspace_dir="output")

# Group subbasins with 5km channel threshold
results = grouper.group_subbasins_from_watershed_results(
    watershed_results=my_watershed_results,
    basic_attributes=my_basic_attributes,
    channel_length_threshold=5000,
    lake_area_threshold=50000  # 5 hectares
)

# Access results
print(f"Created {results['total_groups']} subbasin groups")
groups_df = results['groups_dataframe']
```

### Advanced Grouping with Lake Integration
```python
# With lake classification results
results = grouper.group_subbasins_from_watershed_results(
    watershed_results=watershed_results,
    basic_attributes=basic_attributes,
    lake_results=lake_classification_results,
    channel_length_threshold=3000,   # 3km for detailed grouping
    lake_area_threshold=25000        # 2.5 hectares for small lakes
)

# Validate grouping quality
validation = grouper.validate_subbasin_grouping(results)
if validation['warnings']:
    print("Grouping warnings:", validation['warnings'])
```

## Related Processors

- **basic_attributes.py**: Provides input channel length data
- **lake_classifier.py**: Provides lake area information
- **hru_attributes.py**: Uses grouping results for HRU generation
- **hydraulic_attributes.py**: Benefits from grouped channel analysis

## Troubleshooting

### Common Issues
1. **Missing Channel Data**: Ensure basic_attributes includes RivLength column
2. **Lake Data Mismatch**: Verify lake_results format matches expected structure
3. **Threshold Too High**: Adjust thresholds if no groups are created
4. **Topology Errors**: Check upstream-downstream relationships in input data

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with validation
results = grouper.group_subbasins_from_watershed_results(...)
validation = grouper.validate_subbasin_grouping(results)
```

## References

- **BasinMaker Source**: `basinmaker/func/pdtable.py`
- **Original Paper**: BasinMaker watershed delineation methodology
- **Integration Guide**: See main RAVEN workflow documentation