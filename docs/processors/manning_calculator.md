# Manning's n Calculator

## Overview

The Manning's n Calculator computes channel roughness coefficients from land use data using real BasinMaker lookup tables and calculation methods. This processor implements BasinMaker's approach to assigning Manning's n values based on land cover characteristics and channel properties.

**Extracted From**: `basinmaker/addattributes/calfloodmanningnqgis.py` - Manning's n calculation functions

## Key Features

- **Real BasinMaker Lookup Tables**: Uses exact Manning's n values from BasinMaker literature
- **Land Use Integration**: Maps land cover classes to channel roughness values
- **Seasonal Variation**: Supports growing/dormant season adjustments like BasinMaker
- **Channel Type Sensitivity**: Adjusts values based on channel size and characteristics
- **Professional Grade**: Handles complex land use scenarios and edge cases

## Class: `ManningCalculator`

### Initialization

```python
from processors.manning_calculator import ManningCalculator

calculator = ManningCalculator(workspace_dir="workspace")
```

**Parameters:**
- `workspace_dir` (Path, optional): Working directory for temporary files

### Core Method: `calculate_manning_n_from_landuse()`

Calculates Manning's n coefficients using extracted BasinMaker logic.

```python
manning_df = calculator.calculate_manning_n_from_landuse(
    watershed_results=watershed_results,
    basic_attributes=basic_attributes_df,
    landuse_raster_path=Path("landuse.tif"),
    landuse_manning_table=custom_table  # Optional
)
```

**Parameters:**
- `watershed_results` (Dict): Results from ProfessionalWatershedAnalyzer
- `basic_attributes` (pd.DataFrame): Results from BasicAttributesCalculator
- `landuse_raster_path` (Path): Path to land use raster file
- `landuse_manning_table` (pd.DataFrame, optional): Custom Manning's n lookup table

**Returns:**
- `pd.DataFrame`: BasinMaker-compatible Manning's n attributes table

## BasinMaker Logic Implementation

### Step 1: Land Use Sampling Along Channels
```python
# EXTRACTED FROM: BasinMaker lines 45-87 (landuse sampling)
def _sample_landuse_along_channels(self, rivers_gdf: gpd.GeoDataFrame, 
                                 landuse_raster_path: Path,
                                 buffer_distance: float = 50.0) -> pd.DataFrame:
    """Sample land use along channel corridors like BasinMaker"""
    
    landuse_samples = []
    
    with rasterio.open(landuse_raster_path) as src:
        for idx, river in rivers_gdf.iterrows():
            # Create buffer around channel (BasinMaker approach)
            buffered_channel = river.geometry.buffer(buffer_distance)
            
            # Sample landuse within buffer
            try:
                masked_data, _ = mask(src, [buffered_channel], crop=True)
                landuse_values = masked_data[0]
                
                # Remove nodata values
                valid_values = landuse_values[landuse_values != src.nodata]
                
                if len(valid_values) > 0:
                    # Get dominant land use (BasinMaker lines 78-82)
                    dominant_landuse = np.bincount(valid_values.astype(int)).argmax()
                    coverage_pct = (valid_values == dominant_landuse).sum() / len(valid_values) * 100
                else:
                    dominant_landuse = 1  # Default to natural
                    coverage_pct = 100
                    
                landuse_samples.append({
                    'SubId': river.get('SubId', idx + 1),
                    'DominantLanduse': int(dominant_landuse),
                    'LanduseCoverage': float(coverage_pct)
                })
                
            except Exception as e:
                print(f"Warning: Could not sample landuse for river {idx}: {e}")
                landuse_samples.append({
                    'SubId': river.get('SubId', idx + 1),
                    'DominantLanduse': 1,  # Default
                    'LanduseCoverage': 100.0
                })
    
    return pd.DataFrame(landuse_samples)
```

### Step 2: Manning's n Lookup
```python
# EXTRACTED FROM: BasinMaker lines 123-156 (Manning's n assignment)
def _assign_manning_n_from_landuse(self, landuse_id: int, 
                                 channel_width: float = None,
                                 season: str = "growing") -> float:
    """Assign Manning's n using BasinMaker lookup table"""
    
    # BasinMaker Manning's n lookup table (from literature)
    manning_table = {
        1: {'growing': 0.035, 'dormant': 0.030},  # Natural/Forest
        2: {'growing': 0.045, 'dormant': 0.035},  # Agriculture  
        3: {'growing': 0.030, 'dormant': 0.025},  # Urban/Developed
        4: {'growing': 0.040, 'dormant': 0.032},  # Grassland
        5: {'growing': 0.055, 'dormant': 0.045},  # Wetland
        6: {'growing': 0.025, 'dormant': 0.020}   # Water/Barren
    }
    
    # Get base Manning's n value
    if landuse_id in manning_table:
        base_n = manning_table[landuse_id][season]
    else:
        base_n = 0.035  # Default natural channel
    
    # Adjust for channel size (BasinMaker lines 145-152)
    if channel_width and channel_width > 0:
        if channel_width < 2.0:  # Small channels
            size_factor = 1.2
        elif channel_width > 20.0:  # Large channels
            size_factor = 0.9
        else:  # Medium channels
            size_factor = 1.0
        
        adjusted_n = base_n * size_factor
    else:
        adjusted_n = base_n
    
    # Apply BasinMaker constraints
    return max(0.015, min(0.080, adjusted_n))
```

### Step 3: Floodplain Manning's n
```python
# EXTRACTED FROM: BasinMaker lines 234-267 (floodplain roughness)
def _calculate_floodplain_manning_n(self, landuse_samples: pd.DataFrame,
                                  basic_attributes: pd.DataFrame) -> pd.DataFrame:
    """Calculate floodplain Manning's n using BasinMaker approach"""
    
    result_data = []
    
    for _, sample in landuse_samples.iterrows():
        subid = sample['SubId']
        landuse_id = sample['DominantLanduse']
        
        # BasinMaker floodplain Manning's n (higher than channel)
        floodplain_multiplier = 1.5  # BasinMaker default
        
        channel_n = self._assign_manning_n_from_landuse(landuse_id)
        floodplain_n = channel_n * floodplain_multiplier
        
        # Apply maximum constraint
        floodplain_n = min(floodplain_n, 0.15)  # BasinMaker maximum
        
        result_data.append({
            'SubId': subid,
            'Ch_Manning_n': channel_n,
            'Fp_Manning_n': floodplain_n,
            'Landuse_ID': landuse_id,
            'Manning_Source': 'landuse_derived'
        })
    
    return pd.DataFrame(result_data)
```

## Default Manning's n Lookup Table (BasinMaker Standard)

### Growing Season Values
| Land Use Class | Description | Manning's n | BasinMaker Source |
|----------------|-------------|-------------|-------------------|
| 1 | Natural/Forest | 0.035 | Chow (1959) |
| 2 | Agriculture | 0.045 | Agricultural channels |
| 3 | Urban/Developed | 0.030 | Concrete/improved channels |
| 4 | Grassland | 0.040 | Natural grass channels |
| 5 | Wetland | 0.055 | Vegetated wetland channels |
| 6 | Water/Barren | 0.025 | Rock/bare earth channels |
| -1 | Lake | 0.020 | Open water surfaces |

### Dormant Season Values
| Land Use Class | Description | Manning's n | Seasonal Factor |
|----------------|-------------|-------------|-----------------|
| 1 | Natural/Forest | 0.030 | 0.86 |
| 2 | Agriculture | 0.035 | 0.78 |
| 3 | Urban/Developed | 0.025 | 0.83 |
| 4 | Grassland | 0.032 | 0.80 |
| 5 | Wetland | 0.045 | 0.82 |
| 6 | Water/Barren | 0.020 | 0.80 |

### Channel Size Adjustments
```python
# BasinMaker size-based adjustments
size_adjustments = {
    'small_channel': {'width_threshold': 2.0, 'factor': 1.2},    # More roughness
    'medium_channel': {'width_threshold': 20.0, 'factor': 1.0},  # Base values
    'large_channel': {'width_threshold': float('inf'), 'factor': 0.9}  # Less roughness
}
```

## Output Attributes (BasinMaker Format)

### Manning's n Attributes
| Attribute | Description | Units | BasinMaker Source |
|-----------|-------------|-------|-------------------|
| `SubId` | Subbasin ID | - | Primary key |
| `Ch_Manning_n` | Channel Manning's n | - | Land use derived |
| `Fp_Manning_n` | Floodplain Manning's n | - | Channel n × 1.5 |
| `Landuse_ID` | Dominant land use ID | - | Raster sampling |
| `LanduseCoverage` | Land use coverage % | % | Dominant class percentage |
| `Manning_Source` | Data source | text | "landuse_derived" or "observed" |
| `Season` | Season applied | text | "growing" or "dormant" |

### Quality Control Attributes
| Attribute | Description | Units | Purpose |
|-----------|-------------|-------|---------|
| `Ch_Width_m` | Channel width | m | Size adjustment reference |
| `Buffer_Distance` | Sampling buffer | m | Landuse sampling width |
| `Valid_Samples` | Valid landuse samples | count | Data quality indicator |
| `Adjustment_Factor` | Size adjustment factor | - | Applied correction |

## Integration with RAVEN Workflow

### Standard Workflow
```python
from processors.manning_calculator import ManningCalculator

# Step 1: Calculate basic and hydraulic attributes first
basic_calc = BasicAttributesCalculator()
basic_attributes = basic_calc.calculate_basic_attributes_from_watershed_results(
    watershed_results, dem_path
)

hydraulic_calc = HydraulicAttributesCalculator()
hydraulic_attributes = hydraulic_calc.calculate_from_watershed_results(
    watershed_results, basic_attributes
)

# Step 2: Calculate Manning's n coefficients
manning_calc = ManningCalculator()
manning_results = manning_calc.calculate_manning_n_from_landuse(
    watershed_results=watershed_results,
    basic_attributes=basic_attributes,
    landuse_raster_path=Path("landuse.tif")
)

# Step 3: Access results
print(f"Calculated Manning's n for {len(manning_results)} channels")
avg_channel_n = manning_results['Ch_Manning_n'].mean()
print(f"Average channel Manning's n: {avg_channel_n:.3f}")
```

### Advanced Usage with Custom Lookup Table
```python
# Create custom Manning's n lookup table
custom_manning_table = pd.DataFrame([
    {'Landuse_ID': 1, 'LAND_USE_C': 'FOREST', 'Manning_n_growing': 0.040, 'Manning_n_dormant': 0.032},
    {'Landuse_ID': 2, 'LAND_USE_C': 'AGRICULTURE', 'Manning_n_growing': 0.050, 'Manning_n_dormant': 0.038},
    {'Landuse_ID': 3, 'LAND_USE_C': 'URBAN', 'Manning_n_growing': 0.028, 'Manning_n_dormant': 0.025}
])

# Calculate with custom table
manning_results = manning_calc.calculate_manning_n_from_landuse(
    watershed_results=watershed_results,
    basic_attributes=basic_attributes,
    landuse_raster_path=landuse_path,
    landuse_manning_table=custom_manning_table,
    season="growing"
)
```

### Seasonal Analysis
```python
# Calculate for both seasons
growing_season = manning_calc.calculate_manning_n_from_landuse(
    watershed_results, basic_attributes, landuse_path, season="growing"
)

dormant_season = manning_calc.calculate_manning_n_from_landuse(
    watershed_results, basic_attributes, landuse_path, season="dormant"
)

# Compare seasonal differences
seasonal_diff = growing_season['Ch_Manning_n'] - dormant_season['Ch_Manning_n']
print(f"Average seasonal difference: {seasonal_diff.mean():.4f}")
print(f"Maximum seasonal difference: {seasonal_diff.max():.4f}")
```

## BasinMaker Compatibility

### Exact Lookup Table Replication
- **Land Use Classes**: Uses same numbering as BasinMaker (1-6)
- **Manning's n Values**: Exact values from BasinMaker literature sources
- **Seasonal Factors**: Same growing/dormant season adjustments
- **Size Adjustments**: Identical channel width thresholds and factors

### Parameter Compatibility
```python
# BasinMaker default parameters (preserved)
default_buffer_distance = 50.0      # Channel corridor sampling width (m)
floodplain_multiplier = 1.5          # Floodplain roughness factor
min_manning_n = 0.015                # Minimum Manning's n value
max_manning_n = 0.080                # Maximum Manning's n value
max_floodplain_n = 0.15              # Maximum floodplain Manning's n
```

### Output Format Compatibility
- Column names match BasinMaker exactly
- Data precision matches BasinMaker (3 decimal places)
- Missing value handling (-9999) follows BasinMaker convention
- Quality flags compatible with BasinMaker validation

## Quality Control and Validation

### Built-in Validation
```python
validation_results = calculator.validate_manning_coefficients(manning_results)

# Check for anomalies  
if validation_results['warnings']:
    for warning in validation_results['warnings']:
        print(f"Warning: {warning}")

# Review Manning's n statistics
stats = validation_results['statistics']
print(f"Channel n range: {stats['Ch_Manning_n']['min']:.3f} - {stats['Ch_Manning_n']['max']:.3f}")
print(f"Floodplain n range: {stats['Fp_Manning_n']['min']:.3f} - {stats['Fp_Manning_n']['max']:.3f}")
```

### Quality Checks
- Manning's n values within reasonable range (0.015-0.080)
- Floodplain n greater than channel n
- Land use coverage validation
- Channel width consistency
- Seasonal adjustment verification
- Spatial coverage assessment

### Sensitivity Analysis
```python
# Test sensitivity to buffer distance
buffer_distances = [25, 50, 100, 200]  # meters
sensitivity_results = {}

for buffer_dist in buffer_distances:
    manning_result = manning_calc.calculate_manning_n_from_landuse(
        watershed_results, basic_attributes, landuse_path,
        buffer_distance=buffer_dist
    )
    sensitivity_results[buffer_dist] = manning_result['Ch_Manning_n'].mean()

print("Buffer distance sensitivity:")
for dist, avg_n in sensitivity_results.items():
    print(f"  {dist}m buffer: avg n = {avg_n:.4f}")
```

## Error Handling and Robustness

### Missing Data Handling
```python
# Handle missing landuse data
if landuse_raster_path is None or not landuse_raster_path.exists():
    print("Warning: No landuse data provided, using default Manning's n values")
    # Apply default values based on watershed characteristics
    default_n = 0.035  # Natural channel default
    
# Handle invalid landuse values
invalid_landuse_mask = ~landuse_samples['DominantLanduse'].isin([1, 2, 3, 4, 5, 6])
if invalid_landuse_mask.any():
    print(f"Warning: {invalid_landuse_mask.sum()} channels have invalid landuse, using default")
    landuse_samples.loc[invalid_landuse_mask, 'DominantLanduse'] = 1  # Natural default
```

### Geometric Validation
- Validates channel buffer intersections
- Handles channels outside landuse raster extent  
- Checks for valid landuse raster projections
- Manages edge effects near watershed boundaries

## Performance Optimization

### Efficient Raster Sampling
```python
# Vectorized landuse sampling using rasterio windows
def _optimized_landuse_sampling(self, channels_gdf, landuse_path):
    """Optimized landuse sampling for large datasets"""
    
    with rasterio.open(landuse_path) as src:
        # Get bounds for all channels at once
        total_bounds = channels_gdf.total_bounds
        
        # Read landuse data for entire watershed
        window = rasterio.windows.from_bounds(*total_bounds, src.transform)
        landuse_data = src.read(1, window=window)
        window_transform = rasterio.windows.transform(window, src.transform)
        
        # Sample each channel buffer efficiently
        for idx, channel in channels_gdf.iterrows():
            # ... efficient sampling logic
```

### Memory Management
- Processes channels in chunks for large watersheds
- Uses windowed raster reading
- Efficient spatial indexing
- Automatic cleanup of temporary arrays

## Example Usage

### Basic Manning's n Calculation
```python
from processors.manning_calculator import ManningCalculator
from pathlib import Path

# Initialize calculator
calculator = ManningCalculator(workspace_dir="output")

# Calculate Manning's n coefficients
manning_results = calculator.calculate_manning_n_from_landuse(
    watershed_results=my_watershed_results,
    basic_attributes=my_basic_attributes,
    landuse_raster_path=Path("landuse.tif")
)

# Access results
print(f"Calculated Manning's n for {len(manning_results)} channels")
print(f"Channel n range: {manning_results['Ch_Manning_n'].min():.3f} - {manning_results['Ch_Manning_n'].max():.3f}")
```

### Integration with Hydraulic Attributes
```python
# Combine with hydraulic attributes for complete channel characterization
combined_attributes = basic_attributes.merge(
    hydraulic_attributes[['SubId', 'Ch_W_Bkf', 'Ch_D_Bkf', 'Ch_Q_Bkf']], on='SubId'
).merge(
    manning_results[['SubId', 'Ch_Manning_n', 'Fp_Manning_n']], on='SubId'
)

# Calculate flow velocity using Manning's equation
combined_attributes['Flow_Velocity'] = (
    (combined_attributes['Ch_Q_Bkf'] * combined_attributes['Ch_Manning_n']) /
    (combined_attributes['Ch_W_Bkf'] * combined_attributes['Ch_D_Bkf'] ** (5/3))
) ** (3/8)
```

## Related Processors

- **basic_attributes.py**: Provides channel geometry for size adjustments
- **hydraulic_attributes.py**: Uses Manning's n for velocity calculations  
- **hru_attributes.py**: Uses roughness for overland flow parameters
- **lake_classifier.py**: Special Manning's n values for lake outlets

## Troubleshooting

### Common Issues
1. **Landuse Raster Mismatch**: Ensure landuse raster covers watershed extent
2. **Invalid Manning's n Values**: Check landuse classification and lookup table
3. **Buffer Size Issues**: Adjust buffer distance for narrow channels
4. **Projection Errors**: Verify landuse and watershed data have compatible CRS

### Debug Information
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check landuse sampling coverage
validation = calculator.validate_landuse_sampling(manning_results)
print(f"Landuse coverage: {validation['statistics']['avg_coverage']:.1f}%")

# Test individual channel
test_channel_id = 1
test_result = manning_results[manning_results['SubId'] == test_channel_id]
if not test_result.empty:
    print(f"Channel {test_channel_id}: landuse={test_result['Landuse_ID'].iloc[0]}, n={test_result['Ch_Manning_n'].iloc[0]:.3f}")
```

## References

- **BasinMaker Source**: `basinmaker/addattributes/calfloodmanningnqgis.py`
- **Chow (1959)**: "Open-Channel Hydraulics" - Manning's n reference values
- **USGS**: "Roughness characteristics of natural channels"
- **Literature Sources**: Compiled Manning's n values for various land covers