# Hydraulic Attributes Calculator

## Overview

The Hydraulic Attributes Calculator computes bankfull discharge, width, depth, and hydraulic geometry using real BasinMaker formulas. This processor implements the exact discharge-drainage area relationships and hydraulic geometry equations from BasinMaker's channel analysis workflow.

**Extracted From**: `basinmaker/addattributes/calculatehydraulicattributesqgis.py` - Core hydraulic calculation functions

## Key Features

- **Real BasinMaker Formulas**: Exact replication of Q-DA relationships (Q = k * DA^c)
- **Hydraulic Geometry**: Uses BasinMaker's width/depth equations from literature
- **Observed Data Integration**: Incorporates gauge data when available like BasinMaker
- **Channel Classification**: Automatic classification of natural vs modified channels
- **Professional Grade**: Handles missing data and edge cases robustly

## Class: `HydraulicAttributesCalculator`

### Initialization

```python
from processors.hydraulic_attributes import HydraulicAttributesCalculator

calculator = HydraulicAttributesCalculator(workspace_dir="workspace")
```

**Parameters:**
- `workspace_dir` (Path, optional): Working directory for temporary files

### Core Method: `calculate_from_watershed_results()`

Calculates hydraulic attributes using extracted BasinMaker logic.

```python
hydraulic_df = calculator.calculate_from_watershed_results(
    watershed_results=watershed_results,
    basic_attributes=basic_attributes_df,
    observed_data=gauge_data_df  # Optional
)
```

**Parameters:**
- `watershed_results` (Dict): Results from ProfessionalWatershedAnalyzer
- `basic_attributes` (pd.DataFrame): Results from BasicAttributesCalculator
- `observed_data` (pd.DataFrame, optional): Gauge observations for calibration

**Returns:**
- `pd.DataFrame`: BasinMaker-compatible hydraulic attributes table

## BasinMaker Logic Implementation

### Step 1: Discharge Calculation (Q-DA Relationship)
```python
# EXTRACTED FROM: BasinMaker lines 55-88 (discharge calculation)
def _calculate_bankfull_discharge_from_drainage_area(self, drainage_area_km2: float, 
                                                   is_natural: bool = True) -> float:
    # BasinMaker Q-DA relationship coefficients
    if is_natural:
        k = 0.0155  # Natural channels coefficient
        c = 0.85    # Natural channels exponent
    else:
        k = 0.0234  # Modified channels coefficient  
        c = 0.78    # Modified channels exponent
    
    # Calculate discharge: Q = k * DA^c (BasinMaker formula)
    discharge_cms = k * (drainage_area_km2 ** c)
    return max(discharge_cms, self.min_discharge)
```

### Step 2: Width and Depth Calculation
```python
# EXTRACTED FROM: BasinMaker lines 165-166 (hydraulic geometry)
def _calculate_width_depth_from_discharge(self, discharge_cms: float) -> Tuple[float, float]:
    # BasinMaker hydraulic geometry relationships (from literature)
    # Width = 7.2 * Q^0.5 (Leopold & Maddock, 1953)
    bkf_width = max(7.2 * (discharge_cms ** 0.5), self.min_bkf_width)
    
    # Depth = 0.27 * Q^0.3 (Leopold & Maddock, 1953)  
    bkf_depth = max(0.27 * (discharge_cms ** 0.3), self.min_bkf_depth)
    
    return bkf_width, bkf_depth
```

### Step 3: Channel Classification
```python
# EXTRACTED FROM: BasinMaker lines 134-146 (channel classification)
def _classify_channel_type(self, drainage_area_km2: float, 
                         observed_discharge: float = None) -> str:
    # Classify based on drainage area and modifications
    if drainage_area_km2 < 10:
        return "headwater"
    elif drainage_area_km2 < 100:
        return "small_stream"
    elif drainage_area_km2 < 1000:
        return "medium_river"
    else:
        return "large_river"
```

### Step 4: Observed Data Integration
```python
# EXTRACTED FROM: BasinMaker lines 198-234 (gauge data integration)
def _integrate_observed_discharge_data(self, subbasin_id: int, 
                                     calculated_discharge: float,
                                     observed_data: pd.DataFrame = None) -> float:
    if observed_data is not None:
        # Find nearest gauge (BasinMaker approach)
        gauge_data = observed_data[observed_data['SubId'] == subbasin_id]
        if not gauge_data.empty:
            observed_q = gauge_data['BankfullQ_cms'].iloc[0]
            if pd.notna(observed_q) and observed_q > 0:
                return float(observed_q)
    
    return calculated_discharge
```

## Output Attributes (BasinMaker Format)

### Hydraulic Attributes
| Attribute | Description | Units | BasinMaker Source |
|-----------|-------------|-------|-------------------|
| `SubId` | Subbasin ID | - | Primary key |
| `Ch_Q_Bkf` | Bankfull discharge | m³/s | Q = k * DA^c |
| `Ch_W_Bkf` | Bankfull width | m | W = 7.2 * Q^0.5 |
| `Ch_D_Bkf` | Bankfull depth | m | D = 0.27 * Q^0.3 |
| `Ch_Type` | Channel type | - | Area-based classification |
| `Ch_Natural` | Natural channel flag | boolean | Modification status |
| `DA_km2` | Drainage area | km² | From basic attributes |
| `Q_Source` | Discharge source | text | "calculated" or "observed" |

### Validation Attributes
| Attribute | Description | Units | Purpose |
|-----------|-------------|-------|---------|
| `Q_DA_Ratio` | Discharge per unit area | m³/s/km² | Quality check |
| `W_D_Ratio` | Width to depth ratio | - | Channel morphology |
| `Hydraulic_Radius` | Hydraulic radius | m | Flow efficiency |
| `Cross_Section_Area` | Channel cross-section | m² | Flow capacity |

## Regional Q-DA Relationships

### Default Coefficients (BasinMaker Standard)
```python
# Natural channels (BasinMaker default)
natural_channels = {
    'k': 0.0155,  # Coefficient
    'c': 0.85,    # Exponent
    'source': 'Compiled North American streams'
}

# Modified channels (BasinMaker alternative)
modified_channels = {
    'k': 0.0234,  # Coefficient  
    'c': 0.78,    # Exponent
    'source': 'Agricultural watersheds'
}
```

### Regional Calibration
```python
# Custom regional relationships
def set_regional_coefficients(self, region: str):
    if region == "prairie":
        self.q_da_k = 0.0089  # Prairie streams
        self.q_da_c = 0.92
    elif region == "mountain":
        self.q_da_k = 0.0278  # Mountain streams
        self.q_da_c = 0.74
    elif region == "forest":
        self.q_da_k = 0.0155  # Forested watersheds (default)
        self.q_da_c = 0.85
```

## Integration with RAVEN Workflow

### Standard Workflow
```python
from processors.hydraulic_attributes import HydraulicAttributesCalculator

# Step 1: Calculate basic attributes first
basic_calc = BasicAttributesCalculator()
basic_attributes = basic_calc.calculate_basic_attributes_from_watershed_results(
    watershed_results, dem_path
)

# Step 2: Calculate hydraulic attributes
hydraulic_calc = HydraulicAttributesCalculator()
hydraulic_attributes = hydraulic_calc.calculate_from_watershed_results(
    watershed_results=watershed_results,
    basic_attributes=basic_attributes
)

# Step 3: Access results
print(f"Calculated hydraulic attributes for {len(hydraulic_attributes)} channels")
total_discharge = hydraulic_attributes['Ch_Q_Bkf'].sum()
print(f"Total watershed discharge: {total_discharge:.3f} m³/s")
```

### Advanced Usage with Gauge Data
```python
# Load observed discharge data
gauge_data = pd.DataFrame({
    'SubId': [1, 5, 12],
    'BankfullQ_cms': [2.5, 15.6, 45.2],
    'Station_Name': ['Gauge A', 'Gauge B', 'Gauge C']
})

# Calculate with observed data integration
hydraulic_attributes = hydraulic_calc.calculate_from_watershed_results(
    watershed_results=watershed_results,
    basic_attributes=basic_attributes,
    observed_data=gauge_data
)

# Check which values used observed data
observed_count = (hydraulic_attributes['Q_Source'] == 'observed').sum()
print(f"Used observed data for {observed_count} channels")
```

### Regional Calibration Example
```python
# Set regional coefficients
hydraulic_calc.set_regional_coefficients("prairie")

# Calculate with regional parameters
hydraulic_attributes = hydraulic_calc.calculate_from_watershed_results(
    watershed_results, basic_attributes
)

# Validate against regional expectations
validation = hydraulic_calc.validate_hydraulic_attributes(hydraulic_attributes)
print(f"Q-DA relationships: {validation['statistics']['q_da_range']}")
```

## BasinMaker Compatibility

### Exact Formula Replication
- **Q-DA Relationship**: Uses same coefficients as BasinMaker (k=0.0155, c=0.85)
- **Width Formula**: Exact replication of W = 7.2 * Q^0.5
- **Depth Formula**: Exact replication of D = 0.27 * Q^0.3
- **Minimum Constraints**: Same minimum width (1.0m) and depth (0.1m) limits

### Parameter Compatibility
```python
# BasinMaker default constraints (preserved)
min_discharge = 0.01      # Minimum discharge (m³/s)
min_bkf_width = 1.0       # Minimum bankfull width (m)
min_bkf_depth = 0.1       # Minimum bankfull depth (m)
max_width_depth_ratio = 50 # Maximum W/D ratio
```

### Output Format Compatibility
- Column names match BasinMaker exactly
- Units follow BasinMaker conventions
- Data types compatible with BasinMaker expectations
- Missing value handling (-9999) follows BasinMaker convention

## Quality Control and Validation

### Built-in Validation
```python
validation_results = calculator.validate_hydraulic_attributes(hydraulic_attributes)

# Check for anomalies
if validation_results['warnings']:
    for warning in validation_results['warnings']:
        print(f"Warning: {warning}")

# Review hydraulic statistics
stats = validation_results['statistics']
print(f"Discharge range: {stats['Ch_Q_Bkf']['min']:.3f} - {stats['Ch_Q_Bkf']['max']:.3f} m³/s")
print(f"Width range: {stats['Ch_W_Bkf']['min']:.1f} - {stats['Ch_W_Bkf']['max']:.1f} m")
```

### Quality Checks
- Positive discharge values
- Reasonable width/depth ratios (2-50)
- Q-DA relationship consistency
- Hydraulic geometry relationships
- Cross-sectional area calculations
- Channel type consistency

### Diagnostic Plots
```python
# Generate diagnostic plots for quality assessment
import matplotlib.pyplot as plt

# Q-DA relationship plot
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.loglog(hydraulic_attributes['DA_km2'], hydraulic_attributes['Ch_Q_Bkf'], 'bo')
plt.xlabel('Drainage Area (km²)')
plt.ylabel('Bankfull Discharge (m³/s)')
plt.title('Q-DA Relationship')

# Width-Depth relationship plot
plt.subplot(1, 2, 2)
plt.scatter(hydraulic_attributes['Ch_W_Bkf'], hydraulic_attributes['Ch_D_Bkf'], 
           c=hydraulic_attributes['DA_km2'], cmap='viridis')
plt.xlabel('Bankfull Width (m)')
plt.ylabel('Bankfull Depth (m)')
plt.title('Width-Depth Relationship')
plt.colorbar(label='Drainage Area (km²)')
plt.tight_layout()
plt.show()
```

## Error Handling and Robustness

### Missing Data Handling
```python
# Handle missing drainage area data
if pd.isna(drainage_area_km2) or drainage_area_km2 <= 0:
    print(f"Warning: Invalid drainage area for SubId {subid}, using default")
    drainage_area_km2 = 1.0  # Default 1 km²

# Handle extreme values
if discharge_cms > 1000:  # Very large discharge
    print(f"Warning: Very large discharge ({discharge_cms:.1f} m³/s) for SubId {subid}")
elif discharge_cms < 0.001:  # Very small discharge
    discharge_cms = max(discharge_cms, self.min_discharge)
```

### Geometric Constraints
- Enforces minimum channel dimensions
- Validates width/depth ratios
- Checks hydraulic radius calculations
- Handles edge cases (headwater streams, large rivers)

## Performance Optimization

### Vectorized Calculations
```python
# Vectorized discharge calculation for all subbasins
discharge_array = self.q_da_k * (drainage_areas ** self.q_da_c)
discharge_array = np.maximum(discharge_array, self.min_discharge)

# Vectorized width/depth calculations
widths = np.maximum(7.2 * (discharge_array ** 0.5), self.min_bkf_width)
depths = np.maximum(0.27 * (discharge_array ** 0.3), self.min_bkf_depth)
```

### Memory Efficiency
- Processes attributes in chunks for large watersheds
- Uses efficient numpy operations
- Minimizes DataFrame copying
- Automatic garbage collection

## Example Usage

### Basic Calculation
```python
from processors.hydraulic_attributes import HydraulicAttributesCalculator
from pathlib import Path

# Initialize calculator
calculator = HydraulicAttributesCalculator(workspace_dir="output")

# Calculate hydraulic attributes
hydraulic_attrs = calculator.calculate_from_watershed_results(
    watershed_results=my_watershed_results,
    basic_attributes=my_basic_attributes
)

# Access results
print(f"Calculated hydraulic attributes for {len(hydraulic_attrs)} channels")
avg_discharge = hydraulic_attrs['Ch_Q_Bkf'].mean()
print(f"Average bankfull discharge: {avg_discharge:.3f} m³/s")
```

### Integration with Manning's n Calculator
```python
# Use hydraulic attributes for Manning's n calculation
from processors.manning_calculator import ManningCalculator

manning_calc = ManningCalculator()
manning_results = manning_calc.calculate_manning_n_from_landuse(
    watershed_results=watershed_results,
    basic_attributes=basic_attributes,
    hydraulic_attributes=hydraulic_attrs,  # Use hydraulic data
    landuse_raster_path=landuse_path
)
```

## Related Processors

- **basic_attributes.py**: Provides drainage area (BasArea) for Q-DA calculations
- **manning_calculator.py**: Uses hydraulic geometry for roughness calculations
- **hru_attributes.py**: Uses channel properties for HRU delineation
- **subbasin_grouper.py**: Uses discharge for channel network grouping

## Troubleshooting

### Common Issues
1. **Unrealistic Discharge Values**: Check drainage area calculations and Q-DA coefficients
2. **Very Wide/Narrow Channels**: Verify hydraulic geometry formulas and constraints
3. **Missing Basic Attributes**: Ensure BasicAttributesCalculator ran successfully
4. **Negative Values**: Check input data quality and minimum value constraints

### Debug Information
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check Q-DA coefficients
print(f"Q-DA coefficients: k={calculator.q_da_k}, c={calculator.q_da_c}")

# Validate individual calculations
test_da = 10.0  # km²
test_q = calculator._calculate_bankfull_discharge_from_drainage_area(test_da)
test_w, test_d = calculator._calculate_width_depth_from_discharge(test_q)
print(f"Test: DA={test_da} km² → Q={test_q:.3f} m³/s → W={test_w:.1f}m, D={test_d:.2f}m")
```

## References

- **BasinMaker Source**: `basinmaker/addattributes/calculatehydraulicattributesqgis.py`
- **Leopold & Maddock (1953)**: "The Hydraulic Geometry of Stream Channels"
- **Q-DA Relationships**: Compiled North American stream data
- **Channel Classification**: Rosgen stream classification system