# Streamlined RAVEN Calibration Workflow

## Overview

The RAVEN calibration workflow has been significantly simplified. You now only need to specify an **observed station ID** or **observed data file** to run comprehensive model calibration. No latitude/longitude coordinates or separate data preparation steps are required.

## Key Features

### ✅ **Station-Centric Interface**
- Just provide a station ID (e.g., `'08NM116'`) or data file path
- No need for lat/long coordinates
- Automatic data download/lookup for Canadian hydrometric stations

### ✅ **Multi-Objective OSTRICH Calibration**
- Nash-Sutcliffe coefficient optimization
- Volume error minimization  
- Peak timing error reduction
- Low flow error minimization
- Comprehensive parameter space exploration

### ✅ **Automatic Data Handling**
- Auto-downloads missing station data
- Creates synthetic data for testing when real data unavailable
- Supports custom observed data files
- Intelligent file format detection

### ✅ **Comprehensive Analytics**
- Interactive hydrograph plotting with Plotly
- Performance metrics calculation
- Parameter sensitivity analysis
- Calibration convergence tracking

## Basic Usage

### Station-Based Calibration
```python
from workflows.project_steps.step6_validate_run_model.step6_calibrate import Step6ValidateRunModel

# Initialize calibration
step6 = Step6ValidateRunModel(workspace_dir="projects/MyProject/models/results")

# Run calibration with just a station ID
result = step6.execute(
    station_id="08NM116",           # Canadian hydrometric station
    calibrate=True,                 # Enable calibration mode
    run_simulation=True,            # Run the simulation
    generate_plots=True,            # Generate visualization
    outlet_name="bigwhite_calibration"
)
```

### Custom Data File Calibration
```python
# Run calibration with your own observed data
result = step6.execute(
    observed_data_file="/path/to/my_streamflow_data.csv",
    calibrate=True,
    run_simulation=True,
    generate_plots=True,
    outlet_name="custom_calibration"
)
```

## Command Line Usage

### Simple Test Script
```bash
# Test with station ID
python test_streamlined_calibration.py --station 08NM116

# Test with custom data file  
python test_streamlined_calibration.py --observed-file path/to/data.csv

# Multi-objective calibration
python test_streamlined_calibration.py --station 08NM116 --multi-objective
```

### Comprehensive Examples
```bash
# Run all calibration examples
python examples_streamlined_calibration.py
```

## Data File Formats

### Expected CSV Format
Your observed streamflow data should be in CSV format with columns:
- `date`: Date in YYYY-MM-DD format
- `flow_cms`: Flow in cubic meters per second

Example:
```csv
date,flow_cms
2020-01-01,15.2
2020-01-02,14.8
2020-01-03,16.1
```

### Auto-Generated Synthetic Data
When real data isn't available, the system automatically creates realistic synthetic streamflow data with:
- Seasonal flow patterns (spring freshet, summer low flow)
- Random variability
- Occasional flood events
- Positive flow constraints

## Calibration Methods

### OSTRICH Integration
The workflow uses OSTRICH (Optimization Software Tool for Research Involving Computational Heuristics) for advanced parameter optimization:

- **DDS (Dynamically Dimensioned Search)**: Efficient for high-dimensional problems
- **SCE (Shuffled Complex Evolution)**: Robust global optimization
- **PSO (Particle Swarm Optimization)**: Nature-inspired optimization

### Multi-Objective Functions
The calibration simultaneously optimizes multiple objectives:

1. **Nash-Sutcliffe Efficiency**: Overall model performance
2. **Volume Error**: Water balance accuracy
3. **Peak Flow Timing**: Flood prediction accuracy  
4. **Low Flow Error**: Baseflow/drought simulation

## Output and Results

### Calibration Results Structure
```python
{
    'success': True,
    'simulation': {
        'calibration': {
            'method': 'OSTRICH_DDS',
            'status': 'converged',
            'best_parameters': {
                'parameter1': 0.123,
                'parameter2': 4.567,
                # ... all calibrated parameters
            },
            'performance_metrics': {
                'nash_sutcliffe': 0.85,
                'volume_error': 0.02,
                'peak_timing_error': 1.2,
                'low_flow_error': 0.15
            },
            'convergence_info': {
                'iterations': 500,
                'function_evaluations': 2500,
                'final_objective': 0.876
            }
        }
    },
    'plots': {
        'hydrographs': ['path/to/plot1.html'],
        'interactive_plots': ['path/to/interactive.html']
    }
}
```

### Generated Plots
- **Hydrograph comparison**: Observed vs. simulated flows
- **Flow duration curves**: Statistical flow analysis
- **Residual plots**: Model error analysis
- **Parameter sensitivity**: Parameter impact analysis
- **Convergence plots**: Optimization progress

## Workflow Comparison

### Old Workflow (Complex)
```python
# 1. Setup coordinates
lat, lon = 49.7313, -118.9439

# 2. Run separate hydrometric data step
climate_hydro_result = run_climate_hydro_step(lat, lon)

# 3. Load climate and hydrometric data
step6.load_previous_results()

# 4. Run calibration
result = step6.execute(lat, lon, calibrate=True)
```

### New Workflow (Simple)
```python
# Single step - just specify station!
result = step6.execute(station_id="08NM116", calibrate=True)
```

## Advanced Configuration

### Custom OSTRICH Settings
```python
# Advanced calibration with custom settings
result = step6.execute(
    station_id="08NM116",
    calibrate=True,
    # Custom calibration parameters can be set in the method
    outlet_name="advanced_calibration"
)
```

### Climate Data Override
```python
# Use custom climate forcing data
result = step6.execute(
    station_id="08NM116",
    calibrate=True,
    climate_override="/path/to/custom_climate.csv"
)
```

## Error Handling and Troubleshooting

### Common Issues

1. **Station Data Not Found**
   - System will attempt to download data automatically
   - Falls back to synthetic data for testing
   - Provides clear error messages

2. **Invalid Data Format**
   - Checks CSV format automatically
   - Provides specific format requirements
   - Suggests corrections

3. **Calibration Convergence Issues**
   - Adjusts optimization parameters automatically
   - Provides convergence diagnostics
   - Suggests parameter bounds adjustments

### Debugging
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check data loading
station_data = step6._load_station_data("08NM116")
print(station_data)
```

## Migration from Old Workflow

### Update Your Scripts
Replace:
```python
# Old approach
result = step6.execute(49.7313, -118.9439, calibrate=True)
```

With:
```python
# New approach  
result = step6.execute(station_id="08NM116", calibrate=True)
```

### Benefits of Migration
- **50% fewer lines of code** - Simplified interface
- **Automatic data handling** - No manual data preparation
- **Better error handling** - Clear, actionable error messages
- **Enhanced features** - Multi-objective optimization built-in
- **Improved performance** - Optimized data loading and processing

## Next Steps

1. **Try the examples**: Run `python examples_streamlined_calibration.py`
2. **Test with your data**: Use your own station IDs or data files
3. **Explore results**: Check the interactive plots and metrics
4. **Customize calibration**: Adjust parameters for your specific needs

## Support

For questions or issues:
- Check the generated log files in your workspace
- Review the validation diagnostics in the results
- Examine the error messages for specific guidance
- Use the test scripts to verify your setup
