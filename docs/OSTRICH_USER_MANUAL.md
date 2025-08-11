# OSTRICH User Manual for RAVEN Calibration

## Overview

**OSTRICH** (Optimization Software Toolkit for Research Involving Computational Heuristics) is a model-independent optimization tool for automated model calibration and parameter estimation. Version 17.12.19 is integrated with our RAVEN hydrological modeling workflow.

## Key Features

- ✅ **Model-Independent**: Works with any simulation model via template files
- ✅ **Multi-Objective Optimization**: Simultaneously optimize multiple criteria
- ✅ **Parallel Processing**: MPI support for faster optimization
- ✅ **Advanced Algorithms**: DDS, PSO, Genetic Algorithms, Simulated Annealing
- ✅ **Hydrological Focus**: Specialized for watershed model calibration

## Multi-Objective Calibration for RAVEN

### Calibration Objectives

Our RAVEN implementation optimizes **5 key objectives** simultaneously:

1. **Nash-Sutcliffe Efficiency (NSE)**: Overall model performance
2. **Volume Balance Score**: Water balance accuracy
3. **Low Flow NSE**: Baseflow and drought simulation
4. **Mean Flow Error**: Average flow accuracy  
5. **Peak Timing Error**: Flood prediction and timing

### Objective Weights

```
Volume Balance:    20% (0.20)
Low Flow NSE:      25% (0.25)  
Mean Flow Error:   20% (0.20)
Peak Timing:       15% (0.15)
Overall NSE:       20% (0.20)
```

## OSTRICH Input File (ostIn.txt)

### Basic Structure

```
ProgramType         DDS
ObjectiveFunction   WSSE
ModelSubdir         ./
ModelExecutable     "path/to/Raven.exe"
PreserveModelOutput yes
OstrichWarmStart    no

BeginFilePairs
model.rvi.tpl ; model.rvi
EndFilePairs

BeginExtraFiles
model.rvh
model.rvp
model.rvt
model.rvc
EndExtraFiles

BeginParams
# Parameter_Name  Lower_Bound  Upper_Bound  Initial_Value  Transform  Format
FC               100.0        500.0        300.0          none       none
BETA             1.0          5.0          2.0            none       none
LP               0.3          1.0          0.7            none       none
EndParams

BeginResponseVars
NSE  diagnostics.csv  ;  OST_NULL  2  2  ' '
EndResponseVars

BeginDDS
PerturbationValue  0.2
MaxIterations      100
EndDDS
```

### Key Sections Explained

#### 1. Program Configuration
- `ProgramType`: Algorithm type (DDS, PSO, GA, etc.)
- `ObjectiveFunction`: Optimization target (WSSE = minimize)
- `ModelExecutable`: Path to RAVEN executable

#### 2. File Management
- `BeginFilePairs`: Template → Model input file mapping
- `BeginExtraFiles`: Additional files needed by model

#### 3. Parameters
- Define calibration parameters with bounds
- `Transform`: Parameter transformation (none, log, etc.)
- `Format`: Output format specification

#### 4. Response Variables
- Define optimization targets from model output
- Links to CSV files containing performance metrics

## DDS Algorithm Configuration

### Basic DDS Settings

```
BeginDDS
PerturbationValue   0.2     # Search neighborhood size (0.1-0.3)
MaxIterations       100     # Maximum function evaluations
UseInitialParamValues       # Start from initial values
EndDDS
```

### Advanced DDS Options

```
BeginDDS
PerturbationValue   0.15
MaxIterations       500
UseRandomParamValues        # Start from random values
DDS_r              0.2      # Neighborhood reduction factor
EndDDS
```

### Parameter Guidelines

| Parameter | Range | Description | Recommendation |
|-----------|-------|-------------|----------------|
| PerturbationValue | 0.05-0.5 | Search radius | 0.2 for most cases |
| MaxIterations | 50-2000 | Function calls | 100-500 for RAVEN |
| DDS_r | 0.1-0.3 | Reduction rate | Default: 0.2 |

## Multi-Objective Optimization

### PADDS Algorithm (Multi-Objective DDS)

```
ProgramType         PADDS
BeginPADDS
PerturbationValue   0.2
MaxIterations       200
ArchiveSize         50      # Pareto archive size
EndPADDS

BeginResponseVars
NSE         diagnostics.csv  ;  OST_NULL  2  2  ' '
VOLUME_ERR  diagnostics.csv  ;  OST_NULL  3  3  ' '  
RMSE        diagnostics.csv  ;  OST_NULL  4  4  ' '
EndResponseVars
```

### Objective Function Weighting

```
BeginGCOP
CostFunction    COMPOSITE   # Use weighted sum
Minimize        NSE         1.0
Minimize        VOLUME_ERR  0.8
Minimize        RMSE        0.6
EndGCOP
```

## RAVEN-Specific Configuration

### HBV-EC Model Parameters

```
BeginParams
# HBV-EC Snow Parameters
TT      -2.0    2.0     0.0     none  none  # Temperature threshold
CFMAX   1.0     8.0     3.0     none  none  # Degree-day factor
CFR     0.0     0.1     0.05    none  none  # Refreezing factor

# HBV-EC Soil Parameters  
FC      100.0   500.0   300.0   none  none  # Field capacity
BETA    1.0     5.0     2.0     none  none  # Shape parameter
LP      0.3     1.0     0.7     none  none  # Limit for PET

# HBV-EC Routing Parameters
K0      0.1     0.8     0.3     none  none  # Quick flow recession
K1      0.01    0.3     0.05    none  none  # Slow flow recession  
K2      0.001   0.1     0.01    none  none  # Baseflow recession
EndParams
```

### Template File (.rvi.tpl)

```
# RAVEN Input Template
:StartDate        2000-01-01 00:00:00
:EndDate          2020-12-31 00:00:00

# Calibration Parameters (OSTRICH replaces these)
:GlobalParameter FC      FC
:GlobalParameter BETA    BETA
:GlobalParameter LP      LP
:GlobalParameter K0      K0
:GlobalParameter K1      K1
:GlobalParameter K2      K2

# Fixed Parameters
:GlobalParameter TT      0.0
:GlobalParameter CFMAX   3.0
:GlobalParameter CFR     0.05

:EvaluationMetrics NASH_SUTCLIFFE RMSE VOLUME_ERROR
```

## Best Practices

### 1. Parameter Bounds
- **Conservative bounds**: Start with literature values
- **Physical constraints**: Ensure parameters make physical sense
- **Scaling**: Normalize parameters if ranges differ greatly

### 2. Algorithm Selection
- **DDS**: Best for 5-20 parameters, fast convergence
- **PADDS**: Multi-objective problems
- **PSO**: Global optimization, more robust but slower

### 3. Convergence Settings
- **Start small**: 50-100 iterations for testing
- **Scale up**: 200-500 iterations for final calibration
- **Monitor progress**: Check objective function improvement

### 4. Multi-Objective Balancing
- **Equal weights**: Start with balanced objectives
- **Priority weighting**: Increase weights for critical metrics
- **Pareto analysis**: Use PADDS for trade-off analysis

## Performance Optimization

### Parallel Processing
```bash
# Run with MPI (4 processors)
mpiexec -np 4 OSTRICH.exe ostIn.txt
```

### Model Pre-emption
```
# Stop poor-performing runs early
BeginModelPreEmption  
MaxRuntime  300    # seconds
ThresholdValue  0.5
EndModelPreEmption
```

### Restart Capabilities
```
# Enable algorithm restarts
OstrichWarmStart  yes
RestartFileName   ostrich_restart.txt
```

## Output Files

| File | Description |
|------|-------------|
| OstModel0.txt | Best parameter set |
| OstOutput0.txt | Optimization progress |
| OstStatus0.txt | Algorithm status |
| OstErrors0.txt | Error messages |

## Troubleshooting

### Common Issues

1. **"Parameter bounds error"**
   - Check upper > lower bounds
   - Verify parameter names match template

2. **"Model executable not found"**
   - Use absolute paths for executables
   - Check file permissions

3. **"No improvement"**
   - Increase MaxIterations
   - Adjust PerturbationValue
   - Check parameter bounds

4. **"Convergence issues"**
   - Reduce PerturbationValue
   - Use UseInitialParamValues
   - Check objective function definition

### Debug Mode
```
# Enable detailed logging
BeginMathAndStats
WriteReproducibleResults
EnableOptimalObservations  
EndMathAndStats
```

## Integration with RAVEN Workflow

### Step 6 Calibration Process
1. **Data Preparation**: Observed streamflow loaded automatically
2. **Template Creation**: RAVEN .rvi template with parameter placeholders
3. **OSTRICH Execution**: Multi-objective optimization run
4. **Results Processing**: Best parameters applied to final model
5. **Validation**: Performance metrics calculated and plots generated

### Automated Features
- ✅ Hydrometric data auto-loading from workspace
- ✅ Multi-objective function calculation
- ✅ Parameter bounds validation
- ✅ Convergence monitoring
- ✅ Results visualization

## Example Configurations

### Quick Test (Fast)
```
BeginDDS
PerturbationValue  0.3
MaxIterations      25
EndDDS
```

### Production Run (Thorough)  
```
BeginDDS
PerturbationValue  0.15
MaxIterations      500
UseInitialParamValues
EndDDS
```

### Multi-Objective Analysis
```
ProgramType  PADDS
BeginPADDS
PerturbationValue  0.2
MaxIterations      300
ArchiveSize        100
EndPADDS
```

## References

- Matott, L.S. (2017). OSTRICH: an Optimization Software Tool, Documentation and User's Guide, Version 17.12.19
- Tolson, B.A. and Shoemaker, C.A. (2007). Dynamically dimensioned search algorithm for computationally efficient watershed model calibration. Water Resources Research, 43(1)
- Official Documentation: https://www.eng.buffalo.edu/~lsmatott/Ostrich/OstrichMain.html

---

*This manual is specifically adapted for RAVEN hydrological model calibration workflows.*