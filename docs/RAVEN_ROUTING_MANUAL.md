# RAVEN Hydrological Framework - Routing Manual

## Overview

**RAVEN** (University of Waterloo) is a flexible hydrological modeling framework with extensive routing capabilities. Based on research and documentation from raven.uwaterloo.ca and associated repositories.

## Routing Framework

RAVEN supports **80+ interchangeable routing options** for maximum flexibility in model configuration and supports testing of trillions of possible model configurations.

### Spatial Discretization
- **Subbasins**: Collections of hydrological response units (HRUs)
- **HRUs**: Relatively homogeneous land parcels with unique hydrologic signatures
- **Flow Distribution**: Vertical within HRUs, lateral redistribution via routing
- **Stream Network**: Discretized into reaches for flow routing

## CatchmentRoute Options

### Basic Routing Methods

```
:CatchmentRoute TRIANGULAR_UH
:CatchmentRoute GAMMA_UH
:CatchmentRoute NASH_CASCADE
:CatchmentRoute LINEAR_RESERVOIR
```

### Unit Hydrograph Methods

**TRIANGULAR_UH**: Triangular unit hydrograph
- Simple, computationally efficient
- Good for lumped/semi-distributed models
- Requires time to peak parameter

**GAMMA_UH**: Gamma distribution unit hydrograph
- More flexible shape than triangular
- Better representation of natural catchment response

**NASH_CASCADE**: Nash cascade of linear reservoirs
- Represents catchment as series of linear reservoirs
- Good physical interpretation

### Advanced Routing Methods

```
:CatchmentRoute ROUTE_DELAYED_FIRST_ORDER
:CatchmentRoute ROUTE_DUMP
:CatchmentRoute ROUTE_NONE
```

## Channel Routing Options

### :Routing Command Options

```
:Routing ROUTE_DIFFUSIVE_WAVE
:Routing ROUTE_KINEMATIC_WAVE  
:Routing ROUTE_MUSKINGUM
:Routing ROUTE_MUSKINGUM_CUNGE
:Routing ROUTE_HYDROLOGIC
:Routing ROUTE_DELAYED_FIRST_ORDER
:Routing ROUTE_NONE
```

### Routing Method Details

**ROUTE_DIFFUSIVE_WAVE**:
- Solves diffusive wave equation
- Good for complex river networks
- Handles backwater effects
- Higher computational cost

**ROUTE_KINEMATIC_WAVE**:
- Simplified wave routing
- No backwater effects
- Faster computation
- Good for steep channels

**ROUTE_MUSKINGUM**:
- Classic Muskingum routing
- Requires K and X parameters
- Well-established method

**ROUTE_HYDROLOGIC**:
- Simple hydrologic routing
- Unit hydrograph approach
- Fast computation

## Model Configuration Examples

### Basic HBV-EC with TRIANGULAR_UH

```
# Basic HBV-EC Configuration
:StartDate        2000-01-01 00:00:00
:EndDate          2020-12-31 00:00:00
:TimeStep         1.0
:Method           ORDERED_SERIES

:SoilModel        SOIL_TWO_LAYER

# Routing Configuration
:CatchmentRoute   TRIANGULAR_UH
:Routing          ROUTE_NONE

# HBV-EC Parameters
:GlobalParameter FC 300.0
:GlobalParameter BETA 2.0
:GlobalParameter LP 0.7
:GlobalParameter K0 0.3
:GlobalParameter K1 0.05
:GlobalParameter K2 0.01

# Evaluation
:EvaluationMetrics NASH_SUTCLIFFE RMSE LOG_NASH
```

### Advanced Configuration with Channel Routing

```
# Advanced Configuration with Channel Routing
:StartDate        2000-01-01 00:00:00
:EndDate          2020-12-31 00:00:00
:TimeStep         1.0
:Method           ORDERED_SERIES

:SoilModel        SOIL_TWO_LAYER

# Advanced Routing
:CatchmentRoute   TRIANGULAR_UH
:Routing          ROUTE_DIFFUSIVE_WAVE

# Channel Properties (defined in RVP file)
# :ChannelProfile sections with geometry and roughness

# Model Parameters
:GlobalParameter FC 300.0
:GlobalParameter BETA 2.0
# ... other parameters

:EvaluationMetrics NASH_SUTCLIFFE RMSE LOG_NASH VOLUME_ERROR
```

### Distributed Model Configuration

```
# Distributed Model with TIN Routing
:StartDate        2000-01-01 00:00:00
:EndDate          2020-12-31 00:00:00
:TimeStep         1.0
:Method           ORDERED_SERIES

:SoilModel        SOIL_TWO_LAYER

# Distributed Routing
:CatchmentRoute   TRIANGULATED_IRREGULAR_NETWORK
:Routing          ROUTE_DIFFUSIVE_WAVE

# Subbasin network defined in RVH file
# Channel profiles in RVP file

:EvaluationMetrics NASH_SUTCLIFFE RMSE LOG_NASH
```

## Routing Parameters

### Unit Hydrograph Parameters

**TRIANGULAR_UH Parameters**:
```
:UnitHydrographPar TIME_TO_PEAK [days]
:UnitHydrographPar TIME_CONC [days]
```

**NASH_CASCADE Parameters**:
```
:UnitHydrographPar NASH_N [reservoirs]
:UnitHydrographPar NASH_K [days]
```

### Channel Routing Parameters

**Muskingum Parameters**:
```
:ChannelPar MUSKINGUM_K [days]
:ChannelPar MUSKINGUM_X [dimensionless]
```

**Wave Routing Parameters**:
```
:ChannelPar MANNINGS_N [roughness]
:ChannelPar BED_SLOPE [m/m]
```

## Valid Routing Combinations

### Recommended Combinations

**Simple Lumped Model**:
- CatchmentRoute: TRIANGULAR_UH
- Routing: ROUTE_NONE

**Semi-Distributed Model**:
- CatchmentRoute: TRIANGULAR_UH
- Routing: ROUTE_HYDROLOGIC

**Distributed Model**:
- CatchmentRoute: TRIANGULATED_IRREGULAR_NETWORK
- Routing: ROUTE_DIFFUSIVE_WAVE

**Complex Channel Network**:
- CatchmentRoute: GAMMA_UH
- Routing: ROUTE_MUSKINGUM

## Common Configuration Errors

### Invalid Routing Method
```
# INCORRECT - ROUTE_HYDROLOGIC may not be valid in some versions
:CatchmentRoute ROUTE_HYDROLOGIC

# CORRECT - Use proper catchment routing
:CatchmentRoute TRIANGULAR_UH
```

### Missing Channel Properties
```
# If using ROUTE_DIFFUSIVE_WAVE, ensure RVP file has:
:ChannelProfile
  :SurveyPoints
  :RoughnessZones (with non-zero Manning's n)
:EndChannelProfile
```

### Invalid Parameter Values
```
# Ensure Manning's n > 0 in RoughnessZones
:RoughnessZones
  0.0  100.0  0.035  # n = 0.035 (valid)
:EndRoughnessZones
```

## Model Selection Guidelines

### Choose CatchmentRoute Based on:

**TRIANGULAR_UH**:
- ✅ Simple watersheds
- ✅ Limited data available
- ✅ Fast computation needed
- ✅ Lumped/semi-distributed models

**GAMMA_UH**:
- ✅ More complex catchment response needed
- ✅ Better fit to observed data
- ✅ Semi-distributed models

**NASH_CASCADE**:
- ✅ Physical interpretation important
- ✅ Parameter regionalization
- ✅ Research applications

**TRIANGULATED_IRREGULAR_NETWORK**:
- ✅ Distributed models
- ✅ Complex topography
- ✅ Detailed spatial analysis

### Choose Routing Based on:

**ROUTE_NONE**:
- ✅ Lumped models
- ✅ Small watersheds
- ✅ At-outlet modeling

**ROUTE_HYDROLOGIC**:
- ✅ Simple channel routing
- ✅ Data-limited situations
- ✅ Fast computation

**ROUTE_DIFFUSIVE_WAVE**:
- ✅ Complex river networks
- ✅ Backwater effects important
- ✅ Detailed channel representation

**ROUTE_MUSKINGUM**:
- ✅ Well-characterized channels
- ✅ Parameter transfer from other studies
- ✅ Operational forecasting

## File Structure Requirements

### RVI File (Model Input)
- Routing method specifications
- Global parameters
- Evaluation metrics

### RVP File (Properties)
- Channel profiles (if using channel routing)
- Soil/vegetation properties
- Manning's roughness values

### RVH File (Hydrologic)
- Subbasin network topology
- HRU definitions
- Channel connectivity

### RVT File (Time Series)
- Meteorological forcing
- Observed streamflow
- Other time series data

## Troubleshooting

### Common Error Messages

**"Unrecognized catchment routing method"**
- Check spelling of routing method
- Ensure method is supported in your RAVEN version
- Use TRIANGULAR_UH as safe default

**"Manning's n values must be greater than zero"**
- Check RVP file RoughnessZones
- Ensure all Manning's n > 0.0
- Typical values: 0.025-0.1

**"Invalid channel profile"**
- Verify SurveyPoints geometry
- Check that cross-sections are valid
- Ensure left-to-right ordering

## Version Compatibility

### RAVEN 3.x
- TRIANGULAR_UH: ✅
- GAMMA_UH: ✅
- NASH_CASCADE: ✅
- ROUTE_DIFFUSIVE_WAVE: ✅
- ROUTE_MUSKINGUM: ✅

### RAVEN 4.0 (Latest)
- All previous methods: ✅
- Enhanced routing options: ✅
- Improved error handling: ✅
- NetCDF support: ✅

## References

- Raven Hydrological Framework: https://raven.uwaterloo.ca/
- Craig, J.R., et al. (2020). Flexible watershed simulation with the Raven hydrological modelling framework
- RAVEN User's and Developer's Manual (v3.8, v4.0)
- BasinMaker routing products: http://hydrology.uwaterloo.ca/basinmaker/

---

*This manual is based on RAVEN documentation and research from the University of Waterloo.*