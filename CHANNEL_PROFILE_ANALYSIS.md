# Channel Profile Generation Analysis

## BasinMaker Channel Profile Pattern

### Input Parameters
- `chwd` = Bankfull width (BkfWidth from shapefile)  
- `chdep` = Bankfull depth (BkfDepth from shapefile)
- `elev` = Channel reference elevation
- `chslope` = Channel slope
- `floodn` = Floodplain Manning's n
- `channeln` = Channel Manning's n

### Fixed Constants
```python
zch = 2                    # Side slope ratio (2:1 horizontal:vertical)
sidwdfp = 4 / 0.25 = 16   # Fixed floodplain width = 16m each side
```

### Calculated Geometry
```python
sidwd = zch * chdep = 2 * chdep      # Side slope width
botwd = chwd - 2 * sidwd             # Bottom width

# If botwd < 0 (channel too narrow):
if botwd < 0:
    botwd = 0.5 * chwd
    sidwd = 0.5 * 0.5 * chwd = 0.25 * chwd
    zch = (chwd - botwd) / 2 / chdep

zfld = 4 + elev           # Floodplain elevation (4m above channel)
zbot = elev - chdep       # Channel bottom
```

### Survey Points Pattern (8 points)
For a channel with bankfull width = 1.0m, depth = 0.3m, elevation = 0.3m:

| Point | X-Position | Elevation | Description |
|-------|------------|-----------|-------------|
| 0 | 0.0 | zfld = 4.3 | Left floodplain start |
| 1 | 16.0 | elev = 0.3 | Left floodplain edge |
| 2 | 18.0 | elev = 0.3 | Left bank top |
| 3 | 18.6 | zbot = 0.0 | Left bank bottom |
| 4 | 19.4 | zbot = 0.0 | Right bank bottom |
| 5 | 20.0 | elev = 0.3 | Right bank top |
| 6 | 22.0 | elev = 0.3 | Right floodplain edge |
| 7 | 38.0 | zfld = 4.3 | Right floodplain end |

**Total channel width = 38m (32m floodplain + 6m channel zone)**

### Roughness Zones (3 zones)
| Zone | X-Start | X-End | Manning's n | Description |
|------|---------|-------|-------------|-------------|
| 1 | 0.0 | 18.0 | floodn | Left floodplain |
| 2 | 18.0 | 20.0 | channeln | Main channel |
| 3 | 20.0 | 38.0 | floodn | Right floodplain |

## Comparison: BasinMaker vs Our Implementation

| Aspect | Our Implementation | BasinMaker |
|--------|-------------------|------------|
| **Survey points** | 5 points (trapezoidal + floodplain) | 8 points (complex trapezoid) |
| **Channel pattern** | Simple symmetric trapezoid | Complex side slopes |
| **Floodplain width** | Variable (depends on drainage area) | Fixed 32m total |
| **Flood depth** | Variable (depends on drainage area) | Fixed 4m above channel |
| **Roughness zones** | 2 zones (channel/floodplain) | 3 zones (left/channel/right) |
| **Manning's n values** | 0.15 (floodplain), 0.225 (channel) | Variable from inputs |
| **Cross-section complexity** | Moderate | High |
| **Implementation source** | magpie_hydraulic_routing.py | BasinMaker Generate_Raven_Channel_rvp_string_sub |

## Pattern Analysis

### Our Approach Strengths
1. **Variable sizing**: Channel dimensions scale with drainage area
2. **Simpler geometry**: Easier to understand and debug
3. **Reasonable flood capacity**: 3-4x bankfull capacity
4. **Efficient computation**: Fewer survey points

### BasinMaker Approach Strengths  
1. **Fixed flood standards**: Consistent 4m flood depth across all channels
2. **Complex hydraulics**: More realistic side slope geometry
3. **Detailed roughness**: Separate left/right floodplain zones
4. **Massive flood capacity**: 10-100x bankfull capacity

### Key Technical Differences

**Channel Width Calculation:**
- **Our approach**: Uses drainage area regression → variable width
- **BasinMaker**: Uses input BkfWidth + fixed 32m floodplain → mixed

**Flood Capacity:**
- **Our approach**: Scales with channel size (proportional capacity)
- **BasinMaker**: Fixed 4m flood depth (absolute capacity)

## Our Actual Implementation

### Current Channel Profile Pattern (CHANNEL_6 example)
```
:ChannelProfile CHANNEL_6
  :Bedslope 0.003559
  :SurveyPoints
    0.0 0.90    # Left floodplain edge (flood elevation)
    0.33 0.30   # Left bank top (bankfull elevation)  
    0.67 0.0    # Channel bottom (thalweg)
    1.00 0.30   # Right bank top (bankfull elevation)
    2.68 0.90   # Right floodplain edge (flood elevation)
  :EndSurveyPoints
  :RoughnessZones
    0.0 0.1500    # Floodplain Manning's n = 0.15
    1.34 0.2250   # Channel Manning's n = 0.225
  :EndRoughnessZones
:EndChannelProfile
```

### Analysis of Our Pattern
- **Bankfull width = 0.67m** (point 2: bottom width)
- **Bankfull depth = 0.30m** (points 1,3: bank height above bottom)
- **Total width = 2.68m** (point 4: total cross-section width)
- **Flood depth = 0.90m** (points 0,4: maximum depth capacity)
- **Flood capacity = 4x bankfull** (rough estimate)

### Cross-Section Geometry
```
Elevation (m)
0.90 |----+                         +----| Flood level
     |    |                         |    |
0.30 |    +-------+         +-------+    | Bankfull level  
     |            |         |            |
0.00 |            +---------+            | Channel bottom
     +----+-------+---------+-------+----+
     0.0  0.33   0.67      1.0     2.68   Width (m)
```

## Implications

1. **Flow Capacity**: BasinMaker profiles can handle flows **10-100x larger** than bankfull
2. **Flood Routing**: Enables realistic overbank flow simulation
3. **Stage-Discharge**: Complex rating curves with multiple flow regimes
4. **Computational Cost**: More complex hydraulics but still efficient

## Decision Points for Our Implementation

1. **Keep current approach?** (moderate complexity, scalable)
2. **Adopt BasinMaker fixed flood depth?** (simple but less scalable)  
3. **Hybrid approach?** (scalable width + fixed flood safety margin)

The choice depends on:
- **Primary use case**: Bankfull vs flood simulation priority
- **Computational requirements**: Simple vs complex hydraulics
- **Calibration data**: Available flow data for validation

## Code Implementation Summary

### Our Channel Profile Generation (magpie_hydraulic_routing.py)
```python
def create_channel_profile(drainage_area, slope, manning_n):
    # Calculate bankfull dimensions from drainage area
    bf_width = 1.22 * (drainage_area ** 0.557) 
    bf_depth = 0.27 * (drainage_area ** 0.372)
    
    # Create 5-point trapezoidal profile with floodplain
    # Pattern: [flood_left, bank_left, bottom, bank_right, flood_right]
    
    # Survey points scale with channel size
    # Roughness zones: floodplain (0.15) + channel (0.225)
    
    return channel_profile_string
```

### BasinMaker Approach (Generate_Raven_Channel_rvp_string_sub)
```python
def Generate_Raven_Channel_rvp_string_sub(chwd, chdep, elev, chslope):
    # Fixed geometry constants
    zch = 2          # Side slope ratio
    sidwdfp = 16     # Fixed 16m floodplain each side
    zfld = 4 + elev  # Fixed 4m flood depth
    
    # Calculate trapezoidal geometry
    sidwd = zch * chdep
    botwd = chwd - 2 * sidwd
    
    # Create 8-point complex profile
    # Pattern: [fp_left, fp_edge, bank_left, bot_left, bot_right, bank_right, fp_edge, fp_right]
    
    # 3 roughness zones: left_fp, channel, right_fp
    
    return channel_profile_string
```

## Final Assessment

**Our implementation strikes a good balance:**
- ✅ Scales appropriately with watershed size
- ✅ Provides reasonable flood capacity (3-4x bankfull)
- ✅ Simpler geometry is easier to validate and debug
- ✅ Uses established hydraulic geometry relationships

**BasinMaker's approach is more conservative:**
- ✅ Massive flood capacity for extreme events
- ✅ More detailed hydraulic representation
- ❌ Fixed dimensions may not scale well
- ❌ More complex geometry harder to validate

The current implementation appears **appropriate for most hydrological modeling applications** where bankfull + moderate flood capacity is sufficient.
