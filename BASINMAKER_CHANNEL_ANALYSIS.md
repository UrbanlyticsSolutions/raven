# BasinMaker Channel Profile Logic Beyond Bank Analysis

## Location
Found in: `basinmaker-extracted\basinmaker-master\basinmaker\hymodin\raveninput.py`
Function: `Generate_Raven_Channel_rvp_string_sub()` (lines ~890-1020)

## Key Logic That Goes Beyond Bank

### 1. Trapezoidal Channel Design (SWAT-based)
```python
# Following SWAT instructions, assume a trapezoidal shape channel, 
# with channel sides has depth and width ratio of 2. zch = 2
zch = 2                    # Side slope ratio (horizontal:vertical = 2:1)
sidwd = zch * chdep        # Side width = 2 * channel depth
botwd = chwd - 2 * sidwd   # Bottom width = total width - 2 * side widths
```

### 2. Geometric Constraints and Corrections
```python
if botwd < 0:
    botwd = 0.5 * chwd           # Ensure positive bottom width
    sidwd = 0.5 * 0.5 * chwd     # Recalculate side width
    zch = (chwd - botwd) / 2 / chdep  # Recalculate side slope ratio
```

### 3. Floodplain Extension Beyond Channel Banks
```python
zfld = 4 + elev              # Floodplain elevation = 4m above channel elevation
zbot = elev - chdep          # Channel bottom elevation
sidwdfp = 4 / 0.25           # Floodplain side width = 16m (fixed calculation)
```

### 4. Extended Survey Points Beyond Bankfull
The function creates 8 survey points that extend far beyond the bankfull channel:

1. **Point 0**: `(0, zfld)` - Start of left floodplain
2. **Point 1**: `(sidwdfp, elev)` - Left floodplain to channel transition  
3. **Point 2**: `(sidwdfp + 2*chwd, elev)` - Left bank top
4. **Point 3**: `(sidwdfp + 2*chwd + sidwd, zbot)` - Left bank bottom
5. **Point 4**: `(sidwdfp + 2*chwd + sidwd + botwd, zbot)` - Right bank bottom
6. **Point 5**: `(sidwdfp + 2*chwd + 2*sidwd + botwd, elev)` - Right bank top
7. **Point 6**: `(sidwdfp + 4*chwd + 2*sidwd + botwd, elev)` - Right floodplain transition
8. **Point 7**: `(2*sidwdfp + 4*chwd + 2*sidwd + botwd, zfld)` - End of right floodplain

### 5. Manning's Roughness Zones Beyond Banks
```python
# Zone 1: Floodplain (left side)
"    0" + tab + floodn

# Zone 2: Main channel 
"    " + str(sidwdfp + 2 * chwd) + tab + mann

# Zone 3: Floodplain (right side)  
"    " + str(sidwdfp + 2 * chwd + 2 * sidwd + botwd) + tab + floodn
```

## Key Findings

### 1. **Extensive Floodplain Modeling**
- Creates floodplains extending **16m on each side** beyond the channel
- Floodplain is elevated **4m above** the channel reference elevation
- Total channel width becomes: `2*sidwdfp + 4*chwd + 2*sidwd + botwd`
- Where `sidwdfp = 16m`, this means **32m of floodplain** plus channel width

### 2. **Fixed Floodplain Dimensions**
- The floodplain width is **hardcoded** as `4 / 0.25 = 16m` on each side
- Floodplain elevation is **hardcoded** as `4m` above channel elevation
- These are **not based on actual topography or flood studies**

### 3. **Channel Geometry Assumptions**
- Forces a **2:1 side slope** (based on SWAT methodology)
- Assumes **trapezoidal cross-section** regardless of actual channel shape
- Has **fallback logic** when bankfull width is too narrow for the assumed geometry

### 4. **Hydraulic Implications**
- **Overbank flow capacity**: Includes significant flow capacity beyond bankfull
- **Flood routing**: Can route large floods through the extended floodplain
- **Stage-discharge relationships**: Extends well beyond observed bankfull conditions

## Comparison with Our Implementation

Our current implementation in `magpie_hydraulic_routing.py` focuses on:
- Bankfull width and depth from watershed characteristics
- Simple hydraulic geometry relationships
- Manning's equation for velocity and flow capacity

BasinMaker goes further by:
- Adding extensive floodplain capacity
- Creating complex cross-sectional geometry
- Providing multiple roughness zones
- Enabling overbank flow modeling

## How BasinMaker Uses Bankfull Dimensions

In the main channel generation loop (lines ~1630-1650):

```python
if strRlen != "ZERO-":
    bkf_width = max(catinfo_sub["BkfWidth"].values[i], 1)    # Minimum 1m width
    bkf_depth = max(catinfo_sub["BkfDepth"].values[i], 1)    # Minimum 1m depth
else:
    bkf_width = 0.12345
    bkf_depth = 0.12345

# Call channel generation function
output_string_chn_rvp_sub = Generate_Raven_Channel_rvp_string_sub(
    pronam,        # Channel name
    bkf_width,     # Used as chwd parameter  
    bkf_depth,     # Used as chdep parameter
    chslope,       # Channel slope
    catinfo_sub["MeanElev"].values[i],  # Channel elevation
    floodn,        # Floodplain Manning's n
    nchn,          # Channel Manning's n
    iscalmanningn, # Use provided Manning's or default
)
```

### Key Points:
1. **Bankfull width becomes the reference channel width** (`chwd`)
2. **Bankfull depth becomes the reference channel depth** (`chdep`) 
3. **Minimum constraints**: Both width and depth have a 1m minimum
4. **The channel geometry extends far beyond these bankfull dimensions**

So BasinMaker uses bankfull as the **starting point** but then builds an extensive floodplain system around it.

## Comparison with Our Current Implementation

Our implementation in `magpie_hydraulic_routing.py`:
- Calculates bankfull width and depth from drainage area
- Uses these for Manning's equation flow capacity calculations
- **Stops at bankfull** - no floodplain extension

BasinMaker's approach:
- Takes bankfull width/depth as input (from shapefile attributes)
- **Extends far beyond bankfull** with complex cross-section
- Creates **32m of additional floodplain width** (16m each side)
- Provides **4m of additional flood depth** above channel
- Uses **different Manning's roughness** for channel vs. floodplain

## Recommendation

Consider whether we want to adopt BasinMaker's approach of:
1. **Adding floodplain capacity** beyond bankfull channel
2. **Using topographic data** to define floodplain extent (rather than fixed 16m)
3. **Creating multi-zone roughness** for channel vs. floodplain  
4. **Enabling overbank flow routing** for flood events

This would provide more realistic flood modeling but requires additional topographic analysis.

**Key Decision**: Do we want to limit our channel routing to **bankfull capacity only** (current approach) or extend to **flood capacity** (BasinMaker approach)?
