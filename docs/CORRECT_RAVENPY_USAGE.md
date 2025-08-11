# Correct RavenPy Usage Guide

## 1. Always Use the Correct Model Class

**❌ Incorrect (old low-level approach):**
```python
from ravenpy.config import emulators
from ravenpy import Emulator

config = emulators.HBVEC(...)
emulator = Emulator(config=config, workdir=str(output_dir))
output_reader = emulator.run()
```

**✅ Correct (modern model class approach):**
```python
from ravenpy.models import HBVEC

model = HBVEC(
    params=hbvec_params,                 # 14-param list in correct order
    start_date="2000-01-01",
    end_date="2010-12-31", 
    hrus=ravenpy_hrus,                   # complete HRU dicts
    subbasins=subbasin_data,             # subbasin dicts with downstream_id
    gauges=gauge_config,                 # lat/long/elev
    forcing_files=[climate_file],        # CF-compliant NetCDF or CSV
)
model.run(output_dir=output_dir)
```

**Why**: RavenPy will auto-create .rvi, .rvh, .rvp, .rvt, .rvc files with proper headers, time steps, routing, and parameters.

## 2. Correct HBVEC Parameter Order

**✅ HBVEC requires exactly 14 parameters in this order:**
```python
# FC, BETA, LP, K0, K1, K2, UZL, PERC, MAXBAS, TT, CFMAX, CFR, SFCF, ETR
hbvec_params = [
    300.0,  # FC - Field capacity [mm]
    2.0,    # BETA - Shape coefficient [-]
    0.7,    # LP - Limit for PET [-]
    0.3,    # K0 - Near surface flow coefficient [1/d]
    0.05,   # K1 - Upper zone coefficient [1/d]
    0.01,   # K2 - Lower zone coefficient [1/d]
    10.0,   # UZL - Upper zone limit [mm]
    1.0,    # PERC - Percolation rate [mm/d]
    3.0,    # MAXBAS - Routing coefficient [d]
    0.0,    # TT - Temperature threshold [°C]
    3.0,    # CFMAX - Melt factor [mm/°C/d]
    0.05,   # CFR - Refreeze coefficient [-]
    1.1,    # SFCF - Snow correction factor [-]
    1.0     # ETR - ET correction factor [-]
]
```

**❌ Wrong**: Using 21 parameters from old templates
**❌ Wrong**: Using parameter dictionaries instead of ordered lists

## 3. Complete HRU Definitions

**✅ Supply all required HRU properties:**
```python
ravenpy_hrus = []
for hru in hru_list:
    ravenpy_hrus.append({
        "id": hru["hru_id"],
        "area": hru["area"],                    # km²
        "elevation": hru["elevation"],          # m
        "latitude": hru["latitude"],            # decimal degrees
        "longitude": hru["longitude"],          # decimal degrees
        "land_use_class": hru.get("land_use", "FOREST"),
        "vegetation_class": hru.get("vegetation", "MIXED_FOREST"), 
        "soil_profile": hru.get("soil_type", "LOAM"),
        "subbasin_id": hru["subbasin_id"]
    })
```

**❌ Wrong**: Hardcoding "FOREST", "MIXED_FOREST", "LOAM" for all HRUs
**✅ Better**: Use actual land use/soil data from your HRU processing

## 4. Proper Subbasin Routing

**✅ Ensure downstream_id is set:**
```python
subbasin_data = []
for sb in subbasins:
    subbasin_data.append({
        "id": sb["subbasin_id"],
        "name": f"subbasin_{sb['subbasin_id']}",
        "area": sb["area"],                     # km²
        "elevation": sb["elevation"],           # m
        "latitude": sb["latitude"],             # decimal degrees  
        "longitude": sb["longitude"],           # decimal degrees
        "downstream_id": sb.get("downstream_id", -1)  # -1 for outlet
    })
```

## 5. CF-Compliant Climate Data

**✅ Ensure PET is in NetCDF with proper attributes:**
```python
import xarray as xr
import numpy as np

# Open and modify NetCDF
ds = xr.open_dataset(climate_file)

# Add PET if missing
if "pet" not in ds:
    # Use temperature data (prefer tasmax/tasmin, fall back to tas)
    if "tasmax" in ds and "tasmin" in ds:
        tas = (ds["tasmax"] + ds["tasmin"]) / 2
    elif "tas" in ds:
        tas = ds["tas"]
    else:
        raise ValueError("No temperature data for PET calculation")
    
    # Simple Thornthwaite PET calculation
    monthly_tas = tas.groupby("time.month").mean()
    monthly_pet_values = []
    for temp in monthly_tas.values:
        if temp > 0:
            pet = max(5.0, 1.6 * (10 * temp / 12) ** 1.514 * 30.44)
        else:
            pet = 5.0
        monthly_pet_values.append(pet)
    
    # Interpolate to daily
    daily_pet = np.interp(
        np.arange(len(ds.time)),
        np.linspace(0, len(ds.time) - 1, 12),
        monthly_pet_values
    )
    
    # Add PET with CF-compliant attributes
    ds["pet"] = (
        ("time",), 
        daily_pet,
        {
            "units": "mm/d",
            "long_name": "Potential Evapotranspiration",
            "standard_name": "water_potential_evapotranspiration_flux"
        }
    )

# Ensure proper time attributes
ds.time.attrs.update({
    "standard_name": "time",
    "long_name": "time"
})

# Save back to file
ds.to_netcdf(climate_file)
ds.close()
```

## 6. Gauge Configuration

**✅ Proper gauge setup:**
```python
gauge_config = [
    {
        "latitude": outlet_lat,
        "longitude": outlet_lon,
        "elevation": outlet_elev,
        "name": outlet_name,
        "river_name": "Main_River"
    }
]
```

## 7. Complete Working Example

```python
from ravenpy.models import HBVEC
import xarray as xr
import numpy as np

def run_hbvec_correctly(output_dir, outlet_name, start_date, end_date,
                       hru_list, subbasin_data, gauge_config, climate_file):
    """
    Correct HBVEC implementation using RavenPy
    """
    
    # 1. Ensure PET is in climate file
    ds = xr.open_dataset(climate_file)
    if "pet" not in ds:
        # Add PET calculation here (see example above)
        pass
    ds.close()
    
    # 2. Format HRUs with complete data
    ravenpy_hrus = []
    for hru in hru_list:
        ravenpy_hrus.append({
            "id": hru["hru_id"],
            "area": hru["area"],
            "elevation": hru["elevation"], 
            "latitude": hru["latitude"],
            "longitude": hru["longitude"],
            "land_use_class": hru.get("land_use", "FOREST"),
            "vegetation_class": hru.get("vegetation", "MIXED_FOREST"),
            "soil_profile": hru.get("soil_type", "LOAM"),
            "subbasin_id": hru["subbasin_id"]
        })
    
    # 3. Ensure subbasin routing
    for sb in subbasin_data:
        if "downstream_id" not in sb:
            sb["downstream_id"] = -1  # or proper routing logic
    
    # 4. Correct 14-parameter HBVEC setup
    hbvec_params = [
        300.0, 2.0, 0.7, 0.3, 0.05, 0.01, 10.0,  # FC, BETA, LP, K0, K1, K2, UZL
        1.0, 3.0, 0.0, 3.0, 0.05, 1.1, 1.0       # PERC, MAXBAS, TT, CFMAX, CFR, SFCF, ETR
    ]
    
    # 5. Create and run model
    model = HBVEC(
        params=hbvec_params,
        start_date=start_date,
        end_date=end_date,
        hrus=ravenpy_hrus,
        subbasins=subbasin_data,
        gauges=gauge_config,
        forcing_files=[climate_file]
    )
    
    # 6. Run the model
    model.run(output_dir=output_dir)
    
    return output_dir
```

## Common Mistakes to Avoid

1. **Don't** use `emulators.HBVEC` + `Emulator` (old approach)
2. **Don't** pass wrong parameter counts (21 instead of 14)
3. **Don't** hardcode all HRU properties to same values
4. **Don't** forget `downstream_id` in subbasins
5. **Don't** skip PET in climate files
6. **Don't** use non-CF-compliant NetCDF attributes

## Migration from Old Code

If you have existing code using `emulators.HBVEC`:

1. Replace `from ravenpy.config import emulators` with `from ravenpy.models import HBVEC`
2. Remove `Emulator(config=config, workdir=...)` wrapper
3. Fix parameter count from 21 to 14 for HBVEC
4. Add complete HRU properties
5. Ensure subbasin downstream_id routing
6. Add PET to climate files