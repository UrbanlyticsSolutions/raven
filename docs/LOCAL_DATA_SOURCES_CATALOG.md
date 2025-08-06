# Local Data Sources Catalog - RAVEN System

## Overview
This document catalogs all **locally available data sources** in the RAVEN system that can operate **without external network dependencies**. These sources provide complete hydrological modeling capabilities using pre-loaded datasets.

---

## 1. Primary Local GeoPackage - Canadian Hydro Data

### **File Location**: `data/canadian/canadian_hydro.gpkg`

#### **Dataset Details**
- **Records**: 584 Canadian lakes
- **CRS**: EPSG:4326 (WGS84)
- **Geographic Coverage**: 
  - Longitude: -139.06° to -54.37°
  - Latitude: 42.30° to 81.95°
- **Status**: **Local - No Network Required**

#### **Comprehensive Attribute Catalog**

| **Attribute** | **Type** | **Description** | **Units** | **Range** |
|---------------|----------|-----------------|-----------|-----------|
| `lake_id` | INTEGER | Unique identifier | - | 1-584 |
| `lake_name` | TEXT | Official lake name | - | "Great Bear", "Great Slave", etc. |
| `lake_area_km2` | FLOAT | Surface water area | km² | 0.01 - 30,450.64 |
| `lake_depth_m` | FLOAT | Maximum depth | meters | 0.5 - 446.0 |
| `lake_volume_km3` | FLOAT | Total water volume | km³ | 0.001 - 2,236.0 |
| `shoreline_length_km` | FLOAT | Shoreline perimeter | km | 0.5 - 3,826.0 |
| `geometry` | POLYGON | Lake boundary geometry | - | Complete polygons |

#### **Top 10 Largest Lakes (Local Data)**
| **Lake Name** | **Area (km²)** | **Depth (m)** | **Volume (km³)** |
|---------------|----------------|---------------|------------------|
| Great Bear | 30,450.64 | 72.2 | 2,236.00 |
| Great Slave | 26,734.29 | 59.1 | 1,580.00 |
| Winnipeg | 23,923.04 | 11.9 | 284.00 |
| Athabasca | 7,850.00 | 124.0 | 204.00 |
| Reindeer | 6,330.00 | 219.0 | 95.00 |
| Nettilling | 5,542.00 | 132.0 | 51.00 |
| Smallwood Reservoir | 5,430.00 | 75.0 | 35.00 |
| Dubawnt | 3,833.00 | 73.0 | 31.00 |
| Lake of the Woods | 3,846.00 | 64.0 | 19.00 |
| Mistassini | 2,335.00 | 183.0 | 82.00 |

---

## 2. Global Lake Dataset - HydroLAKES

### **File Location**: `data/canadian/lakes/HydroLAKES_polys_v10_shp/`

#### **Dataset Details**
- **Records**: 1,427,688 global lakes
- **CRS**: EPSG:4326
- **Status**: ✅ **Local - No Network Required**

#### **Comprehensive Attribute Catalog**

| **Attribute** | **Type** | **Description** | **Units** | **Sample Values** |
|---------------|----------|-----------------|-----------|-------------------|
| `Hylak_id` | INTEGER | Unique lake identifier | - | 1-1,427,688 |
| `Lake_name` | TEXT | Lake name | - | "Lake Superior", "Lake Victoria" |
| `Country` | TEXT | Country/territory | - | "Canada", "United States" |
| `Lake_area` | FLOAT | Surface area | km² | 0.1 - 82,103.0 |
| `Shore_len` | FLOAT | Shoreline length | km | 0.1 - 7,821.0 |
| `Vol_total` | FLOAT | Total volume | km³ | 0.001 - 12,070.0 |
| `Depth_avg` | FLOAT | Average depth | m | 0.1 - 1,470.0 |
| `Dis_avg` | FLOAT | Average discharge | m³/s | 0.1 - 16,790.0 |
| `Elevation` | FLOAT | Surface elevation | m | -415.0 - 6,893.0 |
| `Wshd_area` | FLOAT | Watershed area | km² | 0.1 - 1,580,000.0 |
| `Pour_long` | FLOAT | Outlet longitude | degrees | -180.0 to 180.0 |
| `Pour_lat` | FLOAT | Outlet latitude | degrees | -90.0 to 90.0 |

---

## 3. Global River Network - HydroRIVERS

### **File Location**: `data/canadian/rivers/HydroRIVERS_v10_na_shp/`

#### **Dataset Details**
- **Records**: 986,463 rivers (North America subset)
- **CRS**: EPSG:4326
- **Status**: **Local - No Network Required**

#### **Comprehensive Attribute Catalog**

| **Attribute** | **Type** | **Description** | **Units** | **Range** |
|---------------|----------|-----------------|-----------|-----------|
| `HYRIV_ID` | INTEGER | Unique river identifier | - | 1-986,463 |
| `NEXT_DOWN` | INTEGER | Downstream river ID | - | 0-986,462 |
| `LENGTH_KM` | FLOAT | River segment length | km | 0.1 - 6,275.0 |
| `CATCH_SKM` | FLOAT | Catchment area | km² | 0.01 - 2,980,000.0 |
| `DIS_AV_CMS` | FLOAT | Average discharge | m³/s | 0.001 - 16,790.0 |
| `ORD_STRA` | INTEGER | Strahler order | - | 1-10 |
| `HYBAS_L12` | INTEGER | Basin level 12 identifier | - | 1-1,580,000 |

---

## 4. BasinMaker Test Data (Local Validation)

### **File Location**: `basinmaker-extracted/basinmaker-master/tests/testdata/`

#### **Watershed Structure Data**

##### **Catchment Files**
- **File**: `catchment_without_merging_lakes.shp`
- **Records**: 1,100 catchments
- **Attributes**: Catchment ID, area, geometry
- **Status**: **Local - Test Dataset**

##### **Final Catchment Files**
- **File**: `finalcat_info.shp`
- **Records**: 844 final catchments
- **Attributes**: Final HRU definitions, lake integration
- **Status**: **Local - Test Dataset**

##### **River Network Files**
- **File**: `river_without_merging_lakes.shp`
- **Records**: 896 river segments
- **Attributes**: River ID, length, connectivity
- **Status**: ✅ **Local - Test Dataset**

##### **Lake Classification Files**
- **File**: `sl_connected_lake.shp`
- **Records**: 164 connected lakes
- **Attributes**: Lake ID, area, connectivity status
- **Status**: ✅ **Local - Test Dataset**

- **File**: `sl_non_connected_lake.shp`
- **Records**: 204 non-connected lakes
- **Attributes**: Lake ID, area, isolation status
- **Status**: ✅ **Local - Test Dataset**

##### **Observation Network**
- **File**: `obs_gauges.shp`
- **Records**: 31 observation gauges
- **Attributes**: Gauge ID, coordinates, measurement type
- **Status**: ✅ **Local - Test Dataset**

#### **Raster Data Files**
- **DEM_big_merit.tif**: High-resolution digital elevation model
- **DEM_small_merit.tif**: Lower-resolution elevation model
- **landuse_modis_250.tif**: Land use classification (MODIS 250m)
- **oih_30_dem.tif**: Alternative 30m DEM
- **hyshed_90_dem.tif**: 90m resolution DEM for testing

#### **Classification Tables**
- **landuse_info.csv**: Land use type definitions and codes
- **soil_info.csv**: Soil classification types and properties
- **veg_info.csv**: Vegetation type classifications

---

## 5. Model Configuration Templates

### **File Location**: `models/`

#### **Available Templates**
- **GR4JCN_template.yaml**: GR4J-CN hydrological model
- **HBVEC_template.yaml**: HBV-EC conceptual model
- **HMETS_template.yaml**: HMETS conceptual model
- **UBCWM_template.yaml**: UBC watershed model

#### **Template Attributes**
| **Template** | **Model Type** | **Parameters** | **Time Step** | **Climate Inputs** |
|--------------|----------------|----------------|---------------|-------------------|
| GR4JCN | Conceptual | 4 parameters | Daily | Precipitation, Temperature |
| HBVEC | Conceptual | 13 parameters | Daily | Precipitation, Temperature, PET |
| HMETS | Conceptual | 21 parameters | Daily | Precipitation, Temperature, PET |
| UBCWM | Conceptual | 17 parameters | Daily | Precipitation, Temperature, Snow |

---

## 6. Local-Only Workflow Capabilities

### **Completely Offline Operations**
✅ **Lake Detection**: Using local HydroLAKES data
✅ **River Network**: Using local HydroRIVERS data
✅ **Watershed Delineation**: Using local DEM and BasinMaker tools
✅ **Soil Classification**: Using coordinate-based regional classification
✅ **Land Cover**: Using predefined classification schemes
✅ **Model Configuration**: Using local YAML templates
✅ **Gauge Validation**: Using local hydrometric GeoPackage

### **Hybrid Operations** (Local + Network Fallback)
⚠️ **DEM Download**: Can pre-download for study area
⚠️ **Climate Data**: Requires external APIs for real-time
⚠️ **Streamflow**: Requires external APIs for observations

---

## 7. Data Quality Assessment

### **Local Data Completeness**
| **Data Type** | **Completeness** | **Coverage** | **Resolution** |
|---------------|------------------|--------------|----------------|
| **Lakes** | ✅ Excellent | Global | 30m-1km |
| **Rivers** | ✅ Excellent | Global | 90m-1km |
| **Watersheds** | ✅ Good | Regional | 30m-90m |
| **Soils** | ✅ Good | Regional | Coordinate-based |
| **Land Cover** | ✅ Good | Global | 250m-1km |
| **Elevation** | ⚠️ Variable | Regional | 30m-90m |

### **Network Dependencies**
| **Operation** | **Local** | **External** |
|---------------|-----------|--------------|
| **Lake/River Analysis** | ✅ Complete | ❌ Not Required |
| **Watershed Delineation** | ✅ Available | ⚠️ Optional |
| **Soil Classification** | ✅ Complete | ❌ Not Required |
| **Model Templates** | ✅ Complete | ❌ Not Required |
| **Climate Data** | ❌ Required | ✅ External |
| **Real-time Flow** | ❌ Required | ✅ External |

---

## 8. Usage Recommendations for Offline Workflows

### **Recommended Local-Only Workflow**
1. **Pre-download** USGS 3DEP DEM for study area
2. **Use local** HydroLAKES/HydroRIVERS for lake/river analysis
3. **Apply** coordinate-based soil classification
4. **Utilize** predefined land cover classes
5. **Validate** outlets using local hydrometric GeoPackage
6. **Generate** complete RAVEN models using local templates

This provides **complete hydrological modeling capability** without external network dependencies, with only climate forcing data requiring external sources.