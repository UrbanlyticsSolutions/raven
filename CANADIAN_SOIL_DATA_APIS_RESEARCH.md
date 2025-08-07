# Canadian Soil Data APIs and Endpoints Research Results

## Executive Summary

This research identifies comprehensive soil data endpoints and APIs specifically for Canada that provide real soil properties like clay%, sand%, silt%, pH, organic matter, and other physical/chemical properties. The focus is on finding APIs that return actual soil property values rather than just classifications.

## 1. Government of Canada Soil Databases and APIs

### 1.1 Canadian Soil Information Service (CanSIS)
- **Primary URL**: https://sis.agr.gc.ca/cansis/
- **Description**: Authoritative source of soil data and land resource information for Canada
- **Coverage**: National coverage for all of Canada
- **Data Source**: National Soil Database (NSDB) - collection of geospatial datasets with soil, landscape, and climatic data
- **Available Properties**: Soil texture (clay, sand, silt percentages), organic matter, pH, drainage, parent material
- **Data Format**: FGDB, GeoJSON, CSV, Shapefile
- **Authentication**: None required for basic access
- **Usage Notes**: Primary federal soil data repository, serves as national archive
- **Contact**: aafc.cansis-siscan.aac@agr.gc.ca

### 1.2 Agriculture and Agri-Food Canada (AAFC) ESRI REST Services
- **Primary Endpoints**:
  - Canada Land Inventory (1:1,000,000): https://services.arcgis.com/lGOekm0RsNxYnT3j/arcgis/rest/services/cli_agr_cap_1M/FeatureServer
  - Canada Land Inventory (1:250,000): https://services.arcgis.com/lGOekm0RsNxYnT3j/arcgis/rest/services/cli_agr_cap_250k/FeatureServer
  - Annual Crop Inventory: https://agriculture.canada.ca/imagery-images/rest/services/annual_crop_inventory/[YEAR]/ImageServer
- **Available Properties**: Agricultural land capability, soil classifications, crop inventory
- **Data Format**: ESRI REST Services (JSON)
- **Authentication**: None required
- **Parameters**: Standard ESRI REST parameters (geometry, where, outFields, etc.)
- **Contact**: aafc.agri-geomatics-agrogeomatiques.aac@agr.gc.ca

### 1.3 Soil Landscapes of Canada (SLC)
- **Primary URL**: https://sis.agr.gc.ca/cansis/nsdb/slc/v3.2/index.html
- **Description**: Latest revision (v3.2) showing major soil and land characteristics
- **Coverage**: National coverage at 1:1 million scale
- **Available Properties**: Soil texture, drainage, parent material, landscape attributes
- **Data Format**: FGDB, GeoJSON, CSV, Shapefile
- **Resolution**: 1:1 million compilation scale
- **Usage**: Suitable for provincial and national level analysis

## 2. Provincial Soil Data Services

### 2.1 British Columbia
#### BC Soil Information Finder Tool (SIFT)
- **Primary URL**: https://www2.gov.bc.ca/gov/content/environment/air-land-water/land/soil/soil-information-finder
- **Description**: Interactive tool providing access to soil survey data, reports and maps
- **Coverage**: Province-wide with detailed coverage in Lower Fraser Valley
- **Available Properties**: Soil texture (sand, silt, clay %), pH, organic matter, drainage
- **Data Format**: Interactive viewer, downloadable packages in FGDB and SHP format
- **Resolution**: 1:50,000 scale for Lower Fraser Valley
- **Usage**: Farm-level and regional planning decisions

#### BC Data Catalogue Soil Services
- **Soil Survey Spatial Data**: https://catalogue.data.gov.bc.ca/dataset/soil-survey-spatial-view
- **Soil Mapping Data**: https://catalogue.data.gov.bc.ca/dataset/soil-mapping-data-packages
- **Available Properties**: Comprehensive soil properties including texture, chemistry, physics
- **Data Format**: FGDB, Shapefile, WMS services
- **License**: Open Government Licence - British Columbia

### 2.2 Alberta
#### Alberta Soil Information Viewer
- **Primary URL**: https://soil.agric.gov.ab.ca/
- **Coverage**: Agricultural regions of Alberta
- **Available Properties**: Organic soils distribution, soil texture (sand, silt, clay), parent material

#### Alberta Geospatial Services
- **ESRI REST Endpoint**: https://geospatial.alberta.ca/titan/rest/services/agriculture/Agricultural_Land_Resource_Atlas/MapServer
- **WMS Endpoint**: https://geospatial.alberta.ca/titan/services/agriculture/agricultural_lan
- **Available Properties**: Organic soils, soil texture distributions
- **Coverage**: Agricultural region of Alberta
- **Data Format**: WMS, ESRI REST Services

### 2.3 Ontario
#### Ontario Ministry of Agriculture (OMAFRA)
- **Description**: Provincial data custodian for soil information
- **Database**: Soil Survey Complex geospatial database
- **Coverage**: Seamless coverage from Windsor to Ottawa Valley
- **Resolution**: 1:50,000 scale
- **Available Properties**: Comprehensive soil properties
- **Data Size**: 201 MB downloadable dataset

### 2.4 Saskatchewan
#### Saskatchewan Geospatial Services
- **WMS/WMTS Services**: Available through Saskatchewan Geospatial Imagery Collaborative (SGIC)
- **Description**: OGC-Standard Web Map Service capabilities
- **Coverage**: Province-wide
- **Access**: Public viewing through Web Mapping Client

### 2.5 Manitoba
#### Manitoba Soil Data
- **Coverage**: Most of the province at multiple scales
- **Data Format**: Downloadable dataset (328 MB)
- **Available Properties**: Comprehensive soil survey data

## 3. International Services Covering Canada

### 3.1 SoilGrids (ISRIC)
- **Primary URL**: https://soilgrids.org/
- **REST API**: https://rest.isric.org/soilgrids/v2.0/docs
- **Point Query Endpoint**: https://rest.isric.org/soilgrids/v2.0/properties/query
- **WCS Endpoint**: Available for spatial data extraction
- **Grid Data Downloads**: https://files.isric.org/soilgrids/latest/data/

#### Available Properties:
- **Physical**: Clay content (clay), Sand content (sand), Silt content (silt), Bulk density (bdod), Coarse fragments (cfvo)
- **Chemical**: pH in H2O (phh2o), Soil organic carbon (soc), Total nitrogen (nitrogen), Cation exchange capacity (cec)
- **Hydrological**: Water content at various tensions (wv0010, wv0033, wv1500)
- **Derived**: Organic carbon density (ocd)

#### Technical Specifications:
- **Resolution**: 250 meters global
- **Depth Intervals**: 0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm
- **Coverage**: Global including complete Canada coverage
- **Data Format**: GeoTIFF, REST JSON responses
- **Authentication**: None required
- **Usage Limits**: Fair use - 5 API calls per 1 minute period

#### Example Usage:
```
# Point query example
https://rest.isric.org/soilgrids/v2.0/properties/query?lon=-123.25&lat=49.25&property=clay&property=sand&property=phh2o&property=soc&depth=0-5cm&depth=5-15cm&value=mean
```

### 3.2 ISRIC World Soil Information Service (WoSIS)
- **WFS Endpoint**: http://wfs.isric.org/geoserver/wosis/wfs
- **Coverage**: Global with 228,000 geo-referenced points from 173 countries
- **Available Properties**: 
  - **Chemical**: Organic carbon, total carbon, total nitrogen, phosphorus, soil pH, cation exchange capacity, electrical conductivity
  - **Physical**: Sand, silt, clay percentages, bulk density, coarse fragments, water retention
- **Data Format**: WFS (Web Feature Service) - OGC compliant
- **Quality**: Quality-assessed and standardized soil profile data
- **Authentication**: None required
- **Access Methods**: 
  - Direct WFS connection
  - QGIS integration
  - R programming language
  - GraphQL API
- **Metadata Portal**: https://data.isric.org

### 3.3 FAO Harmonized World Soil Database (HWSD)
- **Primary URL**: https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/harmonized-world-soil-database-v12/en/
- **Latest Version**: HWSD v2.0 (released 2024)
- **Resolution**: 30 arc-second (~1km) global raster
- **Coverage**: Global including Canada
- **Available Properties**: 
  - Soil texture (clay, sand, silt percentages)
  - Organic carbon content
  - pH values
  - Water storage capacity
  - Cation exchange capacity
  - Chemical properties (lime, gypsum content, salinity)
- **Depth Layers**: 7 standard depth layers (improved from 2 in v1.2)
- **Data Format**: Raster (GeoTIFF), Microsoft Access database
- **Access**: Download through FAO GeoNetwork
- **GeoNetwork Portal**: http://geonetwork.fao.org/geonetwork/

## 4. Recommended Implementation Strategy

### 4.1 Primary Data Sources (Ordered by Priority)

1. **SoilGrids REST API** (Immediate Implementation)
   - Global 250m resolution coverage
   - Comprehensive soil properties
   - Simple REST interface
   - No authentication required
   - Active development and support

2. **ISRIC WoSIS WFS** (Point Data Validation)
   - High-quality point measurements
   - OGC-compliant WFS
   - Useful for validation and calibration
   - Standardized global dataset

3. **Canadian CanSIS/NSDB** (National Detail)
   - Official Canadian soil database
   - Higher resolution for detailed areas
   - Authoritative source for Canada
   - Multiple download formats available

4. **Provincial Services** (Regional Detail)
   - BC SIFT for Vancouver area testing
   - Alberta services for prairie regions
   - High-resolution local data

### 4.2 Technical Implementation Notes

#### For SoilGrids API Integration:
```python
import requests
import json

def get_soilgrids_data(lon, lat, properties=['clay', 'sand', 'silt', 'phh2o', 'soc'], 
                      depths=['0-5cm', '5-15cm', '15-30cm'], value='mean'):
    """
    Retrieve soil properties from SoilGrids API
    """
    base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    
    params = {
        'lon': lon,
        'lat': lat,
        'value': value
    }
    
    # Add multiple properties and depths
    for prop in properties:
        params[f'property'] = prop
    for depth in depths:
        params[f'depth'] = depth
    
    response = requests.get(base_url, params=params)
    return response.json()

# Example usage for Vancouver area
vancouver_soil = get_soilgrids_data(-123.25, 49.25)
```

#### For WFS Integration:
```python
from owslib.wfs import WebFeatureService

def get_wosis_data(bbox):
    """
    Retrieve soil profile data from WoSIS WFS
    """
    wfs = WebFeatureService('http://wfs.isric.org/geoserver/wosis/wfs', version='1.1.0')
    
    # Get feature info
    layer = 'wosis:wosis_latest'
    response = wfs.getfeature(typename=layer, bbox=bbox, outputFormat='json')
    
    return response.read()
```

### 4.3 Data Integration Workflow

1. **Primary Data Retrieval**: Use SoilGrids API for comprehensive coverage
2. **Quality Validation**: Cross-reference with WoSIS point data
3. **Regional Enhancement**: Supplement with provincial/national datasets
4. **Spatial Interpolation**: Apply appropriate interpolation methods for model grid requirements
5. **Property Derivation**: Calculate additional soil parameters as needed for RAVEN models

## 5. Usage Examples and Parameters

### 5.1 Required Parameters for Hydrological Modeling

For RAVEN hydrological modeling, the following soil properties are typically required:
- **Texture**: Clay%, Sand%, Silt% (for hydraulic conductivity estimation)
- **Chemical**: pH, Organic matter% (for nutrient cycling, soil chemistry)
- **Physical**: Bulk density, porosity (for water storage calculations)
- **Hydrological**: Field capacity, wilting point, saturated water content

### 5.2 Spatial Resolution Considerations

- **SoilGrids**: 250m resolution suitable for watershed-scale modeling
- **Provincial data**: Often higher resolution (1:50,000 scale) for detailed areas
- **National data**: 1:250,000 to 1:1,000,000 scale for broad regional studies

### 5.3 Data Quality and Limitations

- **SoilGrids**: Machine learning predictions, validated against global soil profiles
- **WoSIS**: Point measurements with quality assessment, but sparse coverage
- **National/Provincial**: Survey-based data with varying coverage dates and methodologies
- **Temporal Considerations**: Most soil surveys represent historical conditions

## 6. Contact Information and Support

### Government Contacts:
- **AAFC Geomatics**: aafc.agri-geomatics-agrogeomatiques.aac@agr.gc.ca
- **CanSIS Support**: aafc.cansis-siscan.aac@agr.gc.ca

### International Support:
- **ISRIC Support**: Technical documentation available at https://www.isric.org/
- **FAO Support**: Through GeoNetwork portal and FAO Soils Portal

---

**Research Date**: August 6, 2025
**Coverage**: Comprehensive review of Canadian soil data APIs and endpoints
**Focus**: Real soil properties (clay, sand, silt, pH, organic matter) for hydrological modeling applications