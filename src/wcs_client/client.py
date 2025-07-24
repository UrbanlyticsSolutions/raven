import requests
from pathlib import Path
from tqdm import tqdm

def get_wcs_data(endpoint, identifier, bbox, output_path, version="1.1.1",
                 crs="EPSG:2961", grid_offsets="5.0,-5.0", format="image/geotiff"):
    minx, miny, maxx, maxy = bbox
    params = {
        "SERVICE": "WCS",
        "VERSION": version,
        "REQUEST": "GetCoverage",
        "FORMAT": format,
        "IDENTIFIER": identifier,
        "BOUNDINGBOX": f"{minx},{miny},{maxx},{maxy},urn:ogc:def:crs:EPSG::{crs.split(':')[-1]}",
        "GRIDBASECRS": f"urn:ogc:def:crs:EPSG::{crs.split(':')[-1]}",
        "GRIDOFFSETS": grid_offsets,
        "GRIDORIGIN": f"{minx},{maxy}",
        "Gridcs": "urn:ogc:def:cs:OGC:0.0:Grid2dSquareCS",
        "gridtype": "urn:ogc:def:method:WCS:1.1:2dSimpleGrid",
    }
    r = requests.get(endpoint, params=params, stream=True)
    r.raise_for_status()

    total_size = int(r.headers.get('content-length', 0))
    block_size = 8192
    
    with open(output_path, "wb") as f, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc=output_path.name
    ) as pbar:
        for chunk in r.iter_content(chunk_size=block_size):
            f.write(chunk)
            pbar.update(len(chunk))

    if total_size != 0 and pbar.n != total_size:
        print("ERROR, something went wrong")
        
    print(f"✅ Data saved to: {output_path}")

def get_dem(bbox, output_path, grid_offsets="5.0,-5.0"):
    get_wcs_data(
        endpoint="https://datacube.services.geo.ca/ows/elevation",
        identifier="dtm",
        bbox=bbox,
        output_path=output_path,
        grid_offsets=grid_offsets,
    )

def get_landcover(bbox, output_path, grid_offsets="5.0,-5.0", year=2020):
    """
    Download landcover data from NRCan datacube
    
    Args:
        bbox: Tuple of (minx, miny, maxx, maxy)
        output_path: Path object for output file
        grid_offsets: Grid resolution as string
        year: Year for landcover data (2010, 2015, 2020, or 'change' for 2010-2020 change)
    """
    # Map year to identifier
    year_mapping = {
        2010: "landcover-2010",
        2015: "landcover-2015", 
        2020: "landcover-2020",
        "change": "landcover"  # 2010-2020 change
    }
    
    identifier = year_mapping.get(year, "landcover-2020")
    
    get_wcs_data(
        endpoint="https://datacube.services.geo.ca/ows/landcover",
        identifier=identifier,
        bbox=bbox,
        output_path=output_path,
        grid_offsets=grid_offsets,
    )

def get_vegetation(bbox, output_path, grid_offsets="5.0,-5.0", parameter="LAI", year=2020):
    """
    Download vegetation parameters from NRCan datacube
    
    Args:
        bbox: Tuple of (minx, miny, maxx, maxy)
        output_path: Path object for output file
        grid_offsets: Grid resolution as string
        parameter: Vegetation parameter ("LAI" or "fCOVER")
        year: Year for data (2019, 2020, or None for multi-year average)
    """
    # Map parameters and years to identifiers
    if year is None:
        identifier = parameter  # Multi-year average
    else:
        identifier = f"{parameter}_{year}"
    
    get_wcs_data(
        endpoint="https://datacube.services.geo.ca/ows/vegetation",
        identifier=identifier,
        bbox=bbox,
        output_path=output_path,
        grid_offsets=grid_offsets,
    )

def get_elevation(bbox, output_path, grid_offsets="5.0,-5.0", model_type="dtm"):
    """
    Download elevation data from NRCan datacube
    
    Args:
        bbox: Tuple of (minx, miny, maxx, maxy)
        output_path: Path object for output file
        grid_offsets: Grid resolution as string
        model_type: Type of elevation model ("dtm" for Digital Terrain Model or "dsm" for Digital Surface Model)
    """
    get_wcs_data(
        endpoint="https://datacube.services.geo.ca/ows/elevation",
        identifier=model_type,
        bbox=bbox,
        output_path=output_path,
        grid_offsets=grid_offsets,
    )

def get_dem(bbox, output_path, grid_offsets="5.0,-5.0"):
    """Alias for get_elevation with DTM model type"""
    get_elevation(bbox, output_path, grid_offsets, model_type="dtm")

# Note: Soils endpoint is not available at datacube.services.geo.ca
# Available NRCan WCS services:
# - elevation: dtm, dsm
# - landcover: landcover-2010, landcover-2015, landcover-2020, landcover (change)  
# - vegetation: LAI_2019, LAI_2020, fCOVER_2019, fCOVER_2020, LAI, fCOVER
# 
# For soil data, consider alternative sources:
# - Agriculture and Agri-Food Canada (AAFC)
# - Provincial government data portals
# - Federal Geospatial Platform
