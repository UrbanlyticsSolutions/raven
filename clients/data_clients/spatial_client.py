#!/usr/bin/env python3
"""
Spatial Data Client for RAVEN Hydrological Modeling
Downloads spatial layers (DEM, landcover, vegetation) from multiple reliable data sources
Updated to use modern APIs and working endpoints
"""

import requests
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
import json
import numpy as np
warnings.filterwarnings('ignore')

def download_with_progress(url, output_path, params=None, headers=None):
    """Universal download function with progress bar"""
    try:
        response = requests.get(url, params=params, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(output_path, "wb") as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=output_path.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"SUCCESS: Data saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        return False

def get_usgs_dem_data(bbox, output_path, resolution=10):
    """Download DEM data from USGS 3DEP service"""
    try:
        # Convert bbox to USGS format (Web Mercator)
        minx, miny, maxx, maxy = bbox
        
        # USGS 3DEP ImageServer endpoint
        base_url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
        
        params = {
            'f': 'image',
            'bbox': f"{minx},{miny},{maxx},{maxy}",
            'bboxSR': '4326',  # WGS84
            'imageSR': '4326',
            'size': f"{min(4000, int((maxx - minx) * 111000 / resolution))},{min(4000, int((maxy - miny) * 111000 / resolution))}",
            'format': 'tiff',
            'pixelType': 'F32',
            'interpolation': 'RSP_BilinearInterpolation',
            'compression': 'LZ77',
            'renderingRule': '{"rasterFunction":"None"}'
        }
        
        return download_with_progress(base_url, output_path, params=params)
        
    except Exception as e:
        print(f"ERROR: USGS DEM download failed: {e}")
        return False

def get_legacy_wcs_data(endpoint, identifier, bbox, output_path, version="1.1.1",
                       crs="EPSG:4326", grid_offsets="0.001,-0.001", format="image/geotiff"):
    """Legacy WCS function for Canadian services (with fallback support)"""
    try:
        minx, miny, maxx, maxy = bbox
        
        # Calculate reasonable resolution based on bounding box size
        width_deg = maxx - minx
        height_deg = maxy - miny
        # Use a resolution that gives us a reasonable image size (max 2000x2000 pixels)
        resolution = max(width_deg / 500, height_deg / 500, 0.001)  # minimum 0.001 degrees
        
        # Calculate appropriate image dimensions
        width_deg = maxx - minx
        height_deg = maxy - miny
        # Aim for roughly 500-1000 pixels per degree for good resolution
        target_width = max(100, min(2000, int(width_deg * 500)))
        target_height = max(100, min(2000, int(height_deg * 500)))
        
        # Try different parameter combinations for better compatibility
        param_sets = [
            # WCS 1.1.1 request with explicit WIDTH/HEIGHT (preferred for landcover)
            {
                "SERVICE": "WCS",
                "VERSION": version,
                "REQUEST": "GetCoverage",
                "FORMAT": format,
                "IDENTIFIER": identifier,
                "BOUNDINGBOX": f"{minx},{miny},{maxx},{maxy},urn:ogc:def:crs:EPSG::{crs.split(':')[-1]}",
                "GRIDBASECRS": f"urn:ogc:def:crs:EPSG::{crs.split(':')[-1]}",
                "GRIDCS": "urn:ogc:def:cs:OGC::imageCRS",
                "GRIDTYPE": "urn:ogc:def:method:WCS:1.1:2dSimpleGrid",
                "WIDTH": str(target_width),
                "HEIGHT": str(target_height),
            },
            # WCS 1.1.1 request with GRIDOFFSETS (alternative approach)
            {
                "SERVICE": "WCS",
                "VERSION": version,
                "REQUEST": "GetCoverage",
                "FORMAT": format,
                "IDENTIFIER": identifier,
                "BOUNDINGBOX": f"{minx},{miny},{maxx},{maxy},urn:ogc:def:crs:EPSG::{crs.split(':')[-1]}",
                "GRIDBASECRS": f"urn:ogc:def:crs:EPSG::{crs.split(':')[-1]}",
                "GRIDCS": "urn:ogc:def:cs:OGC::imageCRS",
                "GRIDTYPE": "urn:ogc:def:method:WCS:1.1:2dSimpleGrid",
                "GRIDOFFSETS": grid_offsets,
            },
            # REMOVED: Simplified fallback request - no fallback parameter sets allowed
        ]
        
        for params in param_sets:
            try:
                response = requests.get(endpoint, params=params, stream=True, timeout=60)
                if response.status_code == 200 and len(response.content) > 1000:
                    
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 8192
                    
                    with open(output_path, "wb") as f, tqdm(
                        total=total_size, unit='iB', unit_scale=True, desc=output_path.name
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                    
                    print(f"SUCCESS: WCS data saved to: {output_path}")
                    return True
                    
            except Exception as e:
                continue
                
        return False
        
    except Exception as e:
        print(f"ERROR: Legacy WCS download failed: {e}")
        return False

class SpatialLayersClient:
    """Client for downloading spatial data layers needed for RAVEN modeling"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAVEN-Hydrological-Model-Client/2.0'
        })
        
        # Define multiple data source endpoints for redundancy
        self.endpoints = {
            'usgs_dem': 'https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage',
            'nrcan_legacy_elevation': 'https://datacube.services.geo.ca/ows/elevation',
            'nrcan_legacy_landcover': 'https://datacube.services.geo.ca/ows/landcover',
            'nrcan_legacy_vegetation': 'https://datacube.services.geo.ca/ows/vegetation',
            'nrcan_cog_dem': 'https://datacube.services.geo.ca/api/cog/elevation',
            'mghydro_watershed': 'https://mghydro.com/app/watershed_api',
            'mghydro_rivers': 'https://mghydro.com/app/upstream_rivers_api',
            'mghydro_flowpath': 'https://mghydro.com/app/flowpath_api',
        }
    
    def get_dem_for_watershed(self, bbox: Tuple[float, float, float, float], 
                             output_path: Path, resolution: int = 10,
                             source: str = 'auto') -> Dict:
        """Download DEM data for watershed bounding box from multiple sources"""
        
        sources_to_try = []
        
        if source == 'auto':
            # Try USGS first (most reliable), then Canadian sources
            sources_to_try = ['usgs', 'nrcan_cog', 'nrcan_legacy']
        elif source == 'usgs':
            sources_to_try = ['usgs']
        elif source == 'nrcan':
            sources_to_try = ['nrcan_cog', 'nrcan_legacy']
        else:
            sources_to_try = [source]
        
        for source_name in sources_to_try:
            print(f"Attempting DEM download from {source_name}...")
            
            try:
                if source_name == 'usgs':
                    success = self._get_usgs_dem(bbox, output_path, resolution)
                elif source_name == 'nrcan_cog':
                    success = self._get_nrcan_cog_dem(bbox, output_path)
                elif source_name == 'nrcan_legacy':
                    success = self._get_nrcan_legacy_dem(bbox, output_path)
                else:
                    continue
                    
                if success:
                    print(f"SUCCESS: DEM data downloaded from {source_name}")
                    return {
                        'success': True,
                        'data_type': 'DEM',
                        'source': source_name,
                        'file_path': str(output_path),
                        'bbox': bbox,
                        'resolution': resolution
                    }
                else:
                    print(f"Failed to download from {source_name}, trying next source...")
                    
            except Exception as e:
                print(f"Error with {source_name}: {str(e)}")
                continue
        
        return {
            'success': False, 
            'error': 'All DEM sources failed. Check network connectivity and coordinates.',
            'attempted_sources': sources_to_try
        }
    
    def _get_usgs_dem(self, bbox: Tuple[float, float, float, float], 
                      output_path: Path, resolution: int = 10) -> bool:
        """Download DEM from USGS 3DEP service"""
        minx, miny, maxx, maxy = bbox
        
        # Calculate appropriate image size based on resolution - limit to avoid USGS 500 errors
        width = min(2000, int((maxx - minx) * 111000 / resolution))  # Max 2000 pixels
        height = min(2000, int((maxy - miny) * 111000 / resolution))  # Max 2000 pixels
        
        params = {
            'f': 'image',
            'bbox': f"{minx},{miny},{maxx},{maxy}",
            'bboxSR': '4326',  # WGS84
            'imageSR': '4326',
            'size': f"{width},{height}",
            'format': 'tiff',
            'pixelType': 'F32',
            'interpolation': 'RSP_BilinearInterpolation',
            'compression': 'LZ77',
            'renderingRule': '{"rasterFunction":"None"}'
        }
        
        return download_with_progress(
            self.endpoints['usgs_dem'], 
            output_path, 
            params=params
        )
    
    def _get_nrcan_cog_dem(self, bbox: Tuple[float, float, float, float], 
                           output_path: Path) -> bool:
        """Download DEM from NRCan Cloud Optimized GeoTIFF service"""
        # This would be implemented when NRCan provides a working COG API
        # For now, return False to fall back to other methods
        return False
    
    def _get_nrcan_legacy_dem(self, bbox: Tuple[float, float, float, float], 
                              output_path: Path) -> bool:
        """Download DEM from legacy NRCan WCS service (fallback)"""
        return get_legacy_wcs_data(
            endpoint=self.endpoints['nrcan_legacy_elevation'],
            identifier="dtm",
            bbox=bbox,
            output_path=output_path,
            grid_offsets="0.001,-0.001",  # Higher resolution
            crs="EPSG:4326"
        )
    
    def _get_nrcan_landcover(self, bbox: Tuple[float, float, float, float], 
                            output_path: Path, identifier: str) -> bool:
        """Download landcover using native EPSG:3979 projection for better results"""
        try:
            # Import pyproj for coordinate transformation
            from pyproj import Transformer
            
            # Transform WGS84 bbox to EPSG:3979 (LCC Canada)
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3979", always_xy=True)
            minx_wgs, miny_wgs, maxx_wgs, maxy_wgs = bbox
            
            # Transform to EPSG:3979
            minx_3979, miny_3979 = transformer.transform(minx_wgs, miny_wgs)
            maxx_3979, maxy_3979 = transformer.transform(maxx_wgs, maxy_wgs)
            
            # Calculate appropriate image dimensions (aim for ~30m resolution)
            width_m = maxx_3979 - minx_3979
            height_m = maxy_3979 - miny_3979
            target_width = max(100, min(10000, int(width_m / 30)))  # 30m resolution
            target_height = max(100, min(10000, int(height_m / 30)))
            
            params = {
                'SERVICE': 'WCS',
                'VERSION': '1.1.1', 
                'REQUEST': 'GetCoverage',
                'IDENTIFIER': identifier,
                'FORMAT': 'image/geotiff',
                'BOUNDINGBOX': f'{minx_3979},{miny_3979},{maxx_3979},{maxy_3979},urn:ogc:def:crs:EPSG::3979',
                'GRIDBASECRS': 'urn:ogc:def:crs:EPSG::3979',
                'GRIDCS': 'urn:ogc:def:cs:OGC::imageCRS',
                'GRIDTYPE': 'urn:ogc:def:method:WCS:1.1:2dSimpleGrid',
                'WIDTH': str(target_width),
                'HEIGHT': str(target_height)
            }
            
            # Download the raster
            success = download_with_progress(
                self.endpoints['nrcan_legacy_landcover'], 
                output_path, 
                params=params
            )
            
            # Fix CRS if download was successful
            if success and output_path.exists():
                try:
                    import rasterio
                    from rasterio.crs import CRS
                    
                    # Read the raster and fix the CRS
                    with rasterio.open(output_path, 'r+') as src:
                        if src.crs is None or 'LOCAL_CS' in str(src.crs):
                            # Assign EPSG:3979 CRS since that's what we requested
                            src.crs = CRS.from_epsg(3979)
                            print(f"Fixed CRS: Assigned EPSG:3979 to {output_path.name}")
                except Exception as crs_error:
                    print(f"Warning: Could not fix CRS: {crs_error}")
            
            return success
            
        except ImportError:
            print("WARNING: pyproj not available, falling back to WGS84 coordinates")
            return False
        except Exception as e:
            print(f"ERROR: Native projection landcover download failed: {e}")
            return False
    
    def get_landcover_for_watershed(self, bbox: Tuple[float, float, float, float], 
                                   output_path: Path, year: int = 2020, 
                                   grid_offsets: str = "0.0003,-0.0003") -> Dict:
        """Download land cover data for watershed using native EPSG:3979 projection"""
        try:
            year_mapping = {
                2010: "landcover-2010",
                2015: "landcover-2015", 
                2020: "landcover-2020"
            }
            
            identifier = year_mapping.get(year, "landcover-2020")
            
            # Use specialized landcover function that handles projection transformation
            success = self._get_nrcan_landcover(bbox, output_path, identifier)
            
            if not success:
                error_msg = "NRCan landcover data acquisition failed - no synthetic fallback provided"
                raise Exception(error_msg)
            
            if success:
                print(f"SUCCESS: Land cover data downloaded")
                return {
                    'success': True,
                    'data_type': 'landcover',
                    'year': year,
                    'file_path': str(output_path),
                    'bbox': bbox
                }
            else:
                return {
                    'success': False, 
                    'error': 'Land cover download failed - service may be unavailable'
                }
            
        except Exception as e:
            return {'success': False, 'error': f"Land cover download failed: {str(e)}"}
    
    def get_vegetation_for_watershed(self, bbox: Tuple[float, float, float, float], 
                                    output_path: Path, parameter: str = "LAI", 
                                    year: int = 2020, grid_offsets: str = "0.001,-0.001") -> Dict:
        """Download vegetation data for watershed"""
        try:
            if year is None:
                identifier = parameter  # Multi-year average
            else:
                identifier = f"{parameter}_{year}"
            
            success = get_legacy_wcs_data(
                endpoint=self.endpoints['nrcan_legacy_vegetation'],
                identifier=identifier,
                bbox=bbox,
                output_path=output_path,
                grid_offsets=grid_offsets,
                crs="EPSG:4326"
            )
            
            if success:
                print(f"SUCCESS: Vegetation data ({parameter}) downloaded")
                return {
                    'success': True,
                    'data_type': 'vegetation',
                    'parameter': parameter,
                    'year': year,
                    'file_path': str(output_path),
                    'bbox': bbox
                }
            else:
                return {
                    'success': False,
                    'error': 'Vegetation download failed - service may be unavailable'
                }
            
        except Exception as e:
            return {'success': False, 'error': f"Vegetation download failed: {str(e)}"}
    
    def get_watershed_from_mghydro(self, lat: float, lng: float, precision: str = "high", 
                                 output_path: Optional[Path] = None) -> Dict:
        """
        Get watershed boundary from MGHydro API using outlet coordinates
        
        Args:
            lat: Latitude in decimal degrees (-60 to +85)
            lng: Longitude in decimal degrees (-180 to +180)
            precision: "low" or "high" resolution (defaults to high)
            output_path: Optional path to save GeoJSON file
            
        Returns:
            Dict with watershed data including GeoJSON polygon
        """
        try:
            params = {
                'lat': lat,
                'lng': lng,
                'precision': precision
            }
            
            response = self.session.get(
                self.endpoints['mghydro_watershed'], 
                params=params, 
                timeout=60
            )
            
            if response.status_code == 200:
                watershed_data = response.json()
                
                # Save to file if path provided
                if output_path:
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'w') as f:
                        json.dump(watershed_data, f, indent=2)
                    print(f"SUCCESS: Watershed data saved to {output_path}")
                
                # Calculate area from GeoJSON
                area_km2 = 0
                if 'features' in watershed_data and watershed_data['features']:
                    from shapely.geometry import shape
                    watershed_geom = shape(watershed_data['features'][0]['geometry'])
                    area_km2 = watershed_geom.area / 1e6  # Convert m2 to km2
                
                return {
                    'success': True,
                    'data_type': 'watershed',
                    'source': 'mghydro',
                    'lat': lat,
                    'lng': lng,
                    'precision': precision,
                    'area_km2': area_km2,
                    'watershed_geojson': watershed_data,
                    'file_path': str(output_path) if output_path else None
                }
            else:
                return {
                    'success': False,
                    'error': f'MGHydro API returned status {response.status_code}',
                    'lat': lat,
                    'lng': lng
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'lat': lat,
                'lng': lng
            }
    
    def get_upstream_rivers_from_mghydro(self, lat: float, lng: float, precision: str = "high") -> Dict:
        """
        Get upstream river network from MGHydro API
        
        Args:
            lat: Latitude in decimal degrees
            lng: Longitude in decimal degrees
            precision: "low" or "high" resolution
            
        Returns:
            Dict with river network data
        """
        try:
            params = {
                'lat': lat,
                'lng': lng,
                'precision': precision
            }
            
            response = self.session.get(
                self.endpoints['mghydro_rivers'], 
                params=params, 
                timeout=60
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'data_type': 'upstream_rivers',
                    'source': 'mghydro',
                    'rivers_geojson': response.json()
                }
            else:
                return {
                    'success': False,
                    'error': f'MGHydro rivers API returned status {response.status_code}'
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_downstream_flowpath_from_mghydro(self, lat: float, lng: float, precision: str = "high") -> Dict:
        """
        Get downstream flow path from MGHydro API
        
        Args:
            lat: Latitude in decimal degrees
            lng: Longitude in decimal degrees
            precision: "low" or "high" resolution
            
        Returns:
            Dict with flow path data
        """
        try:
            params = {
                'lat': lat,
                'lng': lng,
                'precision': precision
            }
            
            response = self.session.get(
                self.endpoints['mghydro_flowpath'], 
                params=params, 
                timeout=60
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'data_type': 'downstream_flowpath',
                    'source': 'mghydro',
                    'flowpath_geojson': response.json()
                }
            else:
                return {
                    'success': False,
                    'error': f'MGHydro flowpath API returned status {response.status_code}'
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_complete_watershed_data(self, outlet_lat: float, outlet_lng: float, 
                                  output_dir: Path, precision: str = "high") -> Dict:
        """
        Get complete watershed dataset from MGHydro API
        
        Args:
            outlet_lat: Outlet latitude
            outlet_lng: Outlet longitude
            output_dir: Directory to save all data
            precision: "low" or "high" resolution
            
        Returns:
            Dict with all watershed data components
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'outlet': {'lat': outlet_lat, 'lng': outlet_lng},
            'precision': precision,
            'components': {}
        }
        
        # Get watershed boundary
        watershed_file = output_dir / "watershed.geojson"
        watershed_result = self.get_watershed_from_mghydro(
            outlet_lat, outlet_lng, precision, watershed_file
        )
        results['components']['watershed'] = watershed_result
        
        if watershed_result['success']:
            # Get upstream rivers
            rivers_result = self.get_upstream_rivers_from_mghydro(
                outlet_lat, outlet_lng, precision
            )
            results['components']['rivers'] = rivers_result
            
            # Get downstream flowpath
            flowpath_result = self.get_downstream_flowpath_from_mghydro(
                outlet_lat, outlet_lng, precision
            )
            results['components']['flowpath'] = flowpath_result
            
            # Save rivers and flowpath if successful
            if rivers_result['success']:
                with open(output_dir / "upstream_rivers.geojson", 'w') as f:
                    json.dump(rivers_result['rivers_geojson'], f, indent=2)
            
            if flowpath_result['success']:
                with open(output_dir / "downstream_flowpath.geojson", 'w') as f:
                    json.dump(flowpath_result['flowpath_geojson'], f, indent=2)
        
        return results
    
    def test_endpoints(self) -> Dict:
        """Test all configured endpoints for availability"""
        results = {}
        
        for name, url in self.endpoints.items():
            try:
                # Test basic connectivity
                if 'usgs' in name:
                    # Test USGS with a capabilities request
                    test_url = url.replace('/exportImage', '') + '?f=json'
                elif 'mghydro' in name:
                    # Test MGHydro with a simple watershed request
                    test_url = url + '?lat=45.0&lng=-75.0'
                else:
                    # Test NRCan with GetCapabilities
                    test_url = url + '?SERVICE=WCS&REQUEST=GetCapabilities&VERSION=1.1.1'
                
                response = self.session.get(test_url, timeout=15)
                
                if response.status_code == 200:
                    results[name] = {
                        'status': 'available',
                        'response_size': len(response.content),
                        'content_type': response.headers.get('content-type', 'unknown')
                    }
                else:
                    results[name] = {
                        'status': 'error',
                        'error': f'HTTP {response.status_code}'
                    }
                    
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    def get_gauge_data_from_geopackage(self, gpkg_path: Path, outlet_lat: float, outlet_lng: float, max_distance_km: float = 10.0) -> Dict:
        """
        Query Canadian hydrometric data from GeoPackage for nearest gauge to outlet
        
        Args:
            gpkg_path: Path to GeoPackage file
            outlet_lat: Outlet latitude
            outlet_lng: Outlet longitude
            max_distance_km: Maximum search distance in kilometers
            
        Returns:
            Dict with nearest gauge information and watershed data
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            
            # Load GeoPackage
            gdf = gpd.read_file(gpkg_path)
            
            if len(gdf) == 0:
                return {'success': False, 'error': 'Empty GeoPackage'}
            
            # Create point from outlet coordinates
            outlet_point = Point(outlet_lng, outlet_lat)
            
            # Find nearest gauge
            gdf['distance'] = gdf.geometry.distance(outlet_point)
            nearest_gauge = gdf.loc[gdf['distance'].idxmin()]
            
            # Check if within acceptable distance
            actual_distance_km = nearest_gauge['distance'] * 111.0  # 1 degree ≈ 111 km
            if actual_distance_km > max_distance_km:
                return {'success': False, 'error': f'No gauge within {max_distance_km}km'}
            
            return {
                'success': True,
                'data_type': 'gauge_data',
                'source': 'geopackage',
                'gauge_info': {
                    'station_id': str(nearest_gauge.get('STATION_ID', 'unknown')),
                    'station_name': str(nearest_gauge.get('STATION_NAME', 'unknown')),
                    'latitude': float(nearest_gauge.geometry.y),
                    'longitude': float(nearest_gauge.geometry.x),
                    'distance_km': actual_distance_km,
                    'drainage_area_km2': float(nearest_gauge.get('DRAINAGE_AREA', 0)),
                    'province': str(nearest_gauge.get('PROV_TERR_STATE_LOC', 'unknown')),
                    'hydrometric_zone': str(nearest_gauge.get('HYDROMETRIC_ZONE', 'unknown'))
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def validate_outlet_with_gauge_data(self, outlet_lat: float, outlet_lng: float, 
                                      gpkg_path: Path = None) -> Dict:
        """
        Validate outlet coordinates against hydrometric gauge data
        
        Args:
            outlet_lat: Outlet latitude
            outlet_lng: Outlet longitude
            gpkg_path: Path to GeoPackage file (defaults to canadian_hydro.gpkg)
            
        Returns:
            Dict with validation results and gauge mappings
        """
        if gpkg_path is None:
            gpkg_path = Path(__file__).parent.parent.parent / "data" / "canadian" / "canadian_hydro.gpkg"
        
        if not gpkg_path.exists():
            return {'success': False, 'error': f'GeoPackage not found: {gpkg_path}'}
        
        return self.get_gauge_data_from_geopackage(gpkg_path, outlet_lat, outlet_lng, max_distance_km=5.0)
    
    def get_outlet_driven_watershed(self, outlet_lat: float, outlet_lng: float, 
                                  use_gauge_fallback: bool = False, 
                                  gpkg_path: Optional[Path] = None) -> Dict:
        """
        Complete outlet-driven watershed discovery combining MGHydro API and gauge data
        
        Args:
            outlet_lat: Outlet latitude
            outlet_lng: Outlet longitude
            use_gauge_fallback: DEPRECATED - no longer used (no synthetic fallbacks)
            gpkg_path: Path to GeoPackage file
            
        Returns:
            Dict with complete watershed information
        """
        results = {
            'outlet': {'lat': outlet_lat, 'lng': outlet_lng},
            'validation': {},
            'watershed': {},
            'gauge_info': {}
        }
        
        # Get gauge data for reference only (no fallback)
        if gpkg_path:
            gauge_validation = self.validate_outlet_with_gauge_data(outlet_lat, outlet_lng, gpkg_path)
            results['validation'] = gauge_validation
            
            if gauge_validation['success']:
                results['gauge_info'] = gauge_validation['gauge_info']
        
        # Get watershed from MGHydro API
        watershed_data = self.get_watershed_from_mghydro(outlet_lat, outlet_lng)
        results['watershed'] = watershed_data
        
        # Combine results
        if watershed_data['success'] and results['gauge_info']:
            results['combined'] = {
                'outlet_coordinates': [outlet_lat, outlet_lng],
                'gauge_match': results['gauge_info'],
                'watershed_area_km2': watershed_data['area_km2'],
                'validation_status': 'valid'
            }
        
        return results
    
    def test_geopackage_capabilities(self, gpkg_path: Optional[Path] = None) -> Dict:
        """Test GeoPackage query capabilities"""
        if gpkg_path is None:
            gpkg_path = Path(__file__).parent.parent.parent / "data" / "canadian" / "canadian_hydro.gpkg"
        
        try:
            import geopandas as gpd
            
            gdf = gpd.read_file(gpkg_path)
            
            return {
                'success': True,
                'file_path': str(gpkg_path),
                'record_count': len(gdf),
                'columns': list(gdf.columns),
                'crs': str(gdf.crs),
                'bounds': list(gdf.total_bounds),
                'sample_data': gdf.head(3).to_dict('records')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}