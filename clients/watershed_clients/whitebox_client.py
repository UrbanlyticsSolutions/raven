#!/usr/bin/env python3
"""
WhiteboxTools Watershed Client
Professional DEM-based watershed analysis using WhiteboxTools and related libraries

This module contains all watershed delineation functionality that was previously
in the main client.py file. It provides professional-grade watershed analysis
using the most stable and reliable libraries.

Libraries Used:
- whitebox: Official WhiteboxTools Python wrapper
- pyflwdir: High-performance flow analysis by Deltares  
- rasterio: Robust raster I/O and processing
- geopandas: Vector data handling and analysis
- fiona: Shapefile and vector format support

Author: RAVEN Hydrological Modeling System
Date: 2025-07-30
"""

import warnings
import requests
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Core GIS and raster processing
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("WARNING: rasterio not available - install with: pip install rasterio")

# Vector data processing
try:
    import geopandas as gpd
    import fiona
    from shapely.geometry import Point, Polygon, LineString, box
    from shapely.ops import transform
    HAS_VECTOR_TOOLS = True
except ImportError:
    HAS_VECTOR_TOOLS = False
    print("WARNING: Vector tools not available - install with: pip install geopandas fiona shapely pyproj")

# WhiteboxTools - Primary watershed analysis engine
try:
    import whitebox
    HAS_WHITEBOX = True
except ImportError:
    HAS_WHITEBOX = False
    print("WARNING: WhiteboxTools not available - install with: pip install whitebox")

# pyflwdir - High-performance flow analysis
try:
    import pyflwdir
    HAS_PYFLWDIR = True
except ImportError:
    HAS_PYFLWDIR = False
    print("WARNING: pyflwdir not available - install with: pip install pyflwdir")

# Import the professional watershed analyzer
try:
    from .professional_watershed_analyzer import ProfessionalWatershedAnalyzer
    HAS_PROFESSIONAL_ANALYZER = True
except ImportError:
    try:
        from clients.watershed_clients.professional_watershed_analyzer import ProfessionalWatershedAnalyzer
        HAS_PROFESSIONAL_ANALYZER = True
    except ImportError:
        HAS_PROFESSIONAL_ANALYZER = False


# ===== 1. NRCAN WATERSHED ATLAS CLIENT =====

class NRCANWatershedClient:
    """Client for NRCAN Watershed Atlas service integration"""
    
    def __init__(self):
        self.base_url = "https://geoappext.nrcan.gc.ca/arcgis/rest/services/NRCAN/AtlasWatershedsEN/MapServer"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAVEN-Hydrological-Model-Client/1.0'
        })
    
    def get_watershed_for_station(self, station_coords: Tuple[float, float], 
                                 output_path: Path, layer_level: str = "detailed") -> Dict:
        """Download watershed boundary for hydrometric station location"""
        layer_mapping = {"detailed": 0, "sub_basin": 2, "major": 3}
        layer_id = layer_mapping.get(layer_level, 0)
        
        lon, lat = station_coords
        print(f"Getting {layer_level} watershed for station at ({lat:.6f}, {lon:.6f})")
        
        try:
            # Point geometry for spatial query
            geometry = {
                "x": lon,
                "y": lat,
                "spatialReference": {"wkid": 4326}
            }
            
            params = {
                "f": "json",
                "geometry": json.dumps(geometry),
                "geometryType": "esriGeometryPoint",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true"
            }
            
            url = f"{self.base_url}/{layer_id}/query"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            watershed_data = response.json()
            
            if watershed_data.get('features'):
                # Save GeoJSON
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": watershed_data['features']
                }
                
                with open(output_path, 'w') as f:
                    json.dump(watershed_data, f, indent=2)
                
                # Extract watershed info
                feature = watershed_data['features'][0]
                properties = feature.get('attributes', {})
                area_km2 = properties.get('AREA_KM2', 0)
                
                result = {
                    'success': True,
                    'watershed_found': True,
                    'file_path': str(output_path),
                    'detailed_name': properties.get('WSHED_NAME', 'Unknown'),
                    'major_name': properties.get('MAJOR_NAME', 'Unknown'),
                    'area_km2': area_km2,
                    'layer_level': layer_level,
                    'feature_count': len(watershed_data['features'])
                }
                
                print(f"SUCCESS: Watershed found: {result['detailed_name']} ({area_km2:,.1f} km²)")
                return result
            else:
                print(f"   FAILED: No watershed found at coordinates")
                return {'watershed_found': False, 'error': 'No watershed found'}
                
        except Exception as e:
            error_msg = f"Watershed download failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def get_watersheds_in_bbox(self, bbox: Tuple[float, float, float, float], 
                              output_path: Path, layer_level: str = "detailed") -> Dict:
        """Download all watersheds within a bounding box"""
        layer_mapping = {"detailed": 0, "sub_basin": 2, "major": 3}
        layer_id = layer_mapping.get(layer_level, 0)
        
        print(f"Getting {layer_level} watersheds in bbox: {bbox}")
        
        try:
            minx, miny, maxx, maxy = bbox
            
            # Bounding box geometry for spatial query
            geometry = {
                "xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
                "spatialReference": {"wkid": 4326}
            }
            
            params = {
                "f": "json",
                "geometry": json.dumps(geometry),
                "geometryType": "esriGeometryEnvelope",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true"
            }
            
            url = f"{self.base_url}/{layer_id}/query"
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            
            watershed_data = response.json()
            
            if watershed_data.get('features'):
                # Save GeoJSON
                with open(output_path, 'w') as f:
                    json.dump(watershed_data, f, indent=2)
                
                feature_count = len(watershed_data['features'])
                total_area = sum(f.get('attributes', {}).get('AREA_KM2', 0) 
                               for f in watershed_data['features'])
                
                result = {
                    'success': True,
                    'watersheds_found': True,
                    'file_path': str(output_path),
                    'feature_count': feature_count,
                    'total_area_km2': total_area,
                    'layer_level': layer_level,
                    'bbox': bbox
                }
                
                print(f"SUCCESS: {feature_count} watersheds found ({total_area:,.1f} km²)")
                return result
            else:
                print(f"   FAILED: No watersheds found in bounding box")
                return {'watersheds_found': False, 'error': 'No watersheds found'}
                
        except Exception as e:
            error_msg = f"Watersheds download failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'success': False, 'error': error_msg}


# ===== 2. NRCAN HYDROGRAPHIC NETWORK CLIENT =====

class NRCANHydrographicClient:
    """Client for Canadian Hydrographic Network (NRCan National Hydro Network)"""
    
    def __init__(self):
        self.base_url = "https://geoappext.nrcan.gc.ca/arcgis/rest/services/NRCAN/Hydro_Network_EN/MapServer"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAVEN-Hydrological-Model-Client/1.0'
        })
        # Layer IDs for different network components
        self.layer_mapping = {
            'waterbodies': 0,
            'watercourses': 1,
            'constructed': 2,
            'network_flow': 3,
            'drainage_area': 4
        }
    
    def get_hydro_network_for_watershed(self, bbox: Tuple[float, float, float, float], 
                                       output_path: Path, layer_type: str = "watercourses",
                                       detail_level: str = "all") -> Dict:
        """Download hydrographic network data for watershed bounding box"""
        layer_id = self.layer_mapping.get(layer_type, 1)
        
        print(f"Getting {layer_type} hydrographic network for bbox: {bbox}")
        
        try:
            minx, miny, maxx, maxy = bbox
            
            # Bounding box geometry
            geometry = {
                "xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
                "spatialReference": {"wkid": 4326}
            }
            
            params = {
                "f": "json",
                "geometry": json.dumps(geometry),
                "geometryType": "esriGeometryEnvelope", 
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true"
            }
            
            url = f"{self.base_url}/{layer_id}/query"
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            
            hydro_data = response.json()
            
            if hydro_data.get('features'):
                # Save GeoJSON
                with open(output_path, 'w') as f:
                    json.dump(hydro_data, f, indent=2)
                
                feature_count = len(hydro_data['features'])
                
                result = {
                    'success': True,
                    'features_found': True,
                    'file_path': str(output_path),
                    'feature_count': feature_count,
                    'layer_type': layer_type,
                    'bbox': bbox
                }
                
                print(f"SUCCESS: {feature_count} {layer_type} features found")
                return result
            else:
                print(f"   FAILED: No {layer_type} found in bounding box")
                return {'features_found': False, 'error': f'No {layer_type} found'}
                
        except Exception as e:
            error_msg = f"Hydrographic network download failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def get_stream_order_network(self, bbox: Tuple[float, float, float, float], 
                                output_path: Path, min_order: int = 1) -> Dict:
        """Download stream network with Strahler stream order classification"""
        print(f"Getting stream order network (min order: {min_order}) for bbox: {bbox}")
        
        try:
            minx, miny, maxx, maxy = bbox
            
            # Query with stream order filter
            where_clause = f"STRAHLER >= {min_order}" if min_order > 1 else "1=1"
            
            geometry = {
                "xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
                "spatialReference": {"wkid": 4326}
            }
            
            params = {
                "f": "json",
                "geometry": json.dumps(geometry),
                "geometryType": "esriGeometryEnvelope",
                "spatialRel": "esriSpatialRelIntersects",
                "where": where_clause,
                "outFields": "*",
                "returnGeometry": "true"
            }
            
            url = f"{self.base_url}/1/query"  # watercourses layer
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            
            stream_data = response.json()
            
            if stream_data.get('features'):
                with open(output_path, 'w') as f:
                    json.dump(stream_data, f, indent=2)
                
                feature_count = len(stream_data['features'])
                
                # Calculate stream order statistics
                orders = []
                for feature in stream_data['features']:
                    order = feature.get('attributes', {}).get('STRAHLER', 0)
                    if order > 0:
                        orders.append(order)
                
                result = {
                    'success': True,
                    'features_found': True,
                    'file_path': str(output_path),
                    'feature_count': feature_count,
                    'min_order_requested': min_order,
                    'max_order_found': max(orders) if orders else 0,
                    'order_distribution': {str(i): orders.count(i) for i in set(orders)},
                    'bbox': bbox
                }
                
                print(f"SUCCESS: {feature_count} stream segments found")
                print(f"   Stream orders: {min(orders) if orders else 0} to {max(orders) if orders else 0}")
                return result
            else:
                print(f"   FAILED: No streams found with order >= {min_order}")
                return {'features_found': False, 'error': f'No streams found with order >= {min_order}'}
                
        except Exception as e:
            error_msg = f"Stream order network download failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def get_waterbodies_for_watershed(self, bbox: Tuple[float, float, float, float], 
                                     output_path: Path, min_area_km2: float = 0.1) -> Dict:
        """Download waterbodies (lakes, ponds, reservoirs) for watershed"""
        print(f"Getting waterbodies (min area: {min_area_km2} km²) for bbox: {bbox}")
        
        try:
            minx, miny, maxx, maxy = bbox
            
            # Query with area filter
            where_clause = f"AREA_KM2 >= {min_area_km2}" if min_area_km2 > 0 else "1=1"
            
            geometry = {
                "xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
                "spatialReference": {"wkid": 4326}
            }
            
            params = {
                "f": "json",
                "geometry": json.dumps(geometry),
                "geometryType": "esriGeometryEnvelope",
                "spatialRel": "esriSpatialRelIntersects",
                "where": where_clause,
                "outFields": "*",
                "returnGeometry": "true"
            }
            
            url = f"{self.base_url}/0/query"  # waterbodies layer
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            
            waterbody_data = response.json()
            
            if waterbody_data.get('features'):
                with open(output_path, 'w') as f:
                    json.dump(waterbody_data, f, indent=2)
                
                feature_count = len(waterbody_data['features'])
                total_area = sum(f.get('attributes', {}).get('AREA_KM2', 0) 
                               for f in waterbody_data['features'])
                
                result = {
                    'success': True,
                    'features_found': True,
                    'file_path': str(output_path),
                    'feature_count': feature_count,
                    'total_area_km2': total_area,
                    'min_area_requested': min_area_km2,
                    'bbox': bbox
                }
                
                print(f"SUCCESS: {feature_count} waterbodies found ({total_area:,.1f} km²)")
                return result
            else:
                print(f"   FAILED: No waterbodies found with area >= {min_area_km2} km²")
                return {'features_found': False, 'error': f'No waterbodies found with area >= {min_area_km2} km²'}
                
        except Exception as e:
            error_msg = f"Waterbodies download failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'success': False, 'error': error_msg}


# ===== 3. DEM-BASED WATERSHED ANALYSIS =====

class DEMWatershedAnalyzer:
    """Professional DEM-based watershed analysis using stable libraries"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAVEN-Hydrological-Model-Client/1.0'
        })
        
        if HAS_PROFESSIONAL_ANALYZER:
            self.professional_analyzer = ProfessionalWatershedAnalyzer()
        else:
            self.professional_analyzer = None
            print("WARNING: Professional watershed analyzer not available")
        
    def analyze_watershed_from_dem(self, dem_path: Path, outlet_coords: Tuple[float, float], 
                                  output_dir: Path, analysis_type: str = 'complete') -> Dict:
        """
        Perform comprehensive watershed analysis from DEM using professional tools
        
        Parameters:
        -----------
        dem_path : Path
            Path to DEM raster file
        outlet_coords : Tuple[float, float]
            (longitude, latitude) of watershed outlet
        output_dir : Path
            Directory to save analysis outputs
        analysis_type : str
            'complete', 'basic', or 'advanced'
            
        Returns:
        --------
        Dict
            Analysis results with file paths and metadata
        """
        
        print(f"Starting professional DEM-based watershed analysis...")
        print(f"DEM file: {dem_path}")
        print(f"Outlet coordinates: {outlet_coords}")
        print(f"Analysis type: {analysis_type}")
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if not self.professional_analyzer:
            return {
                'success': False, 
                'error': 'Professional watershed analyzer not available. Install required libraries: pip install whitebox pyflwdir rasterio geopandas'
            }
        
        try:
            # Check if DEM file exists
            if not dem_path.exists():
                return {'success': False, 'error': f'DEM file not found: {dem_path}'}
            
            # Configure analysis parameters based on type
            if analysis_type == 'basic':
                stream_threshold = 2000
                output_formats = ['geojson']
                flow_algorithm = 'd8'
            elif analysis_type == 'advanced':
                stream_threshold = 500
                output_formats = ['geojson', 'shapefile', 'geotiff']
                flow_algorithm = 'dinf'
            else:  # complete
                stream_threshold = 1000
                output_formats = ['geojson', 'geotiff']
                flow_algorithm = 'd8'
            
            # Run professional analysis
            results = self.professional_analyzer.analyze_watershed_complete(
                dem_path=dem_path,
                outlet_coords=outlet_coords,
                output_dir=output_dir,
                stream_threshold=stream_threshold,
                flow_algorithm=flow_algorithm,
                output_formats=output_formats
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Watershed analysis failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'analysis_type': analysis_type,
                'files_created': [],
                'metadata': {},
                'errors': [error_msg]
            }
    
    def get_watershed_analysis_requirements(self) -> Dict:
        """Get information about required libraries for watershed analysis"""
        requirements = {
            'required_libraries': {
                'whitebox': {
                    'package': 'whitebox',
                    'install': 'pip install whitebox',
                    'description': 'WhiteboxTools for comprehensive hydrological analysis',
                    'essential': True
                },
                'pyflwdir': {
                    'package': 'pyflwdir',
                    'install': 'pip install pyflwdir',
                    'description': 'High-performance flow analysis by Deltares',
                    'essential': False
                },
                'rasterio': {
                    'package': 'rasterio',
                    'install': 'pip install rasterio',
                    'description': 'Raster data I/O and processing',
                    'essential': True
                },
                'geopandas': {
                    'package': 'geopandas',
                    'install': 'pip install geopandas',
                    'description': 'Vector data processing',
                    'essential': True
                },
                'fiona': {
                    'package': 'fiona',
                    'install': 'pip install fiona',
                    'description': 'Vector file format support',
                    'essential': True
                },
                'shapely': {
                    'package': 'shapely',
                    'install': 'pip install shapely',
                    'description': 'Geometric operations',
                    'essential': True
                }
            },
            'installation_command': 'pip install whitebox pyflwdir rasterio geopandas fiona shapely',
            'professional_analyzer_available': HAS_PROFESSIONAL_ANALYZER,
            'capabilities': {
                'dem_preprocessing': 'Depression filling, breaching, conditioning',
                'flow_analysis': 'D8, D-infinity, and MFD algorithms',
                'stream_extraction': 'Threshold-based with multiple formats',
                'watershed_delineation': 'From outlet points with snapping',
                'subbasin_analysis': 'Automatic subbasin delineation',
                'stream_ordering': 'Strahler and Shreve ordering',
                'quality_assessment': 'Comprehensive validation and statistics',
                'output_formats': 'GeoJSON, Shapefile, GeoTIFF, ASCII'
            }
        }
        
        return requirements


# ===== 4. CONVENIENCE FUNCTIONS =====

def quick_watershed_analysis(dem_path: str, outlet_lon: float, outlet_lat: float, 
                           output_dir: str, **kwargs) -> Dict:
    """
    Convenience function for quick watershed analysis
    
    Parameters:
    -----------
    dem_path : str
        Path to DEM raster file
    outlet_lon : float
        Longitude of watershed outlet
    outlet_lat : float
        Latitude of watershed outlet  
    output_dir : str
        Directory to save analysis outputs
    **kwargs : additional arguments
        Additional arguments passed to professional analyzer
    
    Returns:
    --------
    Dict
        Analysis results with file paths and metadata
    
    Example:
    --------
    >>> results = quick_watershed_analysis(
    ...     'path/to/dem.tif', 
    ...     -75.5, 45.5, 
    ...     'output_directory',
    ...     analysis_type='complete',
    ...     stream_threshold=1000
    ... )
    """
    analyzer = DEMWatershedAnalyzer()
    return analyzer.analyze_watershed_from_dem(
        Path(dem_path), 
        (outlet_lon, outlet_lat), 
        Path(output_dir), 
        **kwargs
    )


def get_watershed_requirements() -> Dict:
    """Get information about watershed analysis requirements"""
    analyzer = DEMWatershedAnalyzer()
    return analyzer.get_watershed_analysis_requirements()


# ===== 5. UNIFIED WATERSHED CLIENT =====

class WhiteboxWatershedClient:
    """
    Unified client for all watershed-related operations
    
    Combines NRCAN Atlas, Hydrographic Network, and DEM-based analysis
    """
    
    def __init__(self):
        self.nrcan_client = NRCANWatershedClient()
        self.hydro_client = NRCANHydrographicClient()
        self.dem_analyzer = DEMWatershedAnalyzer()
        
        print("WhiteboxWatershedClient initialized")
        print(f"Professional analyzer available: {HAS_PROFESSIONAL_ANALYZER}")
        print(f"WhiteboxTools available: {HAS_WHITEBOX}")
        print(f"Vector tools available: {HAS_VECTOR_TOOLS}")
        print(f"Rasterio available: {HAS_RASTERIO}")
    
    def get_watershed_for_station(self, station_coords: Tuple[float, float], 
                                 output_path: Path, layer_level: str = "detailed") -> Dict:
        """Get watershed boundary from NRCAN Atlas"""
        return self.nrcan_client.get_watershed_for_station(station_coords, output_path, layer_level)
    
    def get_watersheds_in_bbox(self, bbox: Tuple[float, float, float, float], 
                              output_path: Path, layer_level: str = "detailed") -> Dict:
        """Get all watersheds in bounding box from NRCAN Atlas"""
        return self.nrcan_client.get_watersheds_in_bbox(bbox, output_path, layer_level)
    
    def get_hydro_network(self, bbox: Tuple[float, float, float, float], 
                         output_path: Path, layer_type: str = "watercourses") -> Dict:
        """Get hydrographic network from NRCAN"""
        return self.hydro_client.get_hydro_network_for_watershed(bbox, output_path, layer_type)
    
    def get_stream_network(self, bbox: Tuple[float, float, float, float], 
                          output_path: Path, min_order: int = 1) -> Dict:
        """Get stream network with order classification"""
        return self.hydro_client.get_stream_order_network(bbox, output_path, min_order)
    
    def get_waterbodies(self, bbox: Tuple[float, float, float, float], 
                       output_path: Path, min_area_km2: float = 0.1) -> Dict:
        """Get waterbodies from NRCAN"""
        return self.hydro_client.get_waterbodies_for_watershed(bbox, output_path, min_area_km2)
    
    def analyze_watershed_from_dem(self, dem_path: Path, outlet_coords: Tuple[float, float], 
                                  output_dir: Path, analysis_type: str = 'complete') -> Dict:
        """Perform DEM-based watershed analysis"""
        return self.dem_analyzer.analyze_watershed_from_dem(dem_path, outlet_coords, output_dir, analysis_type)
    
    def get_requirements(self) -> Dict:
        """Get watershed analysis requirements"""
        return self.dem_analyzer.get_watershed_analysis_requirements()


if __name__ == "__main__":
    # Example usage
    print("WhiteboxTools Watershed Client")
    print("This module provides comprehensive watershed analysis capabilities.")
    print("\nRequired libraries for full functionality:")
    print("- whitebox: pip install whitebox")
    print("- pyflwdir: pip install pyflwdir") 
    print("- rasterio: pip install rasterio")
    print("- geopandas: pip install geopandas")
    print("- fiona: pip install fiona")
    print("- shapely: pip install shapely")
    print("\nExample usage:")
    print("from clients.watershed_clients.whitebox_client import WhiteboxWatershedClient")
    print("client = WhiteboxWatershedClient()")
    print("results = client.analyze_watershed_from_dem('dem.tif', (-75.5, 45.5), 'output_dir')")
