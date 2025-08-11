#!/usr/bin/env python3
"""
Soil Data Client for RAVEN Hydrological Modeling
Uses SoilGrids REST API for real soil property data
"""

import requests
import rasterio
from rasterio.transform import from_origin
import numpy as np
import json
import os
from typing import Dict, Tuple, List, Optional

class SoilDataClient:
    """
    A client to fetch soil texture raster data (clay, sand, silt) using SoilGrids WCS service.
    Provides high-resolution raster maps for watershed modeling applications.
    """
    def __init__(self, output_dir: str = 'output'):
        self.session = requests.Session()
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # SoilGrids WCS endpoint for raster data
        self.wcs_endpoint_base = "https://maps.isric.org/mapserv"
        
        # Available soil texture properties
        self.texture_properties = ['clay', 'sand', 'silt']
        self.available_depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']

    def get_soil_texture_rasters_for_bbox(self, bbox: Tuple[float, float, float, float], 
                                          depth: str = '0-5cm', width: int = 256, height: int = 256) -> Dict[str, Optional[str]]:
        """
        Fetch clay, sand, and silt raster maps for a bounding box using SoilGrids WCS service.
        
        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            depth: Depth interval (default: '0-5cm')
            width: Raster width in pixels
            height: Raster height in pixels
            
        Returns:
            Dictionary with paths to clay, sand, and silt raster files
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        
        print(f"Fetching soil texture rasters for bbox: {bbox}")
        print(f"Depth: {depth}, Resolution: {width}x{height}")
        
        results = {}
        
        for prop in self.texture_properties:
            print(f"\nRequesting {prop} raster...")
            raster_path = self.get_soil_map_for_bbox(bbox, prop, depth, width, height)
            results[prop] = raster_path
            
            if raster_path:
                print(f"SUCCESS: {prop.capitalize()} raster saved: {raster_path}")
                
                # Quick stats
                try:
                    with rasterio.open(raster_path) as src:
                        data = src.read(1, masked=True)
                        if data.count() > 0:
                            print(f"   Data range: {data.min():.1f} - {data.max():.1f}%")
                            print(f"   Mean: {data.mean():.1f}%")
                        else:
                            print(f"   No valid data in this area")
                except Exception as e:
                    print(f"   Could not read raster stats: {e}")
            else:
                print(f"ERROR: Failed to download {prop} raster")
        
        return results

    def create_soil_texture_dataset(self, bbox: Tuple[float, float, float, float], 
                                   depths: List[str] = ['0-5cm', '5-15cm', '15-30cm']) -> Dict:
        """
        Create a complete soil texture dataset with clay, sand, and silt for multiple depths.
        
        Args:
            bbox: Bounding box for the area
            depths: List of depth intervals to fetch
            
        Returns:
            Dictionary with all raster files organized by depth and property
        """
        print(f"Creating comprehensive soil texture dataset for bbox: {bbox}")
        print(f"Depths: {depths}")
        
        dataset = {
            'bbox': bbox,
            'depths': {},
            'summary': {
                'total_files': 0,
                'successful_downloads': 0,
                'failed_downloads': 0
            }
        }
        
        for depth in depths:
            print(f"\n{'='*50}")
            print(f"Processing depth: {depth}")
            print(f"{'='*50}")
            
            depth_results = self.get_soil_texture_rasters_for_bbox(bbox, depth)
            dataset['depths'][depth] = depth_results
            
            # Update summary
            for prop, path in depth_results.items():
                dataset['summary']['total_files'] += 1
                if path:
                    dataset['summary']['successful_downloads'] += 1
                else:
                    dataset['summary']['failed_downloads'] += 1
        
        # Create summary report
        summary_file = os.path.join(self.output_dir, "soil_texture_summary.json")
        with open(summary_file, 'w') as f:
            json.dump({
                'bbox': dataset['bbox'],
                'depths_processed': depths,
                'summary': dataset['summary'],
                'file_paths': dataset['depths']
            }, f, indent=2)
        
        dataset['summary_file'] = summary_file
        
        print(f"\n{'='*60}")
        print(f"DATASET CREATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total files requested: {dataset['summary']['total_files']}")
        print(f"Successful downloads: {dataset['summary']['successful_downloads']}")
        print(f"Failed downloads: {dataset['summary']['failed_downloads']}")
        print(f"Summary saved: {summary_file}")
        
        return dataset

    def download_slc_national_dataset(self, output_filename: str = "slc_v32_canada.zip") -> Optional[str]:
        """
        Download the complete Soil Landscapes of Canada v3.2 national dataset.
        This is a large file (~200MB) with complete Canadian coverage.
        
        Args:
            output_filename: Output filename for the downloaded zip file
            
        Returns:
            Path to the downloaded file, or None if failed
        """
        url = self.canadian_sources['slc_v32']['url']
        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            print(f"Downloading SLC v3.2 national dataset...")
            print(f"Source: {url}")
            print(f"This may take several minutes for the full dataset...")
            
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='')
            
            print(f"\nSUCCESS: SLC v3.2 dataset downloaded: {output_path}")
            print(f"   File size: {os.path.getsize(output_path):,} bytes")
            print(f"   Extract this ZIP file to access soil polygon shapefiles")
            
            return output_path
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading SLC dataset: {e}")
            return None

    def extract_soil_properties_from_polygons(self, geojson_file: str) -> Dict:
        """
        Extract and summarize soil properties from downloaded polygon data.
        
        Args:
            geojson_file: Path to the GeoJSON file with soil polygons
            
        Returns:
            Dictionary with soil property summary and statistics
        """
        try:
            with open(geojson_file, 'r') as f:
                data = json.load(f)
            
            if 'features' not in data or not data['features']:
                return {'error': 'No features found in GeoJSON file'}
            
            features = data['features']
            property_summary = {
                'total_polygons': len(features),
                'available_properties': set(),
                'drainage_classes': {},
                'parent_materials': {},
                'texture_classes': {},
                'slope_classes': {},
                'soil_types': {}
            }
            
            # Analyze all properties across features
            for feature in features:
                props = feature.get('properties', {})
                property_summary['available_properties'].update(props.keys())
                
                # Count drainage classes
                for drain_field in ['DRAINAGE', 'DRAIN', 'DRAINCLASS']:
                    if drain_field in props and props[drain_field]:
                        drain_val = str(props[drain_field])
                        property_summary['drainage_classes'][drain_val] = \
                            property_summary['drainage_classes'].get(drain_val, 0) + 1
                
                # Count parent materials
                for pm_field in ['PARENT_MAT', 'PM1', 'PARENTMAT']:
                    if pm_field in props and props[pm_field]:
                        pm_val = str(props[pm_field])
                        property_summary['parent_materials'][pm_val] = \
                            property_summary['parent_materials'].get(pm_val, 0) + 1
                
                # Count texture classes
                for texture_field in ['TEXTURE', 'TEXTURAL', 'TEXT_CLASS']:
                    if texture_field in props and props[texture_field]:
                        texture_val = str(props[texture_field])
                        property_summary['texture_classes'][texture_val] = \
                            property_summary['texture_classes'].get(texture_val, 0) + 1
                
                # Count slope classes
                for slope_field in ['SLOPE', 'SLOPE_CLASS', 'SLOPECLASS']:
                    if slope_field in props and props[slope_field]:
                        slope_val = str(props[slope_field])
                        property_summary['slope_classes'][slope_val] = \
                            property_summary['slope_classes'].get(slope_val, 0) + 1
            
            # Convert set to list for JSON serialization
            property_summary['available_properties'] = list(property_summary['available_properties'])
            
            return property_summary
            
        except Exception as e:
            return {'error': f'Failed to analyze soil polygons: {str(e)}'}

    def create_comprehensive_canadian_soil_dataset(self, bbox: Tuple[float, float, float, float]) -> Dict:
        """
        Create a comprehensive Canadian soil dataset for a watershed including:
        - Polygon data with detailed soil properties
        - Property analysis and summary
        - Hydrological property derivations
        
        Args:
            bbox: Bounding box for the watershed
            
        Returns:
            Dictionary with all soil data and analysis
        """
        print(f"Creating comprehensive Canadian soil dataset for bbox: {bbox}")
        
        results = {
            'bbox': bbox,
            'data_sources': [],
            'files': {},
            'property_analysis': {},
            'success': False
        }
        
        # 1. Download polygon data for the specific area
        polygon_file = self.get_canadian_soil_polygons_for_bbox(
            bbox, 
            output_filename=f"soil_polygons_{bbox[0]:.3f}_{bbox[1]:.3f}.geojson"
        )
        
        if polygon_file:
            results['files']['soil_polygons'] = polygon_file
            results['data_sources'].append('AAFC Soil Landscapes of Canada REST Service')
            
            # 2. Analyze the polygon properties
            analysis = self.extract_soil_properties_from_polygons(polygon_file)
            results['property_analysis'] = analysis
            
            # 3. Create summary report
            summary_file = os.path.join(self.output_dir, "soil_property_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            results['files']['property_summary'] = summary_file
            
            results['success'] = True
            print(f"SUCCESS: Comprehensive soil dataset created successfully")
            
        else:
            print("ERROR: Failed to download soil polygon data")
        
        return results

    def get_soil_map_for_bbox(self, bbox: Tuple[float, float, float, float], 
                              prop: str, depth: str, width: int = 256, height: int = 256) -> Optional[str]:
        """
        Fetches a soil property map for a given bounding box using the WCS service.
        Each soil property has its own map file on the server.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        
        coverage_id = f"{prop}_{depth}_mean"
        
        wcs_url = f"{self.wcs_endpoint_base}?map=/map/{prop}.map"

        # Corrected parameters based on WCS documentation:
        # - Use 'X' and 'Y' for subset dimensions.
        # - Specify SUBSETTINGCRS for the input bounding box coordinates.
        # - Specify OUTPUTCRS for the desired output projection.
        params = [
            ('service', 'WCS'),
            ('request', 'GetCoverage'),
            ('version', '2.0.1'),
            ('coverageId', coverage_id),
            ('subset', f"Y({min_lat},{max_lat})"),
            ('subset', f"X({min_lon},{max_lon})"),
            ('subsettingcrs', 'http://www.opengis.net/def/crs/EPSG/0/4326'),
            ('outputCrs', 'http://www.opengis.net/def/crs/EPSG/0/4326'),
            ('format', 'image/tiff'),
            ('width', str(width)),
            ('height', str(height))
        ]
        
        try:
            print(f"Requesting soil map from WCS for property: {prop}, depth: {depth}")
            
            req = requests.Request('GET', wcs_url, params=params)
            prepared = req.prepare()
            print(f"WCS Request URL: {prepared.url}")

            response = self.session.send(prepared, stream=True)
            response.raise_for_status()

            if 'image/tiff' not in response.headers.get('Content-Type', ''):
                error_text = response.text
                print(f"WCS service did not return a GeoTIFF. Response:\n{error_text}")
                return None

            output_filename = os.path.join(self.output_dir, f"{coverage_id}_bbox.tif")
            
            with open(output_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Successfully downloaded soil map to: {output_filename}")
            return output_filename

        except requests.exceptions.RequestException as e:
            print(f"Error querying WCS service: {e}")
            return None


# Test the soil client with real API calls
if __name__ == '__main__':
    client = SoilDataClient(output_dir='soil_output')

    # --- Test Case 1: Point data for a location in a non-urban area ---
    print("--- Testing Point Data ---")
    rural_lat, rural_lon = 52.5, -106.0
    print(f"Querying soil properties for point: ({rural_lat}, {rural_lon})")
    soil_data = client.get_soil_properties_for_point(rural_lat, rural_lon)

    if soil_data:
        print("Successfully retrieved soil data.")
        if 'sand' in soil_data and soil_data['sand']['depths']:
            sand_content = soil_data['sand']['depths'].get('0-5cm')
            unit = soil_data['sand']['unit']
            if sand_content is not None:
                print(f"Sand content at 0-5cm: {sand_content:.2f} {unit}")
            else:
                print("Sand content at 0-5cm is NULL.")
        else:
            print("Sand content not available for this location.")
    else:
        print("Failed to retrieve soil data for the point.")

    print("\n" + "="*50 + "\n")

    # --- Test Case 2: Bounding box query using WCS ---
    print("--- Testing Bounding Box (WCS) Data ---")
    bbox = (-106.1, 52.4, -106.0, 52.5)
    prop_to_get = 'clay'
    depth_to_get = '0-5cm'
    
    print(f"Requesting '{prop_to_get}' map for bbox: {bbox}")
    raster_path = client.get_soil_map_for_bbox(bbox, prop=prop_to_get, depth=depth_to_get)

    if raster_path and os.path.exists(raster_path):
        print(f"Successfully created soil raster map: {raster_path}")
        try:
            with rasterio.open(raster_path) as src:
                print(f"  - Raster dimensions: {src.width}x{src.height}")
                print(f"  - CRS: {src.crs}")
                print(f"  - Bounding box: {src.bounds}")
                data = src.read(1, masked=True)
                # Check if there is any unmasked data
                if data.count() > 0:
                    print(f"  - Min: {data.min():.2f}, Max: {data.max():.2f}, Mean: {data.mean():.2f}")
                else:
                    print(f"  - Min: N/A, Max: N/A, Mean: N/A (all values masked or no data)")
        except rasterio.errors.RasterioIOError as e:
            print(f"  - Could not read raster file details: {e}")
    else:
        print("Failed to create soil raster map from WCS.")