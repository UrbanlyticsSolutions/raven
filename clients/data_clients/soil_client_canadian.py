#!/usr/bin/env python3
"""
Canadian Soil Data Client for RAVEN Hydrological Modeling
Uses Canadian government soil data sources only
"""

import requests
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SoilDataClient:
    """Client for Canadian soil data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAVEN-Hydrological-Model-Client/1.0'
        })
        
        # Canadian soil data endpoints
        self.canadian_endpoints = {
            'aafc_rest': 'https://services.arcgis.com/lGOekm0RsNxYnT3j/arcgis/rest/services',
            'cansis_base': 'https://sis.agr.gc.ca/cansis'
        }
    
    def get_canadian_soil_data(self, coords: Tuple[float, float]) -> Dict:
        """Get soil data from Canadian government sources"""
        lat, lon = coords
        
        # Check if coordinates are in Canada
        if not (-141 <= lon <= -52 and 41.5 <= lat <= 83.5):
            return {
                'success': False,
                'error': 'Coordinates are outside Canada bounds',
                'coordinates': coords
            }
        
        print(f"Querying Canadian soil data for point ({lat:.6f}, {lon:.6f})")
        
        try:
            # Try AAFC Soil Landscapes of Canada
            result = self._query_aafc_soil_data(coords)
            if result.get('success'):
                return result
            
            # If AAFC fails, try CanSIS data
            result = self._query_cansis_data(coords)
            if result.get('success'):
                return result
            
            # If both fail, return error
            return {
                'success': False,
                'error': 'No Canadian soil data available for this location',
                'coordinates': coords
            }
            
        except Exception as e:
            error_msg = f"Canadian soil data query failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'success': False, 'error': error_msg, 'coordinates': coords}
    
    def _query_aafc_soil_data(self, coords: Tuple[float, float]) -> Dict:
        """Query AAFC soil data services"""
        lat, lon = coords
        
        try:
            # Try Soil Landscapes of Canada service first
            url = f"{self.canadian_endpoints['aafc_rest']}/Soil_landscapes_of_Canada/FeatureServer/0/query"
            
            params = {
                'geometry': f'{lon},{lat}',
                'geometryType': 'esriGeometryPoint',
                'spatialRel': 'esriSpatialRelIntersects',
                'outFields': '*',
                'returnGeometry': 'false',
                'f': 'json'
            }
            
            print(f"  Querying Soil Landscapes of Canada service...")
            print(f"  URL: {url}")
            
            response = self.session.get(url, params=params, timeout=30)
            print(f"  Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"  Response text: {response.text[:500]}")
                return {
                    'success': False,
                    'error': f'AAFC Soil Landscapes request failed with status {response.status_code}'
                }
            
            data = response.json()
            print(f"  Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            if 'features' in data and len(data['features']) > 0:
                feature = data['features'][0]
                attributes = feature.get('attributes', {})
                print(f"  Feature attributes: {list(attributes.keys())}")
                
                # Extract soil-related attributes from Soil Landscapes
                soil_data = {}
                
                # Include all attributes since we don't know the exact field names
                for field, value in attributes.items():
                    if value is not None and field != 'OBJECTID':
                        soil_data[field.lower()] = value
                
                if soil_data:
                    result = {
                        'success': True,
                        'coordinates': coords,
                        'soil_properties': soil_data,
                        'source': 'Agriculture and Agri-Food Canada - Soil Landscapes of Canada',
                        'service_url': url
                    }
                    
                    print(f"  SUCCESS: Soil Landscapes data found")
                    print(f"    Available fields: {list(soil_data.keys())}")
                    for key, value in soil_data.items():
                        print(f"    {key}: {value}")
                    
                    return result
                else:
                    return {
                        'success': False,
                        'error': 'No soil data found in feature attributes'
                    }
            else:
                return {
                    'success': False,
                    'error': 'No features found in AAFC Soil Landscapes response'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'AAFC Soil Landscapes query failed: {str(e)}'
            }
    
    # REMOVED: _query_aafc_land_capability - NO FALLBACK METHODS ALLOWED
    
    def _query_cansis_data(self, coords: Tuple[float, float]) -> Dict:
        """Query CanSIS soil data (placeholder - would need specific API endpoints)"""
        # Note: CanSIS doesn't have a direct REST API, would need to download
        # and process spatial datasets
        return {
            'success': False,
            'error': 'CanSIS direct API not available - requires dataset download'
        }
    
    def get_soil_properties_for_point(self, coords: Tuple[float, float], 
                                     output_path: Optional[Path] = None) -> Dict:
        """Get soil properties from Canadian government sources"""
        lat, lon = coords
        
        # Check if coordinates are in Canada
        if not (-141 <= lon <= -52 and 41.5 <= lat <= 83.5):
            return {
                'success': False,
                'error': 'Coordinates are outside Canada bounds',
                'coordinates': coords
            }
        
        try:
            print(f"Querying Canadian soil data for point ({lat:.6f}, {lon:.6f})")
            canadian_result = self.get_canadian_soil_data(coords)
            
            if canadian_result.get('success'):
                # Save to file if requested
                if output_path:
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    with open(output_path, 'w') as f:
                        json.dump(canadian_result, f, indent=2)
                    print(f"Soil properties saved to: {output_path}")
                    canadian_result['file_path'] = str(output_path)
                return canadian_result
            else:
                return {
                    'success': False,
                    'error': f"Canadian soil data query failed: {canadian_result.get('error')}",
                    'coordinates': coords
                }
            
        except Exception as e:
            error_msg = f"Soil properties retrieval failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'success': False, 'error': error_msg, 'coordinates': coords}


# Test the Canadian soil client with real API calls
if __name__ == '__main__':
    print("Testing Canadian Soil Data Client with real API calls...")
    
    client = SoilDataClient()
    
    # Test locations in Canada - specifically agricultural areas
    test_locations = [
        (52.1579, -106.6702),  # Saskatoon, SK - agricultural region
        (49.8951, -97.1384),   # Winnipeg, MB - agricultural region  
        (43.6532, -79.3832),   # Toronto, ON - settled area
        (51.0447, -113.4909),  # Calgary, AB - near agricultural areas
    ]
    
    for coords in test_locations:
        print(f"\n{'='*60}")
        print(f"Testing location: {coords}")
        print(f"{'='*60}")
        
        result = client.get_soil_properties_for_point(coords)
        
        if result['success']:
            print(f"SUCCESS: Got soil data for {coords}")
            print(f"Source: {result['source']}")
            print(f"Properties: {result['soil_properties']}")
        else:
            print(f"FAILED: {result['error']}")
    
    print(f"\n{'='*60}")
    print("Canadian Soil Data Client testing complete")
