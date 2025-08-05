#!/usr/bin/env python3
"""
Soil Data Client for RAVEN Hydrological Modeling
Provides coordinate-based soil classification using Canadian and FAO systems
Includes estimated soil properties for modeling
"""

import requests
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SoilDataClient:
    """Client for soil data using coordinate-based classification"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAVEN-Hydrological-Model-Client/1.0'
        })
    
    def get_soil_classification_for_point(self, coords: Tuple[float, float], 
                                         output_path: Optional[Path] = None) -> Dict:
        """Get soil classification for a point using coordinate-based lookup"""
        lat, lon = coords
        print(f"Getting soil classification for point ({lat:.6f}, {lon:.6f})")
        
        try:
            result = {}
            
            # Canadian Soil Classification System
            if -141 <= lon <= -60 and 42 <= lat <= 83:  # Canada bounds
                if lat > 60:
                    soil_type = "Cryosol"
                    description = "Soils of very cold climates with permafrost"
                elif -95 <= lon <= -75:  # Prairie/Central Canada
                    soil_type = "Chernozem" 
                    description = "Dark, fertile grassland soils"
                elif lon < -95:  # Western Canada
                    soil_type = "Luvisol"
                    description = "Forest soils with clay accumulation"
                else:  # Eastern Canada
                    soil_type = "Podzol"
                    description = "Acidic forest soils with distinct horizons"
                
                result['canadian_classification'] = {
                    'soil_order': soil_type,
                    'description': description,
                    'confidence': 'Low (regional approximation)',
                    'source': 'Canadian Soil Classification System'
                }
            
            # FAO World Reference Base
            if 23.5 <= lat <= 66.5:  # Temperate zone
                if -180 <= lon <= -30:  # Americas
                    fao_unit = "Luvisols"
                    fao_desc = "Soils with clay accumulation in subsoil"
                elif -30 <= lon <= 60:  # Europe/Africa  
                    fao_unit = "Cambisols"
                    fao_desc = "Young soils with beginning of horizon development"
                else:  # Asia/Pacific
                    fao_unit = "Acrisols"
                    fao_desc = "Acidic soils with low fertility"
            else:
                fao_unit = "Cryosols"
                fao_desc = "Soils of very cold climates"
                
            result['fao_classification'] = {
                'soil_unit': fao_unit,
                'description': fao_desc,
                'climate_zone': 'Temperate' if 23.5 <= lat <= 66.5 else 'Polar/Tropical',
                'source': 'FAO World Reference Base (simplified)'
            }
            
            # Add coordinate info
            result['coordinates'] = {
                'latitude': lat,
                'longitude': lon,
                'decimal_degrees': True
            }
            
            # Save to file if requested
            if output_path:
                output_path.parent.mkdir(exist_ok=True, parents=True)
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Soil classification saved to: {output_path}")
            
            print(f"SUCCESS: Soil classification found")
            if 'canadian_classification' in result:
                print(f"Canadian: {result['canadian_classification']['soil_order']}")
            print(f"FAO: {result['fao_classification']['soil_unit']}")
            
            return {
                'success': True,
                'coordinates': coords,
                'classification': result,
                'file_path': str(output_path) if output_path else None
            }
            
        except Exception as e:
            error_msg = f"Soil classification failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def get_soil_properties_estimate(self, coords: Tuple[float, float]) -> Dict:
        """Get estimated soil properties based on classification"""
        lat, lon = coords
        
        try:
            # Get classification first
            classification = self.get_soil_classification_for_point(coords)
            
            if not classification.get('success'):
                return classification
            
            soil_data = classification['classification']
            
            # Estimate properties based on soil type
            if 'canadian_classification' in soil_data:
                soil_order = soil_data['canadian_classification']['soil_order']
                
                # Typical property ranges for Canadian soil orders
                properties = {
                    'Chernozem': {'ph': 7.2, 'organic_matter': 4.5, 'clay': 25, 'drainage': 'well'},
                    'Podzol': {'ph': 4.8, 'organic_matter': 8.0, 'clay': 15, 'drainage': 'rapid'},
                    'Luvisol': {'ph': 6.5, 'organic_matter': 3.2, 'clay': 35, 'drainage': 'moderate'},
                    'Cryosol': {'ph': 6.0, 'organic_matter': 12.0, 'clay': 20, 'drainage': 'poor'}
                }
                
                if soil_order in properties:
                    props = properties[soil_order]
                    result = {
                        'success': True,
                        'coordinates': coords,
                        'soil_properties': {
                            'ph': props['ph'],
                            'organic_matter_percent': props['organic_matter'],
                            'clay_percent': props['clay'],
                            'drainage_class': props['drainage'],
                            'source': f'Typical values for {soil_order}',
                            'confidence': 'Very Low (estimated from soil order)'
                        }
                    }
                    
                    print(f"SUCCESS: Soil properties estimated (pH: {props['ph']}, OM: {props['organic_matter']}%, Clay: {props['clay']}%)")
                    
                    return result
            
            return {'success': False, 'error': 'Unable to estimate properties'}
            
        except Exception as e:
            error_msg = f"Soil properties estimation failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'success': False, 'error': error_msg}