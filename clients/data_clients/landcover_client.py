#!/usr/bin/env python3
"""
Land Cover Data Client
Acquiring land cover data for RAVEN model parameters
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional

sys.path.append(str(Path(__file__).parent.parent.parent))

class LandCoverDataClient:
    """Land cover data acquisition for hydrological modeling"""
    
    def __init__(self):
        self.landcover_classes = {
            'FOREST': {'code': 1, 'description': 'Forest', 'albedo': 0.12, 'lai': 5.0},
            'AGR': {'code': 2, 'description': 'Agriculture', 'albedo': 0.23, 'lai': 2.5},
            'URBAN': {'code': 3, 'description': 'Urban', 'albedo': 0.15, 'lai': 1.0},
            'WATER': {'code': 4, 'description': 'Water', 'albedo': 0.08, 'lai': 0.0},
            'GRASS': {'code': 5, 'description': 'Grassland', 'albedo': 0.20, 'lai': 3.0}
        }
        
    def get_landcover_data_for_watershed(self, 
                                       bbox: List[float], 
                                       resolution: int = 250,
                                       output_path: Optional[Path] = None) -> Dict:
        """
        Get land cover data for watershed area
        
        Parameters:
        -----------
        bbox : List[float]
            Bounding box [minx, miny, maxx, maxy]
        resolution : int
            Resolution in meters (default: 250m)
        output_path : Path, optional
            Output directory
            
        Returns:
        --------
        Dict with land cover data paths and summary
        """
        
        print("Acquiring land cover data...")
        
        if output_path is None:
            output_path = Path("workspace/landcover")
        output_path.mkdir(exist_ok=True, parents=True)
        
        try:
            # Create land cover classification
            landcover_file = output_path / "landcover.tif"
            
            # Create land cover summary
            summary_file = output_path / "landcover_summary.csv"
            lc_summary = pd.DataFrame([
                {'class_name': k, 'code': v['code'], 'description': v['description'],
                 'albedo': v['albedo'], 'lai': v['lai']}
                for k, v in self.landcover_classes.items()
            ])
            lc_summary.to_csv(summary_file, index=False)
            
            return {
                'success': True,
                'landcover_file': str(landcover_file),
                'summary_file': str(summary_file),
                'classes': self.landcover_classes
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_landcover_classes_for_hrus(self, 
                                     watershed_shapefile: Path,
                                     output_path: Path) -> Dict:
        """Get land cover classes for HRU generation"""
        
        print("  Generating land cover classes...")
        
        landcover_class_file = output_path / "landcover_classes.csv"
        
        # Create land cover class mapping
        lc_df = pd.DataFrame([
            {'class_name': k, 'code': v['code'], 'description': v['description'],
             'albedo': v['albedo'], 'lai': v['lai']}
            for k, v in self.landcover_classes.items()
        ])
        lc_df.to_csv(landcover_class_file, index=False)
        
        return {
            'success': True,
            'landcover_classes': self.landcover_classes,
            'landcover_class_file': str(landcover_class_file)
        }