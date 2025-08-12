#!/usr/bin/env python3
"""
BasinMaker HRU Consolidation Implementation
Reduces HRU count while maintaining complete coverage using area thresholds and importance ordering
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class BasinMakerHRUConsolidator:
    """
    Implementation of BasinMaker HRU consolidation algorithm
    Based on simplidfy_hrus function from BasinMaker
    """
    
    def __init__(self, min_hru_pct_sub_area: float = 0.10, 
                 importance_order: List[str] = None):
        """
        Initialize consolidator
        
        Args:
            min_hru_pct_sub_area: Minimum HRU area as fraction of subbasin area (0.10 = 10%)
            importance_order: Priority order for attribute matching
        """
        self.min_hru_pct_sub_area = min_hru_pct_sub_area
        self.importance_order = importance_order or ['Landuse_ID', 'Soil_ID', 'Veg_ID']
        
    def consolidate_hrus(self, hru_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Apply BasinMaker consolidation to reduce HRU count
        
        Args:
            hru_gdf: Input HRU GeoDataFrame with SubId, HRU_Area, and attribute columns
            
        Returns:
            Consolidated HRU GeoDataFrame with same geometry but merged attributes
        """
        logger.info(f"Starting HRU consolidation with {len(hru_gdf)} initial HRUs")
        logger.info(f"Target: {self.min_hru_pct_sub_area*100}% minimum area per subbasin")
        logger.info(f"Importance order: {self.importance_order}")
        
        # Create working copy
        hruinfo = hru_gdf.copy(deep=True)
        hruinfo['HRU_ID_New2'] = hruinfo['HRU_ID']
        
        # Get unique subbasin IDs
        subids = np.unique(hruinfo['SubId'].values)
        logger.info(f"Processing {len(subids)} subbasins")
        
        consolidation_stats = {
            'initial_hrus': len(hruinfo),
            'subbasins_processed': 0,
            'hrus_consolidated': 0,
            'final_hrus': 0
        }
        
        # Process each subbasin
        for i, subid in enumerate(subids):
            logger.info(f"Processing subbasin {subid} ({i+1}/{len(subids)})")
            
            # Get HRUs for this subbasin
            sub_hru_info = hruinfo.loc[hruinfo['SubId'] == subid].copy(deep=True)
            initial_hru_count = len(sub_hru_info)
            
            # Calculate area threshold for this subbasin
            subasin_area = np.sum(sub_hru_info['HRU_Area'].values)
            subarea_thrs = self.min_hru_pct_sub_area * subasin_area
            
            logger.info(f"  Subbasin area: {subasin_area:.3f} km²")
            logger.info(f"  Area threshold: {subarea_thrs:.3f} km² ({self.min_hru_pct_sub_area*100}%)")
            logger.info(f"  Initial HRUs: {initial_hru_count}")
            
            # Classify HRUs
            small_hrus = sub_hru_info.loc[sub_hru_info['HRU_Area'] < subarea_thrs].copy()
            good_hrus = sub_hru_info.loc[sub_hru_info['HRU_Area'] >= subarea_thrs].copy()
            
            logger.info(f"  Small HRUs: {len(small_hrus)}, Good HRUs: {len(good_hrus)}")
            
            if len(small_hrus) == 0:
                logger.info(f"  No consolidation needed for subbasin {subid}")
                consolidation_stats['subbasins_processed'] += 1
                continue
                
            # Get attribute columns (exclude geometry and ID columns)
            hru_columns = good_hrus.columns
            exclude_cols = ['HRU_ID_New2', 'geometry', 'SHAPE']
            hru_columns = [col for col in hru_columns if col not in exclude_cols]
            
            # Phase 1: Group small HRUs by most important attribute
            colnm1 = self.importance_order[0]
            logger.info(f"  Phase 1: Grouping by {colnm1}")
            
            small_hrus_remaining = small_hrus.copy(deep=True)
            unique_import1 = np.unique(small_hrus_remaining[colnm1].values)
            
            for import_val in unique_import1:
                # Get all small HRUs with this attribute value
                same_attr_hrus = small_hrus_remaining.loc[
                    small_hrus_remaining[colnm1] == import_val
                ].copy(deep=True)
                
                total_area = np.sum(same_attr_hrus['HRU_Area'].values)
                
                # If combined area meets threshold, consolidate them
                if total_area >= subarea_thrs:
                    logger.info(f"    Consolidating {len(same_attr_hrus)} HRUs with {colnm1}={import_val} (total area: {total_area:.3f} km²)")
                    
                    # Sort by area (largest first) to use as template
                    same_attr_hrus = same_attr_hrus.sort_values(by=['HRU_Area'], ascending=False)
                    template_attrs = same_attr_hrus.iloc[0]
                    
                    # Dissolve geometries and apply template attributes
                    hru_ids_to_update = same_attr_hrus['HRU_ID_New2'].values
                    
                    # Dissolve the geometries of HRUs in this group
                    dissolved_geom = same_attr_hrus.geometry.union_all()
                    
                    # Keep only the largest HRU and update its geometry to the dissolved union
                    template_id = same_attr_hrus.iloc[0]['HRU_ID_New2']
                    hruinfo.loc[hruinfo['HRU_ID_New2'] == template_id, 'geometry'] = dissolved_geom
                    
                    # Remove the other HRUs in this group (they're now merged into the template)
                    other_ids = hru_ids_to_update[hru_ids_to_update != template_id]
                    hruinfo = hruinfo[~hruinfo['HRU_ID_New2'].isin(other_ids)]
                    
                    # Apply template attributes to the remaining merged HRU
                    for col in hru_columns:
                        if col in template_attrs.index:
                            hruinfo.loc[hruinfo['HRU_ID_New2'] == template_id, col] = template_attrs[col]
                    
                    # Remove from remaining small HRUs
                    small_hrus_remaining = small_hrus_remaining[
                        ~small_hrus_remaining['HRU_ID_New2'].isin(hru_ids_to_update)
                    ]
                    consolidation_stats['hrus_consolidated'] += len(same_attr_hrus) - 1
            
            # Phase 2: Merge remaining small HRUs to good HRUs
            logger.info(f"  Phase 2: Merging {len(small_hrus_remaining)} remaining small HRUs to good HRUs")
            
            for idx in small_hrus_remaining.index:
                hruid = small_hrus_remaining.loc[idx, 'HRU_ID_New2']
                
                # Find best matching good HRU using importance order
                target_hru = None
                for importance_attr in self.importance_order:
                    if importance_attr not in small_hrus_remaining.columns:
                        continue
                        
                    small_hru_value = small_hrus_remaining.loc[idx, importance_attr]
                    
                    # Find good HRUs with matching attribute
                    matching_good_hrus = good_hrus.loc[
                        good_hrus[importance_attr] == small_hru_value
                    ].copy()
                    
                    if len(matching_good_hrus) > 0:
                        # Use largest matching good HRU as template
                        target_hru = matching_good_hrus.sort_values(
                            by='HRU_Area', ascending=False
                        ).iloc[0]
                        logger.debug(f"    Merging HRU {hruid} to good HRU based on {importance_attr}={small_hru_value}")
                        break
                
                # If no matching good HRU found, use largest good HRU
                if target_hru is None:
                    target_hru = good_hrus.sort_values(by='HRU_Area', ascending=False).iloc[0]
                    logger.debug(f"    Merging HRU {hruid} to largest good HRU (no attribute match)")
                
                # Merge this small HRU geometry into the target HRU
                small_hru_geom = small_hrus_remaining.loc[idx, 'geometry']
                target_hru_id = target_hru['HRU_ID_New2']
                
                # Get current target geometry and merge with small HRU
                current_target_geom = hruinfo.loc[hruinfo['HRU_ID_New2'] == target_hru_id, 'geometry'].iloc[0]
                merged_geom = current_target_geom.union(small_hru_geom)
                
                # Update target HRU geometry
                hruinfo.loc[hruinfo['HRU_ID_New2'] == target_hru_id, 'geometry'] = merged_geom
                
                # Remove the small HRU (it's now merged into target)
                hruinfo = hruinfo[hruinfo['HRU_ID_New2'] != hruid]
                
                # Apply target HRU attributes (already handled by keeping target HRU)
                
                consolidation_stats['hrus_consolidated'] += 1
            
            final_hru_count = len(good_hrus) + len([g for g in small_hrus.groupby(colnm1) 
                                                  if np.sum(g[1]['HRU_Area']) >= subarea_thrs])
            logger.info(f"  Consolidated {initial_hru_count} → ~{final_hru_count} effective HRUs")
            consolidation_stats['subbasins_processed'] += 1
        
        # Clean up and return
        hruinfo = hruinfo.drop(columns=['HRU_ID_New2'])
        consolidation_stats['final_hrus'] = len(hruinfo)
        
        logger.info("=== CONSOLIDATION SUMMARY ===")
        logger.info(f"Initial HRUs: {consolidation_stats['initial_hrus']}")
        logger.info(f"Final HRUs: {consolidation_stats['final_hrus']}")
        logger.info(f"HRUs consolidated: {consolidation_stats['hrus_consolidated']}")
        logger.info(f"Subbasins processed: {consolidation_stats['subbasins_processed']}")
        
        return hruinfo

def consolidate_hrus_basinmaker_method(
    hru_geojson_path: str,
    output_path: str,
    min_hru_pct_sub_area: float = 0.10,
    importance_order: List[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to consolidate HRUs from file
    
    Args:
        hru_geojson_path: Path to input HRU GeoJSON file
        output_path: Path for consolidated output
        min_hru_pct_sub_area: Minimum HRU area as fraction of subbasin area
        importance_order: Priority order for attribute matching
        
    Returns:
        Dictionary with consolidation statistics
    """
    # Load HRUs
    logger.info(f"Loading HRUs from {hru_geojson_path}")
    hru_gdf = gpd.read_file(hru_geojson_path)
    
    # Apply consolidation
    consolidator = BasinMakerHRUConsolidator(
        min_hru_pct_sub_area=min_hru_pct_sub_area,
        importance_order=importance_order
    )
    
    consolidated_hrus = consolidator.consolidate_hrus(hru_gdf)
    
    # Save results
    logger.info(f"Saving consolidated HRUs to {output_path}")
    consolidated_hrus.to_file(output_path, driver='GeoJSON')
    
    # Calculate final statistics
    stats = {
        'initial_hru_count': len(hru_gdf),
        'final_hru_count': len(consolidated_hrus),
        'consolidation_ratio': len(consolidated_hrus) / len(hru_gdf),
        'subbasin_count': len(np.unique(consolidated_hrus['SubId'])),
        'avg_hrus_per_subbasin': len(consolidated_hrus) / len(np.unique(consolidated_hrus['SubId'])),
        'min_hru_pct_threshold': min_hru_pct_sub_area,
        'importance_order': importance_order or ['Landuse_ID', 'Soil_ID', 'Veg_ID']
    }
    
    return stats