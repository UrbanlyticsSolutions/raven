#!/usr/bin/env python3
"""
Subbasin Grouper - Extracted from BasinMaker
Groups subbasins by channel length and lake area using your existing infrastructure
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np


class SubbasinGrouper:
    """
    Group subbasins by various criteria using extracted BasinMaker logic
    Adapted to work with your existing geopandas infrastructure
    """
    
    def __init__(self):
        pass
    
    def create_subbasin_groups(self, subbasins_gdf: gpd.GeoDataFrame,
                              group_config: Dict) -> str:
        """
        Create subbasin groups by channel length and lake area
        EXTRACTED FROM: Create_Subbasin_Groups() in BasinMaker
        
        Parameters:
        -----------
        subbasins_gdf : gpd.GeoDataFrame
            Subbasins with attributes including RivLength, LakeArea
        group_config : Dict
            Configuration for grouping with keys:
            - 'channel_length_groups': List of group names
            - 'length_thresholds': List of length thresholds in km
            - 'lake_area_groups': List of group names  
            - 'area_thresholds': List of area thresholds in km2
            
        Returns:
        --------
        String formatted for RAVEN RVH file
        """
        
        lines = []
        lines.append("#----------------------------------------------")
        lines.append("# Subbasin Groups")
        lines.append("# Generated using enhanced subbasin grouping")
        lines.append("#----------------------------------------------")
        lines.append("")
        
        # Group by channel length
        if 'channel_length_groups' in group_config:
            length_groups = self._create_channel_length_groups(
                subbasins_gdf, group_config
            )
            lines.extend(length_groups)
        
        # Group by lake area
        if 'lake_area_groups' in group_config:
            lake_groups = self._create_lake_area_groups(
                subbasins_gdf, group_config
            )
            lines.extend(lake_groups)
        
        return "\n".join(lines)
    
    def _create_channel_length_groups(self, subbasins_gdf: gpd.GeoDataFrame,
                                     group_config: Dict) -> List[str]:
        """Create groups based on channel length thresholds"""
        
        lines = []
        length_groups = group_config['channel_length_groups']
        length_thresholds = group_config.get('length_thresholds', [-1])
        
        for i, group_name in enumerate(length_groups):
            if i < len(length_thresholds):
                threshold = length_thresholds[i]
                
                if threshold == -1:
                    # All subbasins group
                    group_subbasins = subbasins_gdf['SubId'].tolist()
                else:
                    # Filter by channel length (convert threshold from km to m)
                    threshold_m = threshold * 1000
                    mask = subbasins_gdf['RivLength'] <= threshold_m
                    group_subbasins = subbasins_gdf[mask]['SubId'].tolist()
                
                if group_subbasins:
                    lines.append(f":SubBasinGroup  {group_name}")
                    
                    # Write subbasin IDs (10 per line as per BasinMaker format)
                    for j in range(0, len(group_subbasins), 10):
                        chunk = group_subbasins[j:j+10]
                        line = "  " + "  ".join(map(str, chunk))
                        lines.append(line)
                    
                    lines.append(f":EndSubBasinGroup")
                    lines.append("")
        
        return lines
    
    def _create_lake_area_groups(self, subbasins_gdf: gpd.GeoDataFrame,
                                group_config: Dict) -> List[str]:
        """Create groups based on lake area thresholds"""
        
        lines = []
        lake_groups = group_config['lake_area_groups']
        area_thresholds = group_config.get('area_thresholds', [-1])
        
        for i, group_name in enumerate(lake_groups):
            if i < len(area_thresholds):
                threshold = area_thresholds[i]
                
                if threshold == -1:
                    # All lake subbasins
                    mask = subbasins_gdf['IsLake'] == 1
                    group_subbasins = subbasins_gdf[mask]['SubId'].tolist()
                else:
                    # Filter by lake area (threshold in km2)
                    threshold_m2 = threshold * 1e6
                    mask = (subbasins_gdf['IsLake'] == 1) & \
                           (subbasins_gdf['LakeArea'] <= threshold_m2)
                    group_subbasins = subbasins_gdf[mask]['SubId'].tolist()
                
                if group_subbasins:
                    lines.append(f":SubBasinGroup  {group_name}")
                    
                    # Write subbasin IDs (10 per line)
                    for j in range(0, len(group_subbasins), 10):
                        chunk = group_subbasins[j:j+10]
                        line = "  " + "  ".join(map(str, chunk))
                        lines.append(line)
                    
                    lines.append(f":EndSubBasinGroup")
                    lines.append("")
        
        return lines
    
    def create_default_groups(self, subbasins_gdf: gpd.GeoDataFrame) -> str:
        """Create default subbasin groups commonly used in RAVEN"""
        
        default_config = {
            'channel_length_groups': ['AllSubbasins', 'ShortChannels'],
            'length_thresholds': [-1, 1.0],  # All subbasins, then <= 1 km
            'lake_area_groups': ['AllLakeSubbasins'],
            'area_thresholds': [-1]  # All lake subbasins
        }
        
        return self.create_subbasin_groups(subbasins_gdf, default_config)
    
    def analyze_subbasin_characteristics(self, subbasins_gdf: gpd.GeoDataFrame) -> Dict:
        """Analyze subbasin characteristics to suggest grouping"""
        
        analysis = {}
        
        # Channel length statistics
        if 'RivLength' in subbasins_gdf.columns:
            lengths_km = subbasins_gdf['RivLength'] / 1000  # Convert m to km
            analysis['channel_length'] = {
                'min_km': float(lengths_km.min()),
                'max_km': float(lengths_km.max()),
                'mean_km': float(lengths_km.mean()),
                'median_km': float(lengths_km.median()),
                'std_km': float(lengths_km.std())
            }
            
            # Suggest length thresholds
            percentiles = [25, 50, 75]
            thresholds = [float(np.percentile(lengths_km, p)) for p in percentiles]
            analysis['suggested_length_thresholds'] = thresholds
        
        # Lake area statistics
        if 'LakeArea' in subbasins_gdf.columns and 'IsLake' in subbasins_gdf.columns:
            lake_subbasins = subbasins_gdf[subbasins_gdf['IsLake'] == 1]
            if len(lake_subbasins) > 0:
                lake_areas_km2 = lake_subbasins['LakeArea'] / 1e6  # Convert m2 to km2
                analysis['lake_area'] = {
                    'lake_count': len(lake_subbasins),
                    'min_km2': float(lake_areas_km2.min()),
                    'max_km2': float(lake_areas_km2.max()),
                    'mean_km2': float(lake_areas_km2.mean()),
                    'total_lake_area_km2': float(lake_areas_km2.sum())
                }
                
                # Suggest area thresholds
                if len(lake_areas_km2) > 1:
                    percentiles = [25, 50, 75]
                    thresholds = [float(np.percentile(lake_areas_km2, p)) for p in percentiles]
                    analysis['suggested_area_thresholds'] = thresholds
        
        # Subbasin count statistics
        analysis['subbasin_count'] = len(subbasins_gdf)
        
        return analysis
    
    def suggest_grouping_strategy(self, subbasins_gdf: gpd.GeoDataFrame) -> Dict:
        """Suggest optimal grouping strategy based on subbasin characteristics"""
        
        analysis = self.analyze_subbasin_characteristics(subbasins_gdf)
        suggestions = {}
        
        # Channel length grouping suggestions
        if 'channel_length' in analysis:
            n_subbasins = analysis['subbasin_count']
            
            if n_subbasins <= 10:
                # Small watershed - simple grouping
                suggestions['channel_length_groups'] = ['AllSubbasins']
                suggestions['length_thresholds'] = [-1]
            elif n_subbasins <= 50:
                # Medium watershed - two groups
                median_length = analysis['channel_length']['median_km']
                suggestions['channel_length_groups'] = ['AllSubbasins', 'ShortChannels']
                suggestions['length_thresholds'] = [-1, round(median_length, 1)]
            else:
                # Large watershed - three groups
                thresholds = analysis.get('suggested_length_thresholds', [1.0, 5.0])
                suggestions['channel_length_groups'] = ['AllSubbasins', 'ShortChannels', 'MediumChannels']
                suggestions['length_thresholds'] = [-1] + [round(t, 1) for t in thresholds[:2]]
        
        # Lake area grouping suggestions
        if 'lake_area' in analysis:
            lake_count = analysis['lake_area']['lake_count']
            
            if lake_count > 0:
                if lake_count <= 5:
                    # Few lakes - single group
                    suggestions['lake_area_groups'] = ['AllLakeSubbasins']
                    suggestions['area_thresholds'] = [-1]
                else:
                    # Multiple lakes - group by size
                    mean_area = analysis['lake_area']['mean_km2']
                    suggestions['lake_area_groups'] = ['AllLakeSubbasins', 'SmallLakes']
                    suggestions['area_thresholds'] = [-1, round(mean_area, 2)]
        
        return suggestions


def test_subbasin_grouper():
    """Test the subbasin grouper with mock data"""
    
    print("Testing Subbasin Grouper...")
    
    # Create mock subbasin data
    n_subbasins = 15
    mock_data = {
        'SubId': range(1, n_subbasins + 1),
        'RivLength': np.random.lognormal(3, 0.5, n_subbasins) * 1000,  # m
        'IsLake': np.random.choice([0, 1], n_subbasins, p=[0.8, 0.2]),
        'LakeArea': np.random.exponential(1e5, n_subbasins),  # m2
    }
    
    # Create GeoDataFrame (simplified)
    mock_gdf = pd.DataFrame(mock_data)
    
    # Initialize grouper
    grouper = SubbasinGrouper()
    
    # Test analysis
    analysis = grouper.analyze_subbasin_characteristics(mock_gdf)
    print(f"✓ Analysis completed: {len(analysis)} metrics")
    
    # Test suggestions
    suggestions = grouper.suggest_grouping_strategy(mock_gdf)
    print(f"✓ Grouping suggestions generated")
    
    # Test default grouping
    default_groups = grouper.create_default_groups(mock_gdf)
    print(f"✓ Default groups created")
    
    print("✓ Subbasin Grouper ready for integration")


if __name__ == "__main__":
    test_subbasin_grouper()