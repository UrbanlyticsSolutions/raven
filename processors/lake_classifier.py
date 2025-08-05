#!/usr/bin/env python3
"""
Lake Classifier - Extracted from BasinMaker
Classifies lakes as connected vs non-connected using real BasinMaker logic
EXTRACTED FROM: basinmaker/addlakeandobs/definelaketypeqgis.py and filterlakesqgis.py
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.mask import mask
from shapely.geometry import shape
import sys

# Import your existing infrastructure
sys.path.append(str(Path(__file__).parent.parent))


class LakeClassifier:
    """
    Classify lakes as connected vs non-connected using real BasinMaker logic
    EXTRACTED FROM: define_connected_and_non_connected_lake_type() in BasinMaker definelaketypeqgis.py
    
    This replicates BasinMaker's lake classification workflow:
    1. Overlay stream network with lakes to identify stream-lake intersections
    2. Connected lakes: lakes that intersect with stream network
    3. Non-connected lakes: lakes that do not intersect with stream network
    4. Filter lakes by area thresholds for connected vs non-connected
    """
    
    def __init__(self, workspace_dir: Path = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # BasinMaker default thresholds (from original code)
        self.default_connected_threshold_km2 = 0.01    # 1 hectare for connected lakes
        self.default_non_connected_threshold_km2 = 0.1  # 10 hectares for non-connected lakes
    
    def classify_lakes_from_watershed_results(self, watershed_results: Dict,
                                            lakes_shapefile_path: Path = None,
                                            connected_threshold_km2: float = None,
                                            non_connected_threshold_km2: float = None) -> Dict:
        """
        Classify lakes using real BasinMaker logic adapted to your infrastructure
        EXTRACTED FROM: define_connected_and_non_connected_lake_type() in BasinMaker
        
        Parameters:
        -----------
        watershed_results : Dict
            Results from your ProfessionalWatershedAnalyzer
        lakes_shapefile_path : Path, optional
            Path to lakes shapefile. If None, looks for lakes in watershed_results or workspace
        connected_threshold_km2 : float, optional
            Minimum area for connected lakes (km²). If None, uses BasinMaker default
        non_connected_threshold_km2 : float, optional
            Minimum area for non-connected lakes (km²). If None, uses BasinMaker default
            
        Returns:
        --------
        Dict with classification results and output files
        """
        
        print("Classifying lakes using BasinMaker logic...")
        
        # Use BasinMaker default thresholds if not provided
        if connected_threshold_km2 is None:
            connected_threshold_km2 = self.default_connected_threshold_km2
        if non_connected_threshold_km2 is None:
            non_connected_threshold_km2 = self.default_non_connected_threshold_km2
        
        print(f"   Connected lake threshold: {connected_threshold_km2} km²")
        print(f"   Non-connected lake threshold: {non_connected_threshold_km2} km²")
        
        # Load lakes and streams
        lakes_gdf, streams_gdf = self._load_lakes_and_streams(
            watershed_results, lakes_shapefile_path
        )
        
        if len(lakes_gdf) == 0:
            print("   No lakes found for classification")
            return self._create_empty_classification_result()
        
        # Step 1: Identify connected lakes (BasinMaker lines 20-27)
        print("   Identifying connected lakes...")
        connected_lake_ids = self._identify_connected_lakes(lakes_gdf, streams_gdf)
        
        # Step 2: Classify lakes based on stream intersection (BasinMaker lines 32-52)
        print("   Classifying lakes based on stream connectivity...")
        classified_lakes = self._classify_lakes_by_connectivity(
            lakes_gdf, connected_lake_ids
        )
        
        # Step 3: Filter lakes by area thresholds (BasinMaker select_lakes_by_area_r)
        print("   Filtering lakes by area thresholds...")
        filtered_results = self._filter_lakes_by_area(
            classified_lakes, connected_threshold_km2, non_connected_threshold_km2
        )
        
        # Step 4: Create output files
        print("   Creating output files...")
        output_files = self._create_output_files(filtered_results)
        
        # Summary statistics
        connected_count = len(filtered_results['connected_lakes'])
        non_connected_count = len(filtered_results['non_connected_lakes'])
        total_area_km2 = (filtered_results['connected_lakes']['area_km2'].sum() + 
                         filtered_results['non_connected_lakes']['area_km2'].sum())
        
        print(f"   Classification complete:")
        print(f"     Connected lakes: {connected_count}")
        print(f"     Non-connected lakes: {non_connected_count}")
        print(f"     Total lake area: {total_area_km2:.3f} km²")
        
        return {
            'success': True,
            'connected_lakes_file': output_files['connected_lakes_file'],
            'non_connected_lakes_file': output_files['non_connected_lakes_file'],
            'all_lakes_file': output_files['all_lakes_file'],
            'connected_count': connected_count,
            'non_connected_count': non_connected_count,
            'total_lakes': connected_count + non_connected_count,
            'total_area_km2': total_area_km2,
            'connected_threshold_km2': connected_threshold_km2,
            'non_connected_threshold_km2': non_connected_threshold_km2,
            'classification_summary': {
                'connected_lakes': {
                    'count': connected_count,
                    'total_area_km2': float(filtered_results['connected_lakes']['area_km2'].sum()),
                    'avg_area_km2': float(filtered_results['connected_lakes']['area_km2'].mean()) if connected_count > 0 else 0
                },
                'non_connected_lakes': {
                    'count': non_connected_count,
                    'total_area_km2': float(filtered_results['non_connected_lakes']['area_km2'].sum()),
                    'avg_area_km2': float(filtered_results['non_connected_lakes']['area_km2'].mean()) if non_connected_count > 0 else 0
                }
            }
        }
    
    def _load_lakes_and_streams(self, watershed_results: Dict, 
                               lakes_shapefile_path: Path = None) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Load lakes and streams from watershed results or user-provided files"""
        
        # Load streams from watershed results
        stream_files = [f for f in watershed_results.get('files_created', []) 
                       if 'streams.geojson' in f or 'rivers.geojson' in f]
        
        if stream_files:
            streams_gdf = gpd.read_file(stream_files[0])
            print(f"   Loaded {len(streams_gdf)} stream segments from watershed results")
        else:
            # Create empty streams dataframe
            streams_gdf = gpd.GeoDataFrame()
            print("   No stream network found in watershed results")
        
        # Load lakes
        if lakes_shapefile_path and lakes_shapefile_path.exists():
            lakes_gdf = gpd.read_file(lakes_shapefile_path)
            print(f"   Loaded {len(lakes_gdf)} lakes from {lakes_shapefile_path}")
        else:
            # Look for lakes in workspace (from your existing lake detection)
            workspace_lakes = self.workspace_dir / "lakes.shp"
            if workspace_lakes.exists():
                lakes_gdf = gpd.read_file(workspace_lakes)
                print(f"   Loaded {len(lakes_gdf)} lakes from workspace")
            else:
                # Create empty lakes dataframe
                lakes_gdf = gpd.GeoDataFrame()
                print("   No lakes found for classification")
        
        # Ensure both have same CRS
        if len(lakes_gdf) > 0 and len(streams_gdf) > 0:
            if lakes_gdf.crs != streams_gdf.crs:
                streams_gdf = streams_gdf.to_crs(lakes_gdf.crs)
        
        return lakes_gdf, streams_gdf
    
    def _identify_connected_lakes(self, lakes_gdf: gpd.GeoDataFrame, 
                                 streams_gdf: gpd.GeoDataFrame) -> List[int]:
        """
        Identify lakes that intersect with stream network
        EXTRACTED FROM: BasinMaker lines 20-27 (overlay streams and lakes)
        """
        
        connected_lake_ids = []
        
        if len(streams_gdf) == 0:
            print("   No stream network - all lakes classified as non-connected")
            return connected_lake_ids
        
        # Find lakes that intersect with streams (BasinMaker overlay logic)
        for idx, lake in lakes_gdf.iterrows():
            lake_id = idx + 1  # Use 1-based indexing like BasinMaker
            
            # Check if lake intersects with any stream
            intersects_stream = False
            for _, stream in streams_gdf.iterrows():
                if lake.geometry.intersects(stream.geometry):
                    intersects_stream = True
                    break
            
            if intersects_stream:
                connected_lake_ids.append(lake_id)
        
        print(f"   Found {len(connected_lake_ids)} lakes intersecting with streams")
        return connected_lake_ids
    
    def _classify_lakes_by_connectivity(self, lakes_gdf: gpd.GeoDataFrame, 
                                       connected_lake_ids: List[int]) -> Dict:
        """
        Classify lakes as connected or non-connected
        EXTRACTED FROM: BasinMaker lines 32-52 (classification logic)
        """
        
        # Add lake classification columns
        lakes_classified = lakes_gdf.copy()
        lakes_classified['lake_id'] = range(1, len(lakes_classified) + 1)
        
        # Calculate area in km²
        if lakes_classified.crs.is_geographic:
            # Convert to appropriate projected CRS for area calculation
            utm_crs = lakes_classified.estimate_utm_crs()
            lakes_projected = lakes_classified.to_crs(utm_crs)
            areas_m2 = lakes_projected.geometry.area
            lakes_classified['area_km2'] = areas_m2 / (1000 * 1000)
        else:
            lakes_classified['area_km2'] = lakes_classified.geometry.area / (1000 * 1000)
        
        # Classify based on connectivity (BasinMaker logic)
        lakes_classified['is_connected'] = lakes_classified['lake_id'].isin(connected_lake_ids)
        lakes_classified['lake_type'] = lakes_classified['is_connected'].map({
            True: 'connected',
            False: 'non_connected'
        })
        
        # Separate into connected and non-connected (BasinMaker approach)
        connected_lakes = lakes_classified[lakes_classified['is_connected']].copy()
        non_connected_lakes = lakes_classified[~lakes_classified['is_connected']].copy()
        
        return {
            'all_lakes': lakes_classified,
            'connected_lakes': connected_lakes,
            'non_connected_lakes': non_connected_lakes
        }
    
    def _filter_lakes_by_area(self, classified_lakes: Dict, 
                             connected_threshold_km2: float,
                             non_connected_threshold_km2: float) -> Dict:
        """
        Filter lakes by area thresholds
        EXTRACTED FROM: select_lakes_by_area_r() in BasinMaker filterlakesqgis.py
        """
        
        # Filter connected lakes by area (BasinMaker lines 36-38)
        connected_filtered = classified_lakes['connected_lakes'][
            classified_lakes['connected_lakes']['area_km2'] >= connected_threshold_km2
        ].copy()
        
        # Filter non-connected lakes by area (BasinMaker lines 39-41)
        non_connected_filtered = classified_lakes['non_connected_lakes'][
            classified_lakes['non_connected_lakes']['area_km2'] >= non_connected_threshold_km2
        ].copy()
        
        # Combine all selected lakes
        all_selected = pd.concat([connected_filtered, non_connected_filtered], ignore_index=True)
        
        print(f"     Before filtering: {len(classified_lakes['connected_lakes'])} connected, {len(classified_lakes['non_connected_lakes'])} non-connected")
        print(f"     After filtering: {len(connected_filtered)} connected, {len(non_connected_filtered)} non-connected")
        
        return {
            'all_lakes': all_selected,
            'connected_lakes': connected_filtered,
            'non_connected_lakes': non_connected_filtered
        }
    
    def _create_output_files(self, filtered_results: Dict) -> Dict:
        """Create output shapefiles for classified lakes"""
        
        output_files = {}
        
        # Connected lakes shapefile
        connected_file = self.workspace_dir / "connected_lakes.shp"
        if len(filtered_results['connected_lakes']) > 0:
            filtered_results['connected_lakes'].to_file(connected_file)
            output_files['connected_lakes_file'] = str(connected_file)
            print(f"     Created: {connected_file}")
        else:
            output_files['connected_lakes_file'] = None
        
        # Non-connected lakes shapefile
        non_connected_file = self.workspace_dir / "non_connected_lakes.shp"
        if len(filtered_results['non_connected_lakes']) > 0:
            filtered_results['non_connected_lakes'].to_file(non_connected_file)
            output_files['non_connected_lakes_file'] = str(non_connected_file)
            print(f"     Created: {non_connected_file}")
        else:
            output_files['non_connected_lakes_file'] = None
        
        # All selected lakes shapefile
        all_lakes_file = self.workspace_dir / "all_lakes.shp"
        if len(filtered_results['all_lakes']) > 0:
            filtered_results['all_lakes'].to_file(all_lakes_file)
            output_files['all_lakes_file'] = str(all_lakes_file)
            print(f"     Created: {all_lakes_file}")
        else:
            output_files['all_lakes_file'] = None
        
        return output_files
    
    def _create_empty_classification_result(self) -> Dict:
        """Create empty result when no lakes are found"""
        
        return {
            'success': True,
            'connected_lakes_file': None,
            'non_connected_lakes_file': None,
            'all_lakes_file': None,
            'connected_count': 0,
            'non_connected_count': 0,
            'total_lakes': 0,
            'total_area_km2': 0.0,
            'classification_summary': {
                'connected_lakes': {'count': 0, 'total_area_km2': 0, 'avg_area_km2': 0},
                'non_connected_lakes': {'count': 0, 'total_area_km2': 0, 'avg_area_km2': 0}
            }
        }
    
    def classify_from_existing_lakes(self, lakes_shapefile_path: Path,
                                   streams_shapefile_path: Path = None,
                                   connected_threshold_km2: float = None,
                                   non_connected_threshold_km2: float = None) -> Dict:
        """
        Classify lakes from existing shapefiles
        
        Parameters:
        -----------
        lakes_shapefile_path : Path
            Path to lakes shapefile
        streams_shapefile_path : Path, optional
            Path to streams shapefile  
        connected_threshold_km2 : float, optional
            Minimum area for connected lakes (km²)
        non_connected_threshold_km2 : float, optional
            Minimum area for non-connected lakes (km²)
            
        Returns:
        --------
        Dict with classification results
        """
        
        print("Classifying lakes from existing shapefiles...")
        
        # Create mock watershed results for compatibility
        mock_watershed_results = {'files_created': []}
        if streams_shapefile_path and streams_shapefile_path.exists():
            mock_watershed_results['files_created'].append(str(streams_shapefile_path))
        
        return self.classify_lakes_from_watershed_results(
            mock_watershed_results, 
            lakes_shapefile_path,
            connected_threshold_km2,
            non_connected_threshold_km2
        )
    
    def validate_lake_classification(self, classification_results: Dict) -> Dict:
        """Validate lake classification results"""
        
        validation = {
            'success': classification_results.get('success', False),
            'total_lakes': classification_results.get('total_lakes', 0),
            'warnings': [],
            'statistics': {}
        }
        
        if not validation['success']:
            validation['warnings'].append("Classification failed")
            return validation
        
        # Statistical validation
        connected_stats = classification_results.get('classification_summary', {}).get('connected_lakes', {})
        non_connected_stats = classification_results.get('classification_summary', {}).get('non_connected_lakes', {})
        
        validation['statistics'] = {
            'connected_lakes': connected_stats,
            'non_connected_lakes': non_connected_stats,
            'total_area_km2': classification_results.get('total_area_km2', 0)
        }
        
        # Check for reasonable distribution
        total_lakes = validation['total_lakes']
        if total_lakes > 0:
            connected_ratio = connected_stats.get('count', 0) / total_lakes
            if connected_ratio > 0.9:
                validation['warnings'].append("Very high connected lake ratio - check stream network")
            elif connected_ratio < 0.1:
                validation['warnings'].append("Very low connected lake ratio - check stream network")
        
        # Check area thresholds
        if connected_stats.get('avg_area_km2', 0) < non_connected_stats.get('avg_area_km2', 0):
            validation['warnings'].append("Connected lakes smaller on average than non-connected - unusual pattern")
        
        return validation
    
    def calculate_lake_active_depth_and_lake_evap(self,
                                                 finalcat_info_path: Path,
                                                 reservoir_stages_path: Path,
                                                 reservoir_mass_balance_path: Path,
                                                 output_folder: Path = None) -> Dict:
        """
        Calculate lake active depth and evaporation from RAVEN model outputs
        EXTRACTED FROM: Caluculate_Lake_Active_Depth_and_Lake_Evap() in BasinMaker raveninput.py
        
        This function processes RAVEN model outputs to calculate lake depth statistics
        and evaporation losses for reservoir/lake subbasins.
        
        Parameters:
        -----------
        finalcat_info_path : Path
            Path to finalcat_info shapefile with lake HRU information
        reservoir_stages_path : Path
            Path to RAVEN ReservoirStages.csv output file
        reservoir_mass_balance_path : Path
            Path to RAVEN ReservoirMassBalance.csv output file
        output_folder : Path, optional
            Output folder for analysis results
            
        Returns:
        --------
        Dict with lake depth and evaporation analysis results
        """
        
        print(f"Calculating lake active depth and evaporation using BasinMaker logic...")
        
        if output_folder is None:
            output_folder = self.workspace_dir / "lake_depth_evap_analysis"
        output_folder.mkdir(exist_ok=True, parents=True)
        
        try:
            # Load finalcat info
            finalcat_info = gpd.read_file(finalcat_info_path)
            
            # Load RAVEN model outputs
            reservoir_stages = pd.read_csv(reservoir_stages_path)
            reservoir_mb = pd.read_csv(reservoir_mass_balance_path)
            
            # Process date columns
            reservoir_stages['Date_2'] = pd.to_datetime(
                reservoir_stages['date'] + ' ' + reservoir_stages['hour']
            )
            reservoir_stages = reservoir_stages.set_index('Date_2')
            
            reservoir_mb['Date_2'] = pd.to_datetime(
                reservoir_mb['date'] + ' ' + reservoir_mb['hour']
            )
            reservoir_mb = reservoir_mb.set_index('Date_2')
            
            print(f"   Processing {len(reservoir_stages)} time steps")
            print(f"   Date range: {reservoir_stages.index.min()} to {reservoir_stages.index.max()}")
            
            # Filter lake HRUs
            lake_hrus = finalcat_info[
                (finalcat_info.get('Lake_Cat', 0) > 0) & 
                (finalcat_info.get('HRU_Type', 0) == 1)
            ]
            
            print(f"   Found {len(lake_hrus)} lake HRUs")
            
            # Calculate lake statistics
            stage_statistics = self._calculate_lake_stage_statistics(
                lake_hrus, reservoir_stages, reservoir_mb
            )
            
            # Calculate annual evaporation
            evaporation_statistics = self._calculate_annual_evaporation(
                lake_hrus, reservoir_mb
            )
            
            # Save results
            stage_output = output_folder / "lake_stage_statistics.csv"
            stage_statistics.to_csv(stage_output, index=False)
            
            evap_output = output_folder / "lake_evaporation_statistics.csv"
            evaporation_statistics.to_csv(evap_output, index=False)
            
            # Create summary
            results = {
                'success': True,
                'output_folder': str(output_folder),
                'stage_statistics_file': str(stage_output),
                'evaporation_statistics_file': str(evap_output),
                'analysis_summary': {
                    'total_lake_hrus': len(lake_hrus),
                    'time_steps_processed': len(reservoir_stages),
                    'date_range': {
                        'start': str(reservoir_stages.index.min()),
                        'end': str(reservoir_stages.index.max())
                    },
                    'years_analyzed': len(evaporation_statistics),
                    'total_lakes_with_data': len(stage_statistics)
                },
                'stage_statistics': stage_statistics,
                'evaporation_statistics': evaporation_statistics
            }
            
            print(f"   ✓ Lake depth and evaporation analysis complete")
            print(f"   ✓ Processed {len(lake_hrus)} lake HRUs")
            print(f"   ✓ Generated statistics for {len(stage_statistics)} lakes")
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'output_folder': str(output_folder) if output_folder else None
            }
    
    def _calculate_lake_stage_statistics(self,
                                       lake_hrus: gpd.GeoDataFrame,
                                       reservoir_stages: pd.DataFrame,
                                       reservoir_mb: pd.DataFrame) -> pd.DataFrame:
        """Calculate lake stage statistics from RAVEN outputs"""
        
        stage_statistics = []
        stage_columns = list(reservoir_stages.columns)
        mb_columns = list(reservoir_mb.columns)
        
        for _, lake_hru in lake_hrus.iterrows():
            lake_id = lake_hru.get('HyLakeId', -1)
            lake_area = lake_hru.get('HRU_Area', 0)  # m²
            lake_subid = lake_hru.get('SubId', -1)
            lake_da = lake_hru.get('DrainArea', 0)
            
            # Find corresponding columns in RAVEN outputs
            stage_col_name = f"sub{int(lake_subid)} "
            mb_col_name = f"sub{int(lake_subid)} losses [m3]"
            
            if stage_col_name in stage_columns and mb_col_name in mb_columns:
                # Get stage data for this lake
                stage_data = reservoir_stages[stage_col_name].dropna()
                
                if len(stage_data) > 0:
                    # Calculate stage statistics
                    active_days = len(stage_data[stage_data > 0])
                    min_stage = float(stage_data.min())
                    max_stage = float(stage_data.max())
                    avg_stage = float(stage_data.mean())
                    
                    stage_statistics.append({
                        'Lake_Id': lake_id,
                        'Lake_Area': lake_area,
                        'Lake_DA': lake_da,
                        'Lake_SubId': lake_subid,
                        'Days_Active_Stage': active_days,
                        'Min_Stage': min_stage,
                        'Max_Stage': max_stage,
                        'Ave_Stage': avg_stage,
                        'Stage_Range': max_stage - min_stage,
                        'Active_Ratio': active_days / len(stage_data) if len(stage_data) > 0 else 0
                    })
        
        return pd.DataFrame(stage_statistics)
    
    def _calculate_annual_evaporation(self,
                                    lake_hrus: gpd.GeoDataFrame,
                                    reservoir_mb: pd.DataFrame) -> pd.DataFrame:
        """Calculate annual evaporation statistics"""
        
        evaporation_stats = []
        year_begin = reservoir_mb.index.min().year
        year_end = reservoir_mb.index.max().year
        
        for year in range(year_begin, year_end + 1):
            year_data = reservoir_mb[reservoir_mb.index.year == year]
            
            total_lake_evap_loss = 0.0
            
            for _, lake_hru in lake_hrus.iterrows():
                lake_subid = lake_hru.get('SubId', -1)
                mb_col_name = f"sub{int(lake_subid)} losses [m3]"
                
                if mb_col_name in reservoir_mb.columns:
                    year_lake_losses = year_data[mb_col_name].sum()
                    total_lake_evap_loss += year_lake_losses
            
            evaporation_stats.append({
                'Year': year,
                'Total_Lake_Evap_Loss': total_lake_evap_loss,
                'Average_Daily_Loss': total_lake_evap_loss / len(year_data) if len(year_data) > 0 else 0
            })
        
        return pd.DataFrame(evaporation_stats)


def test_lake_classifier():
    """Test the lake classifier using real BasinMaker logic"""
    
    print("Testing Lake Classifier with BasinMaker logic...")
    
    # Initialize classifier
    classifier = LakeClassifier()
    
    # Test with mock data (using your existing infrastructure patterns)
    from shapely.geometry import Polygon, LineString
    
    # Create mock lakes
    lake1 = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])  # Will intersect stream
    lake2 = Polygon([(200, 200), (300, 200), (300, 300), (200, 300)])  # Won't intersect
    
    lakes_gdf = gpd.GeoDataFrame({
        'geometry': [lake1, lake2],
        'area_m2': [10000, 10000]  # 1 hectare each
    }, crs='EPSG:4326')
    
    # Create mock stream
    stream = LineString([(50, -10), (50, 110)])  # Intersects lake1
    streams_gdf = gpd.GeoDataFrame({
        'geometry': [stream]
    }, crs='EPSG:4326')
    
    # Test connectivity identification
    connected_ids = classifier._identify_connected_lakes(lakes_gdf, streams_gdf)
    print(f"✓ Connected lake identification: {len(connected_ids)} connected")
    
    # Test classification
    classified = classifier._classify_lakes_by_connectivity(lakes_gdf, connected_ids)
    print(f"✓ Lake classification: {len(classified['connected_lakes'])} connected, {len(classified['non_connected_lakes'])} non-connected")
    
    # Test area filtering
    filtered = classifier._filter_lakes_by_area(
        classified, 
        connected_threshold_km2=0.005,  # 0.5 hectares
        non_connected_threshold_km2=0.01  # 1 hectare
    )
    print(f"✓ Area filtering completed")
    
    print("✓ Lake Classifier ready for integration")
    print("✓ Uses real BasinMaker stream-lake intersection and area filtering logic")


if __name__ == "__main__":
    test_lake_classifier()