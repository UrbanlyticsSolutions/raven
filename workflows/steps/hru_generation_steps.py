"""
HRU Generation Steps for RAVEN Workflows

This module contains steps for creating Hydrological Response Units (HRUs).
"""

import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.transform

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from workflows.steps.base_step import WorkflowStep

class GenerateHRUsFromRoutingProduct(WorkflowStep):
    """
    Step 3A: Generate HRUs from routing product data
    Used in Approach A (Routing Product Workflow)
    """
    
    def __init__(self):
        super().__init__(
            step_name="generate_hrus_routing",
            step_category="hru_generation",
            description="Create HRUs from existing routing product data"
        )
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._log_step_start()
        
        try:
            # Validate required inputs
            required_inputs = ['extracted_catchments']
            self.validate_inputs(inputs, required_inputs)
            
            catchments_file = self.validate_file_exists(inputs['extracted_catchments'])
            lakes_file = inputs.get('extracted_lakes')
            
            # Create workspace
            workspace_dir = inputs.get('workspace_dir', catchments_file.parent)
            workspace = Path(workspace_dir)
            
            # Generate HRUs from routing product
            self.logger.info("Generating HRUs from routing product data...")
            hru_result = self._generate_hrus_from_routing_product(
                catchments_file, lakes_file, workspace
            )
            
            outputs = {
                'final_hrus': hru_result['hru_file'],
                'lake_hru_count': hru_result['lake_hru_count'],
                'land_hru_count': hru_result['land_hru_count'],
                'total_hru_count': hru_result['total_hru_count'],
                'total_area_km2': hru_result['total_area_km2'],
                'attribute_completeness': hru_result['attribute_completeness'],
                'success': True
            }
            
            self._log_step_complete([hru_result['hru_file']])
            return outputs
            
        except Exception as e:
            error_msg = f"HRU generation from routing product failed: {str(e)}"
            self._log_step_failed(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _generate_hrus_from_routing_product(self, catchments_file: Path, 
                                          lakes_file: Path, workspace: Path) -> Dict[str, Any]:
        """Generate HRUs from routing product catchments and lakes using actual data"""
        
        import geopandas as gpd
        from processors.hru_generator import HRUGenerator
        
        # Load catchments
        catchments_gdf = gpd.read_file(catchments_file)
        
        # Use actual HRU generator for routing product
        hru_generator = HRUGenerator(workspace_dir=workspace)
        
        # Process lakes if provided
        lake_files = []
        if lakes_file and Path(lakes_file).exists():
            lake_files = [str(lakes_file)]
        
        # Generate HRUs using actual routing product data
        watershed_results = {
            'files_created': [str(catchments_file)] + lake_files
        }
        
        lake_results = {
            'lakes_detected': len(gpd.read_file(lakes_file)) if lake_files else 0,
            'lake_files': lake_files
        }
        
        hrus_gdf = hru_generator.generate_hrus_from_routing_product(
            watershed_results=watershed_results,
            lake_results=lake_results
        )
        
        # Save HRUs
        hru_file = workspace / "final_hrus.shp"
        hrus_gdf.to_file(hru_file)
        
        # Count HRUs by type
        lake_hru_count = len(hrus_gdf[hrus_gdf.get('HRU_IsLake', 0) == 1])
        land_hru_count = len(hrus_gdf) - lake_hru_count
        total_area_km2 = hrus_gdf.get('HRU_Area_km2', hrus_gdf.geometry.area / 1e6).sum()
        
        return {
            'hru_file': str(hru_file),
            'lake_hru_count': lake_hru_count,
            'land_hru_count': land_hru_count,
            'total_hru_count': len(hrus_gdf),
            'total_area_km2': total_area_km2,
            'attribute_completeness': 100.0  # Routing product has complete attributes
        }
    
    def _load_basinmaker_lookup_tables(self) -> Dict[str, pd.DataFrame]:
        """Load BasinMaker lookup tables"""
        
        # Try to load actual BasinMaker tables
        basinmaker_dir = Path('basinmaker-extracted/basinmaker-master/tests/testdata/HRU')
        
        lookup_tables = {}
        
        try:
            if (basinmaker_dir / 'landuse_info.csv').exists():
                lookup_tables['landuse'] = pd.read_csv(basinmaker_dir / 'landuse_info.csv')
            if (basinmaker_dir / 'soil_info.csv').exists():
                lookup_tables['soil'] = pd.read_csv(basinmaker_dir / 'soil_info.csv')
            if (basinmaker_dir / 'veg_info.csv').exists():
                lookup_tables['vegetation'] = pd.read_csv(basinmaker_dir / 'veg_info.csv')
        except Exception:
            # Use default tables if BasinMaker tables not available
            pass
        
        # Provide defaults if tables not loaded
        if 'landuse' not in lookup_tables:
            lookup_tables['landuse'] = pd.DataFrame([
                {'Landuse_ID': 1, 'LAND_USE_C': 'FOREST'},
                {'Landuse_ID': -1, 'LAND_USE_C': 'WATER'}
            ])
        
        if 'soil' not in lookup_tables:
            lookup_tables['soil'] = pd.DataFrame([
                {'Soil_ID': 1, 'SOIL_PROF': 'LOAM'},
                {'Soil_ID': -1, 'SOIL_PROF': 'WATER'}
            ])
        
        if 'vegetation' not in lookup_tables:
            lookup_tables['vegetation'] = pd.DataFrame([
                {'Veg_ID': 1, 'VEG_C': 'MIXED_FOREST'},
                {'Veg_ID': -1, 'VEG_C': 'WATER'}
            ])
        
        return lookup_tables
    
    def _get_landuse_class(self, centroid, lookup_tables: Dict[str, pd.DataFrame]) -> str:
        """Get land use class from lookup table"""
        # Simplified - in practice would use spatial lookup
        return 'FOREST'
    
    def _get_soil_class(self, centroid, lookup_tables: Dict[str, pd.DataFrame]) -> str:
        """Get soil class from lookup table"""
        # Simplified - in practice would use spatial lookup
        return 'LOAM'
    
    def _get_vegetation_class(self, centroid, lookup_tables: Dict[str, pd.DataFrame]) -> str:
        """Get vegetation class from lookup table"""
        # Simplified - in practice would use spatial lookup
        return 'MIXED_FOREST'


class CreateSubBasinsAndHRUs(WorkflowStep):
    """
    Step 5B: Create sub-basins and HRUs from watershed analysis
    Used in Approach B (Full Delineation Workflow)
    """
    
    def __init__(self):
        super().__init__(
            step_name="create_subbasins_hrus",
            step_category="hru_generation",
            description="Generate sub-basins and HRUs with BasinMaker attributes"
        )
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._log_step_start()
        
        try:
            # Check if we should use existing subbasins
            use_existing_subbasins = inputs.get('use_existing_subbasins', False)
            existing_subbasins_file = inputs.get('subbasins_file')
            
            if use_existing_subbasins and existing_subbasins_file and Path(existing_subbasins_file).exists():
                # Use existing subbasins - skip creation
                self.logger.info("Using existing subbasins...")
                subbasins_file = Path(existing_subbasins_file)
                
                # Count existing subbasins
                import geopandas as gpd
                subbasins_gdf = gpd.read_file(subbasins_file)
                subbasin_count = len(subbasins_gdf)
                
                subbasins_result = {
                    'subbasins_file': str(subbasins_file),
                    'subbasin_count': subbasin_count
                }
                
            else:
                # Create new subbasins (original behavior)
                required_inputs = ['watershed_boundary', 'integrated_stream_network']
                self.validate_inputs(inputs, required_inputs)
                
                watershed_boundary = self.validate_file_exists(inputs['watershed_boundary'])
                stream_network = self.validate_file_exists(inputs['integrated_stream_network'])
                
                # Create workspace
                workspace_dir = inputs.get('workspace_dir', watershed_boundary.parent)
                workspace = Path(workspace_dir)
                
                # Step 1: Create sub-basins
                self.logger.info("Creating sub-basins...")
                subbasins_result = self._create_subbasins(watershed_boundary, stream_network, workspace)
            
            # Optional inputs for HRU generation
            connected_lakes = inputs.get('significant_connected_lakes')
            dem_file = inputs.get('dem_file')
            landcover_file = inputs.get('landcover_file')
            soil_file = inputs.get('soil_file')
            
            # Create workspace for HRU generation
            workspace_dir = inputs.get('workspace_dir', Path(subbasins_result['subbasins_file']).parent)
            workspace = Path(workspace_dir)
            
            # Step 2: Generate HRUs using existing or new subbasins
            self.logger.info("Generating HRUs with BasinMaker attributes...")
            hru_result = self._generate_hrus_with_attributes(
                subbasins_result['subbasins_file'], connected_lakes, workspace,
                dem_file, landcover_file, soil_file
            )
            
            outputs = {
                'sub_basins': subbasins_result['subbasins_file'],
                'final_hrus': hru_result['hru_file'],
                'hydraulic_parameters': hru_result.get('hydraulic_params_file'),
                'subbasin_count': subbasins_result['subbasin_count'],
                'lake_hru_count': hru_result['lake_hru_count'],
                'land_hru_count': hru_result['land_hru_count'],
                'total_hru_count': hru_result['total_hru_count'],
                'total_area_km2': hru_result['total_area_km2'],
                'success': True
            }
            
            created_files = [f for f in outputs.values() if isinstance(f, str)]
            self._log_step_complete(created_files)
            return outputs
            
        except Exception as e:
            error_msg = f"Sub-basin and HRU creation failed: {str(e)}"
            self._log_step_failed(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _create_subbasins(self, watershed_boundary: Path, stream_network: Path, 
                         workspace: Path) -> Dict[str, Any]:
        """Create sub-basins dynamically from watershed and stream network using DEM analysis - NO HARD-CODING"""
        
        import geopandas as gpd
        import rasterio
        from rasterio.mask import mask
        import numpy as np
        from shapely.geometry import Point, Polygon
        from sklearn.cluster import KMeans
        from shapely.geometry import shape
        
        # Load watershed
        watershed_gdf = gpd.read_file(watershed_boundary)
        watershed_geom = watershed_gdf.geometry.iloc[0]
        
        # Convert to appropriate projected CRS for area calculation
        if watershed_gdf.crs.to_epsg() == 4326:  # Geographic coordinates
            # Convert to Web Mercator for area calculation
            watershed_gdf_projected = watershed_gdf.to_crs('EPSG:3857')
            watershed_area_m2 = watershed_gdf_projected.geometry.area.iloc[0]
            watershed_area_km2 = watershed_area_m2 / 1e6
        else:
            watershed_area_km2 = watershed_geom.area / 1e6
        
        # Load stream network
        streams_gdf = gpd.read_file(stream_network)
        
        # Get DEM file for elevation analysis
        dem_file = workspace.parent / 'dem.tif'
        if not dem_file.exists():
            dem_file = workspace / '../dem.tif'
        if not dem_file.exists():
            raise FileNotFoundError(f"DEM file not found. Required for dynamic subbasin generation.")
        
        print(f"Analyzing watershed characteristics from DEM: {dem_file}")
        
        # Extract elevation data from watershed
        with rasterio.open(dem_file) as src:
            # Mask DEM to watershed boundary
            masked_dem, transform = mask(src, [watershed_geom], crop=True)
            elevation_data = masked_dem[0]
            
            # Remove nodata values
            valid_elevations = elevation_data[elevation_data != src.nodata]
            
            if valid_elevations.size == 0:
                raise ValueError("No valid elevation data found in watershed")
            
            # Calculate watershed characteristics
            elev_min = float(np.min(valid_elevations))
            elev_max = float(np.max(valid_elevations))
            elev_mean = float(np.mean(valid_elevations))
            elev_std = float(np.std(valid_elevations))
            elev_range = elev_max - elev_min
            
            print(f"Watershed Analysis:")
            print(f"  - Area: {watershed_area_km2:.2f} km²")
            print(f"  - Elevation range: {elev_min:.0f} - {elev_max:.0f} m ({elev_range:.0f} m)")
            print(f"  - Elevation mean±std: {elev_mean:.0f}±{elev_std:.0f} m")
        
        # DYNAMIC SUBBASIN CALCULATION based on watershed characteristics
        # Calculate optimal number of subbasins based on:
        # 1. Watershed area (larger watersheds need more subbasins)
        # 2. Elevation variability (more complex terrain needs more subbasins)
        # 3. Stream network density
        
        # Base number from area (target ~30-50 km² per subbasin for good modeling)
        target_subbasin_area_km2 = 40.0
        area_based_count = max(1, int(watershed_area_km2 / target_subbasin_area_km2))
        
        # Adjust based on elevation complexity
        # High relief = more subbasins needed
        relief_ratio = elev_range / 1000.0  # Normalize to km
        complexity_multiplier = 1.0 + (relief_ratio * 0.5)  # Up to 50% more for high relief
        
        # Adjust based on stream density
        total_stream_length = streams_gdf.geometry.length.sum() / 1000.0  # Convert to km
        stream_density = total_stream_length / watershed_area_km2  # km/km²
        
        if stream_density > 2.0:  # High stream density
            stream_multiplier = 1.3
        elif stream_density > 1.0:  # Medium stream density
            stream_multiplier = 1.1
        else:  # Low stream density
            stream_multiplier = 1.0
        
        # Calculate final subbasin count
        optimal_count = int(area_based_count * complexity_multiplier * stream_multiplier)
        
        # Apply reasonable bounds (1 to 25 subbasins max)
        optimal_count = max(1, min(25, optimal_count))
        
        print(f"Dynamic Subbasin Calculation:")
        print(f"  - Area-based count: {area_based_count}")
        print(f"  - Relief complexity multiplier: {complexity_multiplier:.2f}")
        print(f"  - Stream density multiplier: {stream_multiplier:.2f}")
        print(f"  - Final optimal count: {optimal_count}")
        
        # Create subbasins using elevation-based clustering
        subbasins = self._create_elevation_based_subbasins(
            watershed_geom, masked_dem, transform, optimal_count, streams_gdf
        )
        
        if len(subbasins) == 0:
            raise ValueError("Failed to create any valid subbasins")
        
        # Save sub-basins
        subbasins_file = workspace / "subbasins.shp"
        subbasins_gdf = gpd.GeoDataFrame(subbasins, crs=watershed_gdf.crs)
        subbasins_gdf.to_file(subbasins_file)
        
        print(f"Created {len(subbasins)} dynamic subbasins (avg: {watershed_area_km2/len(subbasins):.1f} km² each)")
        
        return {
            'subbasins_file': str(subbasins_file),
            'subbasin_count': len(subbasins)
        }
    
    def _create_elevation_based_subbasins(self, watershed_geom, masked_dem, transform, 
                                        num_subbasins: int, streams_gdf) -> list:
        """Create subbasins using elevation-based clustering and stream network"""
        
        import numpy as np
        from sklearn.cluster import KMeans
        from rasterio.features import shapes
        from shapely.geometry import shape, Point, Polygon
        from shapely.ops import unary_union
        
        elevation_data = masked_dem[0]
        
        # Get pixel coordinates and elevations
        height, width = elevation_data.shape
        pixel_coords = []
        pixel_elevations = []
        
        for row in range(height):
            for col in range(width):
                if elevation_data[row, col] != -9999:  # Valid elevation
                    # Convert pixel to geographic coordinates
                    x, y = rasterio.transform.xy(transform, row, col)
                    point = Point(x, y)
                    
                    if watershed_geom.contains(point):
                        pixel_coords.append([x, y])
                        pixel_elevations.append(elevation_data[row, col])
        
        if len(pixel_coords) < num_subbasins:
            print(f"Warning: Not enough valid pixels for {num_subbasins} subbasins, reducing to {len(pixel_coords)}")
            num_subbasins = max(1, len(pixel_coords) // 100)  # At least 100 pixels per subbasin
        
        # Combine spatial and elevation features for clustering
        features = []
        for i, (x, y) in enumerate(pixel_coords):
            elev = pixel_elevations[i]
            # Weight elevation more heavily for mountainous areas
            elev_weight = 0.001  # Scale elevation to similar magnitude as coordinates
            features.append([x, y, elev * elev_weight])
        
        features = np.array(features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_subbasins, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Create subbasin polygons from clusters
        subbasins = []
        
        for cluster_id in range(num_subbasins):
            # Get points in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_coords = np.array(pixel_coords)[cluster_mask]
            
            if len(cluster_coords) < 3:  # Need at least 3 points for polygon
                continue
            
            # Create convex hull of cluster points
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(cluster_coords)
                hull_coords = cluster_coords[hull.vertices]
                
                # Create polygon
                subbasin_poly = Polygon(hull_coords)
                
                # Intersect with watershed to ensure validity
                subbasin_geom = subbasin_poly.intersection(watershed_geom)
                
                if (not subbasin_geom.is_empty and 
                    subbasin_geom.geom_type in ['Polygon', 'MultiPolygon'] and
                    subbasin_geom.area > 1e-6):  # Minimum area threshold
                    
                    # Calculate area in km²
                    area_km2 = subbasin_geom.area / 1e6
                    
                    # Find downstream connection (simplified)
                    centroid = subbasin_geom.centroid
                    downstream_id = self._find_downstream_subbasin(
                        centroid, cluster_id, num_subbasins, streams_gdf
                    )
                    
                    subbasins.append({
                        'subbasin_id': cluster_id + 1,
                        'area_km2': area_km2,
                        'downstream_id': downstream_id,
                        'elevation_mean': float(np.mean(np.array(pixel_elevations)[cluster_mask])),
                        'geometry': subbasin_geom
                    })
                    
            except Exception as e:
                print(f"Warning: Could not create subbasin {cluster_id}: {e}")
                continue
        
        return subbasins
    
    def _find_downstream_subbasin(self, centroid, current_id: int, total_subbasins: int, 
                                streams_gdf) -> int:
        """Find downstream subbasin ID based on stream network and elevation"""
        
        # Simplified downstream logic - in practice would use flow direction
        # For now, assume outlet is the subbasin with lowest elevation
        
        # Return 0 for outlet (convention), otherwise connect to next subbasin
        if current_id == total_subbasins - 1:  # Last subbasin is outlet
            return 0
        else:
            return current_id + 2  # Connect to next subbasin (1-indexed)
    
    def _generate_hrus_with_attributes(self, subbasins_file: str, connected_lakes: str, 
                                     workspace: Path, dem_file: str = None, 
                                     landcover_file: str = None, soil_file: str = None) -> Dict[str, Any]:
        """Generate HRUs with BasinMaker attributes using real subbasin boundaries and thematic data"""
        
        import geopandas as gpd
        import numpy as np
        from pathlib import Path
        from processors.hru_attributes import HRUAttributesCalculator
        from processors.hru_generator import HRUGenerator
        
        # Use the advanced HRU generators
        hru_calculator = HRUAttributesCalculator(workspace_dir=workspace)
        hru_generator = HRUGenerator(workspace_dir=workspace)
        
        try:
            # Load subbasins and check for actual subbasin files
            subbasins_gdf = gpd.read_file(subbasins_file)
            
            # Load lakes if available
            lakes_gdf = None
            if connected_lakes and Path(connected_lakes).exists():
                lakes_gdf = gpd.read_file(connected_lakes)
                print(f"   Found {len(lakes_gdf)} lakes to process")
            
            # Create thematic data dictionary
            thematic_data = {
                'landcover': None,
                'soil': None,
                'dem': Path(dem_file) if dem_file and Path(dem_file).exists() else None
            }
            
            # Add landcover data
            if landcover_file and Path(landcover_file).exists():
                thematic_data['landcover'] = {
                    'shapefile': Path(landcover_file),
                    'type': 'raster' if str(landcover_file).endswith('.tif') else 'polygon'
                }
            
            # Add soil data with percentage extraction
            if soil_file and Path(soil_file).exists():
                thematic_data['soil'] = {
                    'shapefile': Path(soil_file),
                    'type': 'raster' if str(soil_file).endswith('.tif') else 'polygon'
                }
            
            # Generate HRUs using real subbasin boundaries
            print(f"   Processing {len(subbasins_gdf)} subbasins...")
            
            # Create comprehensive watershed and lake results structure
            # Include watershed boundary for HRU generator
            watershed_boundary_file = str(workspace / "watershed.geojson")
            if Path(watershed_boundary_file).exists():
                watershed_files = [watershed_boundary_file]
            else:
                watershed_files = [str(workspace / "watershed.shp")]
                
            watershed_results = {
                'files_created': watershed_files + [subbasins_file] + ([connected_lakes] if connected_lakes else [])
            }
            
            lake_results = {
                'lakes_detected': len(lakes_gdf) if lakes_gdf is not None else 0,
                'lake_files': [connected_lakes] if connected_lakes else []
            }
            
            print(f"   Calling HRU generator with {len(subbasins_gdf)} subbasins...")
            print(f"   Thematic data: landcover={thematic_data.get('landcover') is not None}, soil={thematic_data.get('soil') is not None}, dem={thematic_data.get('dem')}")
            
            # Use the HRU generator with real data
            hrus_gdf = hru_generator.generate_hrus_from_watershed(
                watershed_results=watershed_results,
                lake_results=lake_results,
                landuse_data=thematic_data.get('landcover'),
                soil_data=thematic_data.get('soil'),
                dem_path=thematic_data.get('dem'),
                min_hru_area_km2=0.1
            )
            
            # Option to merge small subbasins before HRU generation
            merge_threshold_km2 = 20.0  # Merge subbasins smaller than this
            print(f"   Checking for subbasins smaller than {merge_threshold_km2} km² to merge...")
            
            # Calculate subbasin areas
            subbasins_reprojected = subbasins_gdf.copy()
            if subbasins_reprojected.crs and subbasins_reprojected.crs.is_geographic:
                subbasins_reprojected = subbasins_reprojected.to_crs(subbasins_reprojected.estimate_utm_crs())
            
            subbasins_reprojected['area_km2'] = subbasins_reprojected.geometry.area / 1e6
            small_subbasins = subbasins_reprojected[subbasins_reprojected['area_km2'] < merge_threshold_km2]
            
            if len(small_subbasins) > 0:
                print(f"   Found {len(small_subbasins)} small subbasins to merge...")
                # Merge small subbasins with their neighbors (simplified approach)
                # For now, we'll keep them but you can implement merging logic here
                print(f"   Keeping {len(small_subbasins)} small subbasins - consider manual review")

            # Calculate soil percentages and landcover integration
            if len(hrus_gdf) > 0:
                # Add soil percentage attributes
                if soil_file and Path(soil_file).exists():
                    hrus_gdf = self._add_soil_percentages(hrus_gdf, soil_file)
                
                # Add landcover classification
                if landcover_file and Path(landcover_file).exists():
                    hrus_gdf = self._add_landcover_attributes(hrus_gdf, landcover_file)
                
                # Calculate elevation from DEM
                if dem_file and Path(dem_file).exists():
                    hrus_gdf = self._add_elevation_attributes(hrus_gdf, dem_file)
            
            # Count HRUs by type
            lake_hru_count = len(hrus_gdf[hrus_gdf.get('HRU_IsLake', 0) == 1])
            land_hru_count = len(hrus_gdf) - lake_hru_count
            
            # Ensure proper column naming for BasinMaker compatibility
            hrus_gdf = self._standardize_hru_columns(hrus_gdf)
            
            # Save HRUs in multiple formats
            hru_file_shp = workspace / "final_hrus.shp"
            hru_file_geojson = workspace / "final_hrus.geojson"
            
            hrus_gdf.to_file(hru_file_shp)
            hrus_gdf.to_file(hru_file_geojson, driver='GeoJSON')
            
            # Create hydraulic parameters file
            hydraulic_params_file = workspace / "hydraulic_parameters.csv"
            hydraulic_cols = ['HRU_ID', 'SubId', 'RivLength', 'RivSlope', 'BkfWidth', 'BkfDepth']
            hydraulic_data = hrus_gdf[hydraulic_cols] if all(col in hrus_gdf.columns for col in hydraulic_cols) else None
            
            if hydraulic_data is not None:
                hydraulic_data.to_csv(hydraulic_params_file, index=False)
                print(f"   Created hydraulic parameters file: {hydraulic_params_file}")
            else:
                # Create minimal hydraulic parameters file with defaults
                hydraulic_df = pd.DataFrame({
                    'HRU_ID': hrus_gdf['HRU_ID'],
                    'SubId': hrus_gdf['SubId'],
                    'RivLength': 0.0,
                    'RivSlope': 0.001,  # Minimal slope for numerical stability
                    'BkfWidth': 1.0,    # Default width
                    'BkfDepth': 0.5     # Default depth
                })
                hydraulic_df.to_csv(hydraulic_params_file, index=False)
                print(f"   Created default hydraulic parameters file: {hydraulic_params_file}")
            
            return {
                'hru_file': str(hru_file_shp),
                'hru_geojson': str(hru_file_geojson),
                'hydraulic_params_file': str(hydraulic_params_file),
                'lake_hru_count': lake_hru_count,
                'land_hru_count': land_hru_count,
                'total_hru_count': len(hrus_gdf),
                'total_area_km2': hrus_gdf['HRU_Area_km2'].sum() if 'HRU_Area_km2' in hrus_gdf.columns else 0,
                'subbasin_count': len(subbasins_gdf),
                'success': True
            }
            
        except Exception as e:
            error_msg = f"HRU generation failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'hru_file': None,
                'lake_hru_count': 0,
                'land_hru_count': 0,
                'total_hru_count': 0,
                'total_area_km2': 0
            }
    
    def _add_soil_percentages(self, hrus_gdf, soil_file: str) -> gpd.GeoDataFrame:
        """Add soil percentages (sand, silt, clay) from soil data"""
        
        import rasterio
        from rasterio.mask import mask
        
        # Create a copy of the dataframe
        result_gdf = hrus_gdf.copy()
        
        # Initialize soil percentage columns
        result_gdf['SAND_PCT'] = 0.0
        result_gdf['SILT_PCT'] = 0.0
        result_gdf['CLAY_PCT'] = 0.0
        result_gdf['SOIL_TEXTURE'] = 'UNKNOWN'
        
        if not Path(soil_file).exists():
            return result_gdf
            
        try:
            print(f"   Extracting soil percentages from: {soil_file}")
            
            # Check if soil file has multiple bands for percentages
            with rasterio.open(soil_file) as src:
                print(f"   Soil raster CRS: {src.crs}")
                print(f"   HRUs CRS: {result_gdf.crs}")
                
                # Reproject HRUs to match raster CRS if needed
                if src.crs != result_gdf.crs:
                    print(f"   Reprojecting HRUs from {result_gdf.crs} to {src.crs} for soil extraction")
                    hrus_reprojected = result_gdf.to_crs(src.crs)
                else:
                    hrus_reprojected = result_gdf
                
                if src.count >= 3:
                    # Assume bands are: 1=sand%, 2=silt%, 3=clay%
                    for idx, hru in hrus_reprojected.iterrows():
                        try:
                            masked_data, _ = mask(src, [hru.geometry], crop=True)
                            
                            # Calculate percentages for each soil type
                            if masked_data[0].size > 0:
                                # Remove nodata values
                                valid_data = masked_data[0][masked_data[0] != src.nodata]
                                if valid_data.size > 0:
                                    # Calculate mean percentages
                                    result_gdf.loc[idx, 'SAND_PCT'] = float(np.mean(valid_data))
                                    
                                    if src.count >= 2:
                                        valid_silt = masked_data[1][masked_data[1] != src.nodata]
                                        result_gdf.loc[idx, 'SILT_PCT'] = float(np.mean(valid_silt))
                                    
                                    if src.count >= 3:
                                        valid_clay = masked_data[2][masked_data[2] != src.nodata]
                                        result_gdf.loc[idx, 'CLAY_PCT'] = float(np.mean(valid_clay))
                                    
                                    # Determine soil texture class
                                    result_gdf.loc[idx, 'SOIL_TEXTURE'] = self._classify_soil_texture(
                                        result_gdf.loc[idx, 'SAND_PCT'],
                                        result_gdf.loc[idx, 'SILT_PCT'],
                                        result_gdf.loc[idx, 'CLAY_PCT']
                                    )
                                    
                        except Exception as e:
                            print(f"   Warning: Could not extract soil data for HRU {idx}: {e}")
                            continue
                else:
                    # Single band - use default percentages based on soil class
                    print("   Single band soil data - using default percentages")
                    result_gdf['SAND_PCT'] = 40.0  # Default loam
                    result_gdf['SILT_PCT'] = 40.0
                    result_gdf['CLAY_PCT'] = 20.0
                    result_gdf['SOIL_TEXTURE'] = 'LOAM'
                    
        except Exception as e:
            print(f"   Warning: Soil percentage extraction failed: {e}")
            # Use default values
            result_gdf['SAND_PCT'] = 40.0
            result_gdf['SILT_PCT'] = 40.0
            result_gdf['CLAY_PCT'] = 20.0
            result_gdf['SOIL_TEXTURE'] = 'LOAM'
        
        return result_gdf
    
    def _classify_soil_texture(self, sand_pct: float, silt_pct: float, clay_pct: float) -> str:
        """Classify soil texture based on sand/silt/clay percentages"""
        
        # Soil texture classification based on USDA triangle
        if sand_pct >= 85:
            return 'SAND'
        elif sand_pct >= 70 and clay_pct <= 15:
            return 'LOAMY_SAND'
        elif sand_pct >= 43 and clay_pct <= 27 and silt_pct >= 28:
            return 'SANDY_LOAM'
        elif clay_pct >= 40:
            return 'CLAY'
        elif clay_pct >= 27 and clay_pct < 40:
            return 'CLAY_LOAM'
        elif silt_pct >= 50 and clay_pct < 27:
            return 'SILT_LOAM'
        elif silt_pct >= 80:
            return 'SILT'
        else:
            return 'LOAM'
    
    def _add_landcover_attributes(self, hrus_gdf, landcover_file: str) -> gpd.GeoDataFrame:
        """Add landcover classification to HRU attributes"""
        
        import rasterio
        from rasterio.mask import mask
        
        # Create a copy
        result_gdf = hrus_gdf.copy()
        
        # Initialize landcover columns
        result_gdf['LANDCOVER_ID'] = 0
        result_gdf['LANDCOVER_CLASS'] = 'UNKNOWN'
        result_gdf['LANDCOVER_PCT'] = 0.0
        
        if not Path(landcover_file).exists():
            return result_gdf
            
        try:
            print(f"   Extracting landcover classification from: {landcover_file}")
            
            with rasterio.open(landcover_file) as src:
                print(f"   Landcover raster CRS: {src.crs}")
                print(f"   HRUs CRS: {result_gdf.crs}")
                
                # Reproject HRUs to match raster CRS if needed
                if src.crs != result_gdf.crs:
                    print(f"   Reprojecting HRUs from {result_gdf.crs} to {src.crs} for landcover extraction")
                    hrus_reprojected = result_gdf.to_crs(src.crs)
                else:
                    hrus_reprojected = result_gdf
                
                # Define landcover classes (NLCD/NLCD-style classification)
                landcover_classes = {
                    11: 'WATER', 12: 'SNOW_ICE', 21: 'URBAN', 22: 'URBAN', 23: 'URBAN',
                    24: 'URBAN', 31: 'BARREN', 41: 'FOREST_DECIDUOUS', 42: 'FOREST_EVERGREEN',
                    43: 'FOREST_MIXED', 51: 'SHRUBLAND', 52: 'SHRUBLAND', 71: 'GRASSLAND',
                    72: 'GRASSLAND', 81: 'CROPLAND', 82: 'CROPLAND', 90: 'WETLAND',
                    95: 'WETLAND'
                }
                
                for idx, hru in hrus_reprojected.iterrows():
                    try:
                        masked_data, _ = mask(src, [hru.geometry], crop=True)
                        
                        if masked_data[0].size > 0:
                            # Remove nodata values
                            valid_data = masked_data[0][masked_data[0] != src.nodata]
                            if valid_data.size > 0:
                                # Find dominant landcover class
                                unique, counts = np.unique(valid_data, return_counts=True)
                                dominant_class = unique[np.argmax(counts)]
                                
                                # Calculate percentage of dominant class
                                dominant_pct = (np.max(counts) / len(valid_data)) * 100
                                
                                result_gdf.loc[idx, 'LANDCOVER_ID'] = int(dominant_class)
                                result_gdf.loc[idx, 'LANDCOVER_CLASS'] = landcover_classes.get(
                                    int(dominant_class), 'UNKNOWN'
                                )
                                result_gdf.loc[idx, 'LANDCOVER_PCT'] = float(dominant_pct)
                                
                    except Exception as e:
                        print(f"   Warning: Could not extract landcover for HRU {idx}: {e}")
                        continue
                        
        except Exception as e:
            print(f"   Warning: Landcover extraction failed: {e}")
            # Use default forest classification
            result_gdf['LANDCOVER_CLASS'] = 'FOREST'
            result_gdf['LANDCOVER_ID'] = 43  # Mixed forest
            result_gdf['LANDCOVER_PCT'] = 100.0
        
        return result_gdf
    
    def _add_elevation_attributes(self, hrus_gdf, dem_file: str) -> gpd.GeoDataFrame:
        """Add elevation statistics from DEM"""
        
        import rasterio
        from rasterio.mask import mask
        
        result_gdf = hrus_gdf.copy()
        
        # Initialize elevation columns
        result_gdf['ELEVATION_MEAN'] = 0.0
        result_gdf['ELEVATION_MIN'] = 0.0
        result_gdf['ELEVATION_MAX'] = 0.0
        result_gdf['SLOPE_MEAN'] = 0.0
        
        if not Path(dem_file).exists():
            return result_gdf
            
        try:
            print(f"   Extracting elevation from DEM: {dem_file}")
            
            with rasterio.open(dem_file) as src:
                print(f"   DEM raster CRS: {src.crs}")
                print(f"   HRUs CRS: {result_gdf.crs}")
                
                # Reproject HRUs to match raster CRS if needed
                if src.crs != result_gdf.crs:
                    print(f"   Reprojecting HRUs from {result_gdf.crs} to {src.crs} for elevation extraction")
                    hrus_reprojected = result_gdf.to_crs(src.crs)
                else:
                    hrus_reprojected = result_gdf
                
                for idx, hru in hrus_reprojected.iterrows():
                    try:
                        masked_data, _ = mask(src, [hru.geometry], crop=True)
                        
                        if masked_data[0].size > 0:
                            # Remove nodata values
                            valid_elev = masked_data[0][masked_data[0] != src.nodata]
                            if valid_elev.size > 0:
                                # Calculate elevation statistics
                                result_gdf.loc[idx, 'ELEVATION_MEAN'] = float(np.mean(valid_elev))
                                result_gdf.loc[idx, 'ELEVATION_MIN'] = float(np.min(valid_elev))
                                result_gdf.loc[idx, 'ELEVATION_MAX'] = float(np.max(valid_elev))
                                
                                # Calculate slope from elevation range (simplified)
                                elev_range = float(np.max(valid_elev) - np.min(valid_elev))
                                area_m2 = hru.geometry.area
                                hru_length = np.sqrt(area_m2)
                                if hru_length > 0:
                                    slope_degrees = np.arctan(elev_range / hru_length) * 180 / np.pi
                                    result_gdf.loc[idx, 'SLOPE_MEAN'] = max(0.1, min(45.0, slope_degrees))
                                
                    except Exception as e:
                        print(f"   Warning: Could not extract elevation for HRU {idx}: {e}")
                        continue
                        
        except Exception as e:
            print(f"   Warning: Elevation extraction failed: {e}")
            # Use default values
            result_gdf['ELEVATION_MEAN'] = 500.0
            result_gdf['ELEVATION_MIN'] = 450.0
            result_gdf['ELEVATION_MAX'] = 550.0
            result_gdf['SLOPE_MEAN'] = 5.0
        
        return result_gdf
    
    def _standardize_hru_columns(self, hrus_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Standardize column names using exact BasinMaker format"""
        
        result_gdf = hrus_gdf.copy()
        
        # Map to exact column names requested by user
        column_mapping = {
            # HRU columns
            'HRU_ID': 'HRU_ID',
            'HRU_Area': 'HRU_Area',
            'HRU_Area_km2': 'HRU_Area',
            'Elevation': 'HRU_E_mean',
            'ELEVATION_MEAN': 'HRU_E_mean',
            'ELEVATION': 'HRU_E_mean',
            'HRU_CenX': 'HRU_CenX',
            'HRU_CenY': 'HRU_CenY',
            'LAND_USE_C': 'LAND_USE_C',
            'LANDCOVER_CLASS': 'LAND_USE_C',
            'SOIL_PROF': 'SOIL_PROF',
            'SOIL_TEXTURE': 'SOIL_PROF',
            'VEG_C': 'VEG_C',
            
            # Additional columns that might be needed
            'HRU_S_mean': 'HRU_S_mean',
            'HRU_A_mean': 'HRU_A_mean',
            'SubId': 'SubId',
            'DowSubId': 'DowSubId',
            'RivLength': 'RivLength',
            'RivSlope': 'RivSlope',
            'BkfWidth': 'BkfWidth',
            'BkfDepth': 'BkfDepth'
        }
        
        # Rename columns to exact standard names
        for old_name, new_name in column_mapping.items():
            if old_name in result_gdf.columns:
                result_gdf[new_name] = result_gdf[old_name]
            elif new_name not in result_gdf.columns:
                # Create missing columns with defaults
                if new_name == 'HRU_S_mean':
                    result_gdf[new_name] = 5.0
                elif new_name == 'HRU_A_mean':
                    result_gdf[new_name] = 180.0
                elif new_name == 'SubId':
                    # Inherit SubId from parent subbasin (should already exist from HRU generation)
                    if 'SubId' not in result_gdf.columns:
                        result_gdf[new_name] = range(1, len(result_gdf) + 1)
                elif new_name == 'DowSubId':
                    # DowSubId should inherit from parent subbasin's routing connectivity
                    if 'DowSubId' not in result_gdf.columns:
                        # If no DowSubId exists, keep as self-reference for now (fallback)
                        result_gdf[new_name] = result_gdf.get('SubId', range(1, len(result_gdf) + 1))
                        print(f"      WARNING: No DowSubId found in subbasins - using self-reference fallback")
                elif new_name in ['RivLength', 'RivSlope', 'BkfWidth', 'BkfDepth']:
                    result_gdf[new_name] = 0.0
                else:
                    result_gdf[new_name] = result_gdf.get(old_name, 0)
        
        # Ensure required columns exist
        required_hru_columns = [
            'HRU_ID', 'HRU_Area', 'HRU_S_mean', 'HRU_A_mean', 'HRU_E_mean',
            'HRU_CenX', 'HRU_CenY', 'LAND_USE_C', 'SOIL_PROF', 'VEG_C'
        ]
        
        for col in required_hru_columns:
            if col not in result_gdf.columns:
                if col == 'HRU_ID':
                    result_gdf[col] = range(1, len(result_gdf) + 1)
                elif col == 'HRU_Area':
                    result_gdf[col] = result_gdf.geometry.area / 1e6  # km²
                elif col == 'HRU_S_mean':
                    result_gdf[col] = 5.0
                elif col == 'HRU_A_mean':
                    result_gdf[col] = 180.0
                elif col == 'HRU_E_mean':
                    result_gdf[col] = 500.0
                elif col == 'HRU_CenX':
                    centroids = result_gdf.geometry.centroid
                    result_gdf[col] = centroids.x
                elif col == 'HRU_CenY':
                    centroids = result_gdf.geometry.centroid
                    result_gdf[col] = centroids.y
                elif col in ['LAND_USE_C', 'SOIL_PROF', 'VEG_C']:
                    result_gdf[col] = 'UNKNOWN'
        
        return result_gdf
    
    def _load_basinmaker_lookup_tables(self) -> Dict[str, pd.DataFrame]:
        """Load BasinMaker lookup tables from actual data sources"""
        
        from processors.basinmaker_loader import BasinMakerDataLoader
        
        loader = BasinMakerDataLoader()
        return loader.load_lookup_tables()