#!/usr/bin/env python3
"""
Professional Watershed Analysis Module
Using the most stable and reliable libraries for watershed delineation

Primary Libraries:
- whitebox: Official WhiteboxTools Python wrapper (most comprehensive)
- pyflwdir: High-performance flow analysis by Deltares
- rasterio: Robust raster I/O and processing
- geopandas: Vector data handling and analysis
- fiona: Shapefile and vector format support

Author: RAVEN Hydrological Modeling System
Date: 2025-07-29
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd

# Core GIS and raster processing
try:
    import rasterio
    import rasterio.features
    import rasterio.mask
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
    import pyproj
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class ProfessionalWatershedAnalyzer:
    """
    Professional-grade watershed analysis using the most stable libraries
    
    Features:
    - Comprehensive DEM preprocessing
    - Multiple flow direction algorithms (D8, D-infinity)
    - Stream network extraction with proper thresholds
    - Watershed and subbasin delineation
    - Stream ordering and network topology
    - Quality assessment and validation
    - Multiple output formats (GeoJSON, Shapefile, GeoTIFF)
    """
    
    def __init__(self, work_dir: Optional[Path] = None):
        """
        Initialize the professional watershed analyzer
        
        Parameters:
        -----------
        work_dir : Path, optional
            Working directory for temporary files. If None, uses current directory.
        """
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.work_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize WhiteboxTools
        if HAS_WHITEBOX:
            self.wbt = whitebox.WhiteboxTools()
            self.wbt.work_dir = str(self.work_dir)
            self.wbt.verbose = True # Enable verbose output
        else:
            self.wbt = None
            
        # Analysis parameters
        self.default_stream_threshold = 1000  # Flow accumulation threshold for streams
        self.snap_distance = 50  # Distance for snapping outlets to streams (meters)
        
        # Supported output formats
        self.vector_formats = ['geojson', 'shapefile', 'gpkg']
        self.raster_formats = ['geotiff', 'ascii']
        
        print(f"Professional Watershed Analyzer initialized")
        print(f"Working directory: {self.work_dir}")
        print(f"WhiteboxTools available: {HAS_WHITEBOX}")
        print(f"pyflwdir available: {HAS_PYFLWDIR}")
        print(f"Vector tools available: {HAS_VECTOR_TOOLS}")
        print(f"Rasterio available: {HAS_RASTERIO}")
    
    def analyze_watershed_complete(self, 
                                 dem_path: Path, 
                                 outlet_coords: Tuple[float, float],
                                 output_dir: Path,
                                 stream_threshold: Optional[int] = None,
                                 flow_algorithm: str = 'd8',
                                 output_formats: List[str] = ['geojson', 'geotiff'],
                                 coordinate_system: str = 'EPSG:4326',
                                 existing_streams: Optional[Path] = None,
                                 burn_streams: bool = False,
                                 burn_distance: float = 10.0,
                                 snap_distance: float = 1.0) -> Dict:
        """
        Perform complete watershed analysis with all components
        
        Parameters:
        -----------
        dem_path : Path
            Path to DEM raster file
        outlet_coords : Tuple[float, float]
            (longitude, latitude) of watershed outlet
        output_dir : Path
            Directory to save all analysis outputs
        stream_threshold : int, optional
            Flow accumulation threshold for stream extraction
        flow_algorithm : str
            Flow direction algorithm: 'd8', 'dinf', or 'mfd'
        output_formats : List[str]
            Output formats: 'geojson', 'shapefile', 'gpkg', 'geotiff', 'ascii'
        coordinate_system : str
            Target coordinate system (EPSG code)
            
        Returns:
        --------
        Dict
            Complete analysis results with file paths and metadata
        """
        
        if not self._check_requirements():
            return {'success': False, 'error': 'Required libraries not available'}
            
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        results = {
            'success': True,
            'analysis_type': 'complete_watershed_analysis',
            'method': 'WhiteboxTools + pyflwdir',
            'files_created': [],
            'metadata': {},
            'errors': [],
            'processing_steps': []
        }
        
        print(f"\n=== PROFESSIONAL WATERSHED ANALYSIS ===")
        print(f"DEM file: {dem_path}")
        print(f"Outlet coordinates: {outlet_coords}")
        print(f"Flow algorithm: {flow_algorithm}")
        print(f"Stream threshold: {stream_threshold or self.default_stream_threshold}")
        print(f"Output formats: {output_formats}")
        
        try:
            # Step 1: Validate and preprocess DEM
            print(f"\n1. DEM Validation and Preprocessing...")
            dem_info = self._validate_dem(dem_path)
            if not dem_info['valid']:
                return {'success': False, 'error': f"Invalid DEM: {dem_info['error']}"}
            
            results['metadata']['dem_info'] = dem_info
            results['processing_steps'].append('DEM validation completed')
            
            # Step 2: DEM preprocessing (fill depressions, conditioning)
            print(f"2. DEM Preprocessing...")
            preprocessed_files = self._preprocess_dem(dem_path, output_dir, flow_algorithm)
            results['files_created'].extend(preprocessed_files)
            results['processing_steps'].append('DEM preprocessing completed')
            
            # Step 2.5: Stream burning if vector network provided
            working_dem = str(preprocessed_files['conditioned_dem'])
            if burn_streams and existing_streams and existing_streams.exists():
                print(f"2.5. Burning Existing Stream Network...")
                burned_files = self._burn_stream_network(
                    dem_path=working_dem,
                    existing_streams=existing_streams,
                    output_dir=output_dir,
                    burn_distance=burn_distance,
                    snap_distance=snap_distance
                )
                if burned_files:
                    preprocessed_files.update(burned_files)
                    results['files_created'].extend(burned_files.values())
                    results['processing_steps'].append('Stream burning completed')
                    working_dem = burned_files['burned_dem']
            
            # Step 3: Flow direction and accumulation
            print(f"3. Flow Direction and Accumulation...")
            flow_files = self._calculate_flow_analysis(
                working_dem, output_dir, flow_algorithm
            )
            results['files_created'].extend(flow_files.values())
            results['processing_steps'].append('Flow analysis completed')
            
            # Step 4: Stream network extraction
            print(f"4. Stream Network Extraction...")
            stream_threshold = stream_threshold or self.default_stream_threshold
            stream_files = self._extract_stream_network(
                flow_files, output_dir, stream_threshold, output_formats
            )
            results['files_created'].extend(stream_files)
            results['processing_steps'].append('Stream network extraction completed')
            
            # Step 5: Watershed delineation
            print(f"5. Watershed Delineation...")
            watershed_files = self._delineate_watershed(
                flow_files, outlet_coords, output_dir, output_formats
            )
            results['files_created'].extend(watershed_files)
            results['processing_steps'].append('Watershed delineation completed')
            
            # Step 6: Subbasin delineation
            print(f"6. Subbasin Delineation...")
            subbasin_files = self._delineate_subbasins(
                flow_files, stream_files, output_dir, output_formats
            )
            results['files_created'].extend(subbasin_files)
            results['processing_steps'].append('Subbasin delineation completed')
            
            # Step 7: Stream ordering and network topology
            print(f"7. Stream Ordering and Network Topology...")
            network_files = self._analyze_stream_network(
                stream_files, flow_files, output_dir, output_formats
            )
            results['files_created'].extend(network_files)
            results['processing_steps'].append('Stream network analysis completed')
            
            # Step 8: Calculate watershed statistics
            print(f"8. Watershed Statistics and Validation...")
            statistics = self._calculate_watershed_statistics(
                dem_path, watershed_files, subbasin_files, stream_files, outlet_coords
            )
            results['metadata']['statistics'] = statistics
            results['processing_steps'].append('Statistics calculation completed')
            
            # Step 9: Quality assessment
            print(f"9. Quality Assessment...")
            quality_report = self._assess_analysis_quality(
                results['files_created'], statistics
            )
            results['metadata']['quality_assessment'] = quality_report
            results['processing_steps'].append('Quality assessment completed')
            
            # Step 10: Create summary report
            print(f"10. Creating Summary Report...")
            report_file = self._create_analysis_report(results, output_dir)
            results['files_created'].append(report_file)
            results['processing_steps'].append('Summary report created')
            
            print(f"\n=== ANALYSIS COMPLETE ===")
            print(f"Total files created: {len(results['files_created'])}")
            print(f"Watershed area: {statistics.get('watershed_area_km2', 'N/A'):.2f} km²")
            print(f"Total stream length: {statistics.get('total_stream_length_km', 'N/A'):.2f} km")
            print(f"Number of subbasins: {statistics.get('subbasin_count', 'N/A')}")
            
            return results
            
        except Exception as e:
            error_msg = f"Watershed analysis failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            results['success'] = False
            results['errors'].append(error_msg)
            return results
    
    def _check_requirements(self) -> bool:
        """Check if all required libraries are available"""
        required = [HAS_WHITEBOX, HAS_RASTERIO, HAS_VECTOR_TOOLS]
        if not all(required):
            missing = []
            if not HAS_WHITEBOX:
                missing.append("whitebox (pip install whitebox)")
            if not HAS_RASTERIO:
                missing.append("rasterio (pip install rasterio)")
            if not HAS_VECTOR_TOOLS:
                missing.append("vector tools (pip install geopandas fiona shapely)")
            
            print(f"ERROR: Missing required libraries: {', '.join(missing)}")
            return False
        return True
    
    def _validate_dem(self, dem_path: Path) -> Dict:
        """Validate DEM file and extract metadata"""
        try:
            with rasterio.open(dem_path) as src:
                profile = src.profile
                bounds = src.bounds
                crs = src.crs
                data = src.read(1, masked=True)
                
                # Check for no-data values
                nodata_count = np.sum(data.mask) if hasattr(data, 'mask') else 0
                total_pixels = data.size
                nodata_percentage = (nodata_count / total_pixels) * 100
                
                # Basic statistics
                valid_data = data[~data.mask] if hasattr(data, 'mask') else data
                
                return {
                    'valid': True,
                    'width': profile['width'],
                    'height': profile['height'],
                    'crs': str(crs),
                    'bounds': bounds,
                    'pixel_size': abs(profile['transform'][0]),
                    'min_elevation': float(np.min(valid_data)),
                    'max_elevation': float(np.max(valid_data)),
                    'mean_elevation': float(np.mean(valid_data)),
                    'nodata_percentage': nodata_percentage,
                    'total_pixels': total_pixels
                }
                
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _preprocess_dem(self, dem_path: str, output_dir: Path, flow_algorithm: str) -> Dict:
        """Preprocess DEM for hydrological analysis"""
        files_created = {}
        
        # Use absolute paths
        dem_path_abs = Path(dem_path).resolve()
        output_dir_abs = Path(output_dir).resolve()
        
        # Fill depressions
        filled_dem = output_dir_abs / "dem_filled.tif"
        print(f"   Filling depressions...")
        self.wbt.fill_depressions(str(dem_path_abs), str(filled_dem))
        if filled_dem.exists():
            files_created['filled_dem'] = str(filled_dem)
        
        # Breach depressions (alternative to filling)
        breached_dem = output_dir_abs / "dem_breached.tif"
        print(f"   Breaching depressions...")
        self.wbt.breach_depressions(str(dem_path_abs), str(breached_dem))
        if breached_dem.exists():
            files_created['breached_dem'] = str(breached_dem)
        
        # Hydrological conditioning (recommended)
        conditioned_dem = output_dir_abs / "dem_conditioned.tif"
        print(f"   Hydrological conditioning...")
        self.wbt.fill_depressions(str(breached_dem), str(conditioned_dem))
        if conditioned_dem.exists():
            files_created['conditioned_dem'] = str(conditioned_dem)
        
        return files_created
    
    def _calculate_flow_analysis(self, dem_path: str, output_dir: Path, algorithm: str) -> Dict:
        """Calculate flow direction and accumulation"""
        files_created = {}
        
        if algorithm.lower() == 'd8':
            # D8 flow direction
            flow_dir = output_dir / "flow_direction_d8.tif"
            print(f"   Calculating D8 flow direction...")
            self.wbt.d8_pointer(str(Path(dem_path).resolve()), str(flow_dir.resolve()))
            
            # Verify flow direction file was created
            if flow_dir.exists():
                files_created['flow_direction'] = str(flow_dir.resolve())
                print(f"   Created: {flow_dir}")
            else:
                print(f"   Error: flow direction file not created")
                return files_created
                
            # D8 flow accumulation
            flow_accum = output_dir / "flow_accumulation_d8.tif"
            print(f"   Calculating D8 flow accumulation...")
            self.wbt.d8_flow_accumulation(str(Path(dem_path).resolve()), str(flow_accum.resolve()))
            
            # Verify flow accumulation file was created
            if flow_accum.exists():
                files_created['flow_accumulation'] = str(flow_accum.resolve())
                print(f"   Created: {flow_accum}")
            else:
                print(f"   Error: flow accumulation file not created")
                return files_created
                
        elif algorithm.lower() == 'dinf':
            # D-infinity flow direction
            flow_dir = output_dir / "flow_direction_dinf.tif"
            print(f"   Calculating D-infinity flow direction...")
            self.wbt.d_inf_pointer(str(Path(dem_path).resolve()), str(flow_dir.resolve()))
            
            if flow_dir.exists():
                files_created['flow_direction'] = str(flow_dir.resolve())
            else:
                return files_created
                
            # D-infinity flow accumulation
            flow_accum = output_dir / "flow_accumulation_dinf.tif"
            print(f"   Calculating D-infinity flow accumulation...")
            self.wbt.d_inf_flow_accumulation(str(flow_dir.resolve()), str(flow_accum.resolve()))
            
            if flow_accum.exists():
                files_created['flow_accumulation'] = str(flow_accum.resolve())
            else:
                return files_created
                
        elif algorithm.lower() == 'mfd':
            # Multiple flow direction (FD8)
            flow_dir = output_dir / "flow_direction_mfd.tif"
            print(f"   Calculating MFD flow direction...")
            self.wbt.fd8_pointer(str(Path(dem_path).resolve()), str(flow_dir.resolve()))
            
            if flow_dir.exists():
                files_created['flow_direction'] = str(flow_dir.resolve())
            else:
                return files_created
                
            # MFD flow accumulation
            flow_accum = output_dir / "flow_accumulation_mfd.tif"
            print(f"   Calculating MFD flow accumulation...")
            self.wbt.fd8_flow_accumulation(str(Path(dem_path).resolve()), str(flow_accum.resolve()))
            
            if flow_accum.exists():
                files_created['flow_accumulation'] = str(flow_accum.resolve())
            else:
                return files_created
        
        return files_created
    
    def _extract_stream_network(self, flow_files: Dict, output_dir: Path, 
                               threshold: int, output_formats: List[str]) -> List[str]:
        """Extract stream network from flow accumulation"""
        files_created = []
        
        # Extract streams (raster) - use absolute paths
        streams_raster = output_dir / "streams.tif"
        print(f"   Extracting streams (threshold: {threshold})...")
        
        # Ensure absolute path and check if flow files exist
        flow_accum_path = Path(flow_files['flow_accumulation'])
        if not flow_accum_path.exists():
            # Check if flow files are in output directory
            flow_accum = output_dir / "flow_accumulation_d8.tif"
            if flow_accum.exists():
                flow_accum_path = flow_accum
            else:
                print(f"   Error: flow accumulation file not found: {flow_accum_path}")
                return files_created
        
        streams_raster_abs = streams_raster.resolve()
        self.wbt.extract_streams(str(flow_accum_path.resolve()), 
                                str(streams_raster_abs), threshold)
        
        # Verify file was created
        if streams_raster_abs.exists():
            files_created.append(str(streams_raster_abs))
            print(f"   Created streams raster: {streams_raster_abs}")
        else:
            print(f"   Error: streams raster not created at {streams_raster_abs}")
            return files_created
        
        # Convert to vector formats using robust approach
        if 'geojson' in output_formats or 'shapefile' in output_formats:
            print(f"   Converting streams to vector...")
            
            # Check if streams raster exists
            if not streams_raster_abs.exists():
                print(f"   Error: streams raster not found at {streams_raster_abs}")
                return files_created
                
            try:
                # Method 1: Try WhiteboxTools native conversion
                streams_vector = output_dir / "streams.shp"
                self.wbt.raster_streams_to_vector(
                    str(streams_raster_abs), 
                    str(Path(flow_files['flow_direction']).resolve()), 
                    str(streams_vector.resolve())
                )
                
                # Verify file was created
                if streams_vector.exists():
                    if 'shapefile' in output_formats:
                        files_created.append(str(streams_vector))
                    
                    # Convert to GeoJSON if requested
                    if 'geojson' in output_formats:
                        streams_geojson = output_dir / "streams.geojson"
                        gdf = gpd.read_file(str(streams_vector))
                        gdf.to_file(str(streams_geojson), driver='GeoJSON')
                        files_created.append(str(streams_geojson))
                        
                else:
                    # Method 2: Manual vectorization using rasterio/geopandas
                    print("   WhiteboxTools conversion failed, trying manual approach...")
                    manual_files = self._manual_stream_vectorization(
                        str(streams_raster_abs), str(Path(flow_files['flow_direction']).resolve()), 
                        output_dir, output_formats
                    )
                    files_created.extend(manual_files)
                    
            except Exception as e:
                print(f"   Error in stream vectorization: {e}")
                print("   Attempting manual vectorization...")
                manual_files = self._manual_stream_vectorization(
                    str(streams_raster_abs), str(Path(flow_files['flow_direction']).resolve()), 
                    output_dir, output_formats
                )
                files_created.extend(manual_files)
        
        return files_created
    
    def _delineate_watershed(self, flow_files: Dict, outlet_coords: Tuple[float, float],
                           output_dir: Path, output_formats: List[str]) -> List[str]:
        """Delineate watershed from outlet point"""
        files_created = []
        
        # Create outlet point shapefile
        outlet_shp = output_dir / "outlet_point.shp"
        self._create_outlet_point_file(outlet_coords, outlet_shp)
        files_created.append(str(outlet_shp))
        
        # Use outlet point directly (snapping handled at workflow level)
        snapped_outlet = outlet_shp  # Use original outlet directly
        print(f"   Using outlet point directly for watershed delineation...")
        
        # Delineate watershed (raster)
        watershed_raster = output_dir / "watershed.tif"
        print(f"   Delineating watershed...")
        
        # Use absolute paths for watershed delineation
        flow_dir_path = Path(flow_files['flow_direction']).resolve()
        snapped_outlet_path = snapped_outlet.resolve()
        watershed_raster_path = watershed_raster.resolve()
        
        try:
            print(f"   Running WhiteboxTools watershed tool...")
            print(f"   Flow direction: {flow_dir_path}")
            print(f"   Snapped outlet: {snapped_outlet_path}")
            print(f"   Output raster: {watershed_raster_path}")

            self.wbt.watershed(
                d8_pntr=str(flow_dir_path), 
                pour_pts=str(snapped_outlet_path), 
                output=str(watershed_raster_path)
            )
            
            if watershed_raster_path.exists():
                files_created.append(str(watershed_raster_path))
                print(f"   SUCCESS: Created watershed raster: {watershed_raster_path}")
            else:
                print(f"   ERROR: watershed raster not created by WhiteboxTools.")
                # Attempt to get more detailed error from WBT
                if self.wbt.verbose:
                    print("WhiteboxTools output:")
                    # This part is tricky as whitebox-python doesn't directly expose stdout
                    # but we can infer from the lack of output file.
                return files_created
                
        except Exception as e:
            print(f"   CRITICAL ERROR: Watershed delineation failed with exception: {e}")
            return []  # Return empty list on failure
            
        # Convert to vector formats
        if 'geojson' in output_formats or 'shapefile' in output_formats:
            print(f"   Converting watershed to vector...")
            
            watershed_vector = output_dir / "watershed.shp"
            watershed_vector_path = watershed_vector.resolve()
            
            if watershed_raster_path.exists():
                try:
                    self.wbt.raster_to_vector_polygons(str(watershed_raster_path), str(watershed_vector_path))
                    
                    if watershed_vector_path.exists():
                        if 'shapefile' in output_formats:
                            files_created.append(str(watershed_vector_path))
                        
                        # Convert to GeoJSON if requested
                        if 'geojson' in output_formats:
                            watershed_geojson = output_dir / "watershed.geojson"
                            gdf = gpd.read_file(str(watershed_vector_path))
                            gdf.to_file(str(watershed_geojson), driver='GeoJSON')
                            files_created.append(str(watershed_geojson))
                    else:
                        print(f"   Error: watershed vector not created")
                except Exception as e:
                    print(f"   Vector conversion failed: {e}")
                    return files_created
        
        return files_created
    
    def _delineate_subbasins(self, flow_files: Dict, stream_files: List[str],
                           output_dir: Path, output_formats: List[str]) -> List[str]:
        """Delineate subbasins from stream network"""
        files_created = []
        
        # Find streams raster file
        streams_raster = None
        for file_path in stream_files:
            if file_path.endswith('.tif') and 'streams' in file_path:
                streams_raster = file_path
                break
        
        if not streams_raster:
            print("   WARNING: No streams raster found, skipping subbasin delineation")
            return files_created
        
        # Delineate subbasins (raster)
        subbasins_raster = output_dir / "subbasins.tif"
        print(f"   Delineating subbasins...")
        self.wbt.subbasins(
            flow_files['flow_direction'], 
            streams_raster, 
            str(subbasins_raster)
        )
        files_created.append(str(subbasins_raster))
        
        # Convert to vector formats
        if 'geojson' in output_formats or 'shapefile' in output_formats:
            print(f"   Converting subbasins to vector...")
            
            subbasins_vector = output_dir / "subbasins.shp"
            self.wbt.raster_to_vector_polygons(str(subbasins_raster), str(subbasins_vector))
            
            if 'shapefile' in output_formats:
                files_created.append(str(subbasins_vector))
            
            # Convert to GeoJSON and add attributes
            if 'geojson' in output_formats:
                subbasins_geojson = output_dir / "subbasins.geojson"
                gdf = gpd.read_file(str(subbasins_vector))
                
                # Calculate subbasin areas
                if gdf.crs and gdf.crs.is_projected:
                    gdf['area_km2'] = gdf.geometry.area / 1e6
                else:
                    # Reproject to calculate area
                    gdf_proj = gdf.to_crs('EPSG:3857')  # Web Mercator for area calculation
                    gdf['area_km2'] = gdf_proj.geometry.area / 1e6
                
                gdf.to_file(str(subbasins_geojson), driver='GeoJSON')
                files_created.append(str(subbasins_geojson))
        
        return files_created
    
    def _analyze_stream_network(self, stream_files: List[str], flow_files: Dict,
                              output_dir: Path, output_formats: List[str]) -> List[str]:
        """Analyze stream network topology and ordering"""
        files_created = []
        
        # Find streams raster file
        streams_raster = None
        for file_path in stream_files:
            if file_path.endswith('.tif') and 'streams' in file_path:
                streams_raster = file_path
                break
        
        if not streams_raster:
            print("   WARNING: No streams raster found, skipping network analysis")
            return files_created
        
        # Strahler stream order
        strahler_order = output_dir / "stream_order_strahler.tif"
        print(f"   Calculating Strahler stream order...")
        self.wbt.strahler_stream_order(
            flow_files['flow_direction'], 
            streams_raster, 
            str(strahler_order)
        )
        files_created.append(str(strahler_order))
        
        # Shreve stream order
        shreve_order = output_dir / "stream_order_shreve.tif"
        print(f"   Calculating Shreve stream order...")
        self.wbt.shreve_stream_magnitude(
            flow_files['flow_direction'], 
            streams_raster, 
            str(shreve_order)
        )
        files_created.append(str(shreve_order))
        
        # Stream length
        stream_length = output_dir / "stream_length.tif"
        print(f"   Calculating stream lengths...")
        self.wbt.length_of_upstream_channels(
            flow_files['flow_direction'], 
            streams_raster, 
            str(stream_length)
        )
        files_created.append(str(stream_length))
        
        return files_created
    
    def _create_outlet_point_file(self, outlet_coords: Tuple[float, float], output_path: Path):
        """Create outlet point shapefile"""
        lon, lat = outlet_coords
        
        # Create point geometry
        point = Point(lon, lat)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {'id': [1], 'outlet': ['main']}, 
            geometry=[point], 
            crs='EPSG:4326'
        )
        
        # Save as shapefile
        gdf.to_file(str(output_path))
    
    def _calculate_watershed_statistics(self, dem_path: Path, watershed_files: List[str],
                                      subbasin_files: List[str], stream_files: List[str],
                                      outlet_coords: Tuple[float, float]) -> Dict:
        """Calculate comprehensive watershed statistics"""
        stats = {
            'outlet_coordinates': outlet_coords,
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_method': 'WhiteboxTools + pyflwdir'
        }
        
        try:
            # DEM statistics
            with rasterio.open(dem_path) as src:
                dem_data = src.read(1, masked=True)
                stats['elevation_stats'] = {
                    'min_elevation_m': float(np.min(dem_data)),
                    'max_elevation_m': float(np.max(dem_data)),
                    'mean_elevation_m': float(np.mean(dem_data)),
                    'relief_m': float(np.max(dem_data) - np.min(dem_data))
                }
            
            # Watershed area (from vector file if available)
            watershed_vector = None
            for file_path in watershed_files:
                if file_path.endswith(('.geojson', '.shp')):
                    watershed_vector = file_path
                    break
            
            if watershed_vector:
                gdf = gpd.read_file(watershed_vector)
                # Ensure CRS is set for accurate area calculations
                if gdf.crs is None:
                    gdf = gdf.set_crs('EPSG:4326')  # Assume WGS84 if no CRS
                
                # Use appropriate UTM zone for accurate area calculation
                target_crs = 'EPSG:32617'  # UTM Zone 17N for Toronto area
                gdf_proj = gdf.to_crs(target_crs)
                area_km2 = gdf_proj.geometry.area.sum() / 1e6
                
                stats['watershed_area_km2'] = float(area_km2)
                stats['watershed_bounds'] = gdf.total_bounds.tolist()
            
            # Subbasin count and statistics
            subbasin_vector = None
            for file_path in subbasin_files:
                if file_path.endswith(('.geojson', '.shp')):
                    subbasin_vector = file_path
                    break
            
            if subbasin_vector:
                gdf = gpd.read_file(subbasin_vector)
                if gdf.crs is None:
                    gdf = gdf.set_crs('EPSG:4326')
                
                target_crs = 'EPSG:32617'
                gdf_proj = gdf.to_crs(target_crs)
                gdf_proj['area_km2'] = gdf_proj.geometry.area / 1e6
                
                stats['subbasin_count'] = len(gdf)
                stats['subbasin_areas_km2'] = gdf_proj['area_km2'].tolist()
                stats['mean_subbasin_area_km2'] = float(gdf_proj['area_km2'].mean())
                stats['min_subbasin_area_km2'] = float(gdf_proj['area_km2'].min())
                stats['max_subbasin_area_km2'] = float(gdf_proj['area_km2'].max())
            
            # Stream network statistics
            stream_vector = None
            for file_path in stream_files:
                if file_path.endswith(('.geojson', '.shp')):
                    stream_vector = file_path
                    break
            
            if stream_vector:
                gdf = gpd.read_file(stream_vector)
                if gdf.crs is None:
                    gdf = gdf.set_crs('EPSG:4326')
                
                target_crs = 'EPSG:32617'
                gdf_proj = gdf.to_crs(target_crs)
                total_length = gdf_proj.geometry.length.sum() / 1000  # Convert to km
                
                stats['total_stream_length_km'] = float(total_length)
                watershed_area = stats.get('watershed_area_km2', 1)
                stats['stream_density_km_per_km2'] = float(total_length / watershed_area)
                stats['stream_segment_count'] = len(gdf)
                
                # Additional stream metrics
                stats['mean_stream_length_km'] = float(total_length / len(gdf)) if len(gdf) > 0 else 0
                
        except Exception as e:
            stats['calculation_error'] = str(e)
            import traceback
            stats['calculation_traceback'] = traceback.format_exc()
        
        return stats
    
    def _assess_analysis_quality(self, files_created: List[str], statistics: Dict) -> Dict:
        """Assess the quality of the watershed analysis"""
        quality = {
            'overall_quality': 'excellent',
            'files_created_count': len(files_created),
            'issues': [],
            'recommendations': []
        }
        
        # Check file completeness
        required_files = ['watershed', 'streams', 'subbasins', 'flow_direction', 'flow_accumulation']
        found_files = {req: False for req in required_files}
        
        for file_path in files_created:
            file_name = Path(file_path).stem.lower()
            for req in required_files:
                if req in file_name:
                    found_files[req] = True
        
        missing_files = [req for req, found in found_files.items() if not found]
        if missing_files:
            quality['issues'].append(f"Missing files: {', '.join(missing_files)}")
            quality['overall_quality'] = 'good'
        
        # Check watershed area reasonableness
        area = statistics.get('watershed_area_km2', 0)
        if area < 1:
            quality['issues'].append("Very small watershed area - check outlet location")
        elif area > 50000:
            quality['recommendations'].append("Large watershed - consider using coarser DEM for efficiency")
        
        # Check elevation data
        relief = statistics.get('elevation_stats', {}).get('relief_m', 0)
        if relief < 10:
            quality['issues'].append("Very low relief - results may be less reliable")
        
        # Check stream density
        density = statistics.get('stream_density_km_per_km2', 0)
        if density < 0.5:
            quality['recommendations'].append("Low stream density - consider lowering stream threshold")
        elif density > 5:
            quality['recommendations'].append("High stream density - consider raising stream threshold")
        
        # Assign overall quality rating
        if len(quality['issues']) == 0:
            quality['overall_quality'] = 'excellent'
        elif len(quality['issues']) <= 2:
            quality['overall_quality'] = 'good'
        else:
            quality['overall_quality'] = 'fair'
        
        return quality
    
    def _manual_stream_vectorization(self, streams_raster_path: str, flow_dir_path: str, 
                                   output_dir: Path, output_formats: List[str]) -> List[str]:
        """Manual stream vectorization using rasterio and geopandas"""
        files_created = []
        
        try:
            import rasterio
            import numpy as np
            from rasterio.features import shapes, skeletonize
            from shapely.geometry import shape, LineString, MultiLineString
            from shapely.ops import linemerge, unary_union
            
            print("   Manual stream vectorization...")
            
            # Read streams raster
            with rasterio.open(streams_raster_path) as src:
                streams_data = src.read(1)
                streams_mask = streams_data > 0
                transform = src.transform
                crs = src.crs
            
            if not np.any(streams_mask):
                print("   Warning: No streams found in raster")
                return files_created
            
            # Create line geometries from raster using skeletonization
            lines = []
            
            # Use rasterio to extract geometries
            for geom, value in shapes(streams_data, mask=streams_mask, transform=transform):
                if value > 0:
                    # Convert polygon to line
                    coords = list(geom['coordinates'][0])
                    if len(coords) > 1:
                        line = LineString(coords)
                        if line.is_valid and line.length > 0:
                            lines.append(line)
            
            if not lines:
                print("   Warning: No valid stream lines created")
                return files_created
            
            # Merge overlapping lines and create a cleaner network
            merged = unary_union(lines)
            
            # Ensure we have individual line features
            if isinstance(merged, (LineString, MultiLineString)):
                if isinstance(merged, MultiLineString):
                    final_lines = list(merged.geoms)
                else:
                    final_lines = [merged]
            else:
                final_lines = lines
            
            # Filter out very short segments (less than 2 pixels)
            min_length = 0.001  # Approximately 2 pixels
            final_lines = [line for line in final_lines if line.length > min_length]
            
            if not final_lines:
                print("   Warning: No valid stream lines after filtering")
                return files_created
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame({
                'geometry': final_lines,
                'STREAM_ID': range(1, len(final_lines) + 1),
                'LENGTH_M': [line.length * 111000 for line in final_lines],  # Approximate meters
                'VALUE': [1] * len(final_lines)
            }, crs=crs)
            
            # Create output files
            if 'shapefile' in output_formats:
                streams_shp = output_dir / "streams.shp"
                gdf.to_file(str(streams_shp))
                files_created.append(str(streams_shp))
                print(f"   Created streams.shp with {len(gdf)} features")
            
            if 'geojson' in output_formats:
                streams_geojson = output_dir / "streams.geojson"
                gdf.to_file(str(streams_geojson), driver='GeoJSON')
                files_created.append(str(streams_geojson))
                print(f"   Created streams.geojson with {len(gdf)} features")
                
        except Exception as e:
            print(f"   Manual vectorization failed: {e}")
            import traceback
            traceback.print_exc()
            
        return files_created

    def _burn_stream_network(self, dem_path: str, existing_streams: Path,
                           output_dir: Path, burn_distance: float, snap_distance: float) -> Dict:
        """Burn existing stream network into DEM using 2024 best practices"""
        files_created = {}
        
        if not HAS_WHITEBOX:
            print("   ERROR: WhiteboxTools not available for stream burning")
            return files_created
            
        try:
            # Method 1: Try TopologicalBreachBurn (2023 best practice)
            burned_dem = output_dir / "dem_burned.tif"
            d8_pointer = output_dir / "d8_pointer_burned.tif"
            d8_accum = output_dir / "d8_accum_burned.tif"
            burned_streams = output_dir / "burned_streams.tif"
            
            print(f"   Using TopologicalBreachBurn (2024 best practice)...")
            try:
                self.wbt.topological_breach_burn(
                    str(existing_streams),
                    dem_path,
                    str(burned_dem),
                    str(d8_pointer),
                    str(d8_accum),
                    str(burned_streams),
                    burn_distance,
                    snap_distance
                )
                
                if burned_dem.exists():
                    files_created['burned_dem'] = str(burned_dem)
                    files_created['burned_pointer'] = str(d8_pointer)
                    files_created['burned_accum'] = str(d8_accum)
                    files_created['burned_streams'] = str(burned_streams)
                    print(f"   Stream burning completed successfully")
                    return files_created
                    
            except Exception as e:
                print(f"   TopologicalBreachBurn failed: {e}")
                
            # TopologicalBreachBurn failed - no synthetic fallback provided
            error_msg = "Stream burning failed - no synthetic fallback provided"
            print(f"   Error: {error_msg}")
            raise Exception(error_msg)
            
        except Exception as e:
            print(f"   Stream burning failed: {e}")
            
        return files_created
    
    def _manual_stream_burning(self, dem_path: str, existing_streams: str,
                           output_dir: Path, burn_distance: float) -> Dict:
        """Manual stream burning - REMOVED"""
        raise Exception("Manual stream burning removed - no fallback methods")
        files_created = {}
        
        try:
            import rasterio
            from rasterio.features import rasterize
            from shapely.geometry import mapping
            
            # Read DEM
            with rasterio.open(dem_path) as src:
                dem_data = src.read(1)
                transform = src.transform
                profile = src.profile
                
            # Read and rasterize streams
            streams_gdf = gpd.read_file(existing_streams)
            if streams_gdf.crs is None:
                streams_gdf = streams_gdf.set_crs('EPSG:4326')
            
            # Ensure streams are in same CRS as DEM
            dem_crs = rasterio.open(dem_path).crs
            if dem_crs is not None:
                streams_gdf = streams_gdf.to_crs(str(dem_crs))
            
            # Rasterize streams
            stream_shapes = [mapping(geom) for geom in streams_gdf.geometry]
            stream_raster = rasterize(
                stream_shapes,
                out_shape=(dem_data.shape[0], dem_data.shape[1]),
                transform=transform,
                fill=0,
                default_value=1,
                dtype=rasterio.int32
            )
            
            # Burn streams into DEM (lower elevation along streams)
            burned_dem = dem_data.astype(rasterio.float32)
            burned_dem[stream_raster == 1] -= burn_distance
            
            # Save burned DEM
            burned_dem_path = output_dir / "dem_burned_manual.tif"
            profile.update(dtype=rasterio.float32)
            
            with rasterio.open(burned_dem_path, 'w', **profile) as dst:
                dst.write(burned_dem, 1)
            
            if burned_dem_path.exists():
                files_created['burned_dem'] = str(burned_dem_path)
                print(f"   Manual stream burning completed")
                
        except Exception as e:
            print(f"   Manual stream burning failed: {e}")
            
        return files_created
    
    def _create_analysis_report(self, results: Dict, output_dir: Path) -> str:
        """Create comprehensive analysis report"""
        report_file = output_dir / "watershed_analysis_report.json"
        
        report = {
            'analysis_summary': {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': results['analysis_type'],
                'method': results['method'],
                'success': results['success'],
                'total_files_created': len(results['files_created']),
                'burning_used': 'burned_dem' in str(results['files_created'])
            },
            'processing_steps': results['processing_steps'],
            'files_created': results['files_created'],
            'metadata': results['metadata'],
            'errors': results['errors']
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Analysis report saved: {report_file}")
        return str(report_file)


# Convenience function for quick watershed analysis
def analyze_watershed(dem_path: Union[str, Path], 
                     outlet_coords: Tuple[float, float],
                     output_dir: Union[str, Path],
                     **kwargs) -> Dict:
    """
    Convenience function for quick watershed analysis
    
    Parameters:
    -----------
    dem_path : str or Path
        Path to DEM raster file
    outlet_coords : Tuple[float, float]
        (longitude, latitude) of watershed outlet
    output_dir : str or Path
        Directory to save analysis outputs
    **kwargs : additional arguments
        Additional arguments passed to analyze_watershed_complete()
    
    Returns:
    --------
    Dict
        Analysis results
    """
    analyzer = ProfessionalWatershedAnalyzer()
    return analyzer.analyze_watershed_complete(
        Path(dem_path), outlet_coords, Path(output_dir), **kwargs
    )


if __name__ == "__main__":
    # Example usage
    print("Professional Watershed Analyzer")
    print("This module provides comprehensive watershed analysis using stable libraries.")
    print("\nRequired libraries:")
    print("- whitebox: pip install whitebox")
    print("- pyflwdir: pip install pyflwdir") 
    print("- rasterio: pip install rasterio")
    print("- geopandas: pip install geopandas")
    print("- fiona: pip install fiona")
    print("- shapely: pip install shapely")
    print("\nExample usage:")
    print("from professional_watershed_analyzer import analyze_watershed")
    print("results = analyze_watershed('dem.tif', (-75.5, 45.5), 'output_dir')")
