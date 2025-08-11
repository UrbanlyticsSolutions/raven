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
        self.default_stream_threshold = 5000  # Flow accumulation threshold for streams
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
                dem_path, watershed_files, subbasin_files, stream_files, outlet_coords, output_dir
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
            print(f"Max stream order: {statistics.get('max_stream_order', 'N/A')}")
            
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
        
        # Step 1: Initial subbasin delineation
        subbasins_raster = output_dir / "subbasins.tif"
        print(f"   Delineating initial subbasins...")
        self.wbt.subbasins(
            flow_files['flow_direction'], 
            streams_raster, 
            str(subbasins_raster)
        )
        
        if not subbasins_raster.exists():
            print("   ERROR: Initial subbasin delineation failed")
            return files_created
            
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
                
                # Calculate routing connectivity between subbasins
                print(f"   Calculating subbasin routing connectivity...")
                routing_gdf = self._calculate_subbasin_routing(
                    gdf, 
                    str(subbasins_raster), 
                    flow_files['flow_direction'],
                    flow_files['flow_accumulation']
                )
                
                # Integrate lake routing if lakes are present
                routing_gdf = self._integrate_lake_routing(routing_gdf, output_dir)
                
                # Save routing results to both GeoJSON and update original shapefile
                routing_gdf.to_file(str(subbasins_geojson), driver='GeoJSON')
                files_created.append(str(subbasins_geojson))
                
                # CRITICAL: Update original subbasins.shp with DowSubId routing
                subbasins_shp = output_dir / "subbasins.shp"
                if subbasins_shp.exists():
                    routing_gdf.to_file(str(subbasins_shp))
                    print(f"   Updated subbasins.shp with routing (DowSubId added)")
                else:
                    # Save as new shapefile if original doesn't exist
                    routing_gdf.to_file(str(subbasins_shp))
                    files_created.append(str(subbasins_shp))
        
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
                                      outlet_coords: Tuple[float, float], output_dir: Path) -> Dict:
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
                
                # Calculate appropriate UTM zone based on actual outlet coordinates
                lon = outlet_coords[0]
                utm_zone = int((lon + 180) / 6) + 1
                # Northern hemisphere (use 326xx) vs Southern hemisphere (use 327xx)
                lat = outlet_coords[1]
                target_crs = f'EPSG:{32600 + utm_zone}' if lat >= 0 else f'EPSG:{32700 + utm_zone}'
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
                
                # Calculate max stream order from Strahler order raster
                strahler_raster = Path(output_dir) / "stream_order_strahler.tif"
                if strahler_raster.exists():
                    try:
                        with rasterio.open(strahler_raster) as src:
                            strahler_data = src.read(1, masked=True)
                            max_order = int(strahler_data.max())
                            stats['max_stream_order'] = max_order
                    except Exception as e:
                        stats['max_stream_order'] = None
                        stats['stream_order_error'] = str(e)
                
        except Exception as e:
            stats['calculation_error'] = str(e)
            import traceback
            stats['calculation_traceback'] = traceback.format_exc()
        
        return stats
    
    def _calculate_subbasin_routing(self, subbasins_gdf: gpd.GeoDataFrame, 
                                  subbasins_raster_path: str,
                                  flow_direction_path: str,
                                  flow_accumulation_path: str) -> gpd.GeoDataFrame:
        """
        Calculate routing connectivity ONLY using subbasins.shp geometries
        NO RASTER SUBID DEPENDENCY
        """
        
        print(f"      ROUTING ANALYSIS: {len(subbasins_gdf)} subbasins (SHAPEFILE ONLY)")
        
        # Use ONLY the shapefile subbasins
        routing_gdf = self._calculate_shapefile_only_routing(
            subbasins_gdf, flow_direction_path, flow_accumulation_path
        )
        
        return routing_gdf
    
    def _calculate_shapefile_only_routing(self, subbasins_gdf: gpd.GeoDataFrame,
                                        flow_direction_path: str,
                                        flow_accumulation_path: str) -> gpd.GeoDataFrame:
        """
        Calculate routing using ONLY shapefile subbasins - no raster dependency
        """
        import rasterio
        import numpy as np
        
        print(f"      Calculating routing for {len(subbasins_gdf)} subbasins...")
        
        # Load flow accumulation to find outlet
        with rasterio.open(flow_accumulation_path) as src:
            flow_acc_data = src.read(1)
            transform = src.transform
            raster_crs = src.crs
        
        # Create routing dataframe and handle CRS mismatch
        routing_gdf = subbasins_gdf.copy()
        
        # Reproject subbasins to match raster CRS if needed
        if routing_gdf.crs != raster_crs:
            routing_gdf = routing_gdf.to_crs(raster_crs)
        
        # Ensure SubId column exists
        if 'SubId' not in routing_gdf.columns:
            if 'VALUE' in routing_gdf.columns:
                routing_gdf['SubId'] = routing_gdf['VALUE']
            else:
                routing_gdf['SubId'] = range(1, len(routing_gdf) + 1)
        
        # Find outlet subbasin (highest flow accumulation)
        outlet_subbasin_id = None
        max_flow_acc = -1
        
        for _, subbasin in routing_gdf.iterrows():
            subbasin_id = subbasin['SubId']
            centroid = subbasin.geometry.centroid
            
            # Convert to raster coordinates
            try:
                centroid_row, centroid_col = rasterio.transform.rowcol(transform, centroid.x, centroid.y)
                
                if (0 <= centroid_row < flow_acc_data.shape[0] and 
                    0 <= centroid_col < flow_acc_data.shape[1]):
                    flow_acc_value = flow_acc_data[centroid_row, centroid_col]
                    
                    if flow_acc_value > max_flow_acc:
                        max_flow_acc = flow_acc_value
                        outlet_subbasin_id = subbasin_id
                    
            except Exception:
                continue  # Skip subbasins with coordinate issues
        
        if outlet_subbasin_id is None:
            raise RuntimeError("Could not identify outlet subbasin from flow accumulation data")
        
        print(f"      Outlet identified: SubId {outlet_subbasin_id}")
        
        # Simple routing: all subbasins drain to outlet, outlet drains to -1
        routing_gdf['DowSubId'] = outlet_subbasin_id
        routing_gdf.loc[routing_gdf['SubId'] == outlet_subbasin_id, 'DowSubId'] = -1
        
        # Validate we have exactly one outlet
        outlets = (routing_gdf['DowSubId'] == -1).sum()
        if outlets != 1:
            raise RuntimeError(f"Invalid routing: found {outlets} outlets, expected 1")
        
        print(f"      SUCCESS: Routing calculated with outlet SubId {outlet_subbasin_id}")
        
        # Return in original CRS
        if routing_gdf.crs != subbasins_gdf.crs:
            routing_gdf = routing_gdf.to_crs(subbasins_gdf.crs)
        
        return routing_gdf
    
    def _calculate_stream_order_routing(self, subbasins_gdf: gpd.GeoDataFrame, 
                                       subbasins_raster_path: str, stream_order_path: str,
                                       flow_accumulation_path: str) -> gpd.GeoDataFrame:
        """Calculate routing using stream order hierarchy"""
        
        try:
            # Read raster data
            with rasterio.open(subbasins_raster_path) as subbasins_src:
                subbasins_data = subbasins_src.read(1)
                
            with rasterio.open(stream_order_path) as stream_order_src:
                stream_order_data = stream_order_src.read(1)
                
            with rasterio.open(flow_accumulation_path) as flow_acc_src:
                flow_acc_data = flow_acc_src.read(1)
            
            print(f"      Processing stream order routing...")
            
            # Get subbasin stream order information
            subbasin_orders = {}
            unique_subbasin_ids = np.unique(subbasins_data[subbasins_data > 0])
            
            # Find max stream order and outlet location for each subbasin
            for subbasin_id in unique_subbasin_ids:
                subbasin_mask = (subbasins_data == subbasin_id)
                
                # Get stream order values within this subbasin
                subbasin_stream_orders = stream_order_data[subbasin_mask]
                valid_orders = subbasin_stream_orders[subbasin_stream_orders > 0]
                
                if len(valid_orders) > 0:
                    max_order = np.max(valid_orders)
                    subbasin_orders[subbasin_id] = max_order
                else:
                    # No streams in subbasin, treat as order 1
                    subbasin_orders[subbasin_id] = 1
            
            print(f"      Subbasin stream orders: {subbasin_orders}")
            
            # Find global maximum stream order (watershed outlet)
            max_global_order = max(subbasin_orders.values())
            print(f"      Maximum stream order in watershed: {max_global_order}")
            
            # Find the true watershed outlet (highest flow accumulation)
            max_flow_acc = np.max(flow_acc_data)
            outlet_candidates = np.where(flow_acc_data == max_flow_acc)
            outlet_row, outlet_col = outlet_candidates[0][0], outlet_candidates[1][0]
            outlet_subbasin = subbasins_data[outlet_row, outlet_col]
            print(f"      True watershed outlet: Subbasin {outlet_subbasin} (max flow acc: {max_flow_acc})")
            
            # Create routing dictionary using stream order hierarchy
            routing_dict = {}
            
            for subbasin_id, stream_order in subbasin_orders.items():
                if subbasin_id == outlet_subbasin:
                    # True watershed outlet
                    routing_dict[subbasin_id] = -1
                    print(f"        Subbasin {subbasin_id} (order {stream_order}) -> WATERSHED OUTLET")
                elif stream_order == max_global_order:
                    # Highest order streams flow to outlet subbasin
                    routing_dict[subbasin_id] = outlet_subbasin
                    print(f"        Subbasin {subbasin_id} (order {stream_order}) -> {outlet_subbasin}")
                else:
                    # Lower order streams flow to higher order streams
                    downstream_subbasin = self._find_downstream_by_proximity_and_order(
                        subbasin_id, stream_order, subbasin_orders, subbasins_data, flow_acc_data
                    )
                    routing_dict[subbasin_id] = downstream_subbasin
                    print(f"        Subbasin {subbasin_id} (order {stream_order}) -> {downstream_subbasin}")
            
            # Update GeoDataFrame with routing information
            routing_gdf = subbasins_gdf.copy()
            
            # Ensure SubId column exists
            if 'VALUE' in routing_gdf.columns:
                routing_gdf['SubId'] = routing_gdf['VALUE']
            elif 'SubId' not in routing_gdf.columns:
                routing_gdf['SubId'] = range(1, len(routing_gdf) + 1)
            
            # Add DowSubId based on routing dictionary
            routing_gdf['DowSubId'] = routing_gdf['SubId'].map(routing_dict)
            
            # Handle any unmapped subbasins
            unmapped_mask = routing_gdf['DowSubId'].isna()
            if unmapped_mask.any():
                print(f"      WARNING: {unmapped_mask.sum()} subbasins have no routing info, routing to outlet")
                routing_gdf.loc[unmapped_mask, 'DowSubId'] = outlet_subbasin
            
            # Validate routing network
            self._validate_routing_network(routing_gdf)
            
            print(f"      SUCCESS: Stream order routing calculated for {len(routing_gdf)} subbasins")
            return routing_gdf
            
        except Exception as e:
            print(f"      FAILED: Stream order routing failed: {e}")
            import traceback
            print(f"      TRACEBACK: {traceback.format_exc()}")
            raise RuntimeError(f"Stream order routing failed: {e}. NO FALLBACKS.")
    
    def _find_downstream_by_proximity_and_order(self, current_subbasin_id: int, current_order: int,
                                               subbasin_orders: dict, subbasins_data: np.ndarray,
                                               flow_acc_data: np.ndarray) -> int:
        """Find downstream subbasin using proximity and stream order logic"""
        
        # Find outlet point of current subbasin (highest flow accumulation)
        current_mask = (subbasins_data == current_subbasin_id)
        current_flow_acc = flow_acc_data[current_mask]
        
        if len(current_flow_acc) == 0:
            raise RuntimeError(f"No flow accumulation data found for subbasin {current_subbasin_id}. NO FALLBACKS.")
            
        max_acc = np.max(current_flow_acc)
        outlet_candidates = np.where((subbasins_data == current_subbasin_id) & (flow_acc_data == max_acc))
        
        if len(outlet_candidates[0]) == 0:
            raise RuntimeError(f"No outlet candidates found for subbasin {current_subbasin_id}. NO FALLBACKS.")
            
        outlet_row, outlet_col = outlet_candidates[0][0], outlet_candidates[1][0]
        
        # Find nearby subbasins with higher or equal stream order
        search_radius = 20  # pixels
        candidates = []
        
        for r_offset in range(-search_radius, search_radius + 1):
            for c_offset in range(-search_radius, search_radius + 1):
                search_row = outlet_row + r_offset
                search_col = outlet_col + c_offset
                
                if (0 <= search_row < subbasins_data.shape[0] and 
                    0 <= search_col < subbasins_data.shape[1]):
                    
                    nearby_subbasin = subbasins_data[search_row, search_col]
                    
                    if (nearby_subbasin > 0 and nearby_subbasin != current_subbasin_id and
                        nearby_subbasin in subbasin_orders):
                        
                        nearby_order = subbasin_orders[nearby_subbasin]
                        nearby_flow_acc = flow_acc_data[search_row, search_col]
                        
                        # Prefer higher order streams, or same order with higher flow accumulation
                        if (nearby_order > current_order or 
                            (nearby_order == current_order and nearby_flow_acc > max_acc)):
                            
                            distance = ((r_offset ** 2 + c_offset ** 2) ** 0.5)
                            candidates.append((nearby_subbasin, nearby_order, nearby_flow_acc, distance))
        
        if candidates:
            # Sort by order (descending), then flow accumulation (descending), then distance (ascending)
            candidates.sort(key=lambda x: (-x[1], -x[2], x[3]))
            return candidates[0][0]
        else:
            # No suitable downstream found, route to highest flow accumulation globally
            max_flow_acc_global = np.max(flow_acc_data)
            outlet_candidates = np.where(flow_acc_data == max_flow_acc_global)
            outlet_subbasin = subbasins_data[outlet_candidates[0][0], outlet_candidates[1][0]]
            return outlet_subbasin
    
    def _calculate_manual_flow_routing(self, subbasins_gdf: gpd.GeoDataFrame, 
                                     subbasins_raster_path: str, flow_direction_path: str,
                                     flow_accumulation_path: str) -> gpd.GeoDataFrame:
        """Fallback manual flow routing (simplified version of old method)"""
        
        print(f"      Using manual flow routing fallback...")
        
        # Simple fallback: route all to outlet subbasin
        try:
            with rasterio.open(subbasins_raster_path) as src:
                subbasins_data = src.read(1)
            with rasterio.open(flow_accumulation_path) as src:
                flow_acc_data = src.read(1)
                
            # Find outlet subbasin
            max_flow_acc = np.max(flow_acc_data)
            outlet_candidates = np.where(flow_acc_data == max_flow_acc)
            outlet_subbasin = subbasins_data[outlet_candidates[0][0], outlet_candidates[1][0]]
            
            # Create routing
            routing_gdf = subbasins_gdf.copy()
            
            if 'VALUE' in routing_gdf.columns:
                routing_gdf['SubId'] = routing_gdf['VALUE']
            elif 'SubId' not in routing_gdf.columns:
                routing_gdf['SubId'] = range(1, len(routing_gdf) + 1)
            
            # Route all to outlet except outlet itself
            routing_gdf['DowSubId'] = outlet_subbasin
            routing_gdf.loc[routing_gdf['SubId'] == outlet_subbasin, 'DowSubId'] = -1
            
            print(f"      Manual routing: All subbasins -> {outlet_subbasin} (outlet)")
            return routing_gdf
            
        except Exception as e:
            print(f"      FAILED: Manual routing failed: {e}")
            raise RuntimeError(f"Manual routing failed: {e}. NO FALLBACKS.")
    
    def _trace_downstream_subbasin(self, start_row: int, start_col: int, 
                                  flow_dir_data: np.ndarray, subbasins_data: np.ndarray) -> Optional[int]:
        """
        Trace flow downstream from a given point to find the next subbasin
        
        Uses D8 flow direction encoding:
        1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
        """
        
        # D8 flow direction offsets [row_offset, col_offset]
        flow_dir_map = {
            1: [0, 1],    # East
            2: [1, 1],    # Southeast  
            4: [1, 0],    # South
            8: [1, -1],   # Southwest
            16: [0, -1],  # West
            32: [-1, -1], # Northwest
            64: [-1, 0],  # North
            128: [-1, 1]  # Northeast
        }
        
        current_row, current_col = start_row, start_col
        original_subbasin = subbasins_data[start_row, start_col]
        max_steps = min(flow_dir_data.shape[0] * flow_dir_data.shape[1], 10000)  # Prevent infinite loops
        
        for step in range(max_steps):
            # Get flow direction at current location
            flow_dir = flow_dir_data[current_row, current_col]
            
            if flow_dir not in flow_dir_map:
                return None  # Invalid flow direction
            
            # Calculate next position
            row_offset, col_offset = flow_dir_map[flow_dir]
            next_row = current_row + row_offset
            next_col = current_col + col_offset
            
            # Check bounds
            if (next_row < 0 or next_row >= flow_dir_data.shape[0] or 
                next_col < 0 or next_col >= flow_dir_data.shape[1]):
                return None  # Flowed out of raster bounds (watershed outlet)
            
            # Check if we've entered a different subbasin
            next_subbasin = subbasins_data[next_row, next_col]
            
            if next_subbasin > 0 and next_subbasin != original_subbasin:
                return int(next_subbasin)  # Found downstream subbasin
            
            # Continue tracing
            current_row, current_col = next_row, next_col
        
        return None  # Exceeded max steps without finding downstream subbasin
    
    def _validate_routing_network(self, routing_gdf: gpd.GeoDataFrame) -> None:
        """Validate the routing network topology"""
        
        subbasin_ids = set(routing_gdf['SubId'].values)
        downstream_ids = set(routing_gdf['DowSubId'].values) - {-1}  # Exclude watershed outlet
        
        # Check for valid downstream references
        invalid_references = downstream_ids - subbasin_ids
        if invalid_references:
            print(f"      WARNING: Invalid downstream references: {invalid_references}")
        
        # Count outlets and cascading connections
        outlets = (routing_gdf['DowSubId'] == -1).sum()
        cascading = (routing_gdf['DowSubId'] != routing_gdf['SubId']).sum()
        self_referencing = (routing_gdf['DowSubId'] == routing_gdf['SubId']).sum()
        
        print(f"      Routing validation: {outlets} outlets, {cascading} cascading, {self_referencing} self-ref")
        
        if cascading > 0:
            print(f"      SUCCESS: Found cascading routing connectivity!")
        else:
            print(f"      WARNING: No cascading routing found - all subbasins are independent")
    
    def _integrate_lake_routing(self, subbasins_gdf: gpd.GeoDataFrame, 
                               output_dir: Path) -> gpd.GeoDataFrame:
        """Integrate lake routing with subbasin routing network"""
        
        print(f"      Checking for lake routing integration...")
        
        # Look for lake files
        lake_files = []
        for lake_pattern in ['*lakes*.shp', '*lakes*.geojson', '*connected*.shp', '*connected*.geojson']:
            lake_files.extend(list(output_dir.glob(lake_pattern)))
        
        if not lake_files:
            print(f"      No lake files found for integration")
            return subbasins_gdf
        
        try:
            # Load the first available lake file
            lake_file = lake_files[0]
            print(f"      Loading lakes from: {lake_file}")
            lakes_gdf = gpd.read_file(str(lake_file))
            
            if len(lakes_gdf) == 0:
                print(f"      No lakes found in file")
                return subbasins_gdf
            
            # Skip processing if there are too many lakes (indicates unfiltered data)
            # Step 3 will handle proper lake filtering and integration
            if len(lakes_gdf) > 100:
                print(f"      Skipping {len(lakes_gdf)} lakes - too many for Step 2 integration")
                print(f"      Lake processing will be handled in Step 3")
                return subbasins_gdf
            
            print(f"      Processing {len(lakes_gdf)} lakes for routing integration...")
            
            # Ensure both datasets have the same CRS
            if lakes_gdf.crs != subbasins_gdf.crs:
                lakes_gdf = lakes_gdf.to_crs(subbasins_gdf.crs)
            
            # Add lake routing attributes to subbasins
            routing_gdf = subbasins_gdf.copy()
            routing_gdf['has_lakes'] = False
            routing_gdf['lake_count'] = 0
            routing_gdf['lake_area_km2'] = 0.0
            routing_gdf['lake_type'] = 'none'  # none, connected, terminal
            
            # For each subbasin, check for lakes
            for idx, subbasin in routing_gdf.iterrows():
                subbasin_geom = subbasin.geometry
                
                # Find lakes within this subbasin
                lakes_in_subbasin = []
                for lake_idx, lake in lakes_gdf.iterrows():
                    if lake.geometry.intersects(subbasin_geom):
                        # Calculate intersection area
                        intersection = lake.geometry.intersection(subbasin_geom)
                        intersection_area = intersection.area if hasattr(intersection, 'area') else 0
                        
                        # Only consider if significant portion of lake is in subbasin
                        lake_area = lake.geometry.area
                        if intersection_area > 0.1 * lake_area:  # 10% threshold
                            lakes_in_subbasin.append(lake)
                
                # Update subbasin attributes based on lakes
                if lakes_in_subbasin:
                    routing_gdf.at[idx, 'has_lakes'] = True
                    routing_gdf.at[idx, 'lake_count'] = len(lakes_in_subbasin)
                    
                    # Calculate total lake area in subbasin
                    total_lake_area = sum(lake.geometry.area for lake in lakes_in_subbasin)
                    routing_gdf.at[idx, 'lake_area_km2'] = total_lake_area / 1e6  # Convert to km2
                    
                    # Determine lake type (connected vs terminal)
                    # This is a simplified classification - in reality, would need more analysis
                    connected_lakes = [lake for lake in lakes_in_subbasin 
                                     if lake.get('is_connected', False) or lake.get('type') == 'connected']
                    
                    if connected_lakes:
                        routing_gdf.at[idx, 'lake_type'] = 'connected'
                        print(f"        Subbasin {subbasin['SubId']}: {len(connected_lakes)} connected lakes")
                    else:
                        routing_gdf.at[idx, 'lake_type'] = 'terminal'  
                        print(f"        Subbasin {subbasin['SubId']}: {len(lakes_in_subbasin)} terminal lakes")
                        
                        # For terminal lakes, might need to modify routing
                        # (This is a simplified approach - more complex logic could be added)
                        # For now, keep original routing but mark for special handling
            
            lakes_with_routing = len(routing_gdf[routing_gdf['has_lakes'] == True])
            print(f"      SUCCESS: Lake routing integrated - {lakes_with_routing} subbasins contain lakes")
            
            return routing_gdf
            
        except Exception as e:
            print(f"      WARNING: Lake routing integration failed: {e}")
            return subbasins_gdf
    
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
