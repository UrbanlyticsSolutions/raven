"""
Watershed Results Mapper

Creates comprehensive maps of watershed delineation results including:
- Watershed boundary
- Stream network with stream orders
- Lakes (connected and non-connected)
- HRU subbasins
- Outlet point
- DEM hillshade background
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd
import rasterio
from rasterio.plot import show
import numpy as np
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

class WatershedMapper:
    """
    Creates comprehensive maps of watershed delineation results
    """
    
    def __init__(self, workspace_dir: Path = None):
        """
        Initialize watershed mapper
        
        Parameters:
        -----------
        workspace_dir : Path, optional
            Working directory containing results
        """
        self.workspace_dir = workspace_dir or Path.cwd()
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the mapper"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def create_comprehensive_map(self, 
                               watershed_file: str,
                               streams_file: str,
                               outlet_coords: Tuple[float, float],
                               dem_file: str = None,
                               lakes_file: str = None,
                               connected_lakes_file: str = None,
                               non_connected_lakes_file: str = None,
                               subbasins_file: str = None,
                               output_file: str = None,
                               title: str = "Watershed Delineation Results",
                               **kwargs) -> Dict[str, Any]:
        """
        Create comprehensive watershed map
        
        Parameters:
        -----------
        watershed_file : str
            Path to watershed boundary shapefile
        streams_file : str
            Path to streams file (geojson or shapefile)
        outlet_coords : tuple
            (longitude, latitude) of outlet point
        dem_file : str, optional
            Path to DEM file for hillshade background
        lakes_file : str, optional
            Path to all lakes shapefile
        connected_lakes_file : str, optional
            Path to connected lakes shapefile  
        non_connected_lakes_file : str, optional
            Path to non-connected lakes shapefile
        subbasins_file : str, optional
            Path to HRU subbasins shapefile
        output_file : str, optional
            Output PNG file path
        title : str, optional
            Map title
        **kwargs : additional parameters
            
        Returns:
        --------
        Dict[str, Any]
            Mapping results with file paths and statistics
        """
        
        self.logger.info(f"Creating comprehensive watershed map: {title}")
        
        try:
            # Setup output file
            if not output_file:
                output_file = str(self.workspace_dir / f"watershed_map.png")
            
            # Load required data
            watershed_gdf = gpd.read_file(watershed_file)
            
            # Load streams
            if streams_file.endswith('.geojson'):
                streams_gdf = gpd.read_file(streams_file)
            else:
                streams_gdf = gpd.read_file(streams_file)
            
            # Create outlet point
            outlet_point = gpd.GeoDataFrame(
                [{'name': 'Outlet'}], 
                geometry=[Point(outlet_coords[0], outlet_coords[1])],
                crs=watershed_gdf.crs
            )
            
            # Setup figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Add DEM hillshade background if available
            if dem_file and Path(dem_file).exists():
                self._add_hillshade_background(ax, dem_file, watershed_gdf.total_bounds)
            
            # Plot watershed boundary
            watershed_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2, alpha=0.8)
            
            # Plot streams with different colors by stream order if available
            self._plot_streams(ax, streams_gdf)
            
            # Plot lakes if available
            legend_elements = []
            if lakes_file and Path(lakes_file).exists():
                self._plot_lakes(ax, lakes_file, connected_lakes_file, non_connected_lakes_file, legend_elements)
            
            # Plot subbasins if available
            if subbasins_file and Path(subbasins_file).exists():
                self._plot_subbasins(ax, subbasins_file, legend_elements)
            
            # Plot outlet point
            outlet_point.plot(ax=ax, color='red', markersize=100, marker='*', 
                            edgecolor='white', linewidth=2, zorder=10)
            
            # Add legend elements
            legend_elements.extend([
                mpatches.Patch(facecolor='none', edgecolor='red', linewidth=2, label='Watershed Boundary'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                          markersize=15, markeredgecolor='white', markeredgewidth=2, label='Outlet Point'),
                plt.Line2D([0], [0], color='blue', linewidth=2, label='Streams')
            ])
            
            # Customize map
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            
            # Add legend
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add north arrow and scale bar
            self._add_map_elements(ax, watershed_gdf.total_bounds)
            
            # Tight layout
            plt.tight_layout()
            
            # Save map
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            self.logger.info(f"Map saved to: {output_file}")
            
            # Calculate statistics
            watershed_area = watershed_gdf.geometry.area.sum() / 1e6  # Convert to km²
            stream_length = streams_gdf.geometry.length.sum() / 1000  # Convert to km
            
            plt.close()
            
            return {
                'success': True,
                'map_file': output_file,
                'watershed_area_km2': watershed_area,
                'stream_length_km': stream_length,
                'title': title,
                'components_mapped': self._get_components_mapped(
                    dem_file, lakes_file, subbasins_file
                )
            }
            
        except Exception as e:
            error_msg = f"Mapping failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def _add_hillshade_background(self, ax, dem_file: str, bounds: np.ndarray):
        """Add DEM hillshade as background"""
        try:
            with rasterio.open(dem_file) as src:
                # Read DEM data
                dem_data = src.read(1)
                
                # Create hillshade
                from rasterio.plot import show
                show(src, ax=ax, cmap='terrain', alpha=0.3, zorder=0)
                
        except Exception as e:
            self.logger.warning(f"Could not add hillshade background: {e}")
    
    def _plot_streams(self, ax, streams_gdf):
        """Plot stream network with stream order coloring if available"""
        try:
            # Check if stream order is available
            if 'strmOrder' in streams_gdf.columns:
                # Color by stream order
                unique_orders = sorted(streams_gdf['strmOrder'].unique())
                colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(unique_orders)))
                
                for i, order in enumerate(unique_orders):
                    order_streams = streams_gdf[streams_gdf['strmOrder'] == order]
                    order_streams.plot(ax=ax, color=colors[i], linewidth=max(1, order/2), 
                                     alpha=0.8, zorder=5)
            else:
                # Single color for all streams
                streams_gdf.plot(ax=ax, color='blue', linewidth=1, alpha=0.8, zorder=5)
                
        except Exception as e:
            self.logger.warning(f"Error plotting streams: {e}")
            # Fallback to simple plotting
            streams_gdf.plot(ax=ax, color='blue', linewidth=1, alpha=0.8, zorder=5)
    
    def _plot_lakes(self, ax, lakes_file: str, connected_lakes_file: str, 
                   non_connected_lakes_file: str, legend_elements: List):
        """Plot lakes with different colors for connected vs non-connected"""
        try:
            # Plot connected lakes
            if connected_lakes_file and Path(connected_lakes_file).exists():
                connected_lakes = gpd.read_file(connected_lakes_file)
                if not connected_lakes.empty:
                    connected_lakes.plot(ax=ax, color='lightblue', edgecolor='blue', 
                                       linewidth=1, alpha=0.7, zorder=6)
                    legend_elements.append(
                        mpatches.Patch(color='lightblue', label=f'Connected Lakes ({len(connected_lakes)})')
                    )
            
            # Plot non-connected lakes
            if non_connected_lakes_file and Path(non_connected_lakes_file).exists():
                non_connected_lakes = gpd.read_file(non_connected_lakes_file)
                if not non_connected_lakes.empty:
                    non_connected_lakes.plot(ax=ax, color='lightcyan', edgecolor='cyan', 
                                           linewidth=1, alpha=0.7, zorder=6)
                    legend_elements.append(
                        mpatches.Patch(color='lightcyan', label=f'Non-connected Lakes ({len(non_connected_lakes)})')
                    )
            
        except Exception as e:
            self.logger.warning(f"Error plotting lakes: {e}")
    
    def _plot_subbasins(self, ax, subbasins_file: str, legend_elements: List):
        """Plot HRU subbasins with transparent colors"""
        try:
            subbasins_gdf = gpd.read_file(subbasins_file)
            if not subbasins_gdf.empty:
                # Create colormap for subbasins
                n_subbasins = len(subbasins_gdf)
                colors = plt.cm.Set3(np.linspace(0, 1, n_subbasins))
                
                subbasins_gdf.plot(ax=ax, color=colors, alpha=0.3, edgecolor='gray', 
                                 linewidth=0.5, zorder=3)
                
                legend_elements.append(
                    mpatches.Patch(color='lightgray', alpha=0.3, 
                                 label=f'HRU Subbasins ({n_subbasins})')
                )
                
        except Exception as e:
            self.logger.warning(f"Error plotting subbasins: {e}")
    
    def _add_map_elements(self, ax, bounds: np.ndarray):
        """Add north arrow and basic map elements"""
        try:
            # Add north arrow (simple)
            ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
                       ha='center', va='center', fontsize=16, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Add coordinate info
            lon_range = bounds[2] - bounds[0]
            lat_range = bounds[3] - bounds[1]
            
            coord_text = f"Extent: {lon_range:.3f}° × {lat_range:.3f}°"
            ax.text(0.02, 0.02, coord_text, transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
        except Exception as e:
            self.logger.warning(f"Error adding map elements: {e}")
    
    def _get_components_mapped(self, dem_file: str, lakes_file: str, subbasins_file: str) -> List[str]:
        """Get list of components that were successfully mapped"""
        components = ['watershed_boundary', 'streams', 'outlet']
        
        if dem_file and Path(dem_file).exists():
            components.append('hillshade')
        if lakes_file and Path(lakes_file).exists():
            components.append('lakes')
        if subbasins_file and Path(subbasins_file).exists():
            components.append('subbasins')
            
        return components

    def create_summary_plot(self, results_dict: Dict[str, Any], 
                          output_file: str = None) -> Dict[str, Any]:
        """
        Create summary statistics plot
        
        Parameters:
        -----------
        results_dict : dict
            Watershed delineation results
        output_file : str, optional
            Output file path
            
        Returns:
        --------
        Dict with plot results
        """
        try:
            if not output_file:
                output_file = str(self.workspace_dir / "watershed_summary.png")
            
            # Extract statistics
            stats = {
                'Watershed Area (km²)': results_dict.get('watershed_area_km2', 0),
                'Stream Length (km)': results_dict.get('stream_length_km', 0),
                'Connected Lakes': results_dict.get('connected_lake_count', 0),
                'Non-connected Lakes': results_dict.get('non_connected_lake_count', 0),
                'Lake Area (km²)': results_dict.get('total_lake_area_km2', 0),
                'Max Stream Order': results_dict.get('max_stream_order', 0)
            }
            
            # Create bar plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            names = list(stats.keys())
            values = list(stats.values())
            colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
            
            bars = ax.bar(names, values, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('Watershed Delineation Statistics', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('Value', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"Summary plot saved to: {output_file}")
            
            return {
                'success': True,
                'summary_plot': output_file,
                'statistics': stats
            }
            
        except Exception as e:
            error_msg = f"Summary plot creation failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }