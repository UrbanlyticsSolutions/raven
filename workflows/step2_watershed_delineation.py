#!/usr/bin/env python3
"""
Step 2: Watershed Delineation for RAVEN Single Outlet Delineation
Delineates watershed boundary and extracts stream network from DEM
"""

import sys
from pathlib import Path
import argparse
import json
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from workflows.steps import DelineateWatershedAndStreams
from processors.outlet_snapping import ImprovedOutletSnapper


class Step2WatershedDelineation:
    """Step 2: Delineate watershed boundary and stream network"""
    
    def __init__(self, workspace_dir: str = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "data"
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize watershed delineation step and outlet snapper
        self.watershed_step = DelineateWatershedAndStreams()
        self.outlet_snapper = ImprovedOutletSnapper(self.workspace_dir / "outlet_snapping")
    
    def load_step1_results(self) -> Dict[str, Any]:
        """Load results from Step 1"""
        results_file = self.workspace_dir / "step1_results.json"
        
        if not results_file.exists():
            return {
                'success': False,
                'error': 'Step 1 results not found. Run step1_data_preparation.py first.'
            }
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def execute(self, latitude: float, longitude: float, outlet_name: str = None) -> Dict[str, Any]:
        """Execute Step 2: Watershed delineation"""
        print(f"STEP 2: Delineating watershed for outlet ({latitude}, {longitude})")
        
        # Load Step 1 results
        step1_results = self.load_step1_results()
        if not step1_results.get('success'):
            return step1_results
        
        dem_file = step1_results['files']['dem']
        print(f"Using DEM: {dem_file}")
        
        if not Path(dem_file).exists():
            return {
                'success': False,
                'error': f'DEM file not found: {dem_file}'
            }
        
        # Create outlet-specific workspace
        if not outlet_name:
            outlet_name = f"outlet_{latitude:.4f}_{longitude:.4f}"
        
        outlet_workspace = self.workspace_dir / outlet_name
        outlet_workspace.mkdir(exist_ok=True)
        
        print(f"Workspace: {outlet_workspace}")
        
        # Step 1: Check for existing flow files or create them
        print("Looking for existing flow accumulation and streams data...")
        flow_accum_file = None
        streams_file = None
        
        # Check for existing files in the outlet workspace
        for f in outlet_workspace.rglob("*"):
            if 'flow_accumulation' in f.name and f.suffix == '.tif':
                flow_accum_file = str(f)
            elif 'streams' in f.name and f.suffix in ['.geojson', '.shp']:
                streams_file = str(f)
        
        # If files don't exist, create them
        if not flow_accum_file or not streams_file:
            print("Creating flow accumulation data...")
            flow_accum_result = self.watershed_step.analyzer.analyze_watershed_complete(
                dem_path=dem_file,
                outlet_coords=(latitude, longitude),
                output_dir=outlet_workspace
            )
            
            if not flow_accum_result.get('success'):
                return {
                    'success': False,
                    'error': f'Flow accumulation preparation failed: {flow_accum_result.get("error")}'
                }
            
            # Find the newly created files
            for f in outlet_workspace.rglob("*"):
                if 'flow_accumulation' in f.name and f.suffix == '.tif':
                    flow_accum_file = str(f)
                elif 'streams' in f.name and f.suffix in ['.geojson', '.shp']:
                    streams_file = str(f)
        
        # Step 2: Snap outlet to stream network using local processor
        if not flow_accum_file or not streams_file:
            return {
                'success': False,
                'error': 'Could not find flow accumulation or streams files for outlet snapping'
            }
            
        print(f"Snapping outlet to stream network using local processor...")
        snap_result = self.outlet_snapper.snap_outlet_downstream(
            (longitude, latitude), 
            streams_file, 
            flow_accum_file,
            max_search_distance=1000  # 1000m search radius
        )
        
        if not snap_result['success']:
            return {
                'success': False,
                'error': f'Outlet snapping failed: {snap_result["error"]}'
            }
        
        print(f"✅ Outlet snapped: {snap_result['snap_distance_m']:.1f}m from original")
        # Use snapped coordinates for final delineation
        final_lat, final_lon = snap_result['snapped_coords'][1], snap_result['snapped_coords'][0]
        
        # Step 3: Execute final watershed delineation with snapped coordinates
        print(f"Delineating watershed boundary and streams at ({final_lat:.6f}, {final_lon:.6f})...")
        watershed_result = self.watershed_step.analyzer.analyze_watershed_complete(
            dem_path=dem_file,
            outlet_coords=(final_lon, final_lat),
            output_dir=outlet_workspace
        )
        
        if not watershed_result.get('success'):
            return {
                'success': False,
                'error': f"Final watershed delineation failed: {watershed_result.get('error')}"
            }

        # Extract file paths from the comprehensive results
        final_watershed_boundary = None
        final_stream_network = None
        for f in watershed_result.get('files_created', []):
            if 'watershed' in Path(f).name and Path(f).suffix in ['.geojson', '.shp']:
                final_watershed_boundary = f
            if 'streams' in Path(f).name and Path(f).suffix in ['.geojson', '.shp']:
                final_stream_network = f

        # Extract statistics
        stats = watershed_result.get('metadata', {}).get('statistics', {})
        
        # Prepare results with snapping information
        results = {
            'success': True,
            'outlet_coordinates': [latitude, longitude],  # Original coordinates
            'outlet_coordinates_snapped': [final_lat, final_lon],  # Snapped coordinates
            'outlet_snapped': final_lat != latitude or final_lon != longitude,
            'snap_distance_m': snap_result['snap_distance_m'],
            'outlet_name': outlet_name,
            'workspace': str(outlet_workspace),
            'files': {
                'watershed_boundary': final_watershed_boundary,
                'stream_network': final_stream_network,
                'dem_processed': watershed_result.get('files_created', [None])[0],  # Example processed DEM
                'snapped_outlet_file': snap_result['snapped_outlet_file']
            },
            'characteristics': {
                'watershed_area_km2': stats.get('watershed_area_km2', 0),
                'stream_length_km': stats.get('total_stream_length_km', 0),
                'max_stream_order': stats.get('max_strahler_order', 0) # Assuming this is available
            }
        }
        
        # Verify output files exist
        for file_type, file_path in results['files'].items():
            if file_path and Path(file_path).exists():
                print(f"✅ {file_type}: {file_path}")
            else:
                print(f"⚠️  {file_type}: Missing or not created")
        
        # Save results to JSON
        results_file = self.workspace_dir / "step2_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"STEP 2 COMPLETE: Results saved to {results_file}")
        print(f"Watershed area: {results['characteristics']['watershed_area_km2']:.2f} km²")
        print(f"✅ Outlet was snapped {results['snap_distance_m']:.1f}m to stream network")
        
        # Generate visualization
        self.create_watershed_plot(results, latitude, longitude)
        
        return results
    
    def create_watershed_plot(self, results: Dict, outlet_lat: float, outlet_lon: float):
        """Create a plot showing watershed boundary, streams, and outlet point"""
        try:
            import matplotlib.pyplot as plt
            import geopandas as gpd
            from matplotlib.patches import Circle
            import numpy as np
            
            print("\nGenerating watershed visualization...")
            
            # Load the watershed boundary and streams
            watershed_shp = results['files'].get('watershed_boundary')
            streams_shp = results['files'].get('stream_network')
            
            if not watershed_shp or not Path(watershed_shp).exists():
                print("⚠️  Cannot create plot: watershed boundary file missing")
                return
            
            # Create figure and axis
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Load and plot watershed boundary
            watershed_gdf = gpd.read_file(watershed_shp)
            watershed_gdf.plot(ax=ax, color='lightblue', edgecolor='darkblue', 
                              linewidth=2, alpha=0.3, label='Watershed Boundary')
            
            # Load and plot streams if available
            if streams_shp and Path(streams_shp).exists():
                streams_gdf = gpd.read_file(streams_shp)
                if len(streams_gdf) > 0:
                    streams_gdf.plot(ax=ax, color='blue', linewidth=1.5, 
                                    alpha=0.7, label='Stream Network')
            
            # Plot outlet points (original and snapped if different)
            original_coords = results['outlet_coordinates']
            snapped_coords = results['outlet_coordinates_snapped']
            
            # Plot original outlet
            ax.scatter(original_coords[1], original_coords[0], color='orange', s=150, 
                      marker='s', edgecolor='darkorange', linewidth=2, 
                      zorder=4, label=f'Original Outlet ({original_coords[0]:.4f}, {original_coords[1]:.4f})')
            
            # Plot snapped outlet if different
            if results['outlet_snapped']:
                ax.scatter(snapped_coords[1], snapped_coords[0], color='red', s=200, 
                          marker='o', edgecolor='darkred', linewidth=2, 
                          zorder=5, label=f'Snapped Outlet ({snapped_coords[0]:.4f}, {snapped_coords[1]:.4f})')
                
                # Draw line between original and snapped
                ax.plot([original_coords[1], snapped_coords[1]], 
                       [original_coords[0], snapped_coords[0]], 
                       'k--', linewidth=2, alpha=0.7, label=f'Snap Distance: {results["snap_distance_m"]:.1f}m')
                
                # Add outlet point annotation
                ax.annotate('SNAPPED OUTLET', xy=(snapped_coords[1], snapped_coords[0]), 
                           xytext=(snapped_coords[1] + 0.005, snapped_coords[0] + 0.005),
                           fontsize=10, fontweight='bold', color='darkred',
                           arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
            else:
                # Add outlet point annotation
                ax.annotate('OUTLET', xy=(original_coords[1], original_coords[0]), 
                           xytext=(original_coords[1] + 0.005, original_coords[0] + 0.005),
                           fontsize=10, fontweight='bold', color='darkorange',
                           arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))
            
            # Set labels and title
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.set_title(f'Step 2: Watershed Delineation Results\n' + 
                        f'Area: {results["characteristics"]["watershed_area_km2"]:.2f} km² | ' +
                        f'Stream Length: {results["characteristics"]["stream_length_km"]:.1f} km | ' +
                        f'Max Stream Order: {results["characteristics"]["max_stream_order"]}',
                        fontsize=14, fontweight='bold')
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add legend
            ax.legend(loc='upper right', fontsize=10)
            
            # Add north arrow
            x_range = ax.get_xlim()
            y_range = ax.get_ylim()
            arrow_x = x_range[0] + (x_range[1] - x_range[0]) * 0.95
            arrow_y = y_range[0] + (y_range[1] - y_range[0]) * 0.95
            arrow_length = (y_range[1] - y_range[0]) * 0.05
            ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - arrow_length),
                       ha='center', va='bottom', fontsize=12, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
            # Add scale bar (approximate)
            bounds = watershed_gdf.total_bounds  # minx, miny, maxx, maxy
            width_deg = bounds[2] - bounds[0]
            # Approximate km per degree at this latitude
            km_per_deg = 111.0 * np.cos(np.radians(outlet_lat))
            width_km = width_deg * km_per_deg
            scalebar_km = 10 if width_km > 50 else 5 if width_km > 20 else 2
            scalebar_deg = scalebar_km / km_per_deg
            
            scalebar_x = bounds[0] + width_deg * 0.1
            scalebar_y = bounds[1] + (bounds[3] - bounds[1]) * 0.1
            
            ax.plot([scalebar_x, scalebar_x + scalebar_deg], [scalebar_y, scalebar_y], 
                   'k-', linewidth=3)
            ax.text(scalebar_x + scalebar_deg/2, scalebar_y - (bounds[3] - bounds[1]) * 0.02,
                   f'{scalebar_km} km', ha='center', va='top', fontsize=10)
            
            # Save the plot
            plot_file = self.workspace_dir / "step2_watershed_plot.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"✅ Watershed plot saved: {plot_file}")
            
            # Also display if in interactive environment
            plot_file_display = self.workspace_dir / "step2_plot_display.png"
            plt.savefig(plot_file_display, dpi=72)
            plt.show()
            plt.close()
            
        except ImportError as e:
            print(f"⚠️  Cannot create plot: Missing required library ({e})")
        except Exception as e:
            print(f"⚠️  Error creating watershed plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 2: Watershed Delineation')
    parser.add_argument('latitude', type=float, help='Outlet latitude')
    parser.add_argument('longitude', type=float, help='Outlet longitude')
    parser.add_argument('--outlet-name', type=str, help='Name for the outlet')
    parser.add_argument('--workspace-dir', type=str, help='Workspace directory')
    
    args = parser.parse_args()
    
    step2 = Step2WatershedDelineation(workspace_dir=args.workspace_dir)
    results = step2.execute(args.latitude, args.longitude, args.outlet_name)
    
    if results.get('success', False):
        print("SUCCESS: Step 2 watershed delineation completed")
        print(f"Workspace: {results['workspace']}")
        print(f"Area: {results['characteristics']['watershed_area_km2']:.2f} km²")
    else:
        print(f"FAILED: {results.get('error', 'Unknown error')}")
        sys.exit(1)