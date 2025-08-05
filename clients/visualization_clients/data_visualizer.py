#!/usr/bin/env python3
"""
Data Visualization Client for RAVEN
Provides comprehensive visualization capabilities for climate and hydrometric data
"""

import json
from pathlib import Path
from typing import Dict, Union

# Core imports
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Optional visualization enhancements
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Optional mapping capabilities
try:
    import geopandas as gpd
    import folium
    HAS_MAPPING = True
except ImportError:
    HAS_MAPPING = False


class DataVisualizationClient:
    """Client for visualizing downloaded climate and hydrometric data"""
    
    def __init__(self):
        self.output_dir = None
    
    def visualize_climate_data(self, csv_path: Path, output_dir: Path, 
                              station_name: str = "Climate Station") -> Dict:
        """Create comprehensive visualization of climate data"""
        if not HAS_PANDAS:
            return {'success': False, 'error': 'Pandas not available for visualization'}
        
        try:
            # Read data
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle(f'Climate Data Analysis - {station_name}', fontsize=16, fontweight='bold')
            
            # 1. Temperature time series
            axes[0, 0].plot(df.index, df['Tmax'], label='Max Temp', color='red', alpha=0.7)
            axes[0, 0].plot(df.index, df['Tmin'], label='Min Temp', color='blue', alpha=0.7)
            axes[0, 0].plot(df.index, df['Tmean'], label='Mean Temp', color='green', alpha=0.8)
            axes[0, 0].fill_between(df.index, df['Tmin'], df['Tmax'], alpha=0.2, color='gray')
            axes[0, 0].set_title('Temperature Time Series')
            axes[0, 0].set_ylabel('Temperature (°C)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Precipitation time series
            axes[0, 1].bar(df.index, df['Precip'], alpha=0.7, color='skyblue', width=0.8)
            axes[0, 1].set_title('Daily Precipitation')
            axes[0, 1].set_ylabel('Precipitation (mm)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Temperature distribution
            if HAS_SEABORN:
                temp_data = pd.melt(df[['Tmax', 'Tmin', 'Tmean']], 
                                  var_name='Temperature Type', value_name='Temperature')
                sns.boxplot(data=temp_data, x='Temperature Type', y='Temperature', ax=axes[1, 0])
            else:
                axes[1, 0].boxplot([df['Tmax'].dropna(), df['Tmin'].dropna(), df['Tmean'].dropna()],
                                 labels=['Tmax', 'Tmin', 'Tmean'])
            axes[1, 0].set_title('Temperature Distribution')
            axes[1, 0].set_ylabel('Temperature (°C)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Precipitation distribution
            axes[1, 1].hist(df['Precip'][df['Precip'] > 0], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
            axes[1, 1].set_title('Precipitation Distribution (Wet Days Only)')
            axes[1, 1].set_xlabel('Precipitation (mm)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 5. Monthly statistics
            monthly_stats = df.groupby(df.index.month).agg({
                'Tmean': 'mean',
                'Precip': 'sum'
            })
            
            ax5 = axes[2, 0]
            ax5_twin = ax5.twinx()
            
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            bars = ax5.bar(range(1, 13), monthly_stats['Precip'], alpha=0.6, color='lightblue', label='Precipitation')
            line = ax5_twin.plot(range(1, 13), monthly_stats['Tmean'], 'ro-', color='red', label='Temperature')
            
            ax5.set_xlabel('Month')
            ax5.set_ylabel('Precipitation (mm)', color='blue')
            ax5_twin.set_ylabel('Temperature (°C)', color='red')
            ax5.set_xticks(range(1, 13))
            ax5.set_xticklabels(months)
            ax5.set_title('Monthly Climate Summary')
            ax5.grid(True, alpha=0.3)
            
            # 6. Data quality assessment
            quality_data = {
                'Tmax': (1 - df['Tmax'].isnull().mean()) * 100,
                'Tmin': (1 - df['Tmin'].isnull().mean()) * 100,
                'Tmean': (1 - df['Tmean'].isnull().mean()) * 100,
                'Precip': (1 - df['Precip'].isnull().mean()) * 100
            }
            
            vars_list = list(quality_data.keys())
            completeness = list(quality_data.values())
            
            bars = axes[2, 1].bar(vars_list, completeness, color=['red', 'blue', 'green', 'skyblue'], alpha=0.7)
            axes[2, 1].set_title('Data Completeness')
            axes[2, 1].set_ylabel('Completeness (%)')
            axes[2, 1].set_ylim(0, 100)
            axes[2, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, completeness):
                height = bar.get_height()
                axes[2, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{value:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = output_dir / f"{station_name.replace(' ', '_')}_climate_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate statistics
            stats = {
                'temperature_stats': {
                    'mean_tmax': df['Tmax'].mean(),
                    'mean_tmin': df['Tmin'].mean(),
                    'mean_temp': df['Tmean'].mean(),
                    'temp_range': df['Tmax'].max() - df['Tmin'].min()
                },
                'precipitation_stats': {
                    'total_precip': df['Precip'].sum(),
                    'mean_daily_precip': df['Precip'].mean(),
                    'wet_days': (df['Precip'] > 0).sum(),
                    'dry_days': (df['Precip'] == 0).sum()
                },
                'data_quality': quality_data,
                'record_count': len(df),
                'date_range': [df.index.min().strftime('%Y-%m-%d'), df.index.max().strftime('%Y-%m-%d')]
            }
            
            # Save statistics
            stats_file = output_dir / f"{station_name.replace(' ', '_')}_climate_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            print(f"SUCCESS: Climate visualization and statistics created")
            
            return {
                'success': True,
                'plot_file': str(plot_file),
                'stats_file': str(stats_file),
                'statistics': stats
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Climate visualization failed: {str(e)}"}
    
    def visualize_streamflow_data(self, csv_path: Path, output_dir: Path,
                                 station_name: str = "Hydrometric Station") -> Dict:
        """Create comprehensive visualization of streamflow data"""
        if not HAS_PANDAS:
            return {'success': False, 'error': 'Pandas not available for visualization'}
        
        try:
            # Read data
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Streamflow Data Analysis - {station_name}', fontsize=16, fontweight='bold')
            
            # 1. Discharge time series
            discharge_data = df['Discharge_cms'].dropna()
            if len(discharge_data) > 0:
                axes[0, 0].plot(discharge_data.index, discharge_data.values, color='blue', alpha=0.7)
                axes[0, 0].set_title('Discharge Time Series')
                axes[0, 0].set_ylabel('Discharge (m³/s)')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Add flow statistics as text
                mean_flow = discharge_data.mean()
                max_flow = discharge_data.max()
                min_flow = discharge_data.min()
                
                axes[0, 0].axhline(y=mean_flow, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_flow:.1f} m³/s')
                axes[0, 0].legend()
            else:
                axes[0, 0].text(0.5, 0.5, 'No discharge data available', 
                              ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Discharge Time Series - No Data')
            
            # 2. Water level time series
            level_data = df['WaterLevel_m'].dropna()
            if len(level_data) > 0:
                axes[0, 1].plot(level_data.index, level_data.values, color='green', alpha=0.7)
                axes[0, 1].set_title('Water Level Time Series')
                axes[0, 1].set_ylabel('Water Level (m)')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'No water level data available',
                              ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Water Level Time Series - No Data')
            
            # 3. Flow duration curve
            if len(discharge_data) > 0:
                sorted_flows = discharge_data.sort_values(ascending=False)
                exceedance = (range(1, len(sorted_flows) + 1)) / len(sorted_flows) * 100
                
                axes[1, 0].semilogy(exceedance, sorted_flows, color='blue', linewidth=2)
                axes[1, 0].set_title('Flow Duration Curve')
                axes[1, 0].set_xlabel('Exceedance Probability (%)')
                axes[1, 0].set_ylabel('Discharge (m³/s)')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add percentile markers
                percentiles = [10, 50, 90]
                for p in percentiles:
                    flow_value = discharge_data.quantile(1 - p/100)
                    axes[1, 0].axhline(y=flow_value, color='red', linestyle=':', alpha=0.7)
                    axes[1, 0].text(5, flow_value, f'Q{p}: {flow_value:.1f}', 
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            else:
                axes[1, 0].text(0.5, 0.5, 'No discharge data for flow duration curve',
                              ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Flow Duration Curve - No Data')
            
            # 4. Data availability and quality
            data_availability = {
                'Discharge': (1 - df['Discharge_cms'].isnull().mean()) * 100,
                'Water Level': (1 - df['WaterLevel_m'].isnull().mean()) * 100
            }
            
            vars_list = list(data_availability.keys())
            completeness = list(data_availability.values())
            
            bars = axes[1, 1].bar(vars_list, completeness, color=['blue', 'green'], alpha=0.7)
            axes[1, 1].set_title('Data Completeness')
            axes[1, 1].set_ylabel('Completeness (%)')
            axes[1, 1].set_ylim(0, 100)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, completeness):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{value:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = output_dir / f"{station_name.replace(' ', '_')}_streamflow_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate statistics
            stats = {
                'discharge_stats': {
                    'mean_discharge': discharge_data.mean() if len(discharge_data) > 0 else None,
                    'max_discharge': discharge_data.max() if len(discharge_data) > 0 else None,
                    'min_discharge': discharge_data.min() if len(discharge_data) > 0 else None,
                    'std_discharge': discharge_data.std() if len(discharge_data) > 0 else None,
                    'discharge_records': len(discharge_data)
                },
                'water_level_stats': {
                    'mean_level': level_data.mean() if len(level_data) > 0 else None,
                    'max_level': level_data.max() if len(level_data) > 0 else None,
                    'min_level': level_data.min() if len(level_data) > 0 else None,
                    'level_records': len(level_data)
                },
                'flow_percentiles': {
                    'Q10': discharge_data.quantile(0.9) if len(discharge_data) > 0 else None,
                    'Q50': discharge_data.quantile(0.5) if len(discharge_data) > 0 else None,
                    'Q90': discharge_data.quantile(0.1) if len(discharge_data) > 0 else None
                },
                'data_quality': data_availability,
                'record_count': len(df),
                'date_range': [df.index.min().strftime('%Y-%m-%d'), df.index.max().strftime('%Y-%m-%d')]
            }
            
            # Save statistics
            stats_file = output_dir / f"{station_name.replace(' ', '_')}_streamflow_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            print(f"SUCCESS: Streamflow visualization and statistics created")
            
            return {
                'success': True,
                'plot_file': str(plot_file),
                'stats_file': str(stats_file),
                'statistics': stats
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Streamflow visualization failed: {str(e)}"}
    
    def create_watershed_map(self, watershed_geojson: Path, stations_geojson: Path,
                           output_path: Path, map_title: str = "Watershed Analysis") -> Dict:
        """Create interactive map of watershed and stations"""
        if not HAS_MAPPING:
            return {'success': False, 'error': 'Mapping libraries not available'}
        
        try:
            # Read watershed data
            watershed_gdf = gpd.read_file(watershed_geojson)
            
            # Get watershed center for map
            bounds = watershed_gdf.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10,
                tiles='OpenStreetMap'
            )
            
            # Add watershed boundary
            folium.GeoJson(
                watershed_geojson,
                style_function=lambda feature: {
                    'fillColor': 'lightblue',
                    'color': 'blue',
                    'weight': 2,
                    'fillOpacity': 0.3
                },
                popup=folium.Popup('Watershed Boundary')
            ).add_to(m)
            
            # Add stations if available
            if stations_geojson.exists():
                stations_gdf = gpd.read_file(stations_geojson)
                
                for idx, station in stations_gdf.iterrows():
                    if station.geometry.geom_type == 'Point':
                        coords = [station.geometry.y, station.geometry.x]
                        station_name = station.get('STATION_NAME', 'Unknown Station')
                        station_id = station.get('STATION_NUMBER', 'Unknown ID')
                        
                        folium.Marker(
                            coords,
                            popup=f"<b>{station_name}</b><br>ID: {station_id}",
                            icon=folium.Icon(color='red', icon='tint')
                        ).add_to(m)
            
            # Add title
            title_html = f'''
                <h3 align="center" style="font-size:20px"><b>{map_title}</b></h3>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Save map
            output_path.parent.mkdir(exist_ok=True, parents=True)
            m.save(output_path)
            
            print(f"SUCCESS: Interactive watershed map created")
            
            return {
                'success': True,
                'map_file': str(output_path),
                'center_coordinates': [center_lat, center_lon],
                'watershed_area': watershed_gdf.geometry.area.sum()
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Map creation failed: {str(e)}"}