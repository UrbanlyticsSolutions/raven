#!/usr/bin/env python3
"""
Plotting Client for RAVEN Data Visualization

This client provides methods to generate insightful plots from the various
data files downloaded during the RAVEN workflow. It saves the plots to a
'plots' subdirectory within the specified workspace.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class PlottingClient:
    """
    A client for generating and saving plots of hydrological and climate data.
    """

    def __init__(self, workspace_dir: Path, station_id: str):
        self.workspace = Path(workspace_dir)
        self.station_id = station_id
        self.plot_dir = self.workspace / "plots"
        self.plot_dir.mkdir(exist_ok=True)
        
        # Set a consistent and professional plot style
        sns.set_theme(style="whitegrid")
        print(f"--- Initialized PlottingClient for Station: {self.station_id} ---")
        print(f"--- Plots will be saved to: {self.plot_dir.resolve()} ---")

    def plot_watershed_overview(self, watershed_path: Path):
        """
        Plots the watershed boundary.
        """
        if not watershed_path.exists():
            print(f"⚠️ Cannot plot watershed: File not found at {watershed_path}")
            return

        print("   - Generating watershed overview plot...")
        try:
            gdf = gpd.read_file(watershed_path)
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf.plot(ax=ax, edgecolor='black', facecolor='lightblue', linewidth=2)
            
            # Add title and labels
            ax.set_title(f"Watershed Boundary for Station {self.station_id}", fontsize=16)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True)
            
            # Save the plot
            save_path = self.plot_dir / "watershed_overview.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"   - ✅ Saved watershed plot to: {save_path}")

        except Exception as e:
            print(f"   - ❌ Error plotting watershed: {e}")

    def plot_hydrometric_data(self, hydro_path: Path):
        """
        Generates a time series plot of daily streamflow.
        """
        if not hydro_path.exists():
            print(f"⚠️ Cannot plot hydrometric data: File not found at {hydro_path}")
            return

        print("   - Generating hydrometric time series plot...")
        try:
            df = pd.read_csv(hydro_path, parse_dates=['Date'])
            df.set_index('Date', inplace=True)

            # Check for the correct column name
            flow_col = None
            if 'Value' in df.columns:
                flow_col = 'Value'
            elif 'Flow' in df.columns:
                flow_col = 'Flow'
            
            if flow_col is None:
                raise ValueError("Could not find 'Value' or 'Flow' column in hydrometric data.")

            fig, ax = plt.subplots(figsize=(15, 7))
            df[flow_col].plot(ax=ax, color='royalblue', linewidth=1.5)
            
            ax.set_title(f"Daily Streamflow for Station {self.station_id}", fontsize=16)
            ax.set_xlabel("Date")
            ax.set_ylabel("Discharge (m³/s)")
            ax.set_yscale('log')
            ax.grid(True, which="both", ls="--", linewidth=0.5)
            
            save_path = self.plot_dir / "hydrometric_timeseries.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"   - ✅ Saved hydrometric plot to: {save_path}")

        except Exception as e:
            print(f"   - ❌ Error plotting hydrometric data: {e}")

    def plot_climate_data(self, climate_path: Path):
        """
        Generates a summary plot of precipitation and temperature.
        """
        if not climate_path.exists():
            print(f"⚠️ Cannot plot climate data: File not found at {climate_path}")
            return

        print("   - Generating climate summary plot...")
        try:
            df = pd.read_csv(climate_path, parse_dates=['date'])
            df.set_index('date', inplace=True)

            # Resample to monthly for a cleaner plot
            monthly_df = df.resample('M').agg({
                'total_precipitation': 'sum',
                'mean_temperature': 'mean'
            })

            fig, ax1 = plt.subplots(figsize=(15, 7))

            # Plot precipitation as bars
            ax1.bar(monthly_df.index, monthly_df['total_precipitation'], 
                    width=25, align='center', color='skyblue', label='Monthly Precipitation')
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Total Monthly Precipitation (mm)", color='skyblue')
            ax1.tick_params(axis='y', labelcolor='skyblue')

            # Create a second y-axis for temperature
            ax2 = ax1.twinx()
            ax2.plot(monthly_df.index, monthly_df['mean_temperature'], 
                     color='crimson', marker='o', linestyle='-', markersize=4, label='Mean Monthly Temperature')
            ax2.set_ylabel("Mean Monthly Temperature (°C)", color='crimson')
            ax2.tick_params(axis='y', labelcolor='crimson')

            fig.suptitle(f"Climate Summary near Station {self.station_id}", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            
            save_path = self.plot_dir / "climate_summary.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"   - ✅ Saved climate plot to: {save_path}")

        except Exception as e:
            print(f"   - ❌ Error plotting climate data: {e}")
