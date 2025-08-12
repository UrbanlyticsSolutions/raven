#!/usr/bin/env python3
"""
Plot hydrographs comparing observed vs simulated streamflow using best calibrated parameters
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import os

def calculate_metrics(obs, sim):
    """Calculate performance metrics"""
    # Remove NaN values
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs_clean = obs[mask]
    sim_clean = sim[mask]
    
    if len(obs_clean) == 0:
        return {"NSE": np.nan, "RMSE": np.nan, "BIAS": np.nan, "R2": np.nan}
    
    # Nash-Sutcliffe Efficiency
    nse = 1 - np.sum((obs_clean - sim_clean)**2) / np.sum((obs_clean - np.mean(obs_clean))**2)
    
    # RMSE
    rmse = np.sqrt(np.mean((obs_clean - sim_clean)**2))
    
    # Bias (%)
    bias = (np.mean(sim_clean) - np.mean(obs_clean)) / np.mean(obs_clean) * 100
    
    # R-squared
    corr_matrix = np.corrcoef(obs_clean, sim_clean)
    r2 = corr_matrix[0,1]**2 if not np.isnan(corr_matrix[0,1]) else np.nan
    
    return {"NSE": nse, "RMSE": rmse, "BIAS": bias, "R2": r2}

def plot_hydrographs():
    """Plot observed vs simulated hydrographs with climate data"""
    
    # Load observed data
    obs_file = "hydrometric/observed_streamflow.csv"
    obs_df = pd.read_csv(obs_file)
    obs_df['Date'] = pd.to_datetime(obs_df['Date'])
    obs_df = obs_df.set_index('Date')
    
    # Load simulated data (best calibrated)
    sim_file = "models/files/outlet_49.5738_-119.0368/output/Hydrographs.csv"
    sim_df = pd.read_csv(sim_file)
    sim_df['date'] = pd.to_datetime(sim_df['date'])
    sim_df = sim_df.set_index('date')
    
    # Load climate data
    climate_file = "climate/climate_forcing.csv"
    try:
        climate_df = pd.read_csv(climate_file, index_col=0)
        climate_df.index = pd.to_datetime(climate_df.index)
        has_climate = True
        print(f"Climate data loaded: {list(climate_df.columns)}")
    except Exception as e:
        has_climate = False
        print(f"Climate data not found: {e}")
    
    # Get outlet subbasin (should be subbasin 39 based on RVH)
    outlet_col = None
    for col in sim_df.columns:
        if 'sub39 [m3/s]' in col:
            outlet_col = col
            break
    
    if outlet_col is None:
        # Try to find any discharge column
        for col in sim_df.columns:
            if '[m3/s]' in col and 'observed' not in col:
                outlet_col = col
                break
        
    print(f"Using simulated column: {outlet_col}")
    print(f"Available columns: {list(sim_df.columns)}")
    
    # Merge data
    combined = pd.merge(obs_df[['Discharge_cms']], sim_df[[outlet_col]], 
                       left_index=True, right_index=True, how='inner')
    combined.columns = ['Observed', 'Simulated']
    
    # Calculate metrics
    metrics = calculate_metrics(combined['Observed'].values, combined['Simulated'].values)
    
    # Create comprehensive hydrograph plots
    if has_climate:
        fig = plt.figure(figsize=(16, 16))
        gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3, height_ratios=[3, 1, 1, 2, 1])
    else:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Main hydrograph - full time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(combined.index, 0, combined['Observed'], alpha=0.4, color='darkblue', label='Observed Flow')
    ax1.fill_between(combined.index, 0, combined['Simulated'], alpha=0.4, color='darkorange', label='Calibrated Flow')
    ax1.plot(combined.index, combined['Observed'], color='darkblue', label='Observed', linewidth=2)
    ax1.plot(combined.index, combined['Simulated'], color='darkorange', label='Calibrated', linewidth=2)
    ax1.set_ylabel('Discharge (m³/s)', fontsize=12)
    ax1.set_title('Hydrograph Comparison - Outlet 49.5738°N, 119.0368°W\\n(OSTRICH Calibrated HBV Model)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(combined.index[0], combined.index[-1])
    
    # Add metrics text box
    metrics_text = f"Overall Performance:\\nNSE: {metrics['NSE']:.3f}\\nRMSE: {metrics['RMSE']:.2f} m³/s\\nBias: {metrics['BIAS']:.1f}%\\nR²: {metrics['R2']:.3f}"
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=10, fontweight='bold')
    
    # Add climate data if available
    if has_climate:
        # Precipitation subplot
        ax_precip = fig.add_subplot(gs[1, :])
        if 'Precipitation_mm' in climate_df.columns:
            precip_data = climate_df['Precipitation_mm'].reindex(combined.index)
            ax_precip.bar(precip_data.index, precip_data.values, alpha=0.6, color='lightblue', width=1, label='Precipitation')
        ax_precip.set_ylabel('Precipitation\\n(mm/day)', fontsize=10)
        ax_precip.set_title('Daily Precipitation', fontsize=11, fontweight='bold')
        ax_precip.legend()
        ax_precip.grid(True, alpha=0.3)
        ax_precip.invert_yaxis()
        
        # Temperature subplot  
        ax_temp = fig.add_subplot(gs[2, :])
        if 'Temperature_C' in climate_df.columns:
            temp_data = climate_df['Temperature_C'].reindex(combined.index)
            ax_temp.plot(temp_data.index, temp_data.values, color='red', linewidth=1, label='Temperature')
            ax_temp.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='0°C')
        ax_temp.set_ylabel('Temperature\\n(°C)', fontsize=10)
        ax_temp.set_title('Daily Temperature', fontsize=11, fontweight='bold')
        ax_temp.legend()
        ax_temp.grid(True, alpha=0.3)
    
    # Calibration period hydrograph (1982-2010)
    if has_climate:
        ax2 = fig.add_subplot(gs[3, :])
    else:
        ax2 = fig.add_subplot(gs[1, :])
    calib_period = combined['1982':'2010']
    if len(calib_period) > 0:
        ax2.fill_between(calib_period.index, 0, calib_period['Observed'], alpha=0.4, color='darkblue')
        ax2.fill_between(calib_period.index, 0, calib_period['Simulated'], alpha=0.4, color='darkorange')
        ax2.plot(calib_period.index, calib_period['Observed'], color='darkblue', label='Observed', linewidth=2)
        ax2.plot(calib_period.index, calib_period['Simulated'], color='darkorange', label='Calibrated', linewidth=2)
        ax2.set_ylabel('Discharge (m³/s)', fontsize=12)
        ax2.set_title('Calibration Period Hydrograph (1982-2010)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Add temperature on secondary axis if available
        if has_climate:
            temp_col = None
            # Check for temperature columns
            for col in climate_df.columns:
                if 'TEMP' in col.upper() or 'Temperature' in col:
                    temp_col = col
                    break
            
            if temp_col:
                # Calculate average daily temperature from min/max if available, otherwise use existing temp
                if 'TEMP_MAX' in climate_df.columns and 'TEMP_MIN' in climate_df.columns:
                    temp_data = (climate_df['TEMP_MAX'] + climate_df['TEMP_MIN']) / 2
                else:
                    temp_data = climate_df[temp_col]
                
                # Get calibration period temperature
                temp_calib = temp_data['1982':'2010']
                if len(temp_calib) > 0:
                    ax2_temp = ax2.twinx()
                    ax2_temp.plot(temp_calib.index, temp_calib.values, color='red', alpha=0.7, linewidth=1, label='Temperature')
                    ax2_temp.axhline(y=0, color='red', linestyle='--', alpha=0.3)
                    ax2_temp.set_ylabel('Temperature (°C)', fontsize=10, color='red')
                    ax2_temp.tick_params(axis='y', labelcolor='red')
                    ax2_temp.legend(loc='upper right')
        
        # Calculate calibration period metrics
        calib_metrics = calculate_metrics(calib_period['Observed'].values, calib_period['Simulated'].values)
        calib_text = f"Calibration Period:\\nNSE: {calib_metrics['NSE']:.3f}\\nRMSE: {calib_metrics['RMSE']:.2f} m³/s"
        ax2.text(0.02, 0.98, calib_text, transform=ax2.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                 fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No calibration period data available', 
                 transform=ax2.transAxes, ha='center', va='center')
    
    # Scatter plot for correlation analysis
    if has_climate:
        ax3 = fig.add_subplot(gs[4, 0])
    else:
        ax3 = fig.add_subplot(gs[2, 0])
    # Sample data for better visualization if too many points
    sample_size = min(5000, len(combined))
    if len(combined) > sample_size:
        sample_idx = np.random.choice(len(combined), sample_size, replace=False)
        obs_sample = combined['Observed'].iloc[sample_idx]
        sim_sample = combined['Simulated'].iloc[sample_idx]
    else:
        obs_sample = combined['Observed']
        sim_sample = combined['Simulated']
    
    ax3.scatter(obs_sample, sim_sample, alpha=0.5, s=10, color='darkorange')
    max_val = max(combined['Observed'].max(), combined['Simulated'].max())
    min_val = min(combined['Observed'].min(), combined['Simulated'].min())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line', linewidth=2)
    ax3.set_xlabel('Observed Discharge (m³/s)', fontsize=10)
    ax3.set_ylabel('Simulated Discharge (m³/s)', fontsize=10)
    ax3.set_title('Observed vs Simulated\\nScatter Plot', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    # Flow duration curve
    if has_climate:
        ax4 = fig.add_subplot(gs[4, 1])
    else:
        ax4 = fig.add_subplot(gs[2, 1])
    obs_sorted = np.sort(combined['Observed'].dropna())[::-1]
    sim_sorted = np.sort(combined['Simulated'].dropna())[::-1]
    obs_exceedance = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100
    sim_exceedance = np.arange(1, len(sim_sorted) + 1) / len(sim_sorted) * 100
    
    ax4.semilogy(obs_exceedance, obs_sorted, color='darkblue', label='Observed', linewidth=2)
    ax4.semilogy(sim_exceedance, sim_sorted, color='darkorange', label='Calibrated', linewidth=2)
    ax4.set_xlabel('Exceedance Probability (%)', fontsize=10)
    ax4.set_ylabel('Discharge (m³/s)', fontsize=10)
    ax4.set_title('Flow Duration Curve', fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 100)
    
    # Format dates for time series plots
    time_axes = [ax1, ax2]
    if has_climate:
        time_axes.extend([ax_precip, ax_temp])
    
    for ax in time_axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save comprehensive hydrograph plot
    output_file = "models/files/outlet_49.5738_-119.0368/best/comprehensive_hydrograph_analysis.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive hydrograph analysis: {output_file}")
    
    # Also save to main plots folder
    main_plot = "models/files/outlet_49.5738_-119.0368/plots/comprehensive_hydrograph_analysis.png"
    os.makedirs(os.path.dirname(main_plot), exist_ok=True)
    plt.savefig(main_plot, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive hydrograph analysis: {main_plot}")
    
    plt.show()
    
    # Print summary
    print("\\n" + "="*60)
    print("CALIBRATED MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Nash-Sutcliffe Efficiency: {metrics['NSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.3f} m³/s")
    print(f"Bias: {metrics['BIAS']:.1f}%")
    print(f"R-squared: {metrics['R2']:.4f}")
    print(f"Data period: {combined.index[0]} to {combined.index[-1]}")
    print(f"Number of data points: {len(combined)}")
    print("="*60)
    
    return metrics

if __name__ == "__main__":
    metrics = plot_hydrographs()