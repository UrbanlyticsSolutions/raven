#!/usr/bin/env python3
"""
Standalone Interactive HTML Plot Generator for RAVEN Model Results

This script generates interactive HTML plots from RAVEN simulation outputs including:
- Interactive hydrographs with zoom, pan, and range selectors
- Flow duration curves with log scales
- Performance dashboards with multiple metrics
- Calibration results visualization

Usage:
    python generate_plots.py --sim-file simulation_results.csv --obs-file observed_data.csv --output-dir plots/
    python generate_plots.py --sim-file simulation_results.csv --output-dir plots/ --no-obs
    python generate_plots.py --model-dir path/to/model --auto-detect
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

# Plotting imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import matplotlib.pyplot as plt
    PLOTLY_AVAILABLE = True
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Plotting libraries not available: {e}")
    PLOTLY_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = False


class InteractivePlotGenerator:
    """Generate interactive HTML plots for RAVEN model results"""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def generate_all_plots(self, sim_file: str, obs_file: str = None, 
                          outlet_name: str = "Outlet", auto_detect: bool = False) -> Dict[str, Any]:
        """Generate all available interactive plots"""
        results = {
            'success': False,
            'plots_created': [],
            'interactive_plots': [],
            'errors': []
        }
        
        try:
            # Load simulation data
            sim_df = pd.read_csv(sim_file)
            print(f"Loaded simulation data: {len(sim_df)} time steps")
            
            # Load observed data if available
            obs_df = None
            if obs_file and Path(obs_file).exists():
                obs_df = pd.read_csv(obs_file)
                print(f"Loaded observed data: {len(obs_df)} time steps")
            elif auto_detect:
                # Try to auto-detect observed data file
                obs_candidates = [
                    Path(sim_file).parent / "observed_streamflow.csv",
                    Path(sim_file).parent / "obs_flow.csv",
                    Path(sim_file).parent / "observations.csv"
                ]
                for candidate in obs_candidates:
                    if candidate.exists():
                        obs_df = pd.read_csv(candidate)
                        print(f"Auto-detected observed data: {candidate}")
                        break
            
            # Parse data columns
            sim_data = self._parse_data_columns(sim_df, "simulation")
            obs_data = self._parse_data_columns(obs_df, "observation") if obs_df is not None else None
            
            if not sim_data['valid']:
                results['errors'].append("Could not parse simulation data columns")
                return results
            
            # Generate interactive hydrograph
            hydrograph_result = self._create_interactive_hydrograph(
                sim_data, obs_data, outlet_name
            )
            if hydrograph_result['success']:
                results['interactive_plots'].extend(hydrograph_result['files'])
            else:
                results['errors'].extend(hydrograph_result['errors'])
            
            # Generate flow duration curve if observed data available
            if obs_data and obs_data['valid']:
                fdc_result = self._create_interactive_flow_duration_curve(
                    sim_data, obs_data, outlet_name
                )
                if fdc_result['success']:
                    results['interactive_plots'].extend(fdc_result['files'])
                else:
                    results['errors'].extend(fdc_result['errors'])
                
                # Generate performance dashboard
                dashboard_result = self._create_performance_dashboard(
                    sim_data, obs_data, outlet_name
                )
                if dashboard_result['success']:
                    results['interactive_plots'].extend(dashboard_result['files'])
                else:
                    results['errors'].extend(dashboard_result['errors'])
            
            # Generate summary static plots as backup
            static_result = self._create_static_summary_plots(
                sim_data, obs_data, outlet_name
            )
            if static_result['success']:
                results['plots_created'].extend(static_result['files'])
            else:
                results['errors'].extend(static_result['errors'])
            
            results['success'] = len(results['interactive_plots']) > 0 or len(results['plots_created']) > 0
            
            # Print summary
            print(f"\nPlot Generation Summary:")
            print(f"  Interactive plots: {len(results['interactive_plots'])}")
            print(f"  Static plots: {len(results['plots_created'])}")
            print(f"  Errors: {len(results['errors'])}")
            
            if results['interactive_plots']:
                print(f"\nInteractive HTML plots created:")
                for plot_path in results['interactive_plots']:
                    print(f"  - {Path(plot_path).name}")
            
        except Exception as e:
            results['errors'].append(f"Plot generation failed: {str(e)}")
        
        return results
    
    def _parse_data_columns(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Parse data columns to identify date and flow columns"""
        data_info = {
            'valid': False,
            'date_col': None,
            'flow_col': None,
            'df': df
        }
        
        if df is None:
            return data_info
        
        try:
            # Find date column
            date_candidates = ['Date', 'date', 'time', 'Time', 'datetime', 'DateTime']
            for col in date_candidates:
                if col in df.columns:
                    data_info['date_col'] = col
                    df[col] = pd.to_datetime(df[col])
                    break
            
            # Find flow column
            flow_candidates = [col for col in df.columns 
                             if any(keyword in col.lower() for keyword in 
                                   ['flow', 'discharge', 'streamflow', 'runoff', 'q_'])]
            
            if flow_candidates:
                data_info['flow_col'] = flow_candidates[0]  # Use first match
            
            data_info['valid'] = data_info['date_col'] is not None and data_info['flow_col'] is not None
            
            if data_info['valid']:
                print(f"  {data_type.title()} data: {data_info['date_col']} (date), {data_info['flow_col']} (flow)")
            else:
                print(f"  Warning: Could not identify columns in {data_type} data")
                
        except Exception as e:
            print(f"  Error parsing {data_type} data: {str(e)}")
        
        return data_info
    
    def _create_interactive_hydrograph(self, sim_data: Dict, obs_data: Dict = None, 
                                     outlet_name: str = "Outlet") -> Dict[str, Any]:
        """Create interactive hydrograph with Plotly"""
        result = {'success': False, 'files': [], 'errors': []}
        
        if not PLOTLY_AVAILABLE:
            result['errors'].append("Plotly not available for interactive plots")
            return result
        
        try:
            fig = go.Figure()
            
            # Add simulation trace
            if sim_data['valid']:
                sim_df = sim_data['df']
                fig.add_trace(go.Scatter(
                    x=sim_df[sim_data['date_col']],
                    y=sim_df[sim_data['flow_col']],
                    mode='lines',
                    name='Simulated',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='<b>Simulated</b><br>Date: %{x}<br>Discharge: %{y:.3f} m³/s<extra></extra>'
                ))
            
            # Add observed trace if available
            if obs_data and obs_data['valid']:
                obs_df = obs_data['df']
                fig.add_trace(go.Scatter(
                    x=obs_df[obs_data['date_col']],
                    y=obs_df[obs_data['flow_col']],
                    mode='lines',
                    name='Observed',
                    line=dict(color='#d62728', width=2),
                    hovertemplate='<b>Observed</b><br>Date: %{x}<br>Discharge: %{y:.3f} m³/s<extra></extra>'
                ))
            
            # Update layout with enhanced features
            fig.update_layout(
                title=dict(
                    text=f'Interactive Hydrograph - {outlet_name}',
                    font=dict(size=18),
                    x=0.5
                ),
                xaxis_title='Date',
                yaxis_title='Discharge (m³/s)',
                template='plotly_white',
                showlegend=True,
                hovermode='x unified',
                width=1200,
                height=600,
                font=dict(size=12)
            )
            
            # Add range selector buttons and slider
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, label="7d", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all", label="All")
                        ]),
                        bgcolor="rgba(0,0,0,0.1)",
                        bordercolor="rgba(0,0,0,0.2)",
                        borderwidth=1
                    ),
                    rangeslider=dict(
                        visible=True,
                        bgcolor="rgba(0,0,0,0.05)",
                        bordercolor="rgba(0,0,0,0.2)",
                        borderwidth=1,
                        thickness=0.1
                    ),
                    type="date"
                ),
                yaxis=dict(
                    fixedrange=False,
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.1)"
                )
            )
            
            # Add modebar buttons
            config = {
                'displayModeBar': True,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                'displaylogo': False,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'hydrograph_{outlet_name}',
                    'height': 600,
                    'width': 1200,
                    'scale': 2
                }
            }
            
            # Save interactive plot
            plot_path = self.output_dir / f"interactive_hydrograph_{outlet_name}.html"
            fig.write_html(str(plot_path), config=config)
            
            result['files'].append(str(plot_path))
            result['success'] = True
            print(f"Created interactive hydrograph: {plot_path}")
            
        except Exception as e:
            result['errors'].append(f"Interactive hydrograph failed: {str(e)}")
        
        return result
    
    def _create_interactive_flow_duration_curve(self, sim_data: Dict, obs_data: Dict, 
                                              outlet_name: str = "Outlet") -> Dict[str, Any]:
        """Create interactive flow duration curve"""
        result = {'success': False, 'files': [], 'errors': []}
        
        if not PLOTLY_AVAILABLE:
            result['errors'].append("Plotly not available for interactive plots")
            return result
        
        try:
            fig = go.Figure()
            
            # Calculate flow duration curves
            sim_flow = sim_data['df'][sim_data['flow_col']].dropna()
            obs_flow = obs_data['df'][obs_data['flow_col']].dropna()
            
            sim_sorted = np.sort(sim_flow)[::-1]
            obs_sorted = np.sort(obs_flow)[::-1]
            
            sim_exceedance = np.arange(1, len(sim_sorted) + 1) / len(sim_sorted) * 100
            obs_exceedance = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100
            
            # Add simulation trace
            fig.add_trace(go.Scatter(
                x=sim_exceedance,
                y=sim_sorted,
                mode='lines',
                name='Simulated',
                line=dict(color='#1f77b4', width=3),
                hovertemplate='<b>Simulated</b><br>Exceedance: %{x:.1f}%<br>Discharge: %{y:.3f} m³/s<extra></extra>'
            ))
            
            # Add observed trace
            fig.add_trace(go.Scatter(
                x=obs_exceedance,
                y=obs_sorted,
                mode='lines',
                name='Observed',
                line=dict(color='#d62728', width=3),
                hovertemplate='<b>Observed</b><br>Exceedance: %{x:.1f}%<br>Discharge: %{y:.3f} m³/s<extra></extra>'
            ))
            
            # Update layout for log scale
            fig.update_layout(
                title=dict(
                    text=f'Interactive Flow Duration Curve - {outlet_name}',
                    font=dict(size=18),
                    x=0.5
                ),
                xaxis_title='Exceedance Probability (%)',
                yaxis_title='Discharge (m³/s)',
                template='plotly_white',
                showlegend=True,
                width=1000,
                height=600,
                xaxis_type="log",
                yaxis_type="log",
                font=dict(size=12)
            )
            
            # Add grid
            fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
            
            # Save interactive plot
            plot_path = self.output_dir / f"interactive_flow_duration_curve_{outlet_name}.html"
            fig.write_html(str(plot_path))
            
            result['files'].append(str(plot_path))
            result['success'] = True
            print(f"Created interactive flow duration curve: {plot_path}")
            
        except Exception as e:
            result['errors'].append(f"Interactive flow duration curve failed: {str(e)}")
        
        return result
    
    def _create_performance_dashboard(self, sim_data: Dict, obs_data: Dict, 
                                    outlet_name: str = "Outlet") -> Dict[str, Any]:
        """Create comprehensive performance dashboard"""
        result = {'success': False, 'files': [], 'errors': []}
        
        if not PLOTLY_AVAILABLE:
            result['errors'].append("Plotly not available for interactive plots")
            return result
        
        try:
            # Merge data for analysis
            sim_df = sim_data['df'][[sim_data['date_col'], sim_data['flow_col']]].copy()
            obs_df = obs_data['df'][[obs_data['date_col'], obs_data['flow_col']]].copy()
            
            sim_df.columns = ['date', 'sim_flow']
            obs_df.columns = ['date', 'obs_flow']
            
            merged_df = pd.merge(sim_df, obs_df, on='date', how='inner')
            
            if len(merged_df) == 0:
                result['errors'].append("No overlapping data for performance analysis")
                return result
            
            # Calculate performance metrics
            sim_values = merged_df['sim_flow'].dropna()
            obs_values = merged_df['obs_flow'].dropna()
            
            if len(sim_values) != len(obs_values) or len(sim_values) == 0:
                result['errors'].append("Data alignment failed for performance metrics")
                return result
            
            # Calculate metrics
            obs_mean = obs_values.mean()
            nse = 1 - (np.sum((obs_values - sim_values) ** 2) / np.sum((obs_values - obs_mean) ** 2))
            rmse = np.sqrt(np.mean((obs_values - sim_values) ** 2))
            mae = np.mean(np.abs(obs_values - sim_values))
            correlation = np.corrcoef(obs_values, sim_values)[0, 1]
            pbias = 100 * np.sum(obs_values - sim_values) / np.sum(obs_values)
            
            # Create dashboard with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Observed vs Simulated Scatter Plot',
                    'Residuals Over Time',
                    'Performance Metrics Summary',
                    'Monthly Statistics'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Scatter plot (obs vs sim)
            fig.add_trace(
                go.Scatter(
                    x=obs_values, y=sim_values, 
                    mode='markers',
                    name='Data Points',
                    marker=dict(color='#1f77b4', size=6, opacity=0.6),
                    hovertemplate='Observed: %{x:.3f}<br>Simulated: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add 1:1 line
            min_val = min(obs_values.min(), sim_values.min())
            max_val = max(obs_values.max(), sim_values.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', name='1:1 Line', 
                    line=dict(color='#d62728', dash='dash', width=2),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 2. Residuals over time
            residuals = sim_values - obs_values
            fig.add_trace(
                go.Scatter(
                    x=merged_df['date'], y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='#2ca02c', size=4, opacity=0.7),
                    hovertemplate='Date: %{x}<br>Residual: %{y:.3f} m³/s<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Add zero line for residuals
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
            
            # 3. Performance metrics text
            metrics_text = f"""
            <b>Performance Metrics</b><br><br>
            Nash-Sutcliffe Efficiency (NSE): <b>{nse:.3f}</b><br>
            Root Mean Square Error (RMSE): <b>{rmse:.3f} m³/s</b><br>
            Mean Absolute Error (MAE): <b>{mae:.3f} m³/s</b><br>
            Correlation Coefficient: <b>{correlation:.3f}</b><br>
            Percent Bias (PBIAS): <b>{pbias:.1f}%</b><br>
            Number of Data Points: <b>{len(sim_values)}</b><br><br>
            
            <b>Performance Categories:</b><br>
            NSE > 0.75: Excellent<br>
            NSE 0.65-0.75: Good<br>
            NSE 0.50-0.65: Satisfactory<br>
            NSE < 0.50: Unsatisfactory
            """
            
            fig.add_annotation(
                text=metrics_text,
                xref="x3", yref="y3",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=11),
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                row=2, col=1
            )
            
            # 4. Monthly box plot comparison
            merged_df['month'] = pd.to_datetime(merged_df['date']).dt.month
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            for month_num in range(1, 13):
                month_data = merged_df[merged_df['month'] == month_num]
                if len(month_data) > 0:
                    # Simulated box plot
                    fig.add_trace(
                        go.Box(
                            y=month_data['sim_flow'], 
                            name=f'Sim {months[month_num-1]}',
                            boxpoints='outliers',
                            marker_color='#1f77b4',
                            showlegend=False
                        ),
                        row=2, col=2
                    )
                    
                    # Observed box plot
                    fig.add_trace(
                        go.Box(
                            y=month_data['obs_flow'], 
                            name=f'Obs {months[month_num-1]}',
                            boxpoints='outliers',
                            marker_color='#d62728',
                            showlegend=False
                        ),
                        row=2, col=2
                    )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f'Performance Dashboard - {outlet_name}',
                    font=dict(size=20),
                    x=0.5
                ),
                height=1000,
                width=1400,
                showlegend=True,
                template='plotly_white',
                font=dict(size=10)
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Observed Discharge (m³/s)", row=1, col=1)
            fig.update_yaxes(title_text="Simulated Discharge (m³/s)", row=1, col=1)
            
            fig.update_xaxes(title_text="Date", row=1, col=2)
            fig.update_yaxes(title_text="Residuals (m³/s)", row=1, col=2)
            
            fig.update_xaxes(title_text="Month", row=2, col=2)
            fig.update_yaxes(title_text="Discharge (m³/s)", row=2, col=2)
            
            # Save dashboard
            plot_path = self.output_dir / f"performance_dashboard_{outlet_name}.html"
            fig.write_html(str(plot_path))
            
            result['files'].append(str(plot_path))
            result['success'] = True
            print(f"Created performance dashboard: {plot_path}")
            
        except Exception as e:
            result['errors'].append(f"Performance dashboard failed: {str(e)}")
        
        return result
    
    def _create_static_summary_plots(self, sim_data: Dict, obs_data: Dict = None, 
                                   outlet_name: str = "Outlet") -> Dict[str, Any]:
        """Create static summary plots as backup"""
        result = {'success': False, 'files': [], 'errors': []}
        
        if not MATPLOTLIB_AVAILABLE:
            result['errors'].append("Matplotlib not available for static plots")
            return result
        
        try:
            # Create summary figure with multiple subplots
            if obs_data and obs_data['valid']:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # 1. Hydrograph
            sim_df = sim_data['df']
            ax1.plot(sim_df[sim_data['date_col']], sim_df[sim_data['flow_col']], 
                    'b-', label='Simulated', linewidth=1)
            
            if obs_data and obs_data['valid']:
                obs_df = obs_data['df']
                ax1.plot(obs_df[obs_data['date_col']], obs_df[obs_data['flow_col']], 
                        'r-', label='Observed', linewidth=1)
            
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Discharge (m³/s)')
            ax1.set_title(f'Hydrograph - {outlet_name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Flow statistics
            sim_flow = sim_df[sim_data['flow_col']].dropna()
            stats_text = f"""Flow Statistics:
Mean: {sim_flow.mean():.2f} m³/s
Median: {sim_flow.median():.2f} m³/s
Min: {sim_flow.min():.2f} m³/s
Max: {sim_flow.max():.2f} m³/s
Std: {sim_flow.std():.2f} m³/s"""
            
            ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_title('Flow Statistics')
            ax2.axis('off')
            
            # Additional plots if observed data available
            if obs_data and obs_data['valid']:
                obs_flow = obs_df[obs_data['flow_col']].dropna()
                
                # 3. Flow duration curves
                sim_sorted = np.sort(sim_flow)[::-1]
                obs_sorted = np.sort(obs_flow)[::-1]
                
                sim_exceedance = np.arange(1, len(sim_sorted) + 1) / len(sim_sorted) * 100
                obs_exceedance = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100
                
                ax3.loglog(sim_exceedance, sim_sorted, 'b-', label='Simulated', linewidth=2)
                ax3.loglog(obs_exceedance, obs_sorted, 'r-', label='Observed', linewidth=2)
                
                ax3.set_xlabel('Exceedance Probability (%)')
                ax3.set_ylabel('Discharge (m³/s)')
                ax3.set_title('Flow Duration Curve')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # 4. Scatter plot
                # Align data for scatter plot
                sim_aligned = sim_df.set_index(sim_data['date_col'])[sim_data['flow_col']]
                obs_aligned = obs_df.set_index(obs_data['date_col'])[obs_data['flow_col']]
                combined = pd.concat([sim_aligned, obs_aligned], axis=1, join='inner')
                combined.columns = ['sim', 'obs']
                combined = combined.dropna()
                
                if len(combined) > 0:
                    ax4.scatter(combined['obs'], combined['sim'], alpha=0.6, s=20)
                    
                    # Add 1:1 line
                    min_val = min(combined['obs'].min(), combined['sim'].min())
                    max_val = max(combined['obs'].max(), combined['sim'].max())
                    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
                    
                    ax4.set_xlabel('Observed Discharge (m³/s)')
                    ax4.set_ylabel('Simulated Discharge (m³/s)')
                    ax4.set_title('Observed vs Simulated')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save static plot
            plot_path = self.output_dir / f"summary_plots_{outlet_name}.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            result['files'].append(str(plot_path))
            result['success'] = True
            print(f"Created static summary plots: {plot_path}")
            
        except Exception as e:
            result['errors'].append(f"Static plots failed: {str(e)}")
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Generate Interactive HTML Plots for RAVEN Model Results')
    parser.add_argument('--sim-file', type=str, required=True, 
                       help='Path to simulation results CSV file')
    parser.add_argument('--obs-file', type=str, 
                       help='Path to observed data CSV file (optional)')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots (default: plots)')
    parser.add_argument('--outlet-name', type=str, default='Outlet',
                       help='Name for the outlet/station (default: Outlet)')
    parser.add_argument('--auto-detect', action='store_true',
                       help='Try to auto-detect observed data file')
    parser.add_argument('--model-dir', type=str,
                       help='Model directory to auto-detect all files')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.sim_file).exists():
        print(f"ERROR: Simulation file not found: {args.sim_file}")
        sys.exit(1)
    
    if args.obs_file and not Path(args.obs_file).exists():
        print(f"ERROR: Observed data file not found: {args.obs_file}")
        sys.exit(1)
    
    # Auto-detect files if model directory provided
    if args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            print(f"ERROR: Model directory not found: {args.model_dir}")
            sys.exit(1)
        
        # Look for simulation output files
        sim_candidates = list(model_dir.glob("*Hydrographs.csv"))
        if sim_candidates:
            args.sim_file = str(sim_candidates[0])
            print(f"Auto-detected simulation file: {args.sim_file}")
        else:
            print(f"ERROR: No hydrograph files found in {args.model_dir}")
            sys.exit(1)
        
        args.auto_detect = True
    
    # Create plot generator
    plot_generator = InteractivePlotGenerator(output_dir=args.output_dir)
    
    print(f"Generating interactive plots...")
    print(f"Simulation file: {args.sim_file}")
    if args.obs_file:
        print(f"Observed file: {args.obs_file}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    # Generate plots
    results = plot_generator.generate_all_plots(
        sim_file=args.sim_file,
        obs_file=args.obs_file,
        outlet_name=args.outlet_name,
        auto_detect=args.auto_detect
    )
    
    # Print results
    print("="*60)
    if results['success']:
        print("SUCCESS: Interactive plot generation completed")
        print(f"Generated {len(results['interactive_plots'])} interactive plots")
        print(f"Generated {len(results['plots_created'])} static plots")
        
        if results['interactive_plots']:
            print("\nInteractive HTML files:")
            for plot_path in results['interactive_plots']:
                print(f"  - {plot_path}")
                
    else:
        print("FAILED: Plot generation failed")
        if results['errors']:
            print("Errors:")
            for error in results['errors']:
                print(f"  - {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
