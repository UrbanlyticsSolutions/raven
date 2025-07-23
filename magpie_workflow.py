#!/usr/bin/env python3
"""
Magpie Workflow - Simplified Local Version
A streamlined hydrological modeling workflow for local execution
"""

import os
import sys
import json
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tempfile
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('magpie_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """Simple configuration class"""
    
    def __init__(self, config_file: str = None):
        self.workspace_dir = Path("./workspace")
        self.model_name = "raven_model"
        self.start_year = 2000
        self.end_year = 2005
        self.min_drainage_area = 50.0
        self.buffer_distance = 5000.0
        
        # Setup paths
        self.paths = {
            'data': self.workspace_dir / 'data',
            'inputs': self.workspace_dir / 'data' / 'inputs',
            'outputs': self.workspace_dir / 'outputs',
            'forcing': self.workspace_dir / 'outputs' / 'forcing',
            'basin': self.workspace_dir / 'outputs' / 'basin',
            'raven': self.workspace_dir / 'outputs' / 'raven',
            'plots': self.workspace_dir / 'outputs' / 'plots',
            'temp': self.workspace_dir / 'temp',
            'logs': self.workspace_dir / 'logs'
        }
        
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            logger.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
    
    def create_directories(self):
        """Create all necessary directories"""
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        logger.info("Workspace directories created")
    
    def save_config(self, output_file: str = None):
        """Save current configuration"""
        if not output_file:
            output_file = self.workspace_dir / 'config' / 'workflow_config.json'
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            'model_name': self.model_name,
            'start_year': self.start_year,
            'end_year': self.end_year,
            'min_drainage_area': self.min_drainage_area,
            'buffer_distance': self.buffer_distance
        }
        
        with open(output_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Configuration saved to {output_file}")


class DataProcessor:
    """Handles data loading and validation"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_study_area(self, shapefile_path: str = None) -> gpd.GeoDataFrame:
        """Load and process study area shapefile"""
        logger.info("Loading study area...")
        
        if not shapefile_path:
            shapefile_path = self.config.paths['inputs'] / 'study_area.shp'
        
        if not Path(shapefile_path).exists():
            logger.error(f"Study area shapefile not found: {shapefile_path}")
            return self._create_sample_study_area()
        
        try:
            gdf = gpd.read_file(shapefile_path)
            
            # Ensure CRS is WGS84
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            
            # Dissolve multiple features
            if len(gdf) > 1:
                gdf = gdf.dissolve().reset_index(drop=True)
            
            # Save processed version
            output_path = self.config.paths['data'] / 'study_area_processed.shp'
            gdf.to_file(output_path)
            
            logger.info(f"Study area processed: {len(gdf)} features")
            return gdf
            
        except Exception as e:
            logger.error(f"Error processing study area: {e}")
            return self._create_sample_study_area()
    
    def _create_sample_study_area(self) -> gpd.GeoDataFrame:
        """Create a sample study area for testing"""
        from shapely.geometry import box
        
        logger.info("Creating sample study area...")
        
        # Create a sample rectangular area
        bbox = box(-76, 45, -75, 46)
        gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs='EPSG:4326')
        
        # Save sample study area
        output_path = self.config.paths['data'] / 'study_area_sample.shp'
        gdf.to_file(output_path)
        
        return gdf
    
    def create_sample_dem(self, study_area: gpd.GeoDataFrame) -> str:
        """Create sample DEM data"""
        logger.info("Creating sample DEM...")
        
        bounds = study_area.total_bounds
        
        # Create simple elevation grid
        lons = np.linspace(bounds[0], bounds[2], 50)
        lats = np.linspace(bounds[1], bounds[3], 50)
        
        # Generate elevation data (simple gradient)
        elevation = np.zeros((len(lats), len(lons)))
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                # Simple elevation based on distance from center
                center_lat, center_lon = (bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2
                dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
                elevation[i, j] = 500 + 300 * np.sin(dist * 50) + np.random.normal(0, 20)
        
        # Save as CSV for simplicity
        dem_data = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                dem_data.append({
                    'lat': lat,
                    'lon': lon,
                    'elevation': elevation[i, j]
                })
        
        dem_df = pd.DataFrame(dem_data)
        dem_path = self.config.paths['data'] / 'dem_sample.csv'
        dem_df.to_csv(dem_path, index=False)
        
        logger.info(f"Sample DEM created: {dem_path}")
        return str(dem_path)
    
    def create_sample_climate_data(self, study_area: gpd.GeoDataFrame) -> str:
        """Create sample climate data"""
        logger.info("Creating sample climate data...")
        
        # Date range
        dates = pd.date_range(
            f'{self.config.start_year}-01-01',
            f'{self.config.end_year}-12-31',
            freq='D'
        )
        
        # Generate synthetic weather data
        np.random.seed(42)  # For reproducible results
        
        climate_data = []
        for date in dates:
            # Simple seasonal temperature pattern
            day_of_year = date.timetuple().tm_yday
            base_temp = 10 + 15 * np.sin(2 * np.pi * day_of_year / 365.25)
            
            tmax = base_temp + 5 + np.random.normal(0, 3)
            tmin = base_temp - 5 + np.random.normal(0, 2)
            
            # Simple precipitation (more in winter/spring)
            precip_base = 2 if day_of_year < 150 or day_of_year > 300 else 1
            precip = np.maximum(0, np.random.exponential(precip_base))
            
            climate_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'tmax': round(tmax, 2),
                'tmin': round(tmin, 2),
                'precip': round(precip, 2)
            })
        
        climate_df = pd.DataFrame(climate_data)
        climate_path = self.config.paths['data'] / 'climate_sample.csv'
        climate_df.to_csv(climate_path, index=False)
        
        logger.info(f"Sample climate data created: {climate_path} ({len(climate_df)} records)")
        return str(climate_path)
    
    def validate_inputs(self) -> Dict[str, bool]:
        """Validate all input data"""
        validation_results = {}
        
        # Check study area (either processed or sample)
        study_area_processed = self.config.paths['data'] / 'study_area_processed.shp'
        study_area_sample = self.config.paths['data'] / 'study_area_sample.shp'
        validation_results['study_area'] = study_area_processed.exists() or study_area_sample.exists()
        
        # Check DEM
        dem_path = self.config.paths['data'] / 'dem_sample.csv'
        validation_results['dem'] = dem_path.exists()
        
        # Check climate data
        climate_path = self.config.paths['data'] / 'climate_sample.csv'
        validation_results['climate'] = climate_path.exists()
        
        logger.info(f"Input validation: {validation_results}")
        return validation_results


class BasinProcessor:
    """Handles basin discretization (simplified)"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_subbasins(self, study_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Create simplified subbasins"""
        logger.info("Creating subbasins...")
        
        # Simple grid-based approach for demonstration
        bounds = study_area.total_bounds
        
        # Calculate grid size based on area
        area_km2 = self._calculate_area_km2(study_area)
        target_subbasins = max(2, min(20, int(area_km2 / self.config.min_drainage_area)))
        
        grid_size = int(np.sqrt(target_subbasins))
        
        # Create grid
        x_coords = np.linspace(bounds[0], bounds[2], grid_size + 1)
        y_coords = np.linspace(bounds[1], bounds[3], grid_size + 1)
        
        subbasins = []
        subbasin_id = 1
        
        for i in range(grid_size):
            for j in range(grid_size):
                from shapely.geometry import box
                
                cell = box(x_coords[j], y_coords[i], x_coords[j+1], y_coords[i+1])
                
                # Check intersection with study area
                intersection = study_area.geometry.iloc[0].intersection(cell)
                
                if intersection.area > 0:
                    subbasins.append({
                        'SubId': subbasin_id,
                        'Area_km2': self._polygon_area_km2(intersection),
                        'geometry': intersection
                    })
                    subbasin_id += 1
        
        subbasins_gdf = gpd.GeoDataFrame(subbasins, crs='EPSG:4326')
        
        # Save subbasins
        output_path = self.config.paths['basin'] / 'subbasins.shp'
        subbasins_gdf.to_file(output_path)
        
        logger.info(f"Created {len(subbasins_gdf)} subbasins")
        return subbasins_gdf
    
    def create_hrus(self, subbasins: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Create simplified HRUs (one per subbasin for now)"""
        logger.info("Creating HRUs...")
        
        hrus = []
        for idx, subbasin in subbasins.iterrows():
            # Simple HRU = one per subbasin
            centroid = subbasin.geometry.centroid
            
            hrus.append({
                'HRU_ID': idx + 1,
                'SubId': subbasin['SubId'],
                'Area_km2': subbasin['Area_km2'],
                'Elevation': 300 + np.random.normal(0, 50),  # Random elevation
                'Latitude': centroid.y,
                'Longitude': centroid.x,
                'LandUse': 1,  # Forest
                'SoilType': 1,  # Default soil
                'geometry': subbasin.geometry
            })
        
        hrus_gdf = gpd.GeoDataFrame(hrus, crs='EPSG:4326')
        
        # Save HRUs
        output_path = self.config.paths['basin'] / 'hrus.shp'
        hrus_gdf.to_file(output_path)
        
        logger.info(f"Created {len(hrus_gdf)} HRUs")
        return hrus_gdf
    
    def _calculate_area_km2(self, gdf: gpd.GeoDataFrame) -> float:
        """Calculate area in km2"""
        # Simple approximation
        bounds = gdf.total_bounds
        area_deg2 = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        area_km2 = area_deg2 * 111 * 111  # Very rough conversion
        return area_km2
    
    def _polygon_area_km2(self, polygon) -> float:
        """Calculate polygon area in km2"""
        bounds = polygon.bounds
        area_deg2 = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        area_km2 = area_deg2 * 111 * 111  # Very rough conversion
        return area_km2


class ForcingProcessor:
    """Handles meteorological forcing data"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def format_for_raven(self, climate_data_path: str, subbasins: gpd.GeoDataFrame) -> str:
        """Format climate data for Raven model"""
        logger.info("Formatting forcing data for Raven...")
        
        # Load climate data
        climate_df = pd.read_csv(climate_data_path)
        climate_df['date'] = pd.to_datetime(climate_df['date'])
        
        # Create RVT file content
        rvt_content = f"""# Raven Time Series File
# Generated by Magpie Workflow
# Model: {self.config.model_name}

:MultiData
{climate_df['date'].iloc[0].strftime('%Y-%m-%d')} 00:00:00 1.0 {len(climate_df)}
:Parameters, PRECIP, TEMP_MAX, TEMP_MIN
:Units, mm/d, C, C

"""
        
        # Add daily data
        for _, row in climate_df.iterrows():
            rvt_content += f"{row['precip']:.2f} {row['tmax']:.2f} {row['tmin']:.2f}\n"
        
        rvt_content += ":EndMultiData\n"
        
        # Save RVT file
        rvt_path = self.config.paths['forcing'] / f'{self.config.model_name}.rvt'
        with open(rvt_path, 'w') as f:
            f.write(rvt_content)
        
        logger.info(f"RVT file created: {rvt_path}")
        return str(rvt_path)
    
    def create_gauge_file(self, subbasins: gpd.GeoDataFrame) -> str:
        """Create observation gauge file"""
        logger.info("Creating gauge file...")
        
        # Use outlet of largest subbasin as gauge location
        largest_subbasin = subbasins.loc[subbasins['Area_km2'].idxmax()]
        centroid = largest_subbasin.geometry.centroid
        
        gauge_content = f"""# Raven Gauge File
# Generated by Magpie Workflow

:Gauge Outlet_Gauge
:Latitude {centroid.y:.6f}
:Longitude {centroid.x:.6f}
:Elevation 250.0
:RedirectToFile Hydrographs.rvt
:EndGauge
"""
        
        gauge_path = self.config.paths['forcing'] / 'gauges.rvg'
        with open(gauge_path, 'w') as f:
            f.write(gauge_content)
        
        logger.info(f"Gauge file created: {gauge_path}")
        return str(gauge_path)


class RavenModelBuilder:
    """Builds Raven model input files"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_rvi_file(self) -> str:
        """Create Raven input file (.rvi)"""
        logger.info("Creating RVI file...")
        
        rvi_content = f"""# Raven Input File
# Generated by Magpie Workflow
# Model: {self.config.model_name}

:StartDate       {self.config.start_year}-01-01 00:00:00
:EndDate         {self.config.end_year}-12-31 00:00:00
:TimeStep        1.0
:Method          ORDERED_SERIES

:SoilClasses
  :Attributes, POROSITY, FIELD_CAPACITY, SAT_WILT, HYD_COND, THERMAL_COND, THERMAL_CAP
  :Units,      none,     none,           none,     mm_per_d,  W_per_mK,    J_per_m3K
  DEFAULT_SOIL, 0.45,    0.30,           0.10,     10.0,      2.0,         2.0e6
:EndSoilClasses

:VegetationClasses
  :Attributes, MAX_HT, MAX_LAI, MAX_LEAF_COND
  :Units,      m,      none,    mm_per_s
  FOREST,      20.0,   5.0,     5.0
:EndVegetationClasses

:LandUseClasses
  :Attributes, IMPERM, FOREST_COV
  :Units,      frac,   frac
  FOREST,      0.0,    1.0
:EndLandUseClasses

:HydrologicProcesses
  :Precipitation        PRECIP_RAVEN        ATMOS_PRECIP    MULTIPLE
  :Infiltration         INF_GREEN_AMPT      PONDED_WATER    MULTIPLE
  :Baseflow             BASE_LINEAR         SOIL[0]         SURFACE_WATER
  :Percolation          PERC_LINEAR         SOIL[0]         SOIL[1] 
:EndHydrologicProcesses

:SubBasinProperties
# [future subbasin properties will be added here]
:EndSubBasinProperties
"""
        
        rvi_path = self.config.paths['raven'] / f'{self.config.model_name}.rvi'
        with open(rvi_path, 'w') as f:
            f.write(rvi_content)
        
        logger.info(f"RVI file created: {rvi_path}")
        return str(rvi_path)
    
    def create_rvh_file(self, subbasins: gpd.GeoDataFrame, hrus: gpd.GeoDataFrame) -> str:
        """Create Raven watershed file (.rvh)"""
        logger.info("Creating RVH file...")
        
        rvh_content = f"""# Raven Watershed File
# Generated by Magpie Workflow
# Model: {self.config.model_name}

:SubBasins
  :Attributes, NAME, DOWNSTREAM_ID, PROFILE, REACH_LENGTH, GAUGED
  :Units,      none, none,          none,    km,           none
"""
        
        for _, subbasin in subbasins.iterrows():
            downstream_id = -1  # All drain to outlet for now
            rvh_content += f"  {subbasin['SubId']}, Sub{subbasin['SubId']}, {downstream_id}, NONE, 1.0, 0\n"
        
        rvh_content += ":EndSubBasins\n\n"
        
        # Add HRUs
        rvh_content += """:HRUs
  :Attributes, AREA, ELEVATION, LATITUDE, LONGITUDE, BASIN_ID, LAND_USE_CLASS, VEG_CLASS, SOIL_PROFILE
  :Units,      km2,  m,         deg,      deg,       none,     none,           none,      none
"""
        
        for _, hru in hrus.iterrows():
            rvh_content += f"  {hru['HRU_ID']}, {hru['Area_km2']:.3f}, {hru['Elevation']:.1f}, {hru['Latitude']:.6f}, {hru['Longitude']:.6f}, {hru['SubId']}, FOREST, FOREST, [DEFAULT_SOIL]\n"
        
        rvh_content += ":EndHRUs\n"
        
        rvh_path = self.config.paths['raven'] / f'{self.config.model_name}.rvh'
        with open(rvh_path, 'w') as f:
            f.write(rvh_content)
        
        logger.info(f"RVH file created: {rvh_path}")
        return str(rvh_path)
    
    def create_rvp_file(self) -> str:
        """Create Raven parameters file (.rvp)"""
        logger.info("Creating RVP file...")
        
        rvp_content = f"""# Raven Parameters File
# Generated by Magpie Workflow
# Model: {self.config.model_name}

# Global Parameters
:GlobalParameter SNOW_SWI 0.05

# Soil Parameters
:SoilParameterList
  :Parameters POROSITY FIELD_CAPACITY SAT_WILT HYD_COND
  :Units      none     none           none     mm_per_d
  DEFAULT_SOIL 0.45   0.30           0.10     10.0
:EndSoilParameterList

# Vegetation Parameters  
:VegetationParameterList
  :Parameters MAX_HT MAX_LAI MAX_LEAF_COND
  :Units      m      none    mm_per_s
  FOREST      20.0   5.0     5.0
:EndVegetationParameterList

# Land Use Parameters
:LandUseParameterList
  :Parameters IMPERM FOREST_COV
  :Units      frac   frac
  FOREST      0.0    1.0
:EndLandUseParameterList
"""
        
        rvp_path = self.config.paths['raven'] / f'{self.config.model_name}.rvp'
        with open(rvp_path, 'w') as f:
            f.write(rvp_content)
        
        logger.info(f"RVP file created: {rvp_path}")
        return str(rvp_path)
    
    def create_rvc_file(self, hrus: gpd.GeoDataFrame) -> str:
        """Create Raven initial conditions file (.rvc)"""
        logger.info("Creating RVC file...")
        
        rvc_content = f"""# Raven Initial Conditions File
# Generated by Magpie Workflow
# Model: {self.config.model_name}

# Initial HRU states
:HRUStateVariableTable
  :Attributes, SOIL[0], SOIL[1]
  :Units,      mm,     mm
"""
        
        for _, hru in hrus.iterrows():
            # Simple initial conditions
            soil0 = 50.0  # mm
            soil1 = 100.0  # mm
            rvc_content += f"  {hru['HRU_ID']}, {soil0}, {soil1}\n"
        
        rvc_content += ":EndHRUStateVariableTable\n"
        
        rvc_path = self.config.paths['raven'] / f'{self.config.model_name}.rvc'
        with open(rvc_path, 'w') as f:
            f.write(rvc_content)
        
        logger.info(f"RVC file created: {rvc_path}")
        return str(rvc_path)


class Visualizer:
    """Creates plots and visualizations"""
    
    def __init__(self, config: Config):
        self.config = config
        plt.style.use('default')
    
    def plot_study_area(self, study_area: gpd.GeoDataFrame) -> str:
        """Plot study area"""
        logger.info("Creating study area plot...")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        study_area.plot(ax=ax, facecolor='lightblue', edgecolor='black', alpha=0.7)
        
        ax.set_title(f'Study Area - {self.config.model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.config.paths['plots'] / 'study_area.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Study area plot saved: {output_path}")
        return str(output_path)
    
    def plot_subbasins(self, subbasins: gpd.GeoDataFrame) -> str:
        """Plot subbasins"""
        logger.info("Creating subbasins plot...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        subbasins.plot(ax=ax, column='SubId', cmap='tab20', 
                      edgecolor='black', linewidth=0.5, alpha=0.8, legend=True)
        
        # Add subbasin labels
        for idx, row in subbasins.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(str(row['SubId']), (centroid.x, centroid.y), 
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        ax.set_title(f'Subbasins - {self.config.model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.config.paths['plots'] / 'subbasins.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Subbasins plot saved: {output_path}")
        return str(output_path)
    
    def plot_climate_data(self, climate_data_path: str) -> str:
        """Plot climate data summary"""
        logger.info("Creating climate data plot...")
        
        # Load climate data
        df = pd.read_csv(climate_data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Temperature time series
        axes[0, 0].plot(df['date'], df['tmax'], label='Tmax', color='red', alpha=0.7)
        axes[0, 0].plot(df['date'], df['tmin'], label='Tmin', color='blue', alpha=0.7)
        axes[0, 0].set_title('Temperature Time Series')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precipitation time series
        axes[0, 1].plot(df['date'], df['precip'], color='green', alpha=0.7)
        axes[0, 1].set_title('Precipitation Time Series')
        axes[0, 1].set_ylabel('Precipitation (mm/day)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Monthly averages
        df['month'] = df['date'].dt.month
        monthly_temp = df.groupby('month')[['tmax', 'tmin']].mean()
        monthly_precip = df.groupby('month')['precip'].sum()
        
        axes[1, 0].plot(monthly_temp.index, monthly_temp['tmax'], 'ro-', label='Tmax')
        axes[1, 0].plot(monthly_temp.index, monthly_temp['tmin'], 'bo-', label='Tmin')
        axes[1, 0].set_title('Monthly Average Temperature')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Temperature (°C)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(monthly_precip.index, monthly_precip.values, alpha=0.7, color='green')
        axes[1, 1].set_title('Monthly Total Precipitation')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Precipitation (mm)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Climate Data Summary - {self.config.model_name}', fontsize=16)
        plt.tight_layout()
        
        output_path = self.config.paths['plots'] / 'climate_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Climate summary plot saved: {output_path}")
        return str(output_path)
    
    def create_summary_report(self, results: Dict[str, Any]) -> str:
        """Create summary report"""
        logger.info("Creating summary report...")
        
        report_content = f"""# Magpie Workflow Summary Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- Model Name: {self.config.model_name}
- Simulation Period: {self.config.start_year} - {self.config.end_year}
- Minimum Drainage Area: {self.config.min_drainage_area} km²

## Processing Results
"""
        
        for step, success in results.items():
            status = "✓ Completed" if success else "✗ Failed"
            report_content += f"- {step.replace('_', ' ').title()}: {status}\n"
        
        report_content += f"""
## Outputs Generated
- Study Area: {self.config.paths['data'] / 'study_area_processed.shp'}
- Subbasins: {self.config.paths['basin'] / 'subbasins.shp'}
- HRUs: {self.config.paths['basin'] / 'hrus.shp'}
- Raven Input Files: {self.config.paths['raven']}
- Plots: {self.config.paths['plots']}

## Next Steps
1. Review generated model files
2. Run Raven simulation
3. Analyze results
4. Calibrate parameters if needed

---
Generated by Magpie Workflow - Simplified Local Version
"""
        
        report_path = self.config.paths['outputs'] / 'workflow_summary.md'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Summary report created: {report_path}")
        return str(report_path)


class MagpieWorkflow:
    """Main workflow orchestration class"""
    
    def __init__(self, config_file: str = None):
        self.config = Config(config_file)
        self.config.create_directories()
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.basin_processor = BasinProcessor(self.config)
        self.forcing_processor = ForcingProcessor(self.config)
        self.model_builder = RavenModelBuilder(self.config)
        self.visualizer = Visualizer(self.config)
        
        logger.info(f"Magpie Workflow initialized: {self.config.model_name}")
    
    def run_full_workflow(self) -> Dict[str, bool]:
        """Execute the complete workflow"""
        logger.info("Starting Magpie Workflow...")
        
        results = {}
        
        try:
            # Step 1: Load/create study area
            logger.info("=== STEP 1: Study Area Processing ===")
            study_area = self.data_processor.load_study_area()
            results['study_area'] = study_area is not None
            
            # Step 2: Create sample data
            logger.info("=== STEP 2: Data Preparation ===")
            dem_path = self.data_processor.create_sample_dem(study_area)
            climate_path = self.data_processor.create_sample_climate_data(study_area)
            results['sample_data'] = True
            
            # Step 3: Basin processing
            logger.info("=== STEP 3: Basin Processing ===")
            subbasins = self.basin_processor.create_subbasins(study_area)
            hrus = self.basin_processor.create_hrus(subbasins)
            results['basin_processing'] = len(subbasins) > 0 and len(hrus) > 0
            
            # Step 4: Forcing data
            logger.info("=== STEP 4: Forcing Data ===")
            rvt_path = self.forcing_processor.format_for_raven(climate_path, subbasins)
            gauge_path = self.forcing_processor.create_gauge_file(subbasins)
            results['forcing_data'] = Path(rvt_path).exists()
            
            # Step 5: Model building
            logger.info("=== STEP 5: Model Building ===")
            rvi_path = self.model_builder.create_rvi_file()
            rvh_path = self.model_builder.create_rvh_file(subbasins, hrus)
            rvp_path = self.model_builder.create_rvp_file()
            rvc_path = self.model_builder.create_rvc_file(hrus)
            
            model_files_exist = all(Path(p).exists() for p in [rvi_path, rvh_path, rvp_path, rvc_path])
            results['model_building'] = model_files_exist
            
            # Step 6: Visualization
            logger.info("=== STEP 6: Visualization ===")
            study_plot = self.visualizer.plot_study_area(study_area)
            subbasins_plot = self.visualizer.plot_subbasins(subbasins)
            climate_plot = self.visualizer.plot_climate_data(climate_path)
            results['visualization'] = True
            
            # Step 7: Summary report
            logger.info("=== STEP 7: Summary Report ===")
            report_path = self.visualizer.create_summary_report(results)
            results['summary_report'] = Path(report_path).exists()
            
            # Final validation
            validation_results = self.data_processor.validate_inputs()
            results['validation'] = all(validation_results.values())
            
            logger.info("=== WORKFLOW COMPLETED ===")
            success_count = sum(results.values())
            total_steps = len(results)
            logger.info(f"Completed {success_count}/{total_steps} steps successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            results['workflow_error'] = False
            return results
    
    def run_step(self, step_name: str) -> bool:
        """Run a single workflow step"""
        logger.info(f"Running step: {step_name}")
        
        if step_name == "study_area":
            study_area = self.data_processor.load_study_area()
            return study_area is not None
        
        elif step_name == "sample_data":
            study_area = self.data_processor.load_study_area()
            dem_path = self.data_processor.create_sample_dem(study_area)
            climate_path = self.data_processor.create_sample_climate_data(study_area)
            return True
        
        elif step_name == "visualization":
            # Load existing data
            study_area_path = self.config.paths['data'] / 'study_area_processed.shp'
            if study_area_path.exists():
                study_area = gpd.read_file(study_area_path)
                self.visualizer.plot_study_area(study_area)
                
                subbasins_path = self.config.paths['basin'] / 'subbasins.shp'
                if subbasins_path.exists():
                    subbasins = gpd.read_file(subbasins_path)
                    self.visualizer.plot_subbasins(subbasins)
                
                climate_path = self.config.paths['data'] / 'climate_sample.csv'
                if climate_path.exists():
                    self.visualizer.plot_climate_data(str(climate_path))
                
                return True
            return False
        
        else:
            logger.warning(f"Unknown step: {step_name}")
            return False


def main():
    """Main entry point"""
    print("=" * 60)
    print("🏔️  MAGPIE WORKFLOW - SIMPLIFIED LOCAL VERSION")
    print("=" * 60)
    
    try:
        # Initialize workflow
        workflow = MagpieWorkflow()
        
        # Run full workflow
        results = workflow.run_full_workflow()
        
        # Print results summary
        print("\n" + "=" * 60)
        print("📊 WORKFLOW RESULTS SUMMARY")
        print("=" * 60)
        
        for step, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{step.replace('_', ' ').title():<25} {status}")
        
        success_count = sum(results.values())
        total_steps = len(results)
        
        print(f"\n🎯 OVERALL: {success_count}/{total_steps} steps completed")
        
        if success_count == total_steps:
            print("🎉 Workflow completed successfully!")
            print(f"📁 Check outputs in: {workflow.config.workspace_dir / 'outputs'}")
        else:
            print("⚠️  Some steps failed. Check logs for details.")
        
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        logger.exception("Fatal workflow error")


if __name__ == "__main__":
    main()
