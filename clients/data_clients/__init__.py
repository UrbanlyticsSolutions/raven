"""
RAVEN Data Clients Package
Provides client classes for downloading data needed for RAVEN hydrological modeling

Available Clients:
- ClimateDataClient: Downloads climate data from ECCC APIs
- HydrometricDataClient: Downloads streamflow data from ECCC APIs  
- SoilDataClient: Provides coordinate-based soil classification
- SpatialLayersClient: Downloads spatial layers from multiple reliable sources (USGS, NRCan)

Recent Updates:
- SpatialLayersClient now supports multiple DEM sources including USGS 3DEP
- Fixed deprecated NRCan WCS endpoints
- Added automatic fallback between data sources
- Improved error handling and progress reporting
"""

from .climate_client import ClimateDataClient
from .hydrometric_client import HydrometricDataClient
from .soil_client import SoilDataClient
from .spatial_client import SpatialLayersClient

# Also provide direct access to standalone functions for backward compatibility
from .spatial_client import download_with_progress, get_usgs_dem_data, get_legacy_wcs_data

__all__ = [
    'ClimateDataClient',
    'HydrometricDataClient', 
    'SoilDataClient',
    'SpatialLayersClient',
    'download_with_progress',
    'get_usgs_dem_data', 
    'get_legacy_wcs_data'
]