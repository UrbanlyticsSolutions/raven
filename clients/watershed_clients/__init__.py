"""
Watershed Clients Package

This package provides watershed analysis and delineation functionality
using various methods and data sources.
"""

try:
    from .whitebox_client import WhiteboxWatershedClient as WhiteboxClient
except ImportError:
    # Create a dummy class if whitebox is not available
    class WhiteboxClient:
        def __init__(self):
            raise ImportError("WhiteboxClient requires additional dependencies. Install with: pip install whitebox pyflwdir rasterio geopandas")

try:
    from .watershed import ProfessionalWatershedAnalyzer as Watershed
except ImportError:
    # Create a dummy class if watershed analysis tools are not available
    class Watershed:
        def __init__(self):
            raise ImportError("Watershed analysis requires additional dependencies. Install with: pip install whitebox pyflwdir rasterio geopandas")

__all__ = ['WhiteboxClient', 'Watershed']