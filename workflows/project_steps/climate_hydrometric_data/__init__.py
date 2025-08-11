"""
Climate and Hydrometric Data Processing
=======================================

Handles external data acquisition for RAVEN modeling:
- Climate data download and processing from ECCC
- Hydrometric station data acquisition
- Data quality control and gap filling
- Format conversion for RAVEN compatibility
- Temporal aggregation and statistics

This module supports the complete workflow by providing
the essential forcing and validation data for hydrological modeling.
"""

from .step_climate_hydrometric_data import ClimateHydrometricDataProcessor

__all__ = ['ClimateHydrometricDataProcessor']
