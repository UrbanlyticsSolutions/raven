"""
Step 4: HRU Generation
======================

Creates Hydrologic Response Units (HRUs) for RAVEN modeling:
- Land cover data integration
- Soil data processing and classification
- Vegetation parameter assignment
- HRU discretization and aggregation
- Parameter estimation and validation

This step creates the fundamental modeling units that represent
the spatial heterogeneity of hydrological processes.
"""

from .step4_hru_generation import Step4HRUGeneration

__all__ = ['Step4HRUGeneration']
