"""
Step 1: Data Preparation
========================

Handles initial data acquisition and preparation for RAVEN workflow:
- DEM data download and processing
- Routing product extraction
- Spatial data validation
- Coordinate system setup
- Boundary delineation

This step prepares all necessary spatial data inputs for subsequent 
watershed delineation and modeling steps.
"""

from .step1_data_preparation import Step1DataPreparation

__all__ = ['Step1DataPreparation']
