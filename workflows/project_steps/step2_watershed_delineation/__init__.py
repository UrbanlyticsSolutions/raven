"""
Step 2: Watershed Delineation
=============================

Performs watershed and stream network delineation:
- DEM processing and conditioning
- Flow direction calculation
- Stream network extraction
- Watershed boundary delineation
- Subbasin identification

This step creates the fundamental hydrological structure that
forms the basis for HRU generation and model setup.
"""

from .step2_watershed_delineation import Step2WatershedDelineation

__all__ = ['Step2WatershedDelineation']
