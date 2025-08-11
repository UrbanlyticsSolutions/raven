"""
Step 3: Lake Processing
=======================

Handles lake detection, classification, and routing integration:
- Lake body identification from DEM and external sources
- Lake classification by size and characteristics
- Lake-stream connectivity analysis
- Routing network modification for lake integration
- Lake parameter estimation

This step ensures proper representation of lake processes
in the hydrological model structure.
"""

from .step3_lake_processing import Step3LakeProcessing

__all__ = ['Step3LakeProcessing']
