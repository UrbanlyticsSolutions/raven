"""
Step 5: RAVEN Model Generation
==============================

Generates complete RAVEN model files and configuration:
- Model structure file (.rvh) creation
- Parameter file (.rvp) generation
- Model instructions (.rvi) configuration
- Time series template (.rvt) setup
- Initial conditions (.rvc) specification
- Model validation and quality checks

This step produces the complete 5-file RAVEN model ready for execution.
"""

from .step5_raven_model import Step5RAVENModel

__all__ = ['Step5RAVENModel']
