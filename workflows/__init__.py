"""
RAVEN Workflows Package
Implements consolidated workflow approaches for RAVEN hydrological model generation

Available Workflows:
- Refactored Full Delineation: Complete watershed modeling with shared datasets
- Multi-Gauge Delineation: Multiple gauge processing with unified data management
"""

from .refactored_full_delineation import RefactoredFullDelineation
from .multi_gauge_delineation import MultiGaugeDelineation

__all__ = [
    'RefactoredFullDelineation',
    'MultiGaugeDelineation'
]