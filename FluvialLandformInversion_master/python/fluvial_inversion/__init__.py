"""Fluvial Landform Inversion Package.

A Python toolkit for linear inversion of river longitudinal profiles to infer
tectonic uplift rate history from fluvial topography.

Author: Liran Goren (MATLAB original)
Python port with modern libraries and comprehensive testing.
"""

__version__ = "1.0.0"
__author__ = "Liran Goren"

# Import all implemented modules
from .calculate_chi import calculate_chi
from .invert_block_uplift import invert_block_uplift, invert_block_uplift_with_stats
from .findm_slope_area import findm_slope_area
from .findm_linear_chi import findm_linear_chi
from .findm_collapse_chi import findm_collapse_chi
from .invert_parabola import invert_parabola
from .invert_with_different_gamma import invert_with_different_gamma
from .calibrate_k_total_uplift import calibrate_k_total_uplift
from .bootstrap_invert_block_uplift import bootstrap_invert_block_uplift
from .prepare_inversion_data import prepare_inversion_data, prepare_inversion_data_simple

__all__ = [
    "calculate_chi",
    "invert_block_uplift",
    "invert_block_uplift_with_stats",
    "findm_slope_area",
    "findm_linear_chi",
    "findm_collapse_chi",
    "invert_parabola",
    "invert_with_different_gamma",
    "calibrate_k_total_uplift",
    "bootstrap_invert_block_uplift",
    "prepare_inversion_data",
    "prepare_inversion_data_simple",
]
