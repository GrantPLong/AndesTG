"""Stone (2000) cosmogenic nuclide production rate scaling.

Extracted from NB02 (production_rates) cell 3.
"""

import numpy as np

# Stone 2000 polynomial coefficients for the 30-40 deg latitude bin
STONE_COEFFICIENTS = {
    "a": 42.0983,
    "b": 512.6857,
    "c": -0.120551,
    "d": 0.00011752,
    "e": -0.000000038809,
}

P_SLHL = 4.01  # Sea-level high-latitude production rate (atoms/g/yr)

# Mean catchment elevations (m) for each sample
SAMPLE_ELEVATIONS = {
    "CT-1": 2261.20,
    "CT-2": 1514.97,
    "CT-3": 285.50,
    "CT-4": 1706.03,
    "CT-5": 2021.01,
    "CT-6": 743.02,
    "CT-7": 2197.73,
    "CT-8": 1656.83,
    "CT-9": 1451.47,
    "CT-10": 1670.98,
    "CT-11": 1660.70,
}


def elevation_to_pressure(elevation):
    """Convert elevation (m) to atmospheric pressure (hPa) using the standard atmosphere."""
    return 1013.25 * np.exp(
        -5.2569 * np.log(288.15 / (288.15 - 0.0065 * elevation))
    )


def stone2000_scaling(pressure):
    """Compute Stone (2000) scaling factor from atmospheric pressure (hPa)."""
    a = STONE_COEFFICIENTS["a"]
    b = STONE_COEFFICIENTS["b"]
    c = STONE_COEFFICIENTS["c"]
    d = STONE_COEFFICIENTS["d"]
    e = STONE_COEFFICIENTS["e"]
    return a + b * np.exp(-pressure / 150.0) + c * pressure + d * pressure**2 + e * pressure**3


def stone2000_production_rate(elevation):
    """Compute surface production rate (atoms/g/yr) from elevation using Stone (2000).

    Parameters
    ----------
    elevation : float
        Mean catchment elevation in metres.

    Returns
    -------
    float
        Spallation production rate in atoms/g/yr.
    """
    pressure = elevation_to_pressure(elevation)
    scaling = stone2000_scaling(pressure)
    return P_SLHL * scaling
