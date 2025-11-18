"""Calculate chi coordinate for river pixels in a fluvial network.

This module implements the chi transformation for river profiles, which is
fundamental to the stream power model analysis.
"""

import numpy as np
from typing import Tuple


def calculate_chi(
    x: np.ndarray,
    y: np.ndarray,
    rec_array: np.ndarray,
    area_array: np.ndarray,
    m: float,
    A0: float = 1.0
) -> np.ndarray:
    """Calculate chi coordinate for river pixels in a fluvial network.

    Chi (χ) is a transformed longitudinal coordinate that linearizes steady-state
    river profiles under the stream power model. It is defined as:

        χ = ∫ (A₀/A(x))^m dx

    where A is drainage area, A₀ is a reference area, and m is the concavity index.

    Parameters
    ----------
    x : np.ndarray
        X coordinates [L] of each pixel (1D array of length n).
    y : np.ndarray
        Y coordinates [L] of each pixel (1D array of length n).
    rec_array : np.ndarray
        Receiver relationships defining flow network (1D array of length n).
        rec_array[i] = j means pixel i flows to pixel j.
        Outlets are defined as their own receivers (rec_array[outlet] = outlet).
    area_array : np.ndarray
        Upstream drainage area [L²] for each pixel (1D array of length n).
    m : float
        Area exponent in stream power law (concavity index).
    A0 : float, optional
        Reference drainage area [L²]. Default is 1.0.

    Returns
    -------
    np.ndarray
        Chi values [L] for each pixel (1D array of length n).

    Notes
    -----
    - Row i in x, y, area_array, rec_array, and output chi refers to the same pixel
    - The algorithm sorts pixels by drainage area (largest first) to ensure
      downstream pixels are processed before upstream pixels
    - Chi increases upstream from the outlet (where chi = 0)

    Examples
    --------
    >>> x = np.array([0, 100, 200, 300])
    >>> y = np.array([0, 0, 0, 0])
    >>> rec_array = np.array([0, 0, 1, 2])  # 0 is outlet
    >>> area_array = np.array([1000, 750, 500, 100])
    >>> m = 0.45
    >>> chi = calculate_chi(x, y, rec_array, area_array, m)
    """
    # Validate inputs
    n = len(x)
    if not (len(y) == len(rec_array) == len(area_array) == n):
        raise ValueError("All input arrays must have the same length")

    if m < 0 or m > 1:
        raise ValueError("m should typically be between 0 and 1")

    if A0 <= 0:
        raise ValueError("A0 (reference area) must be positive")

    if np.any(area_array <= 0):
        raise ValueError("All drainage areas must be positive")

    # Create data matrix: [id, x, y, area]
    # Using 0-based indexing (Python convention)
    ids = np.arange(n)
    data_mat = np.column_stack([ids, x, y, area_array])

    # Sort by drainage area (largest first = outlet first)
    sorted_data = data_mat[np.argsort(-data_mat[:, 3])]

    # Initialize chi array
    chi = np.zeros(n)

    # Calculate chi for each pixel, starting from outlet
    for i in range(n):
        my_id = int(sorted_data[i, 0])
        my_rec = rec_array[my_id]

        # Check if this is an outlet (receiver is itself)
        if my_id == my_rec:
            chi[i] = 0.0
        else:
            # Find the row of the receiver in sorted data
            rec_row = np.where(sorted_data[:, 0] == my_rec)[0]

            if len(rec_row) == 0:
                raise ValueError(f"Receiver {my_rec} for pixel {my_id} not found")

            rec_row = rec_row[0]

            # Calculate distance to receiver
            dx = sorted_data[i, 1] - sorted_data[rec_row, 1]
            dy = sorted_data[i, 2] - sorted_data[rec_row, 2]
            dist = np.sqrt(dx**2 + dy**2)

            # Calculate chi using the integral formulation
            # chi[i] = chi[receiver] + dist * (A0/A[i])^m
            chi[i] = chi[rec_row] + dist * (A0 / sorted_data[i, 3])**m

    # Resort chi to match original pixel ordering
    sorted_data_with_chi = np.column_stack([sorted_data, chi])
    desorted_data = sorted_data_with_chi[np.argsort(sorted_data_with_chi[:, 0])]
    chi_output = desorted_data[:, -1]

    return chi_output


def validate_flow_network(rec_array: np.ndarray) -> Tuple[bool, str]:
    """Validate that receiver array defines a valid flow network.

    Parameters
    ----------
    rec_array : np.ndarray
        Receiver relationships.

    Returns
    -------
    bool
        True if network is valid.
    str
        Error message if invalid, empty string if valid.
    """
    n = len(rec_array)

    # Check that all receivers are valid indices
    if np.any(rec_array < 0) or np.any(rec_array >= n):
        return False, "All receiver indices must be in range [0, n-1]"

    # Check for at least one outlet (pixel that is its own receiver)
    outlets = np.where(rec_array == np.arange(n))[0]
    if len(outlets) == 0:
        return False, "No outlet found (need at least one pixel where rec_array[i] = i)"

    # Check for cycles (each pixel must eventually reach an outlet)
    for i in range(n):
        visited = set()
        current = i

        # Follow flow path
        while current not in visited:
            visited.add(current)
            next_pixel = rec_array[current]

            # If we reach an outlet, this path is valid
            if next_pixel == current:
                break

            current = next_pixel

            # Safety check for infinite loops
            if len(visited) > n:
                return False, f"Cycle detected starting from pixel {i}"

    return True, ""
