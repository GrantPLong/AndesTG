"""Invert fluvial topography for spatially uniform (block) uplift history.

This module implements the core linear inversion method using Tikhonov
regularization to recover uplift rate history from topographic data.
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def invert_block_uplift(
    chi: np.ndarray,
    z: np.ndarray,
    gamma: float,
    q: int,
    to_plot: bool = False,
    fig: Optional[plt.Figure] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Invert fluvial topography for block uplift history using linear inversion.

    Uses Tikhonov regularization to solve for temporal variations in spatially
    uniform uplift rate. Can work with either non-dimensional (chi, z) or
    dimensional (tau, z) coordinates.

    Parameters
    ----------
    chi : np.ndarray
        Chi coordinate [L] or tau [T] values (1D array of length n).
    z : np.ndarray
        Elevation data [L] (1D array of length n).
    gamma : float
        Damping/regularization coefficient. Controls trade-off between
        model smoothness and data fit. Larger gamma = smoother solution.
    q : int
        Number of time intervals in the inversion.
    to_plot : bool, optional
        If True, plot the inferred uplift rate history. Default False.
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None and to_plot=True, creates new figure.

    Returns
    -------
    Ustar : np.ndarray
        Inferred uplift rate history (q×1 array). First element is most recent.
        Units depend on input: if chi input, returns U* (non-dimensional);
        if tau input, returns U [L/T].
    tstar : np.ndarray
        Time interval boundaries (q+1×1 array). First element is 0.
        Units match chi input.
    misfit : float
        Misfit between data and model topography [L].
        Calculated as: sqrt(sum(residuals²)) / (N-q)
        Note: This differs from standard RMSE to match MATLAB implementation.

    Notes
    -----
    - Time intervals are not equally spaced in time, but contain equal
      numbers of data pixels
    - Uses Tikhonov regularization: minimizes ||Az - b||² + gamma² ||U - U_prior||²
    - Prior model assumes constant uplift rate equal to mean(z/chi)
    - Only non-zero elevation data are used in the inversion

    Mathematical Background
    -----------------------
    For block uplift under stream power model with n=1:
        z(χ) = ∫₀^χ U(χ') dχ'

    This is discretized as: z = A·U where A[i,j] is the length of time
    interval j recorded in pixel i's elevation.

    Examples
    --------
    >>> chi = np.array([0, 100, 200, 300])
    >>> z = np.array([0, 50, 120, 200])
    >>> Ustar, tstar, misfit = invert_block_uplift(chi, z, gamma=1.0, q=2)
    """
    # Input validation
    if len(chi) != len(z):
        raise ValueError("chi and z must have the same length")

    if gamma <= 0:
        raise ValueError("gamma must be positive")

    if q <= 0:
        raise ValueError("q (number of time intervals) must be positive")

    if q > len(chi):
        raise ValueError("q cannot exceed number of data points")

    # Remove zero-elevation points (typically outlet or no-data)
    non_zero_elements = z != 0
    nz_chi = chi[non_zero_elements]
    nz_z = z[non_zero_elements]
    N = len(nz_chi)

    if N == 0:
        raise ValueError("All elevation values are zero - no data to invert")

    if q > N:
        raise ValueError(f"q ({q}) cannot exceed number of non-zero data points ({N})")

    # Sort data by chi
    chi_z_mat = np.column_stack([nz_chi, nz_z])
    sorted_indices = np.argsort(chi_z_mat[:, 0])
    sorted_chi_z_mat = chi_z_mat[sorted_indices]
    sorted_chi = sorted_chi_z_mat[:, 0]
    sorted_z = sorted_chi_z_mat[:, 1]

    # Construct time intervals with equal number of pixels in each
    val_per_dt = N // q  # Integer division

    # Construct time interval boundaries
    scaled_t_vec = np.zeros(q + 1)
    scaled_t_vec[0] = 0
    for i in range(1, q):
        scaled_t_vec[i] = sorted_chi[(i) * val_per_dt - 1]
    scaled_t_vec[q] = sorted_chi[-1]

    # Time interval lengths
    scaled_dt_vec = np.diff(scaled_t_vec)

    # Construct forward model matrix A*
    # A*[i,j] = length of time interval j recorded in pixel i
    Astar = np.zeros((N, q))

    for i in range(N):
        # Find which time intervals this pixel records
        # Pixels with small chi record fewer intervals
        pixel_chi = sorted_chi[i]

        # Find the last complete time interval
        filled_full_elements = np.searchsorted(scaled_t_vec, pixel_chi, side='right') - 2

        # Fill in complete time intervals
        if filled_full_elements >= 0:
            Astar[i, :filled_full_elements+1] = scaled_dt_vec[:filled_full_elements+1]

        # Fill in partial interval
        if filled_full_elements + 1 < q:
            partial_time = pixel_chi - np.sum(scaled_dt_vec[:filled_full_elements+1])
            if partial_time > 0:  # Ensure numerical stability
                Astar[i, filled_full_elements + 1] = partial_time

    # Build prior model (constant uplift rate)
    # Avoid division by zero
    valid_chi = sorted_chi > 1e-10
    if np.sum(valid_chi) > 0:
        U_pri_star = np.ones(q) * np.mean(sorted_z[valid_chi] / sorted_chi[valid_chi])
    else:
        U_pri_star = np.ones(q) * np.mean(sorted_z)

    # Solve regularized least squares problem
    # min ||A*U - z||² + gamma² ||U - U_prior||²
    # Solution: U = U_prior + (A'A + gamma²I)^(-1) A'(z - A*U_prior)

    ATA = Astar.T @ Astar
    denom = ATA + gamma**2 * np.eye(q)
    residual = sorted_z - Astar @ U_pri_star
    nom = Astar.T @ residual

    # Solve linear system
    Ustar = U_pri_star + np.linalg.solve(denom, nom)
    tstar = scaled_t_vec

    # Calculate misfit (matches MATLAB formula exactly)
    model_z = Astar @ Ustar
    residuals = sorted_z - model_z
    # MATLAB: Misfit = 1/((N-q))*sqrt(sum((sorted_z - Astar*Ustar).^2))
    misfit = np.sqrt(np.sum(residuals**2)) / (N - q)

    # Plotting
    if to_plot:
        if fig is None:
            fig = plt.figure(figsize=(12, 5))

        # Plot 1: Uplift history
        ax1 = fig.add_subplot(1, 2, 1)
        tstar_plot = []
        Ustar_plot = []
        for i in range(q):
            tstar_plot.extend([tstar[i], tstar[i+1]])
            Ustar_plot.extend([Ustar[i], Ustar[i]])

        ax1.plot(tstar_plot, Ustar_plot, 'b-', linewidth=2)
        ax1.set_xlabel('t* [m]', fontsize=14)
        ax1.set_ylabel('U*', fontsize=14)
        ax1.set_title('Inferred Uplift Rate History', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Data vs model fit
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(sorted_z, model_z, 'x', markersize=8)
        # Plot 1:1 line
        z_range = [min(sorted_z.min(), model_z.min()), max(sorted_z.max(), model_z.max())]
        ax2.plot(z_range, z_range, 'r--', linewidth=1, alpha=0.5, label='1:1 line')
        ax2.set_xlabel('Observed z [m]', fontsize=14)
        ax2.set_ylabel('Modeled z [m]', fontsize=14)
        ax2.set_title(f'Data vs Model (Misfit = {misfit:.2f} m)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')

        plt.tight_layout()
        plt.show()

    return Ustar, tstar, misfit


def invert_block_uplift_with_stats(
    chi: np.ndarray,
    z: np.ndarray,
    gamma: float,
    q: int
) -> dict:
    """Extended version that returns detailed statistics and diagnostics.

    Parameters
    ----------
    chi, z, gamma, q : same as invert_block_uplift

    Returns
    -------
    dict
        Dictionary containing:
        - 'Ustar': uplift rate history
        - 'tstar': time boundaries
        - 'misfit': RMS misfit
        - 'model_z': modeled elevations
        - 'residuals': z - model_z
        - 'r_squared': coefficient of determination
        - 'dof': degrees of freedom (N - q)
    """
    Ustar, tstar, misfit = invert_block_uplift(chi, z, gamma, q, to_plot=False)

    # Recompute for statistics
    non_zero_elements = z != 0
    nz_chi = chi[non_zero_elements]
    nz_z = z[non_zero_elements]
    N = len(nz_chi)

    chi_z_mat = np.column_stack([nz_chi, nz_z])
    sorted_indices = np.argsort(chi_z_mat[:, 0])
    sorted_chi_z_mat = chi_z_mat[sorted_indices]
    sorted_chi = sorted_chi_z_mat[:, 0]
    sorted_z = sorted_chi_z_mat[:, 1]

    # Reconstruct A matrix (same as in main function)
    val_per_dt = N // q
    scaled_t_vec = np.zeros(q + 1)
    scaled_t_vec[0] = 0
    for i in range(1, q):
        scaled_t_vec[i] = sorted_chi[(i) * val_per_dt - 1]
    scaled_t_vec[q] = sorted_chi[-1]
    scaled_dt_vec = np.diff(scaled_t_vec)

    Astar = np.zeros((N, q))
    for i in range(N):
        pixel_chi = sorted_chi[i]
        filled_full_elements = np.searchsorted(scaled_t_vec, pixel_chi, side='right') - 2
        if filled_full_elements >= 0:
            Astar[i, :filled_full_elements+1] = scaled_dt_vec[:filled_full_elements+1]
        if filled_full_elements + 1 < q:
            partial_time = pixel_chi - np.sum(scaled_dt_vec[:filled_full_elements+1])
            if partial_time > 0:
                Astar[i, filled_full_elements + 1] = partial_time

    model_z = Astar @ Ustar
    residuals = sorted_z - model_z

    # Calculate R²
    ss_tot = np.sum((sorted_z - np.mean(sorted_z))**2)
    ss_res = np.sum(residuals**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'Ustar': Ustar,
        'tstar': tstar,
        'misfit': misfit,
        'model_z': model_z,
        'observed_z': sorted_z,
        'residuals': residuals,
        'r_squared': r_squared,
        'dof': N - q,
        'n_data': N,
        'chi_sorted': sorted_chi
    }
