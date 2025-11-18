"""Find concavity index that collapses tributaries in chi-z space."""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from .calculate_chi import calculate_chi


def findm_collapse_chi(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rec_array: np.ndarray,
    area_array: np.ndarray,
    m_range: Optional[Tuple[float, float, float]] = None,
    n_bins: int = 25,
    to_plot: bool = True,
    fig: Optional[plt.Figure] = None
) -> Tuple[float, np.ndarray]:
    """Find concavity index that best collapses tributaries in chi-z space.

    Tests a range of m values and selects the one that minimizes the mean
    standard deviation of elevation across chi bins. This indicates that
    tributaries collapse onto a single curve.

    Parameters
    ----------
    x : np.ndarray
        X coordinates [L] (1D array of length n).
    y : np.ndarray
        Y coordinates [L] (1D array of length n).
    z : np.ndarray
        Elevation data [L] (1D array of length n).
    rec_array : np.ndarray
        Receiver relationships (1D array of length n).
    area_array : np.ndarray
        Drainage area [L²] (1D array of length n).
    m_range : tuple, optional
        (m_min, m_max, m_step) for testing. Default (0.1, 0.95, 0.05).
    n_bins : int, optional
        Number of bins for scatter calculation. Default 25.
    to_plot : bool, optional
        If True, plot results. Default True.
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on.

    Returns
    -------
    m : float
        Optimal concavity index.
    chi : np.ndarray
        Chi values computed with optimal m.

    Notes
    -----
    - Divides chi range into n_bins bins with equal numbers of data points
    - Calculates std(z) within each bin
    - Selects m that minimizes mean(std(z across bins))
    - If scatter metric is monotonic, consider extending m_range

    Examples
    --------
    >>> x = np.array([0, 100, 200, 150, 250])
    >>> y = np.array([0, 0, 0, 50, 50])
    >>> z = np.array([0, 50, 120, 60, 130])
    >>> rec_array = np.array([0, 0, 1, 1, 3])
    >>> area = np.array([1000, 800, 500, 400, 100])
    >>> m, chi = findm_collapse_chi(x, y, z, rec_array, area, to_plot=False)
    """
    # Input validation
    n = len(x)
    if not (len(y) == len(z) == len(rec_array) == len(area_array) == n):
        raise ValueError("All input arrays must have the same length")

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    if n_bins > n // 2:
        raise ValueError(f"n_bins ({n_bins}) too large for data size ({n})")

    # Set default m range
    if m_range is None:
        m_try = np.arange(0.1, 0.96, 0.05)
    else:
        m_min, m_max, m_step = m_range
        m_try = np.arange(m_min, m_max + m_step/2, m_step)

    scat_metric = np.zeros(len(m_try))

    # Test each m value
    for i, m_val in enumerate(m_try):
        chi_test = calculate_chi(x, y, rec_array, area_array, m_val)

        # Sort chi and z together
        sorted_indices = np.argsort(chi_test)
        sorted_chi = chi_test[sorted_indices]
        sorted_z = z[sorted_indices]

        # Calculate scatter metric: mean of std(z) across bins
        val_in_bins = len(chi_test) // n_bins
        chi_scat = np.zeros(n_bins)

        for k in range(n_bins - 1):
            start_idx = val_in_bins * k
            end_idx = val_in_bins * (k + 1)
            chi_scat[k] = np.std(sorted_z[start_idx:end_idx])

        # Last bin gets remaining points
        chi_scat[n_bins - 1] = np.std(sorted_z[val_in_bins * (n_bins - 1):])

        scat_metric[i] = np.mean(chi_scat)

    # Find m with minimum scatter
    min_idx = np.argmin(scat_metric)
    m_best = m_try[min_idx]
    min_scat = scat_metric[min_idx]

    # Calculate chi with best m
    chi = calculate_chi(x, y, rec_array, area_array, m_best)

    # Plotting
    if to_plot:
        if fig is None:
            fig = plt.figure(figsize=(12, 5))

        # Plot 1: Scatter metric vs m
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(m_try, scat_metric, 'ob', markersize=8)
        ax1.axvline(m_best, color='r', linestyle='--', linewidth=2,
                   label=f'Best m = {m_best:.3f}')
        ax1.set_xlabel('m', fontsize=12)
        ax1.set_ylabel('Mean of z scatter in bins', fontsize=12)
        ax1.set_title('Finding Optimal m', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Chi-z relationship with best m
        ax2 = fig.add_subplot(1, 2, 2)

        # Plot river network
        for i in range(len(chi)):
            j = rec_array[i]
            if j != i:  # Not an outlet
                ax2.plot([chi[j], chi[i]], [z[j], z[i]], 'b-', linewidth=0.5)

        ax2.set_xlabel('χ [m]', fontsize=12)
        ax2.set_ylabel('z [m]', fontsize=12)
        ax2.set_title(f'χ-z using m that minimizes scatter', fontsize=14)

        text_str = f'm = {m_best:.3f}\nmin scatter = {min_scat:.2f}'
        ax2.text(0.05, 0.95, text_str, transform=ax2.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return m_best, chi
