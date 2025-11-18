"""Find concavity index that produces most linear chi-z profiles."""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .calculate_chi import calculate_chi


def findm_linear_chi(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rec_array: np.ndarray,
    area_array: np.ndarray,
    m_range: Optional[Tuple[float, float, float]] = None,
    to_plot: bool = True,
    fig: Optional[plt.Figure] = None
) -> Tuple[float, np.ndarray]:
    """Find concavity index that produces most linear chi-z profiles.

    Tests a range of m values and selects the one that maximizes R² for
    a linear relationship through the origin in chi-z space.

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
    - Fits z = a*chi (line through origin) for each m value
    - Selects m that maximizes R²
    - If R² is monotonic across range, consider extending m_range

    Examples
    --------
    >>> x = np.array([0, 100, 200, 300])
    >>> y = np.zeros(4)
    >>> z = np.array([0, 50, 120, 200])
    >>> rec_array = np.array([0, 0, 1, 2])
    >>> area = np.array([1000, 800, 500, 100])
    >>> m, chi = findm_linear_chi(x, y, z, rec_array, area, to_plot=False)
    """
    # Input validation
    n = len(x)
    if not (len(y) == len(z) == len(rec_array) == len(area_array) == n):
        raise ValueError("All input arrays must have the same length")

    # Set default m range
    if m_range is None:
        m_try = np.arange(0.1, 0.96, 0.05)
    else:
        m_min, m_max, m_step = m_range
        m_try = np.arange(m_min, m_max + m_step/2, m_step)

    R2 = np.zeros(len(m_try))

    # Test each m value
    for i, m_val in enumerate(m_try):
        chi_test = calculate_chi(x, y, rec_array, area_array, m_val)

        # Fit z = a * chi (line through origin)
        # Use curve_fit with fixed intercept at 0
        def linear_through_origin(chi_in, a):
            return a * chi_in

        try:
            # Only use non-zero z values
            valid = z > 0
            if np.sum(valid) < 2:
                R2[i] = 0
                continue

            popt, _ = curve_fit(linear_through_origin, chi_test[valid], z[valid])

            # Calculate R²
            z_pred = popt[0] * chi_test[valid]
            ss_res = np.sum((z[valid] - z_pred)**2)
            ss_tot = np.sum((z[valid] - np.mean(z[valid]))**2)
            R2[i] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        except:
            R2[i] = 0

    # Find m with maximum R²
    max_idx = np.argmax(R2)
    m_best = m_try[max_idx]
    max_R2 = R2[max_idx]

    # Calculate chi with best m
    chi = calculate_chi(x, y, rec_array, area_array, m_best)

    # Plotting
    if to_plot:
        if fig is None:
            fig = plt.figure(figsize=(12, 5))

        # Plot 1: R² vs m
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(m_try, R2, 'ob', markersize=8)
        ax1.axvline(m_best, color='r', linestyle='--', linewidth=2,
                   label=f'Best m = {m_best:.3f}')
        ax1.set_xlabel('m', fontsize=12)
        ax1.set_ylabel('R² of linear regression through χ-z domain', fontsize=12)
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

        # Plot linear fit
        valid = z > 0
        if np.sum(valid) > 0:
            def linear_through_origin(chi_in, a):
                return a * chi_in
            popt, _ = curve_fit(linear_through_origin, chi[valid], z[valid])
            chi_line = np.linspace(0, chi.max(), 100)
            z_line = popt[0] * chi_line
            ax2.plot(chi_line, z_line, 'k-', linewidth=2, label='Best fit')

        ax2.set_xlabel('χ [m]', fontsize=12)
        ax2.set_ylabel('z [m]', fontsize=12)
        ax2.set_title(f'χ-z using m = {m_best:.3f}', fontsize=14)

        text_str = f'm = {m_best:.3f}, R² = {max_R2:.3f}'
        ax2.text(0.05, 0.95, text_str, transform=ax2.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return m_best, chi
