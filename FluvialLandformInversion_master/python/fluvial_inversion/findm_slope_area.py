"""Calculate concavity index from slope-area relationships."""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats


def findm_slope_area(
    slope_array: np.ndarray,
    area_array: np.ndarray,
    to_plot: bool = True,
    fig: Optional[plt.Figure] = None
) -> Tuple[float, float, float, float]:
    """Calculate concavity index from log-log slope-area relationship.

    Uses linear regression on log(slope) vs log(area) to determine the
    concavity index m. This assumes the stream power model with n=1:
        S ∝ A^(-m)

    Parameters
    ----------
    slope_array : np.ndarray
        Channel slope values [L/L] (1D array of length n).
    area_array : np.ndarray
        Drainage area values [L²] (1D array of length n).
    to_plot : bool, optional
        If True, plot the slope-area relationship. Default True.
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None and to_plot=True, creates new figure.

    Returns
    -------
    m : float
        Best-fit concavity index.
    lower_bound : float
        Lower bound of 95% confidence interval on m.
    upper_bound : float
        Upper bound of 95% confidence interval on m.
    R2 : float
        Coefficient of determination (R²) for the fit.

    Notes
    -----
    - Only positive slope values are used
    - Uses ordinary least squares regression on log-transformed data
    - The concavity m = -slope of the log(S) vs log(A) relationship

    Examples
    --------
    >>> slope = np.array([0.1, 0.05, 0.02, 0.01])
    >>> area = np.array([100, 500, 1000, 5000])
    >>> m, lb, ub, R2 = findm_slope_area(slope, area, to_plot=False)
    >>> print(f"m = {m:.3f}, R² = {R2:.3f}")
    """
    # Input validation
    if len(slope_array) != len(area_array):
        raise ValueError("slope_array and area_array must have same length")

    if np.any(area_array <= 0):
        raise ValueError("All drainage areas must be positive")

    # Filter for positive slopes only
    pos_slope_index = slope_array > 0

    if np.sum(pos_slope_index) < 2:
        raise ValueError("Need at least 2 positive slope values for regression")

    slope_pos = slope_array[pos_slope_index]
    area_pos = area_array[pos_slope_index]

    # Log-transform data
    log_area = np.log(area_pos)
    log_slope = np.log(slope_pos)

    # Linear regression
    slope_fit, intercept, r_value, p_value, std_err = stats.linregress(log_area, log_slope)

    # Concavity index m = -slope
    m = -slope_fit

    # Calculate confidence interval (95%)
    # For a linear model, the standard error on the slope gives CI
    t_crit = stats.t.ppf(0.975, len(log_area) - 2)  # 97.5% for two-tailed
    slope_ci = t_crit * std_err

    # Confidence intervals on m (flip signs because m = -slope)
    lower_bound = -(slope_fit - slope_ci)
    upper_bound = -(slope_fit + slope_ci)

    R2 = r_value**2

    # Plotting
    if to_plot:
        if fig is None:
            fig = plt.figure(figsize=(8, 6))

        ax = fig.add_subplot(1, 1, 1)

        # Plot data
        ax.plot(log_area, log_slope, 'xr', markersize=8, label='Data')

        # Plot fit line
        log_area_line = np.array([log_area.min(), log_area.max()])
        log_slope_line = slope_fit * log_area_line + intercept
        ax.plot(log_area_line, log_slope_line, 'k-', linewidth=2, label='Best fit')

        ax.set_xlabel('ln(Area)', fontsize=12)
        ax.set_ylabel('ln(Slope)', fontsize=12)
        ax.set_title('Slope-Area Relationship', fontsize=14)

        # Add text with results
        text_str = f'm = {m:.3f}, R² = {R2:.3f}'
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return m, lower_bound, upper_bound, R2
