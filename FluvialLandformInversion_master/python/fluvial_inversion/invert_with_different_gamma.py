"""Generate L-curve for selecting optimal damping coefficient."""

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from .invert_block_uplift import invert_block_uplift


def invert_with_different_gamma(
    chi: np.ndarray,
    z: np.ndarray,
    q: int,
    gamma_range: Optional[np.ndarray] = None,
    to_plot: bool = True,
    fig: Optional[plt.Figure] = None
) -> tuple:
    """Generate L-curve to select optimal damping coefficient.

    Tests a range of gamma values and plots the trade-off between
    model roughness (1/gamma) and data misfit.

    Parameters
    ----------
    chi : np.ndarray
        Chi coordinate values [L].
    z : np.ndarray
        Elevation data [L].
    q : int
        Number of time intervals in inversion.
    gamma_range : np.ndarray, optional
        Array of gamma values to test. Default is logspace(-1, 2, 100).
    to_plot : bool, optional
        If True, plot L-curve. Default True.
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on.

    Returns
    -------
    gamma_vec : np.ndarray
        Tested gamma values.
    misfit_vec : np.ndarray
        Corresponding misfit values.

    Notes
    -----
    - L-curve shows trade-off between smoothness and data fit
    - Optimal gamma is typically at the "elbow" of the L-curve
    - Smaller gamma = rougher model, better data fit
    - Larger gamma = smoother model, worse data fit

    Examples
    --------
    >>> chi = np.linspace(0, 1000, 100)
    >>> z = 0.5 * chi + np.random.normal(0, 10, len(chi))
    >>> gamma_vec, misfit = invert_with_different_gamma(
    ...     chi, z, q=5, to_plot=False
    ... )
    """
    # Input validation
    if len(chi) != len(z):
        raise ValueError("chi and z must have the same length")

    if q <= 0:
        raise ValueError("q must be positive")

    # Set default gamma range
    if gamma_range is None:
        gamma_vec = np.logspace(-1, 2, 100)
    else:
        gamma_vec = gamma_range

    misfit_vec = np.zeros(len(gamma_vec))

    # Test each gamma value
    for i, gamma in enumerate(gamma_vec):
        try:
            _, _, misfit = invert_block_uplift(chi, z, gamma, q, to_plot=False)
            misfit_vec[i] = misfit
        except Exception as e:
            # If inversion fails, set misfit to NaN
            misfit_vec[i] = np.nan
            print(f"Warning: Inversion failed for gamma={gamma}: {e}")

    # Plotting
    if to_plot:
        if fig is None:
            fig = plt.figure(figsize=(8, 6))

        ax = fig.add_subplot(1, 1, 1)

        # Filter out NaN values
        valid = ~np.isnan(misfit_vec)

        # Plot L-curve
        ax.plot(1.0 / gamma_vec[valid], misfit_vec[valid], 'b-', linewidth=2)
        ax.set_xlabel('1/Γ', fontsize=14)
        ax.set_ylabel('Misfit [m]', fontsize=14)
        ax.set_title('L-Curve for Regularization Parameter Selection', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Set reasonable axis limits
        if np.sum(valid) > 0:
            max_inv_gamma = np.max(1.0 / gamma_vec[valid])
            max_misfit = np.max(misfit_vec[valid])
            ax.set_xlim([-1, max_inv_gamma * 1.1])
            ax.set_ylim([0, max_misfit * 1.1])

        # Add guidance text
        text_str = ('Choose Γ at the "elbow"\n'
                   'where curve transitions\n'
                   'from steep to flat')
        ax.text(0.95, 0.05, text_str, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        plt.tight_layout()
        plt.show()

    return gamma_vec, misfit_vec
