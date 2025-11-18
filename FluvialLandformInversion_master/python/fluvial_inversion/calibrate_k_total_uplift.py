"""Calibrate erodibility coefficient from total uplift observations."""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def calibrate_k_total_uplift(
    H: float,
    t_H: float,
    Ustar: np.ndarray,
    tstar: np.ndarray,
    A0: float,
    m: float,
    to_plot: bool = False,
    fig: Optional[plt.Figure] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calibrate erodibility coefficient K from total uplift data.

    Converts non-dimensional inversion results to dimensional units using
    constraints from dated uplifted features.

    Parameters
    ----------
    H : float
        Total uplift [L] from present to time t_H in the past.
    t_H : float
        Age [T] of the dated uplifted feature.
    Ustar : np.ndarray
        Non-dimensional uplift rate history from inversion (length q).
    tstar : np.ndarray
        Scaled time boundaries from inversion (length q+1).
    A0 : float
        Reference drainage area used in chi calculation [L²].
    m : float
        Area exponent used in chi calculation.
    to_plot : bool, optional
        If True, plot dimensional uplift history. Default False.
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on.

    Returns
    -------
    K : float
        Erodibility coefficient [L^(1-2m)/T].
    U : np.ndarray
        Dimensional uplift rate history [L/T] (length q).
    t : np.ndarray
        Dimensional time boundaries [T] (length q+1).

    Notes
    -----
    - Assumes block uplift conditions
    - Uses the relationship: K = t*_H / (t_H * A0^m)
      where t*_H is the scaled time corresponding to total uplift H
    - Total uplift H is calculated as: H = Σ(U* · Δt*)

    Examples
    --------
    >>> Ustar = np.array([0.6, 0.4, 0.3])
    >>> tstar = np.array([0, 300, 600, 1000])
    >>> K, U, t = calibrate_k_total_uplift(
    ...     H=500, t_H=1e6, Ustar=Ustar, tstar=tstar,
    ...     A0=1.0, m=0.45, to_plot=False
    ... )
    """
    # Input validation
    if H <= 0:
        raise ValueError("Total uplift H must be positive")

    if t_H <= 0:
        raise ValueError("Age t_H must be positive")

    if len(Ustar) != len(tstar) - 1:
        raise ValueError("Ustar must have length q, tstar must have length q+1")

    if A0 <= 0:
        raise ValueError("Reference area A0 must be positive")

    if m < 0 or m > 1:
        raise ValueError("m should typically be between 0 and 1")

    # Calculate time interval lengths
    del_t_star = np.diff(tstar)

    # Calculate cumulative uplift for each time interval
    testH_vec = np.cumsum(Ustar * del_t_star)

    # Find the time interval where cumulative uplift exceeds H
    time_ind = np.searchsorted(testH_vec, H, side='right')

    if time_ind == 0:
        # H is achieved before first interval ends
        # Use simple proportionality
        if Ustar[0] > 0:
            del_scaled_time = H / Ustar[0]
            t_H_star = tstar[0] + del_scaled_time
        else:
            raise ValueError("Cannot calibrate: first uplift rate is zero or negative")

    elif time_ind >= len(Ustar):
        # H exceeds total modeled uplift
        raise ValueError(
            f"Total uplift H={H} exceeds modeled uplift {testH_vec[-1]:.2f}. "
            "Cannot calibrate K."
        )

    else:
        # H is achieved partway through interval time_ind
        rem_H = H - testH_vec[time_ind - 1]

        if Ustar[time_ind] > 0:
            del_scaled_time = rem_H / Ustar[time_ind]
            t_H_star = tstar[time_ind] + del_scaled_time
        else:
            # Uplift rate is zero or negative in this interval
            # This is problematic - use end of interval
            t_H_star = tstar[time_ind + 1]

    # Calculate erodibility coefficient
    K = t_H_star / (t_H * A0**m)

    # Convert to dimensional units
    U = Ustar * K
    t = tstar / K

    # Plotting
    if to_plot:
        if fig is None:
            fig = plt.figure(figsize=(8, 6))

        ax = fig.add_subplot(1, 1, 1)

        # Create staircase plot
        t_plot = []
        U_plot = []
        for i in range(len(U)):
            t_plot.extend([t[i], t[i+1]])
            U_plot.extend([U[i], U[i]])

        ax.plot(t_plot, U_plot, 'b-', linewidth=2)
        ax.set_xlabel('t [yr]', fontsize=14)
        ax.set_ylabel('U [m/yr]', fontsize=14)
        ax.set_title('Dimensional Uplift Rate History', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add calibration info
        text_str = f'K = {K:.2e} [L^(1-2m)/T]\nm = {m:.3f}'
        ax.text(0.95, 0.95, text_str, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

    return K, U, t
