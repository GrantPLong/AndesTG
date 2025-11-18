"""Bootstrap analysis for uncertainty quantification in block uplift inversion."""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from .invert_block_uplift import invert_block_uplift


def bootstrap_invert_block_uplift(
    chi: np.ndarray,
    z: np.ndarray,
    gamma: float,
    q: int,
    percent_sample: float,
    num_iterations: int,
    K: float = 1.0,
    to_plot: bool = True,
    fig: Optional[plt.Figure] = None,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform bootstrap analysis for block uplift inversion.

    Repeatedly inverts random subsets of data to quantify uncertainty
    in the uplift rate history.

    Parameters
    ----------
    chi : np.ndarray
        Chi coordinate values [L].
    z : np.ndarray
        Elevation data [L].
    gamma : float
        Damping coefficient.
    q : int
        Number of time intervals.
    percent_sample : float
        Fraction of data to sample in each iteration (0 < x ≤ 1).
    num_iterations : int
        Number of bootstrap iterations.
    K : float, optional
        Erodibility coefficient for plotting. Use 1.0 for non-dimensional.
        Default 1.0.
    to_plot : bool, optional
        If True, plot results. Default True.
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Ustar_mat : np.ndarray
        Matrix of inversion results (num_iterations × q).
        If K != 1, returns dimensional uplift rates.
    tstar_best : np.ndarray
        Time boundaries from full dataset inversion (q+1).

    Notes
    -----
    - Each iteration randomly samples percent_sample of the data
    - Final plot shows all bootstrap realizations (gray), mean (magenta),
      ±1σ bounds (dashed magenta), and best-fit from full dataset (black)
    - If K != 1, results are converted to dimensional units for plotting

    Examples
    --------
    >>> chi = np.linspace(0, 1000, 200)
    >>> z = 0.5 * chi + np.random.normal(0, 10, len(chi))
    >>> Ustar_mat, tstar = bootstrap_invert_block_uplift(
    ...     chi, z, gamma=1.0, q=3, percent_sample=0.8,
    ...     num_iterations=100, to_plot=False, random_seed=42
    ... )
    """
    # Input validation
    if len(chi) != len(z):
        raise ValueError("chi and z must have the same length")

    if percent_sample <= 0 or percent_sample > 1:
        raise ValueError("percent_sample must be between 0 and 1")

    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")

    if K <= 0:
        raise ValueError("K must be positive")

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Remove zero elevations
    non_zero_elements = z != 0
    nz_chi = chi[non_zero_elements]
    nz_z = z[non_zero_elements]
    data_length = len(nz_chi)

    if data_length == 0:
        raise ValueError("All elevation values are zero")

    sample_length = int(np.floor(data_length * percent_sample))

    if sample_length < q:
        raise ValueError(
            f"Sample size ({sample_length}) too small for q={q}. "
            "Increase percent_sample or decrease q."
        )

    # Initialize storage
    Ustar_mat = np.zeros((num_iterations, q))
    tstar_mat = np.zeros((num_iterations, q + 1))

    # Bootstrap iterations
    for i in range(num_iterations):
        # Random sample without replacement
        sample_indices = np.random.choice(data_length, sample_length, replace=False)
        sample_indices = np.sort(sample_indices)  # Sort for numerical stability

        try:
            Ustar, tstar, _ = invert_block_uplift(
                nz_chi[sample_indices],
                nz_z[sample_indices],
                gamma,
                q,
                to_plot=False
            )
            Ustar_mat[i, :] = Ustar
            tstar_mat[i, :] = tstar
        except Exception as e:
            print(f"Warning: Bootstrap iteration {i} failed: {e}")
            # Fill with NaN for failed iterations
            Ustar_mat[i, :] = np.nan
            tstar_mat[i, :] = np.nan

    # Inversion with all data (best fit)
    Ustar_best, tstar_best, _ = invert_block_uplift(nz_chi, nz_z, gamma, q, to_plot=False)

    # Plotting
    if to_plot:
        if fig is None:
            fig = plt.figure(figsize=(10, 6))

        ax = fig.add_subplot(1, 1, 1)

        # Plot all bootstrap realizations
        for i in range(num_iterations):
            if not np.any(np.isnan(Ustar_mat[i, :])):
                Ustar = Ustar_mat[i, :]
                tstar = tstar_mat[i, :]

                # Create staircase plot
                t_plot = []
                U_plot = []
                for j in range(q):
                    t_plot.extend([tstar[j], tstar[j+1]])
                    U_plot.extend([Ustar[j], Ustar[j]])

                # Convert to dimensional units if K != 1
                t_plot_dim = np.array(t_plot) / K / 1e6  # Convert to Ma
                U_plot_dim = np.array(U_plot) * K / 1e-3  # Convert to mm/yr

                ax.plot(t_plot_dim, U_plot_dim, color=[0.9, 0.9, 0.9], linewidth=1, alpha=0.5)

        # Plot best fit solution (black)
        t_best_plot = []
        U_best_plot = []
        for j in range(q):
            t_best_plot.extend([tstar_best[j], tstar_best[j+1]])
            U_best_plot.extend([Ustar_best[j], Ustar_best[j]])

        t_best_dim = np.array(t_best_plot) / K / 1e6
        U_best_dim = np.array(U_best_plot) * K / 1e-3

        ax.plot(t_best_dim, U_best_dim, 'k-', linewidth=2, label='Best fit (all data)')

        # Plot mean solution (magenta)
        valid_mask = ~np.isnan(Ustar_mat).any(axis=1)
        if np.sum(valid_mask) > 0:
            mean_Ustar = np.nanmean(Ustar_mat[valid_mask, :], axis=0)

            t_mean_plot = []
            U_mean_plot = []
            for j in range(q):
                t_mean_plot.extend([tstar_best[j], tstar_best[j+1]])
                U_mean_plot.extend([mean_Ustar[j], mean_Ustar[j]])

            t_mean_dim = np.array(t_mean_plot) / K / 1e6
            U_mean_dim = np.array(U_mean_plot) * K / 1e-3

            ax.plot(t_mean_dim, U_mean_dim, 'm-', linewidth=1.5, label='Mean')

            # Plot ±1σ bounds (dashed magenta)
            std_Ustar = np.nanstd(Ustar_mat[valid_mask, :], axis=0)

            t_std_plot = []
            U_up_plot = []
            U_down_plot = []
            for j in range(q):
                t_std_plot.extend([tstar_best[j], tstar_best[j+1]])
                U_up_plot.extend([mean_Ustar[j] + std_Ustar[j],
                                 mean_Ustar[j] + std_Ustar[j]])
                U_down_plot.extend([mean_Ustar[j] - std_Ustar[j],
                                   mean_Ustar[j] - std_Ustar[j]])

            t_std_dim = np.array(t_std_plot) / K / 1e6
            U_up_dim = np.array(U_up_plot) * K / 1e-3
            U_down_dim = np.array(U_down_plot) * K / 1e-3

            ax.plot(t_std_dim, U_up_dim, 'm:', linewidth=1.5, label='±1σ')
            ax.plot(t_std_dim, U_down_dim, 'm:', linewidth=1.5)

        ax.set_xlabel('t [Ma]', fontsize=14)
        ax.set_ylabel('U [mm/yr]', fontsize=14)
        ax.set_title(f'Bootstrap Analysis (n={num_iterations}, {percent_sample*100:.0f}% sampling)',
                    fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return Ustar_mat, tstar_best
