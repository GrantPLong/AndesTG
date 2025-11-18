"""Invert for spatially varying uplift with parabolic pattern.

This module implements inversion for uplift that varies spatially according
to a parabolic function: U(x,t) = a(t)*x² + b(t)*x + c(t).
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def invert_parabola(
    chi: np.ndarray,
    z: np.ndarray,
    x: np.ndarray,
    rec_array: np.ndarray,
    gamma: float,
    q: int,
    K: float = 1.0,
    to_plot: bool = False,
    fig: Optional[plt.Figure] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Invert for spatially varying uplift with parabolic pattern.

    Assumes uplift varies in space as a parabola: U(x,t) = a(t)*x² + b(t)*x + c(t)
    where x is the along-strike coordinate.

    Parameters
    ----------
    chi : np.ndarray
        Chi coordinate values [L] (length n).
    z : np.ndarray
        Elevation data [L] (length n).
    x : np.ndarray
        Along-strike coordinate [L] (length n). Direction where U changes.
    rec_array : np.ndarray
        Receiver relationships (length n).
    gamma : float
        Damping coefficient.
    q : int
        Number of time intervals.
    K : float, optional
        Erodibility coefficient [L^(1-2m)/T]. Use 1 for non-dimensional.
        Default 1.0.
    to_plot : bool, optional
        If True, plot 3D uplift surface. Default False.
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on.

    Returns
    -------
    Up : np.ndarray
        Parabolic uplift parameters (3q × 1). For each time interval i:
        Up[3*i] = a(i), Up[3*i+1] = b(i), Up[3*i+2] = c(i).
    tstar : np.ndarray
        Time interval boundaries (q+1).
    misfit : float
        Misfit [L]. Calculated as: sqrt(sum(residuals²)) / (N-3q)
        Note: This differs from standard RMSE to match MATLAB implementation.

    Notes
    -----
    - x coordinates are converted to km for numerical stability
    - Time intervals contain equal numbers of data points
    - Uses Tikhonov regularization like block uplift inversion
    - Solves for 3q parameters (3 parabola coefficients per time interval)

    Mathematical Formulation
    ------------------------
    Forward model: z = A * Bp * Up
    where:
    - A is the standard temporal integration matrix (N × Nq)
    - Bp maps parabola parameters to pixel uplift rates (Nq × 3q)
    - Up contains parabola coefficients (3q × 1)
    - Ap = A * Bp is the combined forward operator (N × 3q)

    Examples
    --------
    >>> chi = np.linspace(0, 1000, 100)
    >>> x = np.linspace(0, 50e3, 100)  # 50 km
    >>> z = 0.5 * chi  # Simplified example
    >>> rec_array = np.arange(100) - 1
    >>> rec_array[0] = 0
    >>> Up, tstar, misfit = invert_parabola(
    ...     chi, z, x, rec_array, gamma=1.0, q=2, to_plot=False
    ... )
    """
    # Input validation
    n = len(chi)
    if not (len(z) == len(x) == len(rec_array) == n):
        raise ValueError("All input arrays must have the same length")

    if gamma <= 0:
        raise ValueError("gamma must be positive")

    if q <= 0:
        raise ValueError("q must be positive")

    if K <= 0:
        raise ValueError("K must be positive")

    # Remove zero elevations
    non_zero_elements = z != 0
    nz_chi = chi[non_zero_elements]
    nz_z = z[non_zero_elements]
    nz_x = x[non_zero_elements]
    nz_id = np.arange(n)[non_zero_elements]

    N = len(nz_chi)

    if N == 0:
        raise ValueError("All elevation values are zero")

    if 3 * q > N:
        raise ValueError(f"3*q ({3*q}) exceeds number of non-zero data points ({N})")

    # Sort by chi
    id_chi_z_mat = np.column_stack([nz_id, nz_chi, nz_z])
    sorted_indices = np.argsort(id_chi_z_mat[:, 1])
    sorted_id_chi_z_mat = id_chi_z_mat[sorted_indices]
    sorted_chi = sorted_id_chi_z_mat[:, 1]

    # Construct time intervals with equal pixels
    val_per_dt = N // q

    scaled_t_vec = np.zeros(q + 1)
    scaled_t_vec[0] = 0
    for i in range(1, q):
        scaled_t_vec[i] = sorted_chi[(i) * val_per_dt - 1]
    scaled_t_vec[q] = sorted_chi[-1]

    scaled_dt_vec = np.diff(scaled_t_vec)

    # Build space-time A matrix (similar to block uplift but more complex)
    # A[pixel, time*pixel_within_time] represents contribution
    A = np.zeros((N, N * q))

    for i in range(N):
        global_time = nz_chi[i]
        time_int = 0
        local_time = scaled_dt_vec[time_int] if time_int < q else 0

        id_curr = nz_id[i]
        curr_ind = i

        # Follow flow path backwards in time
        while global_time > 1e-12 and time_int < q:
            col = time_int * N + curr_ind

            # Get receiver
            id_rec = rec_array[id_curr]
            next_ind_array = np.where(nz_id == id_rec)[0]

            if len(next_ind_array) == 0:
                chi_next = 0
                next_ind = curr_ind  # Stay at current if no receiver found
            else:
                next_ind = next_ind_array[0]
                chi_next = nz_chi[next_ind]

            # Calculate contribution to this time interval
            if global_time - chi_next >= local_time + 1e-10:
                A[i, col] = local_time
                global_time -= local_time
                time_int += 1
                if time_int < len(scaled_dt_vec):
                    local_time = scaled_dt_vec[time_int]
            else:
                A[i, col] = global_time - chi_next
                local_time -= (global_time - chi_next)
                global_time = chi_next
                curr_ind = next_ind
                id_curr = id_rec

        # Verify column sum
        if abs(np.sum(A[i, :]) - nz_chi[i]) > 1e-10:
            print(f"Warning: A matrix sum mismatch for pixel {i}")

    # Build parabola matrix Bp
    # Bp[space_time_index, parabola_param] = spatial pattern
    Bp = np.zeros((q * N, 3 * q))

    # Convert x to km for numerical stability
    x_km = nz_x / 1e3

    for i in range(q):
        for j in range(N):
            row = i * N + j
            col_base = i * 3

            Bp[row, col_base] = x_km[j]**2      # a(t) * x²
            Bp[row, col_base + 1] = x_km[j]     # b(t) * x
            Bp[row, col_base + 2] = 1.0         # c(t)

    # Combined forward operator
    Ap = A @ Bp

    # Prior model (constant mean uplift)
    valid_chi = nz_chi > 1e-10
    if np.sum(valid_chi) > 0:
        mean_U = np.mean(nz_z[valid_chi] / nz_chi[valid_chi])
    else:
        mean_U = np.mean(nz_z)

    U_pri_star = np.ones(3 * q) * mean_U

    # Solve regularized least squares
    # min ||Ap*Up - z||² + gamma² ||Up - U_prior||²
    ApTAp = Ap.T @ Ap
    denom = ApTAp + gamma**2 * np.eye(3 * q)
    residual = nz_z - Ap @ U_pri_star
    nom = Ap.T @ residual

    Up = U_pri_star + np.linalg.solve(denom, nom)
    tstar = scaled_t_vec

    # Calculate misfit (matches MATLAB formula exactly)
    model_z = Ap @ Up
    residuals = nz_z - model_z
    # MATLAB: Misfit = 1/((N-3*q))*sqrt(sum((nz_z - Ap*Up).^2))
    misfit = np.sqrt(np.sum(residuals**2)) / (N - 3*q)

    # Plotting
    if to_plot:
        if fig is None:
            fig = plt.figure(figsize=(12, 8))

        ax = fig.add_subplot(111, projection='3d')

        # Create uplift rate surface
        x_domain = np.linspace(np.min(nz_x)/1e3, np.max(nz_x)/1e3, 50)
        U_space_time_mat = np.zeros((q, len(x_domain)))

        for i in range(q):
            a_t = Up[3*i]
            b_t = Up[3*i + 1]
            c_t = Up[3*i + 2]
            U_space_time_mat[i, :] = a_t * x_domain**2 + b_t * x_domain + c_t

        # Create staircase in time
        t_plot = []
        U_mat_plot = []
        for i in range(q):
            t_plot.extend([tstar[i], tstar[i+1]])
            U_mat_plot.append(U_space_time_mat[i, :])
            U_mat_plot.append(U_space_time_mat[i, :])

        t_plot = np.array(t_plot)
        U_mat_plot = np.array(U_mat_plot)

        # Create meshgrid
        X, T = np.meshgrid(x_domain, t_plot)

        # Convert to dimensional units if K != 1
        if K == 1:
            T_plot = T
            U_plot = U_mat_plot
            ax.set_ylabel('Scaled t [m]')
            ax.set_zlabel('Non-dimensional Uplift Rate')
        else:
            T_plot = T / K / 1e6  # Convert to Ma
            U_plot = U_mat_plot * K / 1e-3  # Convert to mm/yr
            ax.set_ylabel('Time [Ma]')
            ax.set_zlabel('Rock Uplift Rate [mm/yr]')

        # Plot surface
        surf = ax.plot_surface(X, T_plot, U_plot, cmap='viridis',
                              edgecolor='none', alpha=0.8)

        ax.set_xlabel('x [km]')
        ax.set_title('Spatially Varying Uplift History', fontsize=14)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        plt.tight_layout()
        plt.show()

    return Up, tstar, misfit
