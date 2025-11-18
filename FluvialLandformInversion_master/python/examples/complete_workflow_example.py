"""Complete workflow example for fluvial landform inversion.

This script demonstrates a complete analysis workflow from loading data
to final calibrated uplift rate history with uncertainty quantification.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fluvial_inversion import (
    calculate_chi,
    invert_block_uplift,
    findm_slope_area,
    findm_linear_chi,
    findm_collapse_chi,
    invert_with_different_gamma,
    calibrate_k_total_uplift,
    bootstrap_invert_block_uplift
)


def complete_workflow():
    """Demonstrate complete inversion workflow."""

    print("\n" + "="*70)
    print("  COMPLETE FLUVIAL INVERSION WORKFLOW")
    print("="*70 + "\n")

    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("STEP 1: Loading fluvial network data")
    print("-" * 70)

    matlab_data = sio.loadmat('../../matlab/BlockUpliftDataHighRes.mat')

    x = matlab_data['x'].flatten()
    y = matlab_data['y'].flatten()
    z = matlab_data['z'].flatten()
    area_array = matlab_data['area_array'].flatten()
    rec_array = matlab_data['rec_array'].flatten().astype(int) - 1
    slope_array = matlab_data['slope_array'].flatten()

    print(f"✓ Loaded {len(x)} pixels")
    print(f"  Elevation range: [{z.min():.1f}, {z.max():.1f}] m")
    print(f"  Drainage area range: [{area_array.min():.1e}, {area_array.max():.1e}] m²")

    # ========================================================================
    # STEP 2: Determine Concavity Index
    # ========================================================================
    print("\nSTEP 2: Determining concavity index (m)")
    print("-" * 70)

    # Method 1: Slope-area relationship (fastest)
    m1, lb, ub, R2 = findm_slope_area(slope_array, area_array, to_plot=False)
    print(f"Method 1 - Slope-Area: m = {m1:.4f} (R² = {R2:.4f})")

    # Method 2: Linear chi-z (more robust for noisy data)
    # m2, _ = findm_linear_chi(x, y, z, rec_array, area_array, to_plot=False)
    # print(f"Method 2 - Linear Chi-Z: m = {m2:.4f}")

    # Method 3: Tributary collapse (best for multiple tributaries)
    # m3, _ = findm_collapse_chi(x, y, z, rec_array, area_array, to_plot=False)
    # print(f"Method 3 - Collapse: m = {m3:.4f}")

    # Use m from slope-area
    m = m1
    print(f"\n✓ Using m = {m:.4f}")

    # ========================================================================
    # STEP 3: Calculate Chi Coordinate
    # ========================================================================
    print("\nSTEP 3: Calculating chi coordinate")
    print("-" * 70)

    chi = calculate_chi(x, y, rec_array, area_array, m, A0=1.0)
    print(f"✓ Chi calculated: range [{chi.min():.2f}, {chi.max():.2f}] m")

    # ========================================================================
    # STEP 4: Select Regularization Parameter (Gamma)
    # ========================================================================
    print("\nSTEP 4: Selecting regularization parameter (gamma)")
    print("-" * 70)

    print("Generating L-curve...")
    gamma_vec, misfit_vec = invert_with_different_gamma(
        chi, z, q=5,
        gamma_range=np.logspace(-1, 2, 20),
        to_plot=False
    )

    # Find "elbow" of L-curve (simple heuristic: max curvature)
    inv_gamma = 1.0 / gamma_vec
    valid = ~np.isnan(misfit_vec)

    # Use gamma = 1.0 as a reasonable default
    gamma_optimal = 1.0
    print(f"✓ Selected gamma = {gamma_optimal}")

    # ========================================================================
    # STEP 5: Perform Block Uplift Inversion
    # ========================================================================
    print("\nSTEP 5: Inverting for uplift rate history")
    print("-" * 70)

    q = 5  # Number of time intervals
    print(f"Running inversion with q = {q} time intervals...")

    Ustar, tstar, misfit = invert_block_uplift(
        chi, z, gamma=gamma_optimal, q=q, to_plot=False
    )

    print(f"\n✓ Inversion complete")
    print(f"  RMS misfit: {misfit:.2f} m")
    print(f"\n  Non-dimensional uplift rates (U*):")
    for i, u in enumerate(Ustar):
        print(f"    Interval {i+1}: {u:.6f}")

    # ========================================================================
    # STEP 6: Calibrate to Dimensional Units
    # ========================================================================
    print("\nSTEP 6: Calibrating to dimensional units")
    print("-" * 70)

    # Example: 500m total uplift over 2 Ma
    H_total = 500.0  # m
    t_H = 2e6        # years
    A0 = 1.0

    print(f"Using constraint: {H_total} m uplift over {t_H/1e6:.1f} Ma")

    K, U, t = calibrate_k_total_uplift(
        H_total, t_H, Ustar, tstar, A0, m, to_plot=False
    )

    print(f"\n✓ Calibration complete")
    print(f"  Erodibility coefficient K = {K:.4e} m^(1-2m)/yr")
    print(f"\n  Dimensional uplift rates [mm/yr]:")
    for i, u_dim in enumerate(U):
        t_start = t[i] / 1e6  # Convert to Ma
        t_end = t[i+1] / 1e6
        print(f"    {t_start:.2f} - {t_end:.2f} Ma: {u_dim*1e3:.4f} mm/yr")

    # ========================================================================
    # STEP 7: Quantify Uncertainty (Bootstrap)
    # ========================================================================
    print("\nSTEP 7: Quantifying uncertainty with bootstrap")
    print("-" * 70)

    print("Running bootstrap analysis (50 iterations)...")

    Ustar_mat, tstar_best = bootstrap_invert_block_uplift(
        chi, z,
        gamma=gamma_optimal,
        q=3,  # Use fewer intervals for faster bootstrap
        percent_sample=0.8,
        num_iterations=50,
        K=K,
        to_plot=False,
        random_seed=42
    )

    # Calculate statistics
    mean_U = np.nanmean(Ustar_mat, axis=0)
    std_U = np.nanstd(Ustar_mat, axis=0)

    print(f"\n✓ Bootstrap complete")
    print(f"\n  Uplift rates with uncertainty [mm/yr]:")
    for i in range(len(mean_U)):
        mean_rate = mean_U[i] * K * 1e3
        std_rate = std_U[i] * K * 1e3
        print(f"    Interval {i+1}: {mean_rate:.4f} ± {std_rate:.4f} mm/yr")

    # ========================================================================
    # STEP 8: Visualize Results
    # ========================================================================
    print("\nSTEP 8: Creating summary visualizations")
    print("-" * 70)

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Chi-elevation relationship
    ax1 = fig.add_subplot(2, 3, 1)
    non_zero = z > 0
    ax1.scatter(chi[non_zero], z[non_zero], c='blue', alpha=0.3, s=1)
    ax1.set_xlabel('χ [m]', fontsize=12)
    ax1.set_ylabel('Elevation [m]', fontsize=12)
    ax1.set_title('Chi-Elevation Relationship', fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Slope-area relationship
    ax2 = fig.add_subplot(2, 3, 2)
    pos_slope = slope_array > 0
    ax2.scatter(np.log(area_array[pos_slope]),
               np.log(slope_array[pos_slope]),
               c='red', alpha=0.3, s=1)
    ax2.set_xlabel('ln(Area)', fontsize=12)
    ax2.set_ylabel('ln(Slope)', fontsize=12)
    ax2.set_title(f'Slope-Area (m = {m:.3f})', fontsize=13)
    ax2.grid(True, alpha=0.3)

    # Plot 3: L-curve
    ax3 = fig.add_subplot(2, 3, 3)
    valid = ~np.isnan(misfit_vec)
    ax3.plot(1.0/gamma_vec[valid], misfit_vec[valid], 'b-', linewidth=2)
    ax3.axvline(1.0/gamma_optimal, color='r', linestyle='--',
               label=f'Selected: γ={gamma_optimal}')
    ax3.set_xlabel('1/γ', fontsize=12)
    ax3.set_ylabel('Misfit [m]', fontsize=12)
    ax3.set_title('L-Curve', fontsize=13)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Non-dimensional uplift history
    ax4 = fig.add_subplot(2, 3, 4)
    t_plot = []
    U_plot = []
    for i in range(len(Ustar)):
        t_plot.extend([tstar[i], tstar[i+1]])
        U_plot.extend([Ustar[i], Ustar[i]])
    ax4.plot(t_plot, U_plot, 'b-', linewidth=2)
    ax4.set_xlabel('t* [m]', fontsize=12)
    ax4.set_ylabel('U*', fontsize=12)
    ax4.set_title('Non-dimensional Uplift History', fontsize=13)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Dimensional uplift history
    ax5 = fig.add_subplot(2, 3, 5)
    t_plot_dim = []
    U_plot_dim = []
    for i in range(len(U)):
        t_plot_dim.extend([t[i]/1e6, t[i+1]/1e6])
        U_plot_dim.extend([U[i]*1e3, U[i]*1e3])
    ax5.plot(t_plot_dim, U_plot_dim, 'g-', linewidth=2)
    ax5.set_xlabel('Time [Ma]', fontsize=12)
    ax5.set_ylabel('Uplift Rate [mm/yr]', fontsize=12)
    ax5.set_title('Dimensional Uplift History', fontsize=13)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Bootstrap uncertainty
    ax6 = fig.add_subplot(2, 3, 6)
    # Show bootstrap distributions for each interval
    for i in range(Ustar_mat.shape[1]):
        valid_samples = ~np.isnan(Ustar_mat[:, i])
        rates = Ustar_mat[valid_samples, i] * K * 1e3
        ax6.violinplot([rates], positions=[i+1], widths=0.7,
                      showmeans=True, showmedians=True)
    ax6.set_xlabel('Time Interval', fontsize=12)
    ax6.set_ylabel('Uplift Rate [mm/yr]', fontsize=12)
    ax6.set_title('Bootstrap Uncertainty', fontsize=13)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('inversion_results.png', dpi=150)
    print("✓ Plots saved to 'inversion_results.png'")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("  WORKFLOW COMPLETE!")
    print("="*70)
    print("\nKey Results:")
    print(f"  • Concavity index: m = {m:.4f}")
    print(f"  • Erodibility: K = {K:.4e} m^(1-2m)/yr")
    print(f"  • Inversion misfit: {misfit:.2f} m")
    print(f"  • Mean uplift rate: {np.mean(U)*1e3:.4f} mm/yr")
    print(f"  • Uplift rate range: {np.min(U)*1e3:.4f} - {np.max(U)*1e3:.4f} mm/yr")
    print("\nOutput:")
    print("  • Numerical results displayed above")
    print("  • Visualization saved to 'inversion_results.png'")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    complete_workflow()
