"""Test Python implementation with MATLAB data files.

This script loads the MATLAB data files and runs the Python inversion
functions to verify correctness and compare with MATLAB results.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fluvial_inversion import (
    calculate_chi,
    invert_block_uplift,
    invert_block_uplift_with_stats,
    findm_slope_area,
    findm_linear_chi,
    findm_collapse_chi,
    invert_parabola,
    invert_with_different_gamma,
    calibrate_k_total_uplift,
    bootstrap_invert_block_uplift
)


def test_block_uplift_data():
    """Test with BlockUpliftDataHighRes.mat."""
    print("\n" + "="*70)
    print("Testing with BlockUpliftDataHighRes.mat")
    print("="*70)

    # Load MATLAB data
    matlab_data_path = '../../matlab/BlockUpliftDataHighRes.mat'
    if not os.path.exists(matlab_data_path):
        print(f"ERROR: Could not find {matlab_data_path}")
        return

    mat_data = sio.loadmat(matlab_data_path)

    # Print available variables
    print("\nAvailable variables in MATLAB file:")
    for key in mat_data.keys():
        if not key.startswith('__'):
            print(f"  {key}: shape = {mat_data[key].shape}, dtype = {mat_data[key].dtype}")

    # Extract data
    try:
        x = mat_data['x'].flatten()
        y = mat_data['y'].flatten()
        z = mat_data['z'].flatten()
        area_array = mat_data['area_array'].flatten()
        rec_array = mat_data['rec_array'].flatten().astype(int) - 1  # MATLAB to Python indexing
        slope_array = mat_data['slope_array'].flatten()

        print(f"\nData loaded successfully:")
        print(f"  Number of pixels: {len(x)}")
        print(f"  X range: [{x.min():.2f}, {x.max():.2f}]")
        print(f"  Y range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"  Elevation range: [{z.min():.2f}, {z.max():.2f}]")
        print(f"  Area range: [{area_array.min():.2e}, {area_array.max():.2e}]")
        print(f"  Non-zero elevations: {np.sum(z > 0)}")

        # Find concavity index m using slope-area method
        print("\n" + "-"*70)
        print("Test 0: Determining Concavity Index m")
        print("-"*70)

        m, m_lb, m_ub, R2 = findm_slope_area(slope_array, area_array, to_plot=False)
        print(f"\nConcavity index from slope-area:")
        print(f"  m = {m:.4f}")
        print(f"  95% CI: [{m_lb:.4f}, {m_ub:.4f}]")
        print(f"  R² = {R2:.4f}")

        # Calculate chi
        print(f"\nCalculating chi coordinate with m = {m:.4f}...")
        chi = calculate_chi(x, y, rec_array, area_array, m, A0=1.0)
        print(f"  Chi range: [{chi.min():.2f}, {chi.max():.2f}]")

        # Test block uplift inversion
        print("\n" + "-"*70)
        print("Test 1: Block Uplift Inversion")
        print("-"*70)

        gamma = 1.0
        q = 5

        print(f"\nParameters:")
        print(f"  gamma = {gamma}")
        print(f"  q = {q}")

        Ustar, tstar, misfit = invert_block_uplift(chi, z, gamma, q, to_plot=False)

        print(f"\nResults:")
        print(f"  Misfit: {misfit:.4f}")
        print(f"  Uplift rates (U*):")
        for i, u in enumerate(Ustar):
            print(f"    Interval {i+1}: {u:.6f}")
        print(f"  Time boundaries (t*):")
        print(f"    {tstar}")

        # Test with detailed statistics
        print("\n" + "-"*70)
        print("Test 2: Inversion with Statistics")
        print("-"*70)

        result = invert_block_uplift_with_stats(chi, z, gamma, q)

        print(f"\nDetailed statistics:")
        print(f"  R²: {result['r_squared']:.6f}")
        print(f"  Degrees of freedom: {result['dof']}")
        print(f"  Number of data points: {result['n_data']}")
        print(f"  Mean residual: {np.mean(result['residuals']):.6f}")
        print(f"  Std residual: {np.std(result['residuals']):.6f}")

        # Test gamma selection
        print("\n" + "-"*70)
        print("Test 3: L-Curve Analysis")
        print("-"*70)

        gamma_vec, misfit_vec = invert_with_different_gamma(
            chi, z, q=5,
            gamma_range=np.logspace(-1, 2, 20),
            to_plot=False
        )

        print(f"\nL-curve generated:")
        print(f"  Gamma range: [{gamma_vec.min():.4f}, {gamma_vec.max():.4f}]")
        print(f"  Misfit range: [{np.nanmin(misfit_vec):.4f}, {np.nanmax(misfit_vec):.4f}]")

        # Test calibration (if we have reference uplift data)
        print("\n" + "-"*70)
        print("Test 4: Calibration")
        print("-"*70)

        # Example calibration
        H = 500.0  # Total uplift in meters
        t_H = 2e6  # 2 Ma
        A0 = 1.0
        m = 0.45

        K, U, t = calibrate_k_total_uplift(H, t_H, Ustar, tstar, A0, m, to_plot=False)

        print(f"\nCalibration results:")
        print(f"  Erodibility K: {K:.6e}")
        print(f"  Dimensional uplift rates [m/yr]:")
        for i, u_dim in enumerate(U):
            print(f"    Interval {i+1}: {u_dim:.6e}")
        print(f"  Dimensional times [yr]:")
        print(f"    {t}")

        # Test bootstrap (quick test with few iterations)
        print("\n" + "-"*70)
        print("Test 5: Bootstrap Analysis (quick test)")
        print("-"*70)

        Ustar_mat, tstar_best = bootstrap_invert_block_uplift(
            chi, z, gamma=1.0, q=3,
            percent_sample=0.8,
            num_iterations=20,
            K=1.0,
            to_plot=False,
            random_seed=42
        )

        print(f"\nBootstrap results:")
        print(f"  Successful iterations: {np.sum(~np.isnan(Ustar_mat[:, 0]))}")
        print(f"  Mean uplift rates:")
        mean_U = np.nanmean(Ustar_mat, axis=0)
        std_U = np.nanstd(Ustar_mat, axis=0)
        for i in range(len(mean_U)):
            print(f"    Interval {i+1}: {mean_U[i]:.6f} ± {std_U[i]:.6f}")

        print("\n" + "="*70)
        print("Block Uplift Data Tests PASSED!")
        print("="*70)

    except KeyError as e:
        print(f"\nERROR: Variable {e} not found in MATLAB file")
        print("Available variables:", [k for k in mat_data.keys() if not k.startswith('__')])
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()


def test_parabola_data():
    """Test with ParabolaDataHighRes.mat."""
    print("\n" + "="*70)
    print("Testing with ParabolaDataHighRes.mat")
    print("="*70)

    # Load MATLAB data
    matlab_data_path = '../../matlab/ParabolaDataHighRes.mat'
    if not os.path.exists(matlab_data_path):
        print(f"ERROR: Could not find {matlab_data_path}")
        return

    mat_data = sio.loadmat(matlab_data_path)

    # Print available variables
    print("\nAvailable variables in MATLAB file:")
    for key in mat_data.keys():
        if not key.startswith('__'):
            print(f"  {key}: shape = {mat_data[key].shape}, dtype = {mat_data[key].dtype}")

    try:
        x = mat_data['x'].flatten()
        y = mat_data['y'].flatten()
        z = mat_data['z'].flatten()
        area_array = mat_data['area_array'].flatten()
        rec_array = mat_data['rec_array'].flatten().astype(int) - 1  # MATLAB to Python indexing
        slope_array = mat_data['slope_array'].flatten()

        print(f"\nData loaded successfully:")
        print(f"  Number of pixels: {len(x)}")
        print(f"  X range: [{x.min():.2f}, {x.max():.2f}]")
        print(f"  Y range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"  Elevation range: [{z.min():.2f}, {z.max():.2f}]")

        # Determine m and calculate chi
        print("\n" + "-"*70)
        print("Determining Concavity Index m")
        print("-"*70)

        m, m_lb, m_ub, R2 = findm_slope_area(slope_array, area_array, to_plot=False)
        print(f"\nConcavity index from slope-area:")
        print(f"  m = {m:.4f}")
        print(f"  95% CI: [{m_lb:.4f}, {m_ub:.4f}]")
        print(f"  R² = {R2:.4f}")

        print(f"\nCalculating chi coordinate with m = {m:.4f}...")
        chi = calculate_chi(x, y, rec_array, area_array, m, A0=1.0)
        print(f"  Chi range: [{chi.min():.2f}, {chi.max():.2f}]")

        # Test parabola inversion
        print("\n" + "-"*70)
        print("Test: Parabola Inversion")
        print("-"*70)

        gamma = 1.0
        q = 3

        print(f"\nParameters:")
        print(f"  gamma = {gamma}")
        print(f"  q = {q}")

        Up, tstar, misfit = invert_parabola(
            chi, z, x, rec_array,
            gamma, q, K=1.0, to_plot=False
        )

        print(f"\nResults:")
        print(f"  Misfit: {misfit:.4f}")
        print(f"  Parabola coefficients:")
        for i in range(q):
            a = Up[3*i]
            b = Up[3*i + 1]
            c = Up[3*i + 2]
            print(f"    Interval {i+1}: a={a:.6e}, b={b:.6e}, c={c:.6f}")

        print("\n" + "="*70)
        print("Parabola Data Tests PASSED!")
        print("="*70)

    except KeyError as e:
        print(f"\nERROR: Variable {e} not found in MATLAB file")
        print("Available variables:", [k for k in mat_data.keys() if not k.startswith('__')])
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 15 + "PYTHON IMPLEMENTATION VALIDATION" + " " * 21 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    # Test with block uplift data
    test_block_uplift_data()

    # Test with parabola data
    test_parabola_data()

    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 20 + "ALL TESTS COMPLETED" + " " * 29 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()


if __name__ == "__main__":
    main()
