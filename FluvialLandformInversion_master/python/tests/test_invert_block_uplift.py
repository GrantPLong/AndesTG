"""Unit tests for invert_block_uplift function."""

import numpy as np
import pytest
from fluvial_inversion.invert_block_uplift import (
    invert_block_uplift,
    invert_block_uplift_with_stats
)


class TestInvertBlockUplift:
    """Test suite for invert_block_uplift function."""

    def test_simple_constant_uplift(self):
        """Test inversion with synthetic constant uplift."""
        # Create synthetic data: z = U * chi for constant U
        chi = np.linspace(0, 1000, 100)
        U_true = 0.5
        z = U_true * chi

        # Invert with single time interval
        Ustar, tstar, misfit = invert_block_uplift(chi, z, gamma=0.1, q=1)

        # Should recover approximately constant uplift
        assert len(Ustar) == 1
        assert len(tstar) == 2
        np.testing.assert_allclose(Ustar[0], U_true, rtol=0.01)
        assert misfit < 1.0  # Low misfit for perfect data

    def test_two_stage_uplift(self):
        """Test inversion with two-stage uplift history."""
        # Create synthetic data with two uplift stages
        chi = np.linspace(0, 1000, 200)
        z = np.zeros_like(chi)

        # Stage 1 (recent): U = 0.8 for chi < 500
        # Stage 2 (old): U = 0.3 for chi >= 500
        mask_recent = chi < 500
        z[mask_recent] = 0.8 * chi[mask_recent]
        z[~mask_recent] = 0.8 * 500 + 0.3 * (chi[~mask_recent] - 500)

        # Invert with 2 time intervals
        Ustar, tstar, misfit = invert_block_uplift(chi, z, gamma=0.1, q=2)

        assert len(Ustar) == 2
        # Recent uplift should be higher
        assert Ustar[0] > Ustar[1]

    def test_gamma_effect(self):
        """Test effect of regularization parameter gamma."""
        # Create noisy data
        np.random.seed(42)
        chi = np.linspace(0, 1000, 100)
        z = 0.5 * chi + np.random.normal(0, 10, len(chi))
        z[0] = 0  # Ensure outlet is zero

        # Invert with different gamma values
        U_small_gamma, _, misfit_small = invert_block_uplift(chi, z, gamma=0.01, q=5)
        U_large_gamma, _, misfit_large = invert_block_uplift(chi, z, gamma=100.0, q=5)

        # Larger gamma should give smoother solution
        roughness_small = np.sum(np.diff(U_small_gamma)**2)
        roughness_large = np.sum(np.diff(U_large_gamma)**2)
        assert roughness_large < roughness_small

        # Smaller gamma should fit data better
        assert misfit_small < misfit_large

    def test_time_interval_structure(self):
        """Test that time intervals are properly constructed."""
        chi = np.linspace(0, 1000, 100)
        z = 0.5 * chi

        Ustar, tstar, misfit = invert_block_uplift(chi, z, gamma=1.0, q=4)

        # tstar should have q+1 elements
        assert len(tstar) == 5

        # tstar should be monotonically increasing
        assert np.all(np.diff(tstar) >= 0)

        # First element should be 0, last should be near max(chi)
        assert tstar[0] == 0
        assert tstar[-1] <= np.max(chi)

    def test_input_validation(self):
        """Test input validation and error handling."""
        chi = np.linspace(0, 1000, 100)
        z = 0.5 * chi

        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            invert_block_uplift(chi[:-1], z, gamma=1.0, q=2)

        # Non-positive gamma
        with pytest.raises(ValueError, match="positive"):
            invert_block_uplift(chi, z, gamma=0, q=2)

        with pytest.raises(ValueError, match="positive"):
            invert_block_uplift(chi, z, gamma=-1, q=2)

        # Non-positive q
        with pytest.raises(ValueError, match="positive"):
            invert_block_uplift(chi, z, gamma=1.0, q=0)

        # q too large
        with pytest.raises(ValueError, match="cannot exceed"):
            invert_block_uplift(chi, z, gamma=1.0, q=200)

        # All zeros
        with pytest.raises(ValueError, match="no data"):
            invert_block_uplift(chi, np.zeros_like(chi), gamma=1.0, q=2)

    def test_zero_removal(self):
        """Test that zero elevations are properly handled."""
        chi = np.linspace(0, 1000, 100)
        z = 0.5 * chi
        z[0] = 0  # Outlet
        z[50:60] = 0  # Some additional zeros

        Ustar, tstar, misfit = invert_block_uplift(chi, z, gamma=1.0, q=3)

        # Should complete without error
        assert len(Ustar) == 3
        assert np.all(np.isfinite(Ustar))

    def test_numerical_stability(self):
        """Test numerical stability with various data scales."""
        # Very small values
        chi_small = np.linspace(0, 1, 50)
        z_small = 0.001 * chi_small
        U, t, m = invert_block_uplift(chi_small, z_small, gamma=0.1, q=2)
        assert np.all(np.isfinite(U))

        # Very large values
        chi_large = np.linspace(0, 1e6, 50)
        z_large = 1e5 * chi_large
        U, t, m = invert_block_uplift(chi_large, z_large, gamma=1.0, q=2)
        assert np.all(np.isfinite(U))

    def test_misfit_calculation(self):
        """Test that misfit is calculated correctly."""
        chi = np.linspace(0, 1000, 100)
        U_true = 0.5
        z = U_true * chi

        Ustar, tstar, misfit = invert_block_uplift(chi, z, gamma=0.1, q=1)

        # For nearly perfect fit, misfit should be very small
        assert misfit < 0.1

        # Add noise
        z_noisy = z + np.random.normal(0, 50, len(z))
        z_noisy[0] = 0
        Ustar_noisy, tstar_noisy, misfit_noisy = invert_block_uplift(
            chi, z_noisy, gamma=1.0, q=1
        )

        # Noisy data should have larger misfit
        assert misfit_noisy > misfit

    def test_physical_constraints(self):
        """Test that results satisfy physical constraints."""
        chi = np.linspace(0, 1000, 100)
        z = 0.5 * chi + 0.3 * chi**1.1  # Nonlinear profile

        Ustar, tstar, misfit = invert_block_uplift(chi, z, gamma=1.0, q=5)

        # Uplift rates should be positive (typically, though technically can be negative)
        # For this synthetic case with positive z, expect positive U
        assert np.all(Ustar > 0)

        # Time intervals should sum to approximately max(chi)
        # (accounting for zero-removal)
        total_time = tstar[-1] - tstar[0]
        assert total_time <= np.max(chi)

    def test_resolution_effect(self):
        """Test effect of number of time intervals q."""
        chi = np.linspace(0, 1000, 200)
        # Create data with variation
        z = 0.5 * chi + 0.1 * np.sin(chi / 100) * chi

        # Invert with different resolutions
        U_low_res, _, _ = invert_block_uplift(chi, z, gamma=1.0, q=2)
        U_high_res, _, _ = invert_block_uplift(chi, z, gamma=1.0, q=10)

        # Higher resolution should capture more detail
        assert len(U_high_res) > len(U_low_res)

    def test_prior_model_effect(self):
        """Test that prior model influences solution for high gamma."""
        chi = np.linspace(0, 1000, 100)
        z = 0.5 * chi

        # With very high gamma, solution should be close to prior
        Ustar_high_gamma, _, _ = invert_block_uplift(chi, z, gamma=1000.0, q=5)

        # All uplift rates should be similar (close to mean)
        assert np.std(Ustar_high_gamma) < np.mean(Ustar_high_gamma) * 0.1


class TestInvertBlockUpliftWithStats:
    """Test suite for extended inversion with statistics."""

    def test_stats_output(self):
        """Test that statistics are computed correctly."""
        chi = np.linspace(0, 1000, 100)
        z = 0.5 * chi

        result = invert_block_uplift_with_stats(chi, z, gamma=1.0, q=3)

        # Check all expected keys are present
        expected_keys = ['Ustar', 'tstar', 'misfit', 'model_z', 'observed_z',
                        'residuals', 'r_squared', 'dof', 'n_data', 'chi_sorted']
        for key in expected_keys:
            assert key in result

        # Check dimensions
        assert len(result['Ustar']) == 3
        assert len(result['tstar']) == 4
        assert len(result['model_z']) == result['n_data']
        assert len(result['residuals']) == result['n_data']

    def test_r_squared_perfect_fit(self):
        """Test R² for perfect fit."""
        chi = np.linspace(0, 1000, 100)
        z = 0.5 * chi

        result = invert_block_uplift_with_stats(chi, z, gamma=0.01, q=1)

        # For perfect fit with right model, R² should be close to 1
        assert result['r_squared'] > 0.99

    def test_residuals(self):
        """Test residuals calculation."""
        chi = np.linspace(0, 1000, 100)
        z = 0.5 * chi + np.random.normal(0, 5, len(chi))
        z[0] = 0

        result = invert_block_uplift_with_stats(chi, z, gamma=1.0, q=2)

        # Residuals should sum to approximately zero
        assert abs(np.mean(result['residuals'])) < 1.0

        # Check residuals match
        computed_residuals = result['observed_z'] - result['model_z']
        np.testing.assert_allclose(result['residuals'], computed_residuals)

    def test_degrees_of_freedom(self):
        """Test degrees of freedom calculation."""
        chi = np.linspace(0, 1000, 100)
        z = 0.5 * chi

        result = invert_block_uplift_with_stats(chi, z, gamma=1.0, q=5)

        # DOF = N - q, where N is number of non-zero data points
        expected_dof = len(chi) - 1 - 5  # -1 for zero at outlet
        assert result['dof'] == expected_dof


class TestInversionProperties:
    """Test mathematical and physical properties of inversion."""

    def test_linearity(self):
        """Test that doubling data doubles solution (for linear problem)."""
        chi = np.linspace(0, 1000, 100)
        z1 = 0.5 * chi

        U1, _, _ = invert_block_uplift(chi, z1, gamma=1.0, q=2)
        U2, _, _ = invert_block_uplift(chi, 2*z1, gamma=1.0, q=2)

        # U2 should be approximately 2 * U1
        np.testing.assert_allclose(U2, 2*U1, rtol=0.01)

    def test_consistency_with_forward_model(self):
        """Test that inverting forward model recovers input."""
        # Create uplift history
        q = 3
        Ustar_true = np.array([0.6, 0.4, 0.3])

        # Create chi and compute forward model z = A * U
        chi = np.linspace(0, 1000, 100)
        N = len(chi)

        # Build A matrix (same as in invert_block_uplift)
        val_per_dt = N // q
        scaled_t_vec = np.zeros(q + 1)
        scaled_t_vec[0] = 0
        for i in range(1, q):
            scaled_t_vec[i] = chi[(i) * val_per_dt - 1]
        scaled_t_vec[q] = chi[-1]
        scaled_dt_vec = np.diff(scaled_t_vec)

        Astar = np.zeros((N, q))
        for i in range(N):
            pixel_chi = chi[i]
            filled_full_elements = np.searchsorted(scaled_t_vec, pixel_chi, side='right') - 2
            if filled_full_elements >= 0:
                Astar[i, :filled_full_elements+1] = scaled_dt_vec[:filled_full_elements+1]
            if filled_full_elements + 1 < q:
                partial_time = pixel_chi - np.sum(scaled_dt_vec[:filled_full_elements+1])
                if partial_time > 0:
                    Astar[i, filled_full_elements + 1] = partial_time

        # Forward model
        z = Astar @ Ustar_true

        # Invert with very small gamma (minimal regularization)
        Ustar_recovered, _, _ = invert_block_uplift(chi, z, gamma=0.001, q=q)

        # Should recover close to true values
        np.testing.assert_allclose(Ustar_recovered, Ustar_true, rtol=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
