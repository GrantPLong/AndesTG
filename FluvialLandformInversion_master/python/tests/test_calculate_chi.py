"""Unit tests for calculate_chi function."""

import numpy as np
import pytest
from fluvial_inversion.calculate_chi import calculate_chi, validate_flow_network


class TestCalculateChi:
    """Test suite for calculate_chi function."""

    def test_simple_linear_channel(self):
        """Test chi calculation for a simple linear channel."""
        # Simple 4-pixel linear channel: 3 -> 2 -> 1 -> 0 (outlet)
        x = np.array([0.0, 100.0, 200.0, 300.0])
        y = np.array([0.0, 0.0, 0.0, 0.0])
        rec_array = np.array([0, 0, 1, 2])  # 0 is outlet
        area_array = np.array([1000.0, 750.0, 500.0, 100.0])
        m = 0.5
        A0 = 1.0

        chi = calculate_chi(x, y, rec_array, area_array, m, A0)

        # Outlet should have chi = 0
        assert chi[0] == 0.0

        # Chi should increase upstream
        assert chi[1] > chi[0]
        assert chi[2] > chi[1]
        assert chi[3] > chi[2]

        # Manual calculation for pixel 1
        dist_1_0 = 100.0
        expected_chi_1 = dist_1_0 * (A0 / 750.0)**0.5
        np.testing.assert_allclose(chi[1], expected_chi_1, rtol=1e-10)

    def test_two_tributaries(self):
        """Test chi calculation for channel with tributaries."""
        # Network:
        #     3
        #      \
        #   2 - 1 - 0
        #
        x = np.array([0.0, 100.0, 200.0, 200.0])
        y = np.array([0.0, 0.0, 0.0, 100.0])
        rec_array = np.array([0, 0, 1, 1])  # Both 2 and 3 flow to 1
        area_array = np.array([1000.0, 800.0, 500.0, 300.0])
        m = 0.45

        chi = calculate_chi(x, y, rec_array, area_array, m)

        # Outlet has chi = 0
        assert chi[0] == 0.0

        # Both tributaries should have positive chi
        assert chi[2] > 0
        assert chi[3] > 0

        # Verify chi values are reasonable
        assert np.all(np.isfinite(chi))

    def test_reference_area_effect(self):
        """Test effect of different reference areas."""
        x = np.array([0.0, 100.0, 200.0])
        y = np.array([0.0, 0.0, 0.0])
        rec_array = np.array([0, 0, 1])
        area_array = np.array([1000.0, 500.0, 100.0])
        m = 0.5

        chi_A0_1 = calculate_chi(x, y, rec_array, area_array, m, A0=1.0)
        chi_A0_10 = calculate_chi(x, y, rec_array, area_array, m, A0=10.0)

        # Chi should scale with A0
        # χ ∝ (A0/A)^m, so larger A0 -> larger chi
        # (skip outlet which is always 0)
        assert np.all(chi_A0_10[1:] > chi_A0_1[1:])

    def test_m_parameter_effect(self):
        """Test effect of different m values."""
        x = np.array([0.0, 100.0, 200.0])
        y = np.array([0.0, 0.0, 0.0])
        rec_array = np.array([0, 0, 1])
        area_array = np.array([1000.0, 500.0, 100.0])

        chi_m_low = calculate_chi(x, y, rec_array, area_array, m=0.3)
        chi_m_high = calculate_chi(x, y, rec_array, area_array, m=0.6)

        # Higher m -> stronger effect of drainage area
        # For A > A0=1, (A0/A)^m < 1, and this decreases with m, so chi decreases
        # For A < A0=1, (A0/A)^m > 1, and this increases with m, so chi increases
        # Pixel 2 has area=100 < A0=1, so chi should increase with m
        # But let's test the physical meaning: chi should differ
        assert chi_m_high[2] != chi_m_low[2]
        # For small areas (A < A0), higher m gives larger (A0/A)^m
        # Since area_array[2] = 100 > 1, actually (1/100)^m decreases with m
        # So higher m gives smaller chi for A > A0=1
        assert chi_m_high[2] < chi_m_low[2]

    def test_diagonal_flow(self):
        """Test chi calculation with diagonal flow."""
        # 2 -> 1 -> 0, with diagonal geometry
        x = np.array([0.0, 100.0, 200.0])
        y = np.array([0.0, 100.0, 200.0])
        rec_array = np.array([0, 0, 1])
        area_array = np.array([1000.0, 500.0, 100.0])
        m = 0.5

        chi = calculate_chi(x, y, rec_array, area_array, m)

        # Distance from 1 to 0 is sqrt(100^2 + 100^2) = 141.42
        expected_dist = np.sqrt(100**2 + 100**2)
        expected_chi_1 = expected_dist * (1.0 / 500.0)**0.5
        np.testing.assert_allclose(chi[1], expected_chi_1, rtol=1e-10)

    def test_input_validation(self):
        """Test input validation and error handling."""
        x = np.array([0.0, 100.0, 200.0])
        y = np.array([0.0, 0.0, 0.0])
        rec_array = np.array([0, 0, 1])
        area_array = np.array([1000.0, 500.0, 100.0])

        # Mismatched array lengths
        with pytest.raises(ValueError, match="same length"):
            calculate_chi(x[:-1], y, rec_array, area_array, m=0.5)

        # Invalid m values
        with pytest.raises(ValueError, match="between 0 and 1"):
            calculate_chi(x, y, rec_array, area_array, m=-0.1)

        with pytest.raises(ValueError, match="between 0 and 1"):
            calculate_chi(x, y, rec_array, area_array, m=1.5)

        # Invalid A0
        with pytest.raises(ValueError, match="must be positive"):
            calculate_chi(x, y, rec_array, area_array, m=0.5, A0=-1.0)

        # Non-positive drainage areas
        bad_area = area_array.copy()
        bad_area[1] = 0
        with pytest.raises(ValueError, match="must be positive"):
            calculate_chi(x, y, rec_array, bad_area, m=0.5)

    def test_multiple_outlets(self):
        """Test network with multiple outlets (disconnected basins)."""
        # Two separate channels: 2->0 and 3->1
        x = np.array([0.0, 200.0, 100.0, 300.0])
        y = np.array([0.0, 0.0, 0.0, 0.0])
        rec_array = np.array([0, 1, 0, 1])  # 0 and 1 are outlets
        area_array = np.array([1000.0, 500.0, 500.0, 100.0])
        m = 0.5

        chi = calculate_chi(x, y, rec_array, area_array, m)

        # Both outlets should have chi = 0
        assert chi[0] == 0.0
        assert chi[1] == 0.0

        # Upstream pixels should have positive chi
        assert chi[2] > 0
        assert chi[3] > 0


class TestValidateFlowNetwork:
    """Test suite for validate_flow_network function."""

    def test_valid_network(self):
        """Test validation of valid flow network."""
        rec_array = np.array([0, 0, 1, 2])  # Simple linear channel
        is_valid, msg = validate_flow_network(rec_array)
        assert is_valid
        assert msg == ""

    def test_no_outlet(self):
        """Test network with no outlet."""
        rec_array = np.array([1, 2, 0])  # Cycle, no outlet
        is_valid, msg = validate_flow_network(rec_array)
        assert not is_valid
        assert "No outlet" in msg

    def test_invalid_receiver_index(self):
        """Test network with out-of-bounds receiver."""
        rec_array = np.array([0, 0, 5])  # Receiver 5 doesn't exist
        is_valid, msg = validate_flow_network(rec_array)
        assert not is_valid
        assert "range" in msg.lower()


class TestChiProperties:
    """Test mathematical properties of chi."""

    def test_chi_monotonic_upstream(self):
        """Chi should increase monotonically along flow path."""
        # Create a longer channel: 9->8->...->1->0
        n = 10
        x = np.arange(n, dtype=float) * 100
        y = np.zeros(n)
        # Each pixel flows to the previous one
        rec_array = np.arange(n) - 1
        rec_array[0] = 0  # Outlet flows to itself
        area_array = np.linspace(1000, 100, n)  # Decreasing upstream
        m = 0.5

        chi = calculate_chi(x, y, rec_array, area_array, m)

        # Chi should increase at each step upstream
        for i in range(1, n):
            assert chi[i] > chi[rec_array[i]]

    def test_chi_zero_at_outlet(self):
        """Chi should be zero at all outlets."""
        x = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        rec_array = np.array([0, 0, 2, 2, 1])  # 0 and 2 are outlets
        area_array = np.array([1000.0, 800.0, 600.0, 200.0, 100.0])
        m = 0.5

        chi = calculate_chi(x, y, rec_array, area_array, m)

        # Check outlets have chi = 0
        assert chi[0] == 0.0
        assert chi[2] == 0.0

    def test_numerical_stability(self):
        """Test chi calculation with extreme values."""
        # Very large areas
        x = np.array([0.0, 100.0, 200.0])
        y = np.array([0.0, 0.0, 0.0])
        rec_array = np.array([0, 0, 1])
        area_array = np.array([1e10, 1e9, 1e8])
        m = 0.5

        chi = calculate_chi(x, y, rec_array, area_array, m)
        assert np.all(np.isfinite(chi))
        assert np.all(chi >= 0)

        # Very small areas (but positive)
        area_array = np.array([1e-3, 1e-4, 1e-5])
        chi = calculate_chi(x, y, rec_array, area_array, m)
        assert np.all(np.isfinite(chi))
        assert np.all(chi >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
