"""Tests for mataquito.fertility — quartz and zircon fertility calculations."""

import numpy as np
import pytest
from mataquito.fertility import load_wct, quartz_fertility, zircon_fertility


class TestLoadWCT:
    def test_load_existing_csv(self):
        """Verify load_wct works on a known mixing coefficient CSV."""
        path = "results/mixing_coefficients/mix_coeffs_all_ct-5_ct-6_to_ct-10.csv"
        wa, wb, n_total, n_kept = load_wct(path)
        assert n_total > 0
        assert n_kept > 0
        assert n_kept <= n_total
        assert len(wa) == n_kept
        assert len(wb) == n_kept

    def test_coefficients_in_range(self):
        """All kept coefficients should be within (THRESHOLD, 1-THRESHOLD)."""
        path = "results/mixing_coefficients/mix_coeffs_all_ct-5_ct-6_to_ct-10.csv"
        wa, wb, _, _ = load_wct(path)
        assert np.all(wa > 0.01)
        assert np.all(wa < 0.99)
        assert np.all(wb > 0.01)
        assert np.all(wb < 0.99)


class TestQuartzFertility:
    def test_positive_output_only(self):
        """quartz_fertility should only return positive Qa/Qb values."""
        np.random.seed(42)
        E_a = np.random.normal(400, 50, 1000)
        E_b = np.random.normal(30, 5, 1000)
        E_c = np.random.normal(250, 30, 1000)
        Q, mask = quartz_fertility(E_a, E_b, E_c, 1500, 2500, 4800, 14.6, 5.9, 11.5)
        assert len(Q) > 0
        assert np.all(Q > 0)

    def test_mask_matches_output(self):
        np.random.seed(42)
        E_a = np.random.normal(400, 50, 1000)
        E_b = np.random.normal(30, 5, 1000)
        E_c = np.random.normal(250, 30, 1000)
        Q, mask = quartz_fertility(E_a, E_b, E_c, 1500, 2500, 4800, 14.6, 5.9, 11.5)
        assert len(Q) == mask.sum()


class TestZirconFertility:
    def test_basic_output_structure(self):
        """zircon_fertility should return dict with ZA_ZB and ZB_ZA sub-dicts."""
        np.random.seed(42)
        wa = np.random.uniform(0.3, 0.7, 500)
        wb = 1.0 - wa
        ER_A = np.random.normal(400, 50, 500)
        ER_B = np.random.normal(30, 5, 500)
        result = zircon_fertility(wa, wb, ER_A, ER_B, 1500, 2500, num_samples=200, seed=17)
        assert "ZA_ZB" in result
        assert "ZB_ZA" in result
        assert "median" in result["ZA_ZB"]
        assert "p25" in result["ZA_ZB"]
        assert "p75" in result["ZA_ZB"]
        assert "log10_median" in result["ZA_ZB"]

    def test_inverse_relationship(self):
        """ZA/ZB * ZB/ZA should equal 1 for all samples."""
        np.random.seed(42)
        wa = np.random.uniform(0.3, 0.7, 500)
        wb = 1.0 - wa
        ER_A = np.random.normal(400, 50, 500)
        ER_B = np.random.normal(30, 5, 500)
        result = zircon_fertility(wa, wb, ER_A, ER_B, 1500, 2500, num_samples=200, seed=17)
        product = result["ZA_ZB"]["samples"] * result["ZB_ZA"]["samples"]
        np.testing.assert_allclose(product, 1.0, atol=1e-10)
