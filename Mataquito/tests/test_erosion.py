"""Tests for mataquito.erosion — MC sampling and flux ordering."""

import numpy as np
import pytest
from mataquito.sample_data import load_samples, get_areas
from mataquito.erosion import generate_mc_samples, flux_order_samples


@pytest.fixture(scope="module")
def df():
    return load_samples()


@pytest.fixture(scope="module")
def er_ext(df):
    return generate_mc_samples(df, N=100_000, seed=17)


@pytest.fixture(scope="module")
def A(df):
    return get_areas(df)


class TestGenerateMCSamples:
    def test_all_samples_present(self, er_ext):
        expected = {f"CT-{i}" for i in range(1, 12)}
        assert set(er_ext.keys()) == expected

    def test_array_length(self, er_ext):
        for sid, arr in er_ext.items():
            assert len(arr) == 100_000, f"{sid} has wrong length"

    def test_ct1_median(self, er_ext):
        """Median should be close to the measured rate (22.9 m/Myr)."""
        assert np.median(er_ext["CT-1"]) == pytest.approx(22.9, abs=0.5)

    def test_ct9_median(self, er_ext):
        assert np.median(er_ext["CT-9"]) == pytest.approx(105.0, abs=1.0)


class TestFluxOrdering:
    @pytest.fixture(scope="class")
    def flux(self, er_ext, A):
        return flux_order_samples(er_ext, A, N=100_000, seed=17)

    def test_ct7_ct5_retention(self, flux):
        """CT-7->CT-5 retention should be ~21% (from NB06 stored output)."""
        pct = 100 * len(flux["m_75"]) / 100_000
        assert 18 <= pct <= 24, f"CT-7->CT-5 retention {pct:.1f}% outside expected range"

    def test_ct1_ct6_retention(self, flux):
        """CT-1->CT-6 retention should be ~100% (from NB06 stored output)."""
        pct = 100 * len(flux["m_16"]) / 100_000
        assert 98 <= pct <= 100, f"CT-1->CT-6 retention {pct:.1f}% outside expected range"

    def test_ct5_res_length(self, flux):
        assert len(flux["CT5_res"]) == 100_000

    def test_ct8_res_length(self, flux):
        assert len(flux["CT8_res"]) == 100_000

    def test_ct3_res_length(self, flux):
        assert len(flux["CT3_res"]) == 100_000

    def test_ordered_ct5_median_higher(self, flux, er_ext):
        """Flux-ordered CT-5 should have higher median than raw CT-5."""
        assert np.median(flux["CT5_ord"]) > np.median(er_ext["CT-5"])

    def test_ct10_f_exists(self, flux):
        """CT-10 flux-ordered array should have entries."""
        assert len(flux["CT10_f"]) > 0

    def test_ct11_f_exists(self, flux):
        assert len(flux["CT11_f"]) > 0

    def test_ct8_f_exists(self, flux):
        assert len(flux["CT8_f"]) > 0
