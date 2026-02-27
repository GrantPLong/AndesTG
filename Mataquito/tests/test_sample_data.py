"""Tests for mataquito.sample_data — verify Excel loading matches expected values."""

import pytest
from mataquito.sample_data import (
    load_samples,
    get_erosion_rates,
    get_areas,
    get_uncertainties,
    get_production_rates,
)


@pytest.fixture(scope="module")
def df():
    return load_samples()


def test_load_samples_row_count(df):
    """Sheet1 should contain 11 samples (CT-1 through CT-11)."""
    assert len(df) == 11


def test_sample_ids(df):
    """All expected sample IDs should be present."""
    expected = {f"CT-{i}" for i in range(1, 12)}
    assert set(df["Sample_ID"]) == expected


def test_ct1_erosion_rate(df):
    E = get_erosion_rates(df)
    assert E["CT-1"] == pytest.approx(22.9, abs=0.1)


def test_ct9_erosion_rate(df):
    E = get_erosion_rates(df)
    assert E["CT-9"] == pytest.approx(105.0, abs=0.1)


def test_ct1_source_area(df):
    A = get_areas(df)
    assert A["CT-1"] == pytest.approx(1385.185, abs=0.5)


def test_ct9_source_area(df):
    A = get_areas(df)
    assert A["CT-9"] == pytest.approx(6189.716, abs=0.5)


def test_get_uncertainties_keys(df):
    U = get_uncertainties(df)
    assert "CT-1" in U and "CT-11" in U
    assert len(U) == 11


def test_get_production_rates_keys(df):
    P = get_production_rates(df)
    assert "CT-1" in P and "CT-11" in P
    assert len(P) == 11


def test_all_erosion_rates_positive(df):
    E = get_erosion_rates(df)
    for sid, val in E.items():
        assert val >= 0, f"{sid} has negative erosion rate {val}"
