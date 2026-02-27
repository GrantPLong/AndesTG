"""Tests for mataquito.production_rates — Stone (2000) scaling."""

import pytest
from mataquito.production_rates import (
    elevation_to_pressure,
    stone2000_scaling,
    stone2000_production_rate,
    SAMPLE_ELEVATIONS,
)

# Expected production rates from NB02 output (atoms/g/yr, spallation only)
EXPECTED_RATES = {
    "CT-1": 17.13,
    "CT-2": 10.35,
    "CT-3": 4.20,
    "CT-4": 11.82,
    "CT-5": 14.63,
    "CT-6": 5.94,
    "CT-7": 16.44,
    "CT-8": 11.42,
    "CT-9": 9.90,
    "CT-10": 11.53,
    "CT-11": 11.45,
}


class TestElevationToPressure:
    def test_sea_level(self):
        """At 0 m elevation, pressure should be ~1013.25 hPa."""
        assert elevation_to_pressure(0) == pytest.approx(1013.25, abs=0.01)

    def test_high_elevation(self):
        """At ~2000 m, pressure should be substantially lower."""
        p = elevation_to_pressure(2000)
        assert 700 < p < 850


class TestStone2000ProductionRate:
    @pytest.mark.parametrize("sample", list(EXPECTED_RATES.keys()))
    def test_production_rate(self, sample):
        elev = SAMPLE_ELEVATIONS[sample]
        rate = stone2000_production_rate(elev)
        assert rate == pytest.approx(EXPECTED_RATES[sample], abs=0.02), (
            f"{sample}: got {rate:.2f}, expected {EXPECTED_RATES[sample]:.2f}"
        )
