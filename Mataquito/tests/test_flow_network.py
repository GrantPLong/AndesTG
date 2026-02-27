"""Tests for mataquito.flow_network — verify topology and deterministic subwatershed rates."""

import pytest
from mataquito.flow_network import (
    FLOW_NETWORK,
    HEADWATERS,
    TRAVERSAL_ORDER,
    subwatershed_area,
    subwatershed_erosion_rate,
    all_subwatershed_erosion_rates,
)

# Hardcoded E and A values from NB03 cell 5 — the regression baseline
E_NB03 = {
    "CT-1": 22.9,
    "CT-2": 94.8,
    "CT-3": 29.5,
    "CT-5": 387.0,
    "CT-6": 29.8,
    "CT-7": 532.0,
    "CT-8": 263.0,
    "CT-9": 105.0,
    "CT-10": 246.0,
    "CT-11": 286.0,
}

A_NB03 = {
    "CT-1": 1384.89776669958,
    "CT-2": 5760.4682392288,
    "CT-3": 189.374842361642,
    "CT-4": 4706.34811609699,
    "CT-5": 1495.57363200431,
    "CT-6": 2573.03043606272,
    "CT-7": 1207.49492554076,
    "CT-8": 4950.1417877331,
    "CT-9": 6190.46808871897,
    "CT-10": 4864.64292867407,
    "CT-11": 4913.05986812238,
}


class TestTopology:
    def test_headwaters_not_in_network(self):
        for hw in HEADWATERS:
            assert hw not in FLOW_NETWORK

    def test_all_parents_defined(self):
        """Every parent referenced in the DAG must be either a headwater or another node."""
        all_nodes = set(FLOW_NETWORK.keys()) | set(HEADWATERS)
        for child, parents in FLOW_NETWORK.items():
            for p in parents:
                assert p in all_nodes, f"Parent {p} of {child} not in network"

    def test_traversal_covers_all_non_headwater(self):
        assert set(TRAVERSAL_ORDER) == set(FLOW_NETWORK.keys())


class TestSubwatershedArea:
    def test_headwater_area(self):
        assert subwatershed_area("CT-1", A_NB03) == A_NB03["CT-1"]

    def test_ct5_area(self):
        expected = A_NB03["CT-5"] - A_NB03["CT-7"]
        assert subwatershed_area("CT-5", A_NB03) == pytest.approx(expected, rel=1e-10)

    def test_ct10_area(self):
        expected = A_NB03["CT-10"] - A_NB03["CT-5"] - A_NB03["CT-6"]
        assert subwatershed_area("CT-10", A_NB03) == pytest.approx(expected, rel=1e-10)


class TestSubwatershedErosionRate:
    """Verify deterministic rates against NB03 cell 5 output."""

    EXPECTED = {
        "CT-1": 22.90,
        "CT-3": 29.50,
        "CT-7": 532.00,
        "CT-5": -220.77,
        "CT-6": 37.84,
        "CT-10": 679.92,
        "CT-11": 4304.96,
        "CT-8": -2784.32,
        "CT-2": -932.70,
        "CT-9": 408.60,
    }

    def test_headwater_rates(self):
        for hw in HEADWATERS:
            assert subwatershed_erosion_rate(hw, E_NB03, A_NB03) == E_NB03[hw]

    @pytest.mark.parametrize(
        "site", ["CT-5", "CT-6", "CT-10", "CT-11", "CT-8", "CT-2", "CT-9"]
    )
    def test_subwatershed_rate(self, site):
        result = subwatershed_erosion_rate(site, E_NB03, A_NB03)
        assert result == pytest.approx(self.EXPECTED[site], abs=0.01)

    def test_all_rates(self):
        result = all_subwatershed_erosion_rates(E_NB03, A_NB03)
        for site, expected in self.EXPECTED.items():
            assert result[site] == pytest.approx(expected, abs=0.01), (
                f"{site}: got {result[site]:.2f}, expected {expected:.2f}"
            )
