"""Tests for mataquito.flowlines — utility functions (no raster data needed)."""

import pytest
from shapely.geometry import LineString, Point
from mataquito.flowlines import calculate_distance_along_line, substring_line


class TestCalculateDistanceAlongLine:
    def test_point_at_start(self):
        line = LineString([(0, 0), (1000, 0)])
        point = Point(0, 0)
        assert calculate_distance_along_line(line, point) == pytest.approx(0.0)

    def test_point_at_end(self):
        line = LineString([(0, 0), (1000, 0)])
        point = Point(1000, 0)
        assert calculate_distance_along_line(line, point) == pytest.approx(1.0)

    def test_point_at_midpoint(self):
        line = LineString([(0, 0), (1000, 0)])
        point = Point(500, 0)
        assert calculate_distance_along_line(line, point) == pytest.approx(0.5)

    def test_point_off_line(self):
        """Projected distance should still work for a point off the line."""
        line = LineString([(0, 0), (1000, 0)])
        point = Point(500, 100)  # 100m off the line
        assert calculate_distance_along_line(line, point) == pytest.approx(0.5)


class TestSubstringLine:
    def test_full_line(self):
        """Requesting the full range should return the original line."""
        line = LineString([(0, 0), (500, 0), (1000, 0)])
        result = substring_line(line, 0, line.length)
        assert result.length == pytest.approx(line.length, rel=1e-6)

    def test_first_half(self):
        line = LineString([(0, 0), (1000, 0)])
        result = substring_line(line, 0, 500)
        assert result.length == pytest.approx(500, rel=1e-3)

    def test_second_half(self):
        line = LineString([(0, 0), (1000, 0)])
        result = substring_line(line, 500, 1000)
        assert result.length == pytest.approx(500, rel=1e-3)

    def test_result_is_linestring(self):
        line = LineString([(0, 0), (1000, 0)])
        result = substring_line(line, 200, 800)
        assert isinstance(result, LineString)
