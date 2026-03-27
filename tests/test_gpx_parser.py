"""Unit tests for ``GPXParser``."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gpxmapper.gpx_parser import GPXParser
from gpxmapper.models import GPXTrackPoint

GPX_HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">'
)
GPX_FOOTER = "</gpx>"


def _write_gpx(path: Path, body: str) -> Path:
    path.write_text(GPX_HEADER + body + GPX_FOOTER, encoding="utf-8")
    return path


def test_parse_file_not_found(tmp_path):
    missing = tmp_path / "missing.gpx"
    parser = GPXParser(str(missing))
    with pytest.raises(FileNotFoundError):
        parser.parse()


def test_parse_invalid_gpx_raises_value_error(tmp_path):
    path = tmp_path / "bad.gpx"
    path.write_text("not valid gpx or xml <<<", encoding="utf-8")
    parser = GPXParser(str(path))
    with pytest.raises(ValueError, match="Invalid GPX file"):
        parser.parse()


def test_parse_minimal_track_extracts_points(tmp_path):
    body = """
  <trk><name>T</name><trkseg>
    <trkpt lat="47.0" lon="19.0"><ele>100</ele><time>2020-01-01T12:00:00Z</time></trkpt>
    <trkpt lat="47.5" lon="19.5"><ele>120</ele><time>2020-01-01T14:00:00Z</time></trkpt>
  </trkseg></trk>
"""
    path = _write_gpx(tmp_path / "t.gpx", body)
    parser = GPXParser(str(path))
    points = parser.parse()

    assert len(points) == 2
    assert parser._parsed is True
    assert all(isinstance(p, GPXTrackPoint) for p in points)
    assert points[0].latitude == pytest.approx(47.0)
    assert points[0].longitude == pytest.approx(19.0)
    assert points[0].elevation == pytest.approx(100.0)
    assert points[0].time == datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert points[1].latitude == pytest.approx(47.5)


def test_parse_multiple_segments_concatenates_points(tmp_path):
    body = """
  <trk><trkseg>
    <trkpt lat="1.0" lon="2.0"></trkpt>
  </trkseg><trkseg>
    <trkpt lat="3.0" lon="4.0"></trkpt>
  </trkseg></trk>
"""
    path = _write_gpx(tmp_path / "multi.gpx", body)
    points = GPXParser(str(path)).parse()
    assert len(points) == 2
    assert points[0].latitude == pytest.approx(1.0)
    assert points[1].longitude == pytest.approx(4.0)


def test_get_time_bounds_requires_parse(tmp_path):
    body = """
  <trk><trkseg>
    <trkpt lat="0" lon="0"><time>2021-06-15T10:00:00Z</time></trkpt>
    <trkpt lat="1" lon="1"><time>2021-06-15T12:30:00Z</time></trkpt>
  </trkseg></trk>
"""
    path = _write_gpx(tmp_path / "time.gpx", body)
    parser = GPXParser(str(path))
    start, end = parser.get_time_bounds()
    assert start == datetime(2021, 6, 15, 10, 0, tzinfo=timezone.utc)
    assert end == datetime(2021, 6, 15, 12, 30, tzinfo=timezone.utc)


def test_get_time_bounds_no_timestamps_returns_none_pair(tmp_path):
    body = """
  <trk><trkseg>
    <trkpt lat="0" lon="0"><ele>10</ele></trkpt>
  </trkseg></trk>
"""
    path = _write_gpx(tmp_path / "notime.gpx", body)
    parser = GPXParser(str(path))
    assert parser.get_time_bounds() == (None, None)


def test_get_time_bounds_empty_track_returns_none_pair(tmp_path):
    path = _write_gpx(tmp_path / "empty.gpx", "<trk><trkseg></trkseg></trk>")
    parser = GPXParser(str(path))
    assert parser.get_time_bounds() == (None, None)


def test_get_coordinate_bounds(tmp_path):
    body = """
  <trk><trkseg>
    <trkpt lat="10.0" lon="5.0"></trkpt>
    <trkpt lat="-2.0" lon="20.0"></trkpt>
  </trkseg></trk>
"""
    path = _write_gpx(tmp_path / "box.gpx", body)
    parser = GPXParser(str(path))
    min_lat, min_lon, max_lat, max_lon = parser.get_coordinate_bounds()
    assert min_lat == pytest.approx(-2.0)
    assert max_lat == pytest.approx(10.0)
    assert min_lon == pytest.approx(5.0)
    assert max_lon == pytest.approx(20.0)


def test_get_coordinate_bounds_no_points_raises(tmp_path):
    path = _write_gpx(tmp_path / "empty.gpx", "<trk><trkseg></trkseg></trk>")
    parser = GPXParser(str(path))
    with pytest.raises(ValueError, match="No track points found"):
        parser.get_coordinate_bounds()


def test_convert_extensions_to_dict_empty_and_indexed():
    parser = GPXParser("/unused")
    assert parser._convert_extensions_to_dict(None) == {}
    assert parser._convert_extensions_to_dict([]) == {}
    assert parser._convert_extensions_to_dict(["a", "b"]) == {
        "extension_0": "a",
        "extension_1": "b",
    }


def test_create_track_point_maps_gpxpy_point():
    parser = GPXParser("/unused")
    point = MagicMock()
    point.latitude = 48.2
    point.longitude = 16.3
    point.elevation = 345.0
    point.time = datetime(2022, 1, 1, tzinfo=timezone.utc)
    point.extensions = ["x"]

    tp = parser._create_track_point(point)
    assert tp.latitude == pytest.approx(48.2)
    assert tp.longitude == pytest.approx(16.3)
    assert tp.elevation == pytest.approx(345.0)
    assert tp.time == point.time
    assert tp.extensions == {"extension_0": "x"}
