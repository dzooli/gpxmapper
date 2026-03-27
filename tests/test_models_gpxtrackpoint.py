from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpxmapper.models import GPXTrackPoint


def _point(
        lat: float,
        lon: float,
        *,
        ele: float | None = None,
        time: datetime | None = None,
        extensions: dict | None = None,
) -> GPXTrackPoint:
    return GPXTrackPoint(lat, lon, ele, time, extensions)


def test_distance_to_known_pair_and_symmetry_and_zero() -> None:
    a = _point(0.0, 0.0)
    b = _point(0.0, 1.0)

    d_ab = a.distance_to(b)
    d_ba = b.distance_to(a)

    assert d_ab == pytest.approx(111194.93, rel=1e-4)
    assert d_ba == pytest.approx(d_ab)
    assert a.distance_to(a) == pytest.approx(0.0, abs=1e-12)


def test_elevation_gain_increasing_decreasing_and_none() -> None:
    low = _point(0, 0, ele=100.0)
    high = _point(0, 0, ele=125.5)
    none_ele = _point(0, 0, ele=None)

    assert low.elevation_gain(high) == pytest.approx(25.5)
    assert high.elevation_gain(low) == pytest.approx(0.0)
    assert low.elevation_gain(none_ele) == pytest.approx(0.0)
    assert none_ele.elevation_gain(high) == pytest.approx(0.0)


def test_time_delta_increasing_equal_and_missing() -> None:
    t0 = datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc)
    t1 = datetime(2020, 1, 1, 12, 0, 30, tzinfo=timezone.utc)

    a = _point(0, 0, time=t0)
    b = _point(0, 0, time=t1)
    c = _point(0, 0, time=t0)
    no_time = _point(0, 0, time=None)

    assert a.time_delta(b) == pytest.approx(30.0)
    assert a.time_delta(c) == pytest.approx(0.0)
    assert a.time_delta(no_time) is None
    assert no_time.time_delta(b) is None


def test_speed_to_normal_zero_dt_and_missing_timestamps() -> None:
    t0 = datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc)
    t10 = datetime(2020, 1, 1, 12, 0, 10, tzinfo=timezone.utc)

    a = _point(0.0, 0.0, time=t0)
    b = _point(0.0, 1.0, time=t10)
    same_time = _point(0.0, 1.0, time=t0)
    no_time = _point(0.0, 1.0, time=None)

    expected_speed = a.distance_to(b) / 10.0
    assert a.speed_to(b) == pytest.approx(expected_speed)
    assert a.speed_to(same_time) is None
    assert a.speed_to(no_time) is None


def test_has_flags_and_to_dict_serialization() -> None:
    t = datetime(2021, 7, 20, 8, 15, 30, tzinfo=timezone.utc)
    ext = {"surface": "asphalt", "source": "device"}
    p = _point(47.5, 19.0, ele=145.2, time=t, extensions=ext)
    p_empty = _point(47.5, 19.0, ele=None, time=None, extensions=None)

    assert p.has_elevation is True
    assert p.has_timestamp is True
    assert p.has_extensions is True

    assert p_empty.has_elevation is False
    assert p_empty.has_timestamp is False
    assert p_empty.has_extensions is False

    data = p.to_dict()
    assert data["lat"] == pytest.approx(47.5)
    assert data["lon"] == pytest.approx(19.0)
    assert data["ele"] == pytest.approx(145.2)
    assert data["time"] == t.isoformat()
    assert data["extensions"] == ext
