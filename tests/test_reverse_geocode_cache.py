"""Tests for :mod:`gpxmapper.reverse_geocode_cache`."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpxmapper.map_renderer.base import MapRendererBase
from gpxmapper.models import GPXTrackPoint
from gpxmapper.reverse_geocode_cache import (
    ReverseGeocodeCache,
    normalize_cache_base_url,
    quantize_coordinates,
    resolve_reverse_geocode_cache_path,
)


def test_quantize_coordinates() -> None:
    assert quantize_coordinates(48.8565789, 2.3522123) == (48.85658, 2.35221)


def test_normalize_cache_base_url() -> None:
    assert normalize_cache_base_url("  HTTPS://Nominatim.Example.COM/foo/  ") == "https://nominatim.example.com/foo"


def test_resolve_reverse_geocode_cache_path_windows_style(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MapRendererBase,
        "resolve_default_cache_directory",
        lambda: str(Path("C:/Users/x/AppData/Local/gpxmapper/cache")),
    )
    assert resolve_reverse_geocode_cache_path() == Path("C:/Users/x/AppData/Local/gpxmapper/reverse_geocode.sqlite")


def test_resolve_reverse_geocode_cache_path_linux_style(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MapRendererBase,
        "resolve_default_cache_directory",
        lambda: str(Path.home() / ".cache" / "gpxmapper"),
    )
    assert resolve_reverse_geocode_cache_path() == Path.home() / ".cache" / "gpxmapper_reverse_geocode.sqlite"


def test_resolve_reverse_geocode_cache_path_macos_style(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MapRendererBase,
        "resolve_default_cache_directory",
        lambda: str(Path.home() / ".gpxmapper" / "cache"),
    )
    assert resolve_reverse_geocode_cache_path() == Path.home() / ".gpxmapper" / "reverse_geocode.sqlite"


def test_reverse_geocode_cache_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "geo.sqlite"
    cache = ReverseGeocodeCache(db)
    assert cache.get_sync("https://example.com", 1.234567, 9.876543) is None
    cache.put_sync("https://example.com", 1.234567, 9.876543, "Somewhere")
    assert cache.get_sync("https://example.com", 1.234567, 9.876543) == "Somewhere"
    # Quantized key: nearby coords hit same cell
    assert cache.get_sync("https://example.com", 1.2345674, 9.8765434) == "Somewhere"
    cache.close_sync()


@pytest.mark.asyncio
async def test_prefetch_second_run_hits_sqlite_cache(tmp_path: Path, mocker) -> None:
    from gpxmapper.geolocation_overlay import prefetch_geolocation_labels

    cache = ReverseGeocodeCache(tmp_path / "prefetch.sqlite")
    mock_client = MagicMock()
    mock_client.reverse_geocode = AsyncMock(
        return_value=MagicMock(display_name="Paris, France"),
    )
    mock_client.aclose = AsyncMock()
    mocker.patch(
        "gpxmapper.geolocation_overlay.GeolocationClientFactory.create_client",
        return_value=mock_client,
    )
    mocker.patch("gpxmapper.geolocation_overlay.is_public_osm_nominatim", return_value=False)

    vg = MagicMock()
    vg._interpolate_position = MagicMock(return_value=(48.8566, 2.3522))

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    points = [GPXTrackPoint(48.8566, 2.3522, None, t0)]

    await prefetch_geolocation_labels(
        vg, points, duration_seconds=1, fps=1, start_time=t0, total_track_seconds=3600.0, geocode_cache=cache
    )
    assert mock_client.reverse_geocode.await_count == 1

    mock_client.reverse_geocode.reset_mock()
    await prefetch_geolocation_labels(
        vg, points, duration_seconds=1, fps=1, start_time=t0, total_track_seconds=3600.0, geocode_cache=cache
    )
    assert mock_client.reverse_geocode.await_count == 0

    cache.close_sync()
