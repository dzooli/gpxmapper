"""Performance and parity checks for async vs sync map renderers (HTTP mocked)."""

from __future__ import annotations

import asyncio
import io
import os
import time
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from gpxmapper.map_renderer import MapRenderer, MapRendererAsync


def _solid_tile_png(rgb: tuple[int, int, int] = (90, 120, 150)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), rgb).save(buf, format="PNG")
    return buf.getvalue()


def _sync_get_response(png_bytes: bytes):
    r = MagicMock()
    r.content = png_bytes
    r.raise_for_status = MagicMock()
    return r


class _FakeHttpxAsyncClient:
    """Minimal async client matching ``async with httpx.AsyncClient()`` usage."""

    def __init__(self, delay_s: float, png_bytes: bytes):
        self._delay = delay_s
        self._png = png_bytes

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):  # noqa: ANN002
        return None

    async def get(self, url: str, **kwargs):  # noqa: ARG002
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        r = MagicMock()
        r.content = self._png
        r.raise_for_status = MagicMock()
        return r


def _bounds_four_tiles_zoom_12() -> tuple[float, float, float, float, int]:
    """Geographic box that requires four tiles at zoom 12 (matches smoke-test scale)."""
    return (47.387014, 18.857873, 47.432039, 18.960122, 12)


@pytest.fixture
def png_bytes() -> bytes:
    return _solid_tile_png()


def test_async_composite_matches_sync_with_mocked_http(tmp_path, png_bytes) -> None:
    """Both backends must build the same composite when tile bytes are identical."""
    min_lat, min_lon, max_lat, max_lon, zoom = _bounds_four_tiles_zoom_12()

    def fake_client(delay: float, data: bytes):
        def _factory(*args, **kwargs):  # noqa: ANN002
            return _FakeHttpxAsyncClient(delay, data)

        return _factory

    with patch("gpxmapper.map_renderer.sync.requests.get", return_value=_sync_get_response(png_bytes)):
        sync_r = MapRenderer(cache_dir=str(tmp_path / "s"), use_cache=False)
        sync_img = sync_r.create_composite_map(min_lat, min_lon, max_lat, max_lon, zoom)

    with patch(
            "gpxmapper.map_renderer.async_renderer.httpx.AsyncClient",
            side_effect=fake_client(0.0, png_bytes),
    ):
        async_r = MapRendererAsync(cache_dir=str(tmp_path / "a"), use_cache=False)
        async_img = async_r.create_composite_map(min_lat, min_lon, max_lat, max_lon, zoom)

    assert sync_img.size == async_img.size
    assert sync_img.tobytes() == async_img.tobytes()
    assert sync_r.composite_map_info == async_r.composite_map_info


def test_sync_fetch_tile_passes_configured_timeout(tmp_path, png_bytes) -> None:
    """Sync renderer should propagate request timeout to requests.get."""
    with patch("gpxmapper.map_renderer.sync.requests.get", return_value=_sync_get_response(png_bytes)) as mock_get:
        renderer = MapRenderer(cache_dir=str(tmp_path / "timeout"), use_cache=False, request_timeout=12.5)
        tile = renderer.fetch_tile(0, 0, 0)

    assert tile is not None
    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs["timeout"] == 12.5


@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("GPXMAPPER_PERF_TESTS"),
    reason="Perf-sensitive; enable with GPXMAPPER_PERF_TESTS=1",
)
def test_async_composite_parallel_fetch_performance(tmp_path, png_bytes) -> None:
    """Per-tile delay × tile count should not apply serially (parallel fetch)."""
    min_lat, min_lon, max_lat, max_lon, zoom = _bounds_four_tiles_zoom_12()
    delay = 0.04

    def fake_client(*args, **kwargs):  # noqa: ANN002
        return _FakeHttpxAsyncClient(delay, png_bytes)

    with patch(
            "gpxmapper.map_renderer.async_renderer.httpx.AsyncClient",
            side_effect=fake_client,
    ):
        renderer = MapRendererAsync(cache_dir=str(tmp_path / "perf"), use_cache=False)
        t0 = time.perf_counter()
        renderer.create_composite_map(min_lat, min_lon, max_lat, max_lon, zoom)
        elapsed = time.perf_counter() - t0

    w = renderer.composite_map_info["width"]  # type: ignore[index]
    tile_count = (w // 256) ** 2  # square composite from smoke path
    serial_floor = delay * tile_count
    max_allowed = max(serial_floor * 0.8, delay + 0.5)

    assert tile_count >= 4
    assert elapsed < serial_floor, (
        f"async composite took {elapsed:.3f}s, expected below serial {serial_floor:.3f}s "
        f"for {tile_count} tiles at {delay}s each"
    )
    assert elapsed < max_allowed, (
        f"async composite unexpectedly slow: elapsed={elapsed:.3f}s, "
        f"max_allowed={max_allowed:.3f}s"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("GPXMAPPER_PERF_TESTS"),
    reason="Perf-sensitive; enable with GPXMAPPER_PERF_TESTS=1",
)
def test_sync_composite_parallel_fetch_performance(tmp_path, png_bytes) -> None:
    """Sync renderer uses a thread pool; ensure fetches are not strictly serial."""
    min_lat, min_lon, max_lat, max_lon, zoom = _bounds_four_tiles_zoom_12()
    delay = 0.04

    def slow_get(*args, **kwargs):  # noqa: ANN002
        time.sleep(delay)
        return _sync_get_response(png_bytes)

    with patch("gpxmapper.map_renderer.sync.requests.get", side_effect=slow_get):
        renderer = MapRenderer(cache_dir=str(tmp_path / "perf_sync"), use_cache=False)
        t0 = time.perf_counter()
        renderer.create_composite_map(min_lat, min_lon, max_lat, max_lon, zoom)
        elapsed = time.perf_counter() - t0

    w = renderer.composite_map_info["width"]  # type: ignore[index]
    tile_count = (w // 256) ** 2
    serial_floor = delay * tile_count
    max_allowed = max(serial_floor * 0.8, delay + 0.5)

    assert tile_count >= 4
    assert elapsed < serial_floor, (
        f"sync composite took {elapsed:.3f}s, expected below ~serial {serial_floor:.3f}s"
    )
    assert elapsed < max_allowed, (
        f"sync composite unexpectedly slow: elapsed={elapsed:.3f}s, "
        f"max_allowed={max_allowed:.3f}s"
    )


@pytest.mark.asyncio
async def test_async_renderer_sync_methods_fail_inside_event_loop(tmp_path, png_bytes) -> None:
    """Blocking helpers that wrap asyncio.run must not run under an active loop."""

    def fake_client(*args, **kwargs):  # noqa: ANN002
        return _FakeHttpxAsyncClient(0.0, png_bytes)

    with patch(
            "gpxmapper.map_renderer.async_renderer.httpx.AsyncClient",
            side_effect=fake_client,
    ):
        r = MapRendererAsync(cache_dir=str(tmp_path / "loop"), use_cache=False)

    with pytest.raises(RuntimeError, match="fetch_tile_async"):
        r.fetch_tile(0, 0, 0)
