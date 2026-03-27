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
from gpxmapper.map_renderer.base import MapRendererBase


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
    assert mock_get.call_args.kwargs["headers"] == renderer.build_tile_headers()


def test_sync_renderer_none_timeout_uses_valid_default(tmp_path) -> None:
    renderer = MapRenderer(cache_dir=str(tmp_path / "default_timeout"), use_cache=False, request_timeout=None)
    assert renderer.request_timeout == MapRendererBase.get_sync_request_timeout()


def test_sync_renderer_invalid_explicit_timeout_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="request_timeout must be positive"):
        MapRenderer(cache_dir=str(tmp_path / "invalid_timeout"), use_cache=False, request_timeout=0)


def test_renderer_build_tile_headers_contains_required_defaults(tmp_path) -> None:
    renderer = MapRenderer(cache_dir=str(tmp_path / "headers"), use_cache=False)
    headers = renderer.build_tile_headers()
    assert headers["User-Agent"] == "gpxmapper/0.1.0"
    assert headers["Accept"] == "image/png,image/*;q=0.9,*/*;q=0.8"


def test_build_tile_headers_rejects_blank_user_agent() -> None:
    original = MapRendererBase.TILE_USER_AGENT
    try:
        MapRendererBase.TILE_USER_AGENT = "   "
        with pytest.raises(ValueError, match="TILE_USER_AGENT must be a non-empty string"):
            MapRendererBase.build_tile_headers()
    finally:
        MapRendererBase.TILE_USER_AGENT = original


def test_build_tile_headers_rejects_blank_accept() -> None:
    original = MapRendererBase.TILE_ACCEPT
    try:
        MapRendererBase.TILE_ACCEPT = "   "
        with pytest.raises(ValueError, match="TILE_ACCEPT must be a non-empty string"):
            MapRendererBase.build_tile_headers()
    finally:
        MapRendererBase.TILE_ACCEPT = original


def test_sync_timeout_configuration_must_be_positive() -> None:
    original = MapRendererBase.SYNC_REQUEST_TIMEOUT
    try:
        MapRendererBase.SYNC_REQUEST_TIMEOUT = 0.0
        with pytest.raises(ValueError, match="SYNC_REQUEST_TIMEOUT must be positive"):
            MapRendererBase.get_sync_request_timeout()
    finally:
        MapRendererBase.SYNC_REQUEST_TIMEOUT = original


def test_async_fetch_tile_uses_shared_headers(tmp_path, png_bytes) -> None:
    captured_headers = {}

    class _CapturingHttpxAsyncClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):  # noqa: ANN002
            return None

        async def get(self, url: str, **kwargs):  # noqa: ARG002
            captured_headers.update(kwargs.get("headers", {}))
            r = MagicMock()
            r.content = png_bytes
            r.raise_for_status = MagicMock()
            return r

    with patch(
            "gpxmapper.map_renderer.async_renderer.httpx.AsyncClient",
            side_effect=_CapturingHttpxAsyncClient,
    ):
        renderer = MapRendererAsync(cache_dir=str(tmp_path / "async_headers"), use_cache=False)
        tile = renderer.fetch_tile(0, 0, 0)

    assert tile is not None
    assert captured_headers == renderer.build_tile_headers()


def test_async_single_profile_config_is_valid() -> None:
    total_timeout, connect_timeout = MapRendererBase.get_async_timeout_values(
        MapRendererBase.ASYNC_TIMEOUT_PROFILE_SINGLE
    )
    max_keepalive, max_connections = MapRendererBase.get_async_limit_values(
        MapRendererBase.ASYNC_LIMITS_PROFILE_SINGLE
    )
    assert total_timeout > 0
    assert connect_timeout > 0
    assert connect_timeout <= total_timeout
    assert max_connections > 0
    assert max_keepalive >= 0
    assert max_keepalive <= max_connections


def test_async_single_profile_invalid_timeout_raises() -> None:
    original = MapRendererBase.ASYNC_TIMEOUTS
    try:
        MapRendererBase.ASYNC_TIMEOUTS = {
            **original,
            MapRendererBase.ASYNC_TIMEOUT_PROFILE_SINGLE: (5.0, 10.0),
        }
        with pytest.raises(ValueError, match="cannot exceed total timeout"):
            MapRendererBase.get_async_timeout_values(MapRendererBase.ASYNC_TIMEOUT_PROFILE_SINGLE)
    finally:
        MapRendererBase.ASYNC_TIMEOUTS = original


def test_async_single_profile_invalid_limits_raises() -> None:
    original = MapRendererBase.ASYNC_LIMITS
    try:
        MapRendererBase.ASYNC_LIMITS = {
            **original,
            MapRendererBase.ASYNC_LIMITS_PROFILE_SINGLE: (30, 20),
        }
        with pytest.raises(ValueError, match="cannot exceed max connections"):
            MapRendererBase.get_async_limit_values(MapRendererBase.ASYNC_LIMITS_PROFILE_SINGLE)
    finally:
        MapRendererBase.ASYNC_LIMITS = original


def test_async_batch_profile_config_is_valid() -> None:
    total_timeout, connect_timeout = MapRendererBase.get_async_timeout_values(
        MapRendererBase.ASYNC_TIMEOUT_PROFILE_BOUNDS
    )
    max_keepalive, max_connections = MapRendererBase.get_async_limit_values(
        MapRendererBase.ASYNC_LIMITS_PROFILE_BATCH
    )
    assert total_timeout > 0
    assert connect_timeout > 0
    assert connect_timeout <= total_timeout
    assert max_connections > 0
    assert max_keepalive >= 0
    assert max_keepalive <= max_connections


def test_async_bounds_uses_fallback_when_batch_profiles_invalid(tmp_path, png_bytes) -> None:
    captured_client_config = {}

    class _CapturingBatchConfigClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002
            timeout = kwargs.get("timeout")
            limits = kwargs.get("limits")
            captured_client_config["timeout"] = timeout
            captured_client_config["limits"] = limits

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):  # noqa: ANN002
            return None

        async def get(self, url: str, **kwargs):  # noqa: ARG002
            r = MagicMock()
            r.content = png_bytes
            r.raise_for_status = MagicMock()
            return r

    original_timeouts = MapRendererBase.ASYNC_TIMEOUTS
    original_limits = MapRendererBase.ASYNC_LIMITS
    try:
        MapRendererBase.ASYNC_TIMEOUTS = {
            k: v
            for k, v in original_timeouts.items()
            if k != MapRendererBase.ASYNC_TIMEOUT_PROFILE_BOUNDS
        }
        MapRendererBase.ASYNC_LIMITS = {
            **original_limits,
            MapRendererBase.ASYNC_LIMITS_PROFILE_BATCH: (80, 50),
        }
        with patch(
                "gpxmapper.map_renderer.async_renderer.httpx.AsyncClient",
                side_effect=_CapturingBatchConfigClient,
        ):
            renderer = MapRendererAsync(cache_dir=str(tmp_path / "batch_fallback"), use_cache=False)
            tiles = renderer.get_tiles_for_bounds(47.387014, 18.857873, 47.432039, 18.960122, 12)
        assert tiles
        timeout = captured_client_config["timeout"]
        limits = captured_client_config["limits"]
        assert timeout.connect == 10.0
        assert timeout.read == 60.0
        assert limits.max_connections <= 50
        assert limits.max_keepalive_connections <= limits.max_connections
    finally:
        MapRendererBase.ASYNC_TIMEOUTS = original_timeouts
        MapRendererBase.ASYNC_LIMITS = original_limits


def test_adaptive_async_config_caps_connections_to_batch_size() -> None:
    (_, _), (max_keepalive, max_connections) = MapRendererBase.resolve_adaptive_async_client_config(
        timeout_profile=MapRendererBase.ASYNC_TIMEOUT_PROFILE_BOUNDS,
        limits_profile=MapRendererBase.ASYNC_LIMITS_PROFILE_BATCH,
        fallback_timeout=(60.0, 10.0),
        fallback_limits=(20, 50),
        task_count=3,
    )
    assert max_connections == 3
    assert max_keepalive <= max_connections


def test_adaptive_async_config_scales_timeout_for_large_composite_batch() -> None:
    assert set(MapRendererBase.ASYNC_TIMEOUTS) == {
        MapRendererBase.ASYNC_TIMEOUT_PROFILE_SINGLE,
        MapRendererBase.ASYNC_TIMEOUT_PROFILE_BOUNDS,
        MapRendererBase.ASYNC_TIMEOUT_PROFILE_COMPOSITE,
    }
    assert set(MapRendererBase.ASYNC_LIMITS) == {
        MapRendererBase.ASYNC_LIMITS_PROFILE_SINGLE,
        MapRendererBase.ASYNC_LIMITS_PROFILE_BATCH,
    }
    (base_total, base_connect), _ = MapRendererBase.resolve_async_client_config(
        timeout_profile=MapRendererBase.ASYNC_TIMEOUT_PROFILE_COMPOSITE,
        limits_profile=MapRendererBase.ASYNC_LIMITS_PROFILE_BATCH,
        fallback_timeout=(120.0, 10.0),
        fallback_limits=(20, 50),
    )
    (adaptive_total, adaptive_connect), _ = MapRendererBase.resolve_adaptive_async_client_config(
        timeout_profile=MapRendererBase.ASYNC_TIMEOUT_PROFILE_COMPOSITE,
        limits_profile=MapRendererBase.ASYNC_LIMITS_PROFILE_BATCH,
        fallback_timeout=(120.0, 10.0),
        fallback_limits=(20, 50),
        task_count=250,
    )
    assert adaptive_total > base_total
    assert adaptive_connect == base_connect


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
