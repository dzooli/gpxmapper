from __future__ import annotations

import os
from typing import Optional
from unittest.mock import patch

import pytest
from PIL import Image

from gpxmapper.map_renderer.base import MapRendererBase
from gpxmapper.models import MapTile, Point


class _TestRenderer(MapRendererBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tiles: dict[tuple[int, int, int], Optional[MapTile]] = {}

    def fetch_tile(self, x: int, y: int, zoom: int) -> Optional[MapTile]:
        return self._tiles.get((x, y, zoom))

    def get_tiles_for_bounds(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int):
        return []

    def create_composite_map(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int):
        img = Image.new("RGB", (256, 256))
        self.composite_map = img
        self.composite_map_info = {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0, "zoom": zoom, "width": 256,
                                   "height": 256}
        return img


@pytest.fixture
def renderer(tmp_path):
    return _TestRenderer(cache_dir=str(tmp_path / "cache"), use_cache=True)


def test_resolve_default_cache_directory_memoizes_by_os() -> None:
    MapRendererBase._default_cache_dir_by_system.clear()
    with patch("gpxmapper.map_renderer.base.platform.system", return_value="Linux"), patch(
            "gpxmapper.map_renderer.base.os.path.expanduser", return_value="/home/tester"
    ):
        first = MapRendererBase.resolve_default_cache_directory()
        second = MapRendererBase.resolve_default_cache_directory()
    assert first.endswith(os.path.join(".cache", "gpxmapper"))
    assert second == first


def test_resolve_cache_directory_returns_custom_or_default(tmp_path) -> None:
    custom = str(tmp_path / "custom")
    assert _TestRenderer.resolve_cache_directory(custom) == custom
    assert _TestRenderer.resolve_cache_directory(None) is not None


def test_cache_and_open_cached_image_roundtrip_rgb(renderer: _TestRenderer) -> None:
    rgba = Image.new("RGBA", (8, 8), (10, 20, 30, 255))
    renderer.cache_image(1, 2, 3, rgba)
    path = renderer.get_tile_path(1, 2, 3)
    assert path is not None and os.path.exists(path)
    loaded = renderer.open_cached_image(1, 2, 3)
    assert loaded is not None
    assert loaded.mode == "RGB"


def test_get_tile_path_returns_none_when_cache_disabled(tmp_path) -> None:
    r = _TestRenderer(cache_dir=str(tmp_path / "disabled"), use_cache=False)
    assert r.get_tile_path(1, 1, 1) is None


def test_build_tile_url_and_coords(renderer: _TestRenderer) -> None:
    renderer.tile_server = "https://tiles/{z}/{x}/{y}.png"
    assert renderer.build_tile_url(4, 5, 6) == "https://tiles/6/4/5.png"
    coords = renderer.build_tile_coords(0, 1, 2, 3, 4)
    assert coords == [(0, 2, 4), (0, 3, 4), (1, 2, 4), (1, 3, 4)]


def test_compute_geometry_and_set_composite_info(renderer: _TestRenderer) -> None:
    min_tile, max_tile, dims = renderer.compute_composite_geometry(47.387014, 18.857873, 47.432039, 18.960122, 12)
    img = Image.new("RGB", (dims.x, dims.y))
    renderer.set_composite_info(min_tile, max_tile, 12, dims, img)
    assert renderer.composite_map_info["width"] == dims.x
    assert renderer.composite_map_info["height"] == dims.y


def test_render_map_for_point_draws_marker(renderer: _TestRenderer) -> None:
    tile_x, tile_y = renderer.deg2num(0.0, 0.0, 1)
    tile_img = Image.new("RGB", (256, 256), (0, 0, 0))
    renderer._tiles[(tile_x, tile_y, 1)] = MapTile(tile_x, tile_y, 1, tile_img)
    out = renderer.render_map_for_point(0.0, 0.0, 1, marker_color=(255, 0, 0), marker_size=10)
    assert out is not None
    assert out.getbbox() is not None


def test_render_map_for_point_returns_none_when_tile_missing(renderer: _TestRenderer) -> None:
    assert renderer.render_map_for_point(0.0, 0.0, 1) is None


def test_render_from_composite_requires_composite(renderer: _TestRenderer) -> None:
    assert renderer.render_from_composite(0.0, 0.0, 128, 128) is None


def test_render_from_composite_returns_cropped_image(renderer: _TestRenderer) -> None:
    renderer.composite_map = Image.new("RGB", (512, 512), (20, 20, 20))
    renderer.composite_map_info = {
        "min_x": 0,
        "min_y": 0,
        "max_x": 1,
        "max_y": 1,
        "zoom": 1,
        "width": 512,
        "height": 512,
    }
    out = renderer.render_from_composite(0.0, 0.0, 200, 150)
    assert out is not None
    assert out.size == (200, 150)


def test_point_geo_roundtrip_at_zoom() -> None:
    p = Point(1, 1)
    geo = MapRendererBase.point2geo(p, 2)
    p2 = MapRendererBase.deg2point(geo, 2)
    assert p2.x == p.x
    assert abs(p2.y - p.y) <= 1


def test_ensure_rgb_converts_non_rgb() -> None:
    rgba = Image.new("RGBA", (4, 4), (1, 2, 3, 255))
    rgb = MapRendererBase.ensure_rgb(rgba)
    assert rgb.mode == "RGB"


def test_open_cached_image_returns_none_for_invalid_file(renderer: _TestRenderer) -> None:
    bad_path = os.path.join(renderer.cache_dir, "3_4_5.png")
    with open(bad_path, "wb") as handle:
        handle.write(b"not-a-valid-image")
    assert renderer.open_cached_image(4, 5, 3) is None


def test_sync_timeout_validation_rejects_non_positive() -> None:
    original = MapRendererBase.SYNC_REQUEST_TIMEOUT
    try:
        MapRendererBase.SYNC_REQUEST_TIMEOUT = 0
        with pytest.raises(ValueError, match="must be positive"):
            MapRendererBase.get_sync_request_timeout()
    finally:
        MapRendererBase.SYNC_REQUEST_TIMEOUT = original


def test_async_timeout_profile_validation_errors() -> None:
    with pytest.raises(ValueError, match="Unknown async timeout profile"):
        MapRendererBase.get_async_timeout_values("does-not-exist")

    original = MapRendererBase.ASYNC_TIMEOUTS
    try:
        MapRendererBase.ASYNC_TIMEOUTS = {**original, "single": (0.0, 0.0)}
        with pytest.raises(ValueError, match="must be positive"):
            MapRendererBase.get_async_timeout_values("single")
        MapRendererBase.ASYNC_TIMEOUTS = {**original, "single": (5.0, 10.0)}
        with pytest.raises(ValueError, match="cannot exceed total timeout"):
            MapRendererBase.get_async_timeout_values("single")
    finally:
        MapRendererBase.ASYNC_TIMEOUTS = original


def test_async_limit_profile_validation_errors() -> None:
    with pytest.raises(ValueError, match="Unknown async limits profile"):
        MapRendererBase.get_async_limit_values("does-not-exist")

    original = MapRendererBase.ASYNC_LIMITS
    try:
        MapRendererBase.ASYNC_LIMITS = {**original, "batch": (-1, 0)}
        with pytest.raises(ValueError, match="must be non-negative/positive"):
            MapRendererBase.get_async_limit_values("batch")
        MapRendererBase.ASYNC_LIMITS = {**original, "batch": (40, 20)}
        with pytest.raises(ValueError, match="cannot exceed max connections"):
            MapRendererBase.get_async_limit_values("batch")
    finally:
        MapRendererBase.ASYNC_LIMITS = original


def test_resolve_async_client_config_uses_fallbacks_on_errors() -> None:
    with patch("gpxmapper.map_renderer.base.logger.warning") as warn_mock:
        timeout_vals, limit_vals = MapRendererBase.resolve_async_client_config(
            timeout_profile="unknown",
            limits_profile="unknown",
            fallback_timeout=(7.0, 2.0),
            fallback_limits=(3, 6),
        )
    assert timeout_vals == (7.0, 2.0)
    assert limit_vals == (3, 6)
    assert warn_mock.call_count == 2


def test_resolve_adaptive_async_client_config_task_count_branches() -> None:
    # task_count <= 0 should keep base config values
    base_cfg = MapRendererBase.resolve_adaptive_async_client_config(
        timeout_profile=MapRendererBase.ASYNC_TIMEOUT_PROFILE_BOUNDS,
        limits_profile=MapRendererBase.ASYNC_LIMITS_PROFILE_BATCH,
        fallback_timeout=(60.0, 10.0),
        fallback_limits=(20, 50),
        task_count=0,
    )
    assert base_cfg[0] == (60.0, 10.0)
    assert base_cfg[1] == (20, 50)

    # large task_count should scale timeout and cap limits to realistic bounds
    adaptive_cfg = MapRendererBase.resolve_adaptive_async_client_config(
        timeout_profile=MapRendererBase.ASYNC_TIMEOUT_PROFILE_COMPOSITE,
        limits_profile=MapRendererBase.ASYNC_LIMITS_PROFILE_BATCH,
        fallback_timeout=(120.0, 10.0),
        fallback_limits=(20, 50),
        task_count=1000,
    )
    assert adaptive_cfg[0][0] > 120.0
    assert adaptive_cfg[0][0] <= 120.0 * 3.0
    assert adaptive_cfg[0][1] == 10.0
    assert adaptive_cfg[1][0] <= adaptive_cfg[1][1]
