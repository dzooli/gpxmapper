"""Async map renderer using httpx."""

from __future__ import annotations

import asyncio
import io
import logging
from typing import List, Optional

import httpx
from PIL import Image

from .base import MapRendererBase
from .constants import DEFAULT_TILE_SERVER
from ..models import MapTile

logger = logging.getLogger(__name__)


class MapRendererAsync(MapRendererBase):
    """Map renderer with async HTTP over httpx.

    For synchronous call sites (e.g. plain functions, threaded code), use the
    :meth:`fetch_tile`, :meth:`get_tiles_for_bounds`, and
    :meth:`create_composite_map` methods; each runs the async implementation via
    :func:`asyncio.run`.

    From **async** code or when a loop is already running, do **not** use those
    synchronous methods—:func:`asyncio.run` cannot nest. Use
    :meth:`fetch_tile_async`, :meth:`get_tiles_for_bounds_async`, and
    :meth:`create_composite_map_async` instead and ``await`` them.

    The synchronous helpers call :meth:`_reject_if_async_context` before invoking
    :func:`asyncio.run` so misuse fails fast with a clear error.
    """

    def __init__(
            self,
            tile_server: Optional[str] = None,
            cache_dir: Optional[str] = None,
            use_cache: bool = True,
    ):
        resolved_server = tile_server if tile_server is not None else DEFAULT_TILE_SERVER
        super().__init__(resolved_server, cache_dir, use_cache)

    @staticmethod
    def _reject_if_async_context() -> None:
        """Raise if called from a running event loop (``asyncio.run`` would fail)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(
            "MapRendererAsync synchronous methods use asyncio.run() and cannot run inside an "
            "active event loop. Use fetch_tile_async, get_tiles_for_bounds_async, or "
            "create_composite_map_async from async code instead."
        )

    async def _fetch_tile_async(
            self, client: httpx.AsyncClient, x: int, y: int, zoom: int
    ) -> Optional[MapTile]:
        """Fetch a single tile using the given async client."""
        cached = self.open_cached_image(x, y, zoom)
        if cached is not None:
            return MapTile(x, y, zoom, cached)

        url = self.build_tile_url(x, y, zoom)

        try:
            response = await client.get(url, headers=self.build_tile_headers())
            response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))
            image = self.ensure_rgb(image)

            self.cache_image(x, y, zoom, image)

            return MapTile(x, y, zoom, image)

        except Exception as e:
            logger.error(f"Failed to fetch tile {x},{y} at zoom {zoom}: {e}")
            return None

    def fetch_tile(self, x: int, y: int, zoom: int) -> Optional[MapTile]:
        """Fetch a map tile using blocking I/O (runs an event loop via :func:`asyncio.run`).

        Sync-only: must not be used from async code; see :meth:`fetch_tile_async`.
        """
        self._reject_if_async_context()
        return asyncio.run(self._fetch_single_tile_async(x, y, zoom))

    async def _fetch_single_tile_async(self, x: int, y: int, zoom: int) -> Optional[MapTile]:
        """Fetch one tile with a dedicated client session."""
        (total_timeout, connect_timeout), (max_keepalive, max_connections) = self.resolve_adaptive_async_client_config(
            timeout_profile=self.ASYNC_TIMEOUT_PROFILE_SINGLE,
            limits_profile=self.ASYNC_LIMITS_PROFILE_SINGLE,
            fallback_timeout=(30.0, 10.0),
            fallback_limits=(10, 20),
            task_count=1,
        )
        timeout = httpx.Timeout(total_timeout, connect=connect_timeout)
        limits = httpx.Limits(max_keepalive_connections=max_keepalive, max_connections=max_connections)

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            return await self._fetch_tile_async(client, x, y, zoom)

    def get_tiles_for_bounds(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> List[MapTile]:
        """Get tiles for bounds using blocking I/O (via :func:`asyncio.run`).

        Sync-only: must not be used from async code; see :meth:`get_tiles_for_bounds_async`.
        """
        self._reject_if_async_context()
        return asyncio.run(self._get_tiles_for_bounds_async(min_lat, min_lon, max_lat, max_lon, zoom))

    async def _get_tiles_for_bounds_async(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> List[MapTile]:
        tile_coords = self.build_tile_coords_for_bounds(min_lat, min_lon, max_lat, max_lon, zoom)

        (total_timeout, connect_timeout), (max_keepalive, max_connections) = self.resolve_adaptive_async_client_config(
            timeout_profile=self.ASYNC_TIMEOUT_PROFILE_BOUNDS,
            limits_profile=self.ASYNC_LIMITS_PROFILE_BATCH,
            fallback_timeout=(60.0, 10.0),
            fallback_limits=(20, 50),
            task_count=len(tile_coords),
        )
        timeout = httpx.Timeout(total_timeout, connect=connect_timeout)
        limits = httpx.Limits(max_keepalive_connections=max_keepalive, max_connections=max_connections)

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            tasks = [self._fetch_tile_async(client, x, y, z) for x, y, z in tile_coords]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            tiles = []
            for result in results:
                if isinstance(result, MapTile):
                    tiles.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Tile fetch failed with exception: {result}")

        return tiles

    def create_composite_map(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> Image.Image:
        """Create composite map using blocking I/O (via :func:`asyncio.run`).

        Sync-only: must not be used from async code; see :meth:`create_composite_map_async`.
        """
        self._reject_if_async_context()
        return asyncio.run(self._create_composite_map_async(min_lat, min_lon, max_lat, max_lon, zoom))

    async def _create_composite_map_async(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> Image.Image:
        min_tile, max_tile, dimensions = self.compute_composite_geometry(min_lat, min_lon, max_lat, max_lon, zoom)

        composite = Image.new("RGB", (dimensions.x, dimensions.y))

        tile_coords = self.build_tile_coords(min_tile.x, max_tile.x, min_tile.y, max_tile.y, zoom)

        (total_timeout, connect_timeout), (max_keepalive, max_connections) = self.resolve_adaptive_async_client_config(
            timeout_profile=self.ASYNC_TIMEOUT_PROFILE_COMPOSITE,
            limits_profile=self.ASYNC_LIMITS_PROFILE_BATCH,
            fallback_timeout=(120.0, 10.0),
            fallback_limits=(20, 50),
            task_count=len(tile_coords),
        )
        timeout = httpx.Timeout(total_timeout, connect=connect_timeout)
        limits = httpx.Limits(max_keepalive_connections=max_keepalive, max_connections=max_connections)

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            tasks = [self._fetch_tile_async(client, x, y, z) for x, y, z in tile_coords]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            tile_dict = {}
            for result in results:
                if isinstance(result, MapTile):
                    tile_dict[(result.x, result.y)] = result
                elif isinstance(result, Exception):
                    logger.warning(f"Tile fetch failed with exception: {result}")

        self.paste_tiles_to_composite(composite, min_tile.x, min_tile.y, max_tile.x, max_tile.y, zoom, tile_dict)

        self.set_composite_info(min_tile, max_tile, zoom, dimensions, composite)

        return composite

    async def fetch_tile_async(self, x: int, y: int, zoom: int) -> Optional[MapTile]:
        """Fetch a single tile (use from ``async`` code instead of :meth:`fetch_tile`)."""
        return await self._fetch_single_tile_async(x, y, zoom)

    async def get_tiles_for_bounds_async(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> List[MapTile]:
        """Fetch tiles for bounds (use from ``async`` code instead of :meth:`get_tiles_for_bounds`)."""
        return await self._get_tiles_for_bounds_async(min_lat, min_lon, max_lat, max_lon, zoom)

    async def create_composite_map_async(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> Image.Image:
        """Build composite map (use from ``async`` code instead of :meth:`create_composite_map`)."""
        return await self._create_composite_map_async(min_lat, min_lon, max_lat, max_lon, zoom)
