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
    """Map renderer with async HTTP; synchronous methods delegate via asyncio.run where needed."""

    def __init__(
            self,
            tile_server: Optional[str] = None,
            cache_dir: Optional[str] = None,
            use_cache: bool = True,
    ):
        resolved_server = tile_server if tile_server is not None else DEFAULT_TILE_SERVER
        super().__init__(resolved_server, cache_dir, use_cache)

    async def _fetch_tile_async(
            self, client: httpx.AsyncClient, x: int, y: int, zoom: int
    ) -> Optional[MapTile]:
        """Fetch a single tile using the given async client."""
        cached = self.open_cached_image(x, y, zoom)
        if cached is not None:
            return MapTile(x, y, zoom, cached)

        url = self.build_tile_url(x, y, zoom)

        try:
            response = await client.get(url, headers={"User-Agent": "gpxmapper/0.1.0"})
            response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))
            image = self.ensure_rgb(image)

            self.cache_image(x, y, zoom, image)

            return MapTile(x, y, zoom, image)

        except Exception as e:
            logger.error(f"Failed to fetch tile {x},{y} at zoom {zoom}: {e}")
            return None

    def fetch_tile(self, x: int, y: int, zoom: int) -> Optional[MapTile]:
        """Fetch a map tile (synchronous interface wrapping async I/O)."""
        return asyncio.run(self._fetch_single_tile_async(x, y, zoom))

    async def _fetch_single_tile_async(self, x: int, y: int, zoom: int) -> Optional[MapTile]:
        """Fetch one tile with a dedicated client session."""
        timeout = httpx.Timeout(30.0, connect=10.0)
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            return await self._fetch_tile_async(client, x, y, zoom)

    def get_tiles_for_bounds(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> List[MapTile]:
        """Get tiles for bounds (synchronous interface)."""
        return asyncio.run(self._get_tiles_for_bounds_async(min_lat, min_lon, max_lat, max_lon, zoom))

    async def _get_tiles_for_bounds_async(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> List[MapTile]:
        tile_coords = self.build_tile_coords_for_bounds(min_lat, min_lon, max_lat, max_lon, zoom)

        timeout = httpx.Timeout(60.0, connect=10.0)
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)

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
        """Create composite map (synchronous interface)."""
        return asyncio.run(self._create_composite_map_async(min_lat, min_lon, max_lat, max_lon, zoom))

    async def _create_composite_map_async(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> Image.Image:
        min_tile, max_tile, dimensions = self.compute_composite_geometry(min_lat, min_lon, max_lat, max_lon, zoom)

        composite = Image.new("RGB", (dimensions.x, dimensions.y))

        tile_coords = self.build_tile_coords(min_tile.x, max_tile.x, min_tile.y, max_tile.y, zoom)

        timeout = httpx.Timeout(120.0, connect=10.0)
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)

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
        """Direct async interface for fetching a single tile."""
        return await self._fetch_single_tile_async(x, y, zoom)

    async def get_tiles_for_bounds_async(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> List[MapTile]:
        """Direct async interface for fetching tiles for bounds."""
        return await self._get_tiles_for_bounds_async(min_lat, min_lon, max_lat, max_lon, zoom)

    async def create_composite_map_async(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> Image.Image:
        """Direct async interface for creating composite maps."""
        return await self._create_composite_map_async(min_lat, min_lon, max_lat, max_lon, zoom)
