"""Async map rendering module for fetching and rendering map tiles.

This module provides an async version of MapRenderer that inherits from the original
MapRenderer class to avoid code duplication while providing async HTTP capabilities.
It's designed as a drop-in replacement for the original MapRenderer class.

Usage Example:

# Simply change the import:
# from .map_renderer import MapRenderer
from .map_renderer_async import MapRendererAsync as MapRenderer

# Everything else stays exactly the same!
renderer = MapRenderer()
tiles = renderer.get_tiles_for_bounds(lat1, lon1, lat2, lon2, zoom)
composite = renderer.create_composite_map(lat1, lon1, lat2, lon2, zoom)
single_tile_map = renderer.render_map_for_point(lat, lon, zoom)
cropped_map = renderer.render_from_composite(lat, lon, width, height)

"""
import asyncio
import io
import logging
from typing import List, Optional

import httpx
from PIL import Image

from .map_renderer import MapRenderer
from .models import MapTile

logger = logging.getLogger(__name__)


class MapRendererAsync(MapRenderer):
    """Async version of MapRenderer that inherits from the original to avoid code duplication.

    This class overrides only the HTTP-related methods to provide async functionality
    while inheriting all other functionality from the parent class.
    """

    def __init__(self, tile_server: str = None, cache_dir: Optional[str] = None, use_cache: bool = True):
        """Initialize the async map renderer.

        Args:
            tile_server: URL template for the tile server. If None, uses default from parent.
            cache_dir: Directory to cache downloaded tiles. If None, uses OS-dependent default.
            use_cache: Whether to use tile caching. Default is True.
        """
        # Import here to avoid circular imports
        from .map_renderer import DEFAULT_TILE_SERVER

        if tile_server is None:
            tile_server = DEFAULT_TILE_SERVER

        # Call parent constructor to handle all initialization logic
        super().__init__(tile_server, cache_dir, use_cache)

    async def _fetch_tile_async(self, client: httpx.AsyncClient, x: int, y: int, zoom: int) -> Optional[MapTile]:
        """Internal async method to fetch a single tile.

        Args:
            client: httpx AsyncClient for making HTTP requests
            x: X coordinate of the tile
            y: Y coordinate of the tile
            zoom: Zoom level

        Returns:
            MapTile object or None if fetching failed
        """
        # Check cache first using inherited helper
        cached = self.open_cached_image(x, y, zoom)
        if cached is not None:
            return MapTile(x, y, zoom, cached)

        # Fetch from a server using async HTTP
        url = self.build_tile_url(x, y, zoom)

        try:
            response = await client.get(url, headers={"User-Agent": "gpxmapper/0.1.0"})
            response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))

            # Convert to RGB mode to ensure full color support
            image = self.ensure_rgb(image)

            # Cache the tile if caching is enabled using inherited logic
            self.cache_image(x, y, zoom, image)

            return MapTile(x, y, zoom, image)

        except Exception as e:
            logger.error(f"Failed to fetch tile {x},{y} at zoom {zoom}: {e}")
            return None

    def fetch_tile(self, x: int, y: int, zoom: int) -> Optional[MapTile]:
        """Override: Fetch a map tile from the server or cache (synchronous interface).

        Args:
            x: X coordinate of the tile
            y: Y coordinate of the tile
            zoom: Zoom level

        Returns:
            MapTile object or None if fetching failed
        """
        return asyncio.run(self._fetch_single_tile_async(x, y, zoom))

    async def _fetch_single_tile_async(self, x: int, y: int, zoom: int) -> Optional[MapTile]:
        """Async method to fetch a single tile with its own client session."""
        timeout = httpx.Timeout(30.0, connect=10.0)
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            return await self._fetch_tile_async(client, x, y, zoom)

    def get_tiles_for_bounds(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float,
                             zoom: int) -> List[MapTile]:
        """Override: Get all tiles needed to cover the given geographic bounds (synchronous interface).

        Args:
            min_lat: Minimum latitude
            min_lon: Minimum longitude
            max_lat: Maximum latitude
            max_lon: Maximum longitude
            zoom: Zoom level

        Returns:
            List of MapTile objects
        """
        return asyncio.run(self._get_tiles_for_bounds_async(min_lat, min_lon, max_lat, max_lon, zoom))

    async def _get_tiles_for_bounds_async(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float,
                                          zoom: int) -> List[MapTile]:
        """Async method to get all tiles needed to cover the given geographic bounds.

        Args:
            min_lat: Minimum latitude
            min_lon: Minimum longitude
            max_lat: Maximum latitude
            max_lon: Maximum longitude
            zoom: Zoom level

        Returns:
            List of MapTile objects
        """
        # Build tile coordinates using a shared helper
        tile_coords = self.build_tile_coords_for_bounds(min_lat, min_lon, max_lat, max_lon, zoom)

        # Configure httpx client for optimal performance
        timeout = httpx.Timeout(60.0, connect=10.0)
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            # Create tasks for all tile fetches
            tasks = [
                self._fetch_tile_async(client, x, y, z)
                for x, y, z in tile_coords
            ]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out None results and exceptions
            tiles = []
            for result in results:
                if isinstance(result, MapTile):
                    tiles.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Tile fetch failed with exception: {result}")

        return tiles

    def create_composite_map(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float,
                             zoom: int) -> Image.Image:
        """Override: Create a large composite image from all tiles in the bounding box (synchronous interface).

        Args:
            min_lat: Minimum latitude
            min_lon: Minimum longitude
            max_lat: Maximum latitude
            max_lon: Maximum longitude
            zoom: Zoom level

        Returns:
            PIL Image of the composite map
        """
        return asyncio.run(self._create_composite_map_async(min_lat, min_lon, max_lat, max_lon, zoom))

    async def _create_composite_map_async(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float,
                                          zoom: int) -> Image.Image:
        """Async method to create a large composite image from all tiles in the bounding box.

        This method reuses the parent's logic for image composition while using async tile fetching.
        """
        # Compute composite geometry using a shared helper
        min_tile, max_tile, dimensions = self.compute_composite_geometry(min_lat, min_lon, max_lat, max_lon, zoom)

        # Create a blank image
        composite = Image.new('RGB', (dimensions.x, dimensions.y))

        # Create a list of all tile coordinates to fetch using helper
        tile_coords = self.build_tile_coords(min_tile.x, max_tile.x, min_tile.y, max_tile.y, zoom)

        # Configure httpx client for optimal performance
        timeout = httpx.Timeout(120.0, connect=10.0)  # Longer timeout for composite maps
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            # Create tasks for all tile fetches
            tasks = [
                self._fetch_tile_async(client, x, y, z)
                for x, y, z in tile_coords
            ]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Store tiles in a dictionary for easy lookup
            tile_dict = {}
            for result in results:
                if isinstance(result, MapTile):
                    tile_dict[(result.x, result.y)] = result
                elif isinstance(result, Exception):
                    logger.warning(f"Tile fetch failed with exception: {result}")

        # Place all tiles on the composite image using a shared helper
        self.paste_tiles_to_composite(composite, min_tile.x, min_tile.y, max_tile.x, max_tile.y, zoom, tile_dict)

        # Store the composite map and its metadata (reuse parent logic)
        self.set_composite_info(min_tile, max_tile, zoom, dimensions, composite)

        return composite

    # Additional async methods for direct async usage (optional)
    async def fetch_tile_async(self, x: int, y: int, zoom: int) -> Optional[MapTile]:
        """Direct async interface for fetching a single tile."""
        return await self._fetch_single_tile_async(x, y, zoom)

    async def get_tiles_for_bounds_async(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float,
                                         zoom: int) -> List[MapTile]:
        """Direct async interface for fetching tiles for bounds."""
        return await self._get_tiles_for_bounds_async(min_lat, min_lon, max_lat, max_lon, zoom)

    async def create_composite_map_async(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float,
                                         zoom: int) -> Image.Image:
        """Direct async interface for creating composite maps."""
        return await self._create_composite_map_async(min_lat, min_lon, max_lat, max_lon, zoom)

    # Note: render_map_for_point and render_from_composite are inherited unchanged
    # since they only use fetch_tile which we've overridden to use async internally
