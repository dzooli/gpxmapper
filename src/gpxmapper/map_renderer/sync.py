"""Synchronous map renderer using requests and thread pools."""

from __future__ import annotations

import concurrent.futures
import io
import logging
from functools import partial
from typing import List, Optional

import requests
from PIL import Image

from .base import MapRendererBase
from .constants import DEFAULT_TILE_SERVER
from ..models import MapTile

logger = logging.getLogger(__name__)


class MapRenderer(MapRendererBase):
    """Fetches and renders map tiles using synchronous HTTP and a thread pool."""

    def __init__(
            self,
            tile_server: Optional[str] = None,
            cache_dir: Optional[str] = None,
            use_cache: bool = True,
            request_timeout: float = 30.0,
    ):
        resolved_server = tile_server if tile_server is not None else DEFAULT_TILE_SERVER
        super().__init__(resolved_server, cache_dir, use_cache)
        self.request_timeout = request_timeout

    def fetch_tile(self, x: int, y: int, zoom: int) -> Optional[MapTile]:
        """Fetch a map tile from the server or cache."""
        cached = self.open_cached_image(x, y, zoom)
        if cached is not None:
            return MapTile(x, y, zoom, cached)

        url = self.build_tile_url(x, y, zoom)

        try:
            response = requests.get(
                url,
                headers={"User-Agent": "gpxmapper/0.1.0"},
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))
            image = self.ensure_rgb(image)

            self.cache_image(x, y, zoom, image)

            return MapTile(x, y, zoom, image)

        except Exception as e:
            logger.error(f"Failed to fetch tile {x},{y} at zoom {zoom}: {e}")
            return None

    def _fetch_tile_wrapper(self, coords):
        """Wrapper for fetch_tile to use with concurrent.futures."""
        x, y, zoom = coords
        return self.fetch_tile(x, y, zoom)

    def get_tiles_for_bounds(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> List[MapTile]:
        """Get all tiles needed to cover the given geographic bounds."""
        tile_coords = self.build_tile_coords_for_bounds(min_lat, min_lon, max_lat, max_lon, zoom)

        tiles = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            fetch_tile_partial = partial(self._fetch_tile_wrapper)
            results = list(executor.map(fetch_tile_partial, tile_coords))
            tiles = [tile for tile in results if tile]

        return tiles

    def create_composite_map(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> Image.Image:
        """Create a large composite image from all tiles in the bounding box."""
        min_tile, max_tile, dimensions = self.compute_composite_geometry(
            min_lat, min_lon, max_lat, max_lon, zoom
        )

        composite = Image.new("RGB", (dimensions.x, dimensions.y))

        tile_coords = self.build_tile_coords(min_tile.x, max_tile.x, min_tile.y, max_tile.y, zoom)

        tile_dict = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            fetch_tile_partial = partial(self._fetch_tile_wrapper)
            results = list(executor.map(fetch_tile_partial, tile_coords))

            for tile in results:
                if tile:
                    tile_dict[(tile.x, tile.y)] = tile

        self.paste_tiles_to_composite(composite, min_tile.x, min_tile.y, max_tile.x, max_tile.y, zoom, tile_dict)

        self.set_composite_info(min_tile, max_tile, zoom, dimensions, composite)

        return composite
