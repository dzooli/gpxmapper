"""Map rendering module for fetching and rendering map tiles."""

import os
import math
import io
import requests
from typing import Tuple, List, Optional
from PIL import Image, ImageDraw
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default tile server URL template
DEFAULT_TILE_SERVER = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

class MapTile:
    """Represents a single map tile with its coordinates and image data."""

    def __init__(self, x: int, y: int, zoom: int, image: Optional[Image.Image] = None):
        """Initialize a map tile.

        Args:
            x: X coordinate of the tile
            y: Y coordinate of the tile
            zoom: Zoom level
            image: Optional PIL Image object of the tile
        """
        self.x = x
        self.y = y
        self.zoom = zoom
        self.image = image

    def __repr__(self) -> str:
        return f"MapTile(x={self.x}, y={self.y}, zoom={self.zoom})"


class MapRenderer:
    """Handles fetching and rendering map tiles."""

    def __init__(self, tile_server: str = DEFAULT_TILE_SERVER, cache_dir: Optional[str] = None):
        """Initialize the map renderer.

        Args:
            tile_server: URL template for the tile server
            cache_dir: Directory to cache downloaded tiles
        """
        self.tile_server = tile_server
        self.cache_dir = cache_dir

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
        """Convert latitude and longitude to tile coordinates.

        Args:
            lat_deg: Latitude in degrees
            lon_deg: Longitude in degrees
            zoom: Zoom level

        Returns:
            Tuple of (x, y) tile coordinates
        """
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        x = int((lon_deg + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    @staticmethod
    def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
        """Convert tile coordinates to latitude and longitude.

        Args:
            xtile: X tile coordinate
            ytile: Y tile coordinate
            zoom: Zoom level

        Returns:
            Tuple of (latitude, longitude) in degrees
        """
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg

    def get_tile_path(self, x: int, y: int, zoom: int) -> Optional[str]:
        """Get the path to a cached tile if it exists.

        Args:
            x: X coordinate of the tile
            y: Y coordinate of the tile
            zoom: Zoom level

        Returns:
            Path to the cached tile or None if not cached
        """
        if not self.cache_dir:
            return None

        tile_path = os.path.join(self.cache_dir, f"{zoom}_{x}_{y}.png")
        if os.path.exists(tile_path):
            return tile_path

        return None

    def fetch_tile(self, x: int, y: int, zoom: int) -> Optional[MapTile]:
        """Fetch a map tile from the server or cache.

        Args:
            x: X coordinate of the tile
            y: Y coordinate of the tile
            zoom: Zoom level

        Returns:
            MapTile object or None if fetching failed
        """
        # Check cache first
        tile_path = self.get_tile_path(x, y, zoom)
        if tile_path:
            try:
                image = Image.open(tile_path)
                # Convert to RGB mode to ensure full color support
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return MapTile(x, y, zoom, image)
            except Exception as e:
                logger.warning(f"Failed to load cached tile: {e}")

        # Fetch from server
        url = self.tile_server.replace("{x}", str(x)).replace("{y}", str(y)).replace("{z}", str(zoom))

        try:
            response = requests.get(url, headers={"User-Agent": "gpxmapper/0.1.0"})
            response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))

            # Convert to RGB mode to ensure full color support
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Cache the tile if cache_dir is set
            if self.cache_dir:
                tile_path = os.path.join(self.cache_dir, f"{zoom}_{x}_{y}.png")
                image.save(tile_path)

            return MapTile(x, y, zoom, image)

        except Exception as e:
            logger.error(f"Failed to fetch tile {x},{y} at zoom {zoom}: {e}")
            return None

    def get_tiles_for_bounds(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, 
                            zoom: int) -> List[MapTile]:
        """Get all tiles needed to cover the given geographic bounds.

        Args:
            min_lat: Minimum latitude
            min_lon: Minimum longitude
            max_lat: Maximum latitude
            max_lon: Maximum longitude
            zoom: Zoom level

        Returns:
            List of MapTile objects
        """
        min_x, max_y = self.deg2num(min_lat, min_lon, zoom)
        max_x, min_y = self.deg2num(max_lat, max_lon, zoom)

        tiles = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tile = self.fetch_tile(x, y, zoom)
                if tile:
                    tiles.append(tile)

        return tiles

    def render_map_for_point(self, lat: float, lon: float, zoom: int, 
                            marker_color: Tuple[int, int, int] = (255, 0, 0),
                            marker_size: int = 10) -> Optional[Image.Image]:
        """Render a map centered on the given coordinates with a marker.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            zoom: Zoom level
            marker_color: RGB color tuple for the marker
            marker_size: Size of the marker in pixels

        Returns:
            PIL Image of the rendered map or None if rendering failed
        """
        # Get tile coordinates
        tile_x, tile_y = self.deg2num(lat, lon, zoom)

        # Fetch the tile
        tile = self.fetch_tile(tile_x, tile_y, zoom)
        if not tile or not tile.image:
            logger.error(f"Failed to fetch tile for coordinates {lat}, {lon} at zoom {zoom}")
            return None

        # Create a copy of the image to draw on
        result = tile.image.copy()
        draw = ImageDraw.Draw(result)

        # Calculate pixel coordinates within the tile
        n = 2.0 ** zoom
        lat_rad = math.radians(lat)
        x_pixel = int((lon + 180.0) / 360.0 * n * 256) % 256
        y_pixel = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * 256) % 256

        # Draw the marker
        draw.ellipse(
            (x_pixel - marker_size//2, y_pixel - marker_size//2, 
             x_pixel + marker_size//2, y_pixel + marker_size//2), 
            fill=marker_color
        )

        return result
