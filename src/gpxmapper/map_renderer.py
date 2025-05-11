"""Map rendering module for fetching and rendering map tiles.

This module has been optimized for performance with the following improvements:
1. Parallel tile fetching using ThreadPoolExecutor
2. Caching of coordinate conversion calculations
3. Efficient composite map creation
"""

import os
import math
import io
import requests
import platform
from typing import Tuple, List, Optional
from PIL import Image, ImageDraw
import logging
import concurrent.futures
from functools import partial

from .models import MapTile

logger = logging.getLogger(__name__)

# Default tile server URL template
DEFAULT_TILE_SERVER = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


class MapRenderer:
    """Handles fetching and rendering map tiles."""

    def __init__(self, tile_server: str = DEFAULT_TILE_SERVER, cache_dir: Optional[str] = None, use_cache: bool = True):
        """Initialize the map renderer.

        Args:
            tile_server: URL template for the tile server
            cache_dir: Directory to cache downloaded tiles. If None, a default directory
                      based on the operating system will be used.
            use_cache: Whether to use tile caching. Default is True.
        """
        self.tile_server = tile_server
        self.composite_map = None
        self.composite_map_info = None
        self.use_cache = use_cache

        # Set default cache directory based on OS if not provided
        if cache_dir is None:
            logger.info("Using OS dependent cache dir")
            system = platform.system()
            if system == "Windows":
                # Windows default: %LOCALAPPDATA%\gpxmapper\cache
                appdata = os.environ.get("LOCALAPPDATA")
                if appdata:
                    cache_dir = os.path.join(appdata, "gpxmapper", "cache")
                else:
                    # Fallback if LOCALAPPDATA is not available
                    cache_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "gpxmapper", "cache")
            elif system == "Linux":
                # Linux default: ~/.cache/gpxmapper
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "gpxmapper")
            else:
                # Default for other OS
                cache_dir = os.path.join(os.path.expanduser("~"), ".gpxmapper", "cache")

        self.cache_dir = cache_dir
        logger.info(f"Using cache dir: {self.cache_dir}")

        # Create cache directory if it doesn't exist
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    # Cache for 2^zoom values to avoid repeated calculations
    _zoom_cache = {}

    @classmethod
    def deg2num(cls, lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
        """Convert latitude and longitude to tile coordinates.

        Args:
            lat_deg: Latitude in degrees
            lon_deg: Longitude in degrees
            zoom: Zoom level

        Returns:
            Tuple of (x, y) tile coordinates
        """
        # Get or calculate 2^zoom
        if zoom not in cls._zoom_cache:
            cls._zoom_cache[zoom] = 2.0 ** zoom
        n = cls._zoom_cache[zoom]

        lat_rad = math.radians(lat_deg)
        x = int((lon_deg + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    @classmethod
    def num2deg(cls, xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
        """Convert tile coordinates to latitude and longitude.

        Args:
            xtile: X tile coordinate
            ytile: Y tile coordinate
            zoom: Zoom level

        Returns:
            Tuple of (latitude, longitude) in degrees
        """
        # Get or calculate 2^zoom
        if zoom not in cls._zoom_cache:
            cls._zoom_cache[zoom] = 2.0 ** zoom
        n = cls._zoom_cache[zoom]

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
            Path to the cached tile or None if not cached or caching is disabled
        """
        if not self.use_cache or not self.cache_dir:
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
        # Check cache first if caching is enabled
        if self.use_cache:
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

            # Cache the tile if caching is enabled and cache_dir is set
            if self.use_cache and self.cache_dir:
                logger.info(f"Caching tile {x},{y} at zoom {zoom}")
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

        # Create a list of all tile coordinates to fetch
        tile_coords = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tile_coords.append((x, y, zoom))

        # Fetch tiles in parallel
        tiles = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Create a partial function with self as the first argument
            fetch_tile_partial = partial(self._fetch_tile_wrapper)
            # Map the function to all tile coordinates
            results = list(executor.map(fetch_tile_partial, tile_coords))
            # Filter out None results
            tiles = [tile for tile in results if tile]

        return tiles

    def _fetch_tile_wrapper(self, coords):
        """Wrapper for fetch_tile to use with concurrent.futures.

        Args:
            coords: Tuple of (x, y, zoom)

        Returns:
            MapTile object or None if fetching failed
        """
        x, y, zoom = coords
        return self.fetch_tile(x, y, zoom)

    def create_composite_map(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int) -> Image.Image:
        """Create a large composite image from all tiles in the bounding box.

        Args:
            min_lat: Minimum latitude
            min_lon: Minimum longitude
            max_lat: Maximum latitude
            max_lon: Maximum longitude
            zoom: Zoom level

        Returns:
            PIL Image of the composite map
        """
        # Get tile coordinates for the bounding box
        min_x, max_y = self.deg2num(min_lat, min_lon, zoom)
        max_x, min_y = self.deg2num(max_lat, max_lon, zoom)

        # Calculate the size of the composite image
        width = (max_x - min_x + 1) * 256
        height = (max_y - min_y + 1) * 256

        # Create a blank image
        composite = Image.new('RGB', (width, height))

        # Create a list of all tile coordinates to fetch
        tile_coords = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tile_coords.append((x, y, zoom))

        # Fetch tiles in parallel
        tile_dict = {}  # Dictionary to store tiles by coordinates
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Create a partial function with self as the first argument
            fetch_tile_partial = partial(self._fetch_tile_wrapper)
            # Map the function to all tile coordinates
            results = list(executor.map(fetch_tile_partial, tile_coords))

            # Store tiles in a dictionary for easy lookup
            for tile in results:
                if tile:
                    tile_dict[(tile.x, tile.y)] = tile

        # Place all tiles on the composite image
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tile = tile_dict.get((x, y))
                if tile and tile.image:
                    # Calculate position in the composite image
                    pos_x = (x - min_x) * 256
                    pos_y = (y - min_y) * 256
                    composite.paste(tile.image, (pos_x, pos_y))
                else:
                    # If tile is missing, create a blank tile
                    logger.warning(f"Missing tile at {x},{y}, zoom {zoom}")
                    blank_tile = Image.new('RGB', (256, 256), (200, 200, 200))
                    pos_x = (x - min_x) * 256
                    pos_y = (y - min_y) * 256
                    composite.paste(blank_tile, (pos_x, pos_y))

        # Store the composite map and its metadata
        self.composite_map = composite
        self.composite_map_info = {
            'min_x': min_x,
            'min_y': min_y,
            'max_x': max_x,
            'max_y': max_y,
            'zoom': zoom,
            'width': width,
            'height': height
        }

        return composite

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

    def render_from_composite(self, lat: float, lon: float, frame_width: int, frame_height: int,
                             marker_color: Tuple[int, int, int] = (255, 0, 0),
                             marker_size: int = 10) -> Optional[Image.Image]:
        """Render a map from the composite map centered on the given coordinates with a marker.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            frame_width: Width of the output frame in pixels
            frame_height: Height of the output frame in pixels
            marker_color: RGB color tuple for the marker
            marker_size: Size of the marker in pixels

        Returns:
            PIL Image of the rendered map or None if rendering failed
        """
        # Check if composite map exists
        if self.composite_map is None or self.composite_map_info is None:
            logger.error("Composite map not created. Call create_composite_map first.")
            return None

        # Get zoom level from composite map info
        zoom = self.composite_map_info['zoom']

        # Calculate pixel coordinates within the composite map
        n = 2.0 ** zoom
        lat_rad = math.radians(lat)
        global_x = int((lon + 180.0) / 360.0 * n * 256)
        global_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * 256)

        # Calculate position in the composite map
        min_x = self.composite_map_info['min_x']
        min_y = self.composite_map_info['min_y']

        # Position in the composite map (pixels)
        x_pixel = (global_x - min_x * 256)
        y_pixel = (global_y - min_y * 256)

        # Calculate the crop box
        left = max(0, x_pixel - frame_width // 2)
        top = max(0, y_pixel - frame_height // 2)
        right = min(self.composite_map_info['width'], left + frame_width)
        bottom = min(self.composite_map_info['height'], top + frame_height)

        # Adjust if we hit the edge of the composite map
        if right - left < frame_width:
            if left == 0:
                right = min(self.composite_map_info['width'], frame_width)
            else:
                left = max(0, right - frame_width)

        if bottom - top < frame_height:
            if top == 0:
                bottom = min(self.composite_map_info['height'], frame_height)
            else:
                top = max(0, bottom - frame_height)

        # Crop the composite map
        cropped = self.composite_map.crop((left, top, right, bottom))

        # Create a copy of the cropped image to draw on
        result = cropped.copy()
        draw = ImageDraw.Draw(result)

        # Calculate marker position in the cropped image
        marker_x = x_pixel - left
        marker_y = y_pixel - top

        # Draw the marker
        draw.ellipse(
            (marker_x - marker_size//2, marker_y - marker_size//2, 
             marker_x + marker_size//2, marker_y + marker_size//2), 
            fill=marker_color
        )

        return result
