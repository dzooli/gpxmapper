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

from .models import MapTile, Point, GeoPoint, Rectangle

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
    def deg2point(cls, geo_point: GeoPoint, zoom: int) -> Point:
        """Convert a GeoPoint to a tile Point.

        Args:
            geo_point: GeoPoint with latitude and longitude
            zoom: Zoom level

        Returns:
            Point with x, y tile coordinates
        """
        x, y = cls.deg2num(geo_point.lat, geo_point.lon, zoom)
        return Point(x=x, y=y)

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

    @classmethod
    def point2geo(cls, tile_point: Point, zoom: int) -> GeoPoint:
        """Convert a tile Point to a GeoPoint.

        Args:
            tile_point: Point with x, y tile coordinates
            zoom: Zoom level

        Returns:
            GeoPoint with latitude and longitude
        """
        lat, lon = cls.num2deg(tile_point.x, tile_point.y, zoom)
        return GeoPoint(lat=lat, lon=lon)

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
        # Create GeoPoint objects for the bounds
        min_geo = GeoPoint(lat=min_lat, lon=min_lon)
        max_geo = GeoPoint(lat=max_lat, lon=max_lon)

        # Get tile coordinates for the bounding box
        min_tile_coords = self.deg2num(min_geo.lat, min_geo.lon, zoom)
        max_tile_coords = self.deg2num(max_geo.lat, max_geo.lon, zoom)

        # Create Point objects for the tile bounds
        min_tile = Point(x=min_tile_coords[0], y=max_tile_coords[1])  # Note: y is swapped due to tile coordinate system
        max_tile = Point(x=max_tile_coords[0], y=min_tile_coords[1])  # Note: y is swapped due to tile coordinate system

        # Calculate the size of the composite image
        dimensions = Point(
            x=(max_tile.x - min_tile.x + 1) * 256,
            y=(max_tile.y - min_tile.y + 1) * 256
        )

        # Create a blank image
        composite = Image.new('RGB', (dimensions.x, dimensions.y))

        # Create a list of all tile coordinates to fetch
        tile_coords = []
        for x in range(min_tile.x, max_tile.x + 1):
            for y in range(min_tile.y, max_tile.y + 1):
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
        for x in range(min_tile.x, max_tile.x + 1):
            for y in range(min_tile.y, max_tile.y + 1):
                tile = tile_dict.get((x, y))
                if tile and tile.image:
                    # Calculate position in the composite image
                    pos = Point(
                        x=(x - min_tile.x) * 256,
                        y=(y - min_tile.y) * 256
                    )
                    composite.paste(tile.image, (pos.x, pos.y))
                else:
                    # If tile is missing, create a blank tile
                    logger.warning(f"Missing tile at {x},{y}, zoom {zoom}")
                    blank_tile = Image.new('RGB', (256, 256), (200, 200, 200))
                    pos = Point(
                        x=(x - min_tile.x) * 256,
                        y=(y - min_tile.y) * 256
                    )
                    composite.paste(blank_tile, (pos.x, pos.y))

        # Store the composite map and its metadata
        self.composite_map = composite
        self.composite_map_info = {
            'min_x': min_tile.x,
            'min_y': min_tile.y,
            'max_x': max_tile.x,
            'max_y': max_tile.y,
            'zoom': zoom,
            'width': dimensions.x,
            'height': dimensions.y
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
        # Create a GeoPoint for the target location
        geo_point = GeoPoint(lat=lat, lon=lon)

        # Get tile coordinates as a Point
        tile_point = self.deg2point(geo_point, zoom)

        # Fetch the tile
        tile = self.fetch_tile(tile_point.x, tile_point.y, zoom)
        if not tile or not tile.image:
            logger.error(f"Failed to fetch tile for coordinates {geo_point.lat}, {geo_point.lon} at zoom {zoom}")
            return None

        # Create a copy of the image to draw on
        result = tile.image.copy()
        draw = ImageDraw.Draw(result)

        # Calculate pixel coordinates within the tile
        n = 2.0 ** zoom
        lat_rad = math.radians(geo_point.lat)

        # Calculate pixel position within the tile
        pixel_pos = Point(
            x=int((geo_point.lon + 180.0) / 360.0 * n * 256) % 256,
            y=int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * 256) % 256
        )

        # Draw the marker
        draw.ellipse(
            (pixel_pos.x - marker_size//2, pixel_pos.y - marker_size//2, 
             pixel_pos.x + marker_size//2, pixel_pos.y + marker_size//2), 
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

        # Create a GeoPoint for the target location
        geo_point = GeoPoint(lat=lat, lon=lon)

        # Calculate global pixel coordinates
        n = 2.0 ** zoom
        lat_rad = math.radians(geo_point.lat)
        global_pixel = Point(
            x=int((geo_point.lon + 180.0) / 360.0 * n * 256),
            y=int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * 256)
        )

        # Calculate position in the composite map
        min_tile = Point(
            x=self.composite_map_info['min_x'],
            y=self.composite_map_info['min_y']
        )

        # Position in the composite map (pixels)
        map_pixel = Point(
            x=global_pixel.x - min_tile.x * 256,
            y=global_pixel.y - min_tile.y * 256
        )

        # Calculate the crop box
        crop_box = Rectangle(
            left=max(0, map_pixel.x - frame_width // 2),
            top=max(0, map_pixel.y - frame_height // 2),
            right=min(self.composite_map_info['width'], max(0, map_pixel.x - frame_width // 2) + frame_width),
            bottom=min(self.composite_map_info['height'], max(0, map_pixel.y - frame_height // 2) + frame_height)
        )

        # Adjust if we hit the edge of the composite map
        if crop_box.width < frame_width:
            if crop_box.left == 0:
                crop_box.right = min(self.composite_map_info['width'], frame_width)
            else:
                crop_box.left = max(0, crop_box.right - frame_width)

        if crop_box.height < frame_height:
            if crop_box.top == 0:
                crop_box.bottom = min(self.composite_map_info['height'], frame_height)
            else:
                crop_box.top = max(0, crop_box.bottom - frame_height)

        # Crop the composite map
        cropped = self.composite_map.crop((crop_box.left, crop_box.top, crop_box.right, crop_box.bottom))

        # Create a copy of the cropped image to draw on
        result = cropped.copy()
        draw = ImageDraw.Draw(result)

        # Calculate marker position in the cropped image
        marker_pos = Point(
            x=map_pixel.x - crop_box.left,
            y=map_pixel.y - crop_box.top
        )

        # Draw the marker
        draw.ellipse(
            (marker_pos.x - marker_size//2, marker_pos.y - marker_size//2, 
             marker_pos.x + marker_size//2, marker_pos.y + marker_size//2), 
            fill=marker_color
        )

        return result
