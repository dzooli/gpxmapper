"""Abstract base class for map tile rendering."""

from __future__ import annotations

import logging
import math
import os
import platform
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

from .constants import DEFAULT_TILE_SERVER
from ..models import GeoPoint, MapTile, Point, Rectangle

logger = logging.getLogger(__name__)


class MapRendererBase(ABC):
    """Abstract base: shared cache, geometry, and rendering helpers."""

    @staticmethod
    def resolve_default_cache_directory() -> str:
        """Return the default tile cache directory for the current OS.

        Does not create the directory. All renderer implementations share this path when
        ``cache_dir`` is omitted at construction time.
        """
        system = platform.system()
        if system == "Windows":
            appdata = os.environ.get("LOCALAPPDATA")
            if appdata:
                return os.path.join(appdata, "gpxmapper", "cache")
            return os.path.join(os.path.expanduser("~"), "AppData", "Local", "gpxmapper", "cache")
        if system == "Linux":
            return os.path.join(os.path.expanduser("~"), ".cache", "gpxmapper")
        return os.path.join(os.path.expanduser("~"), ".gpxmapper", "cache")

    @classmethod
    def resolve_cache_directory(cls, cache_dir: Optional[str]) -> Optional[str]:
        """Normalize ``cache_dir``: use :meth:`resolve_default_cache_directory` when ``None``."""
        if cache_dir is None:
            logger.info("Using OS dependent cache dir")
            return cls.resolve_default_cache_directory()
        return cache_dir

    def __init__(
            self,
            tile_server: str = DEFAULT_TILE_SERVER,
            cache_dir: Optional[str] = None,
            use_cache: bool = True,
    ):
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

        self.cache_dir = self.resolve_cache_directory(cache_dir)
        logger.info(f"Using cache dir: {self.cache_dir}")

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    _zoom_cache: dict = {}

    def build_tile_url(self, x: int, y: int, zoom: int) -> str:
        """Build a tile URL from template."""
        return self.tile_server.replace("{x}", str(x)).replace("{y}", str(y)).replace("{z}", str(zoom))

    @staticmethod
    def ensure_rgb(image: Image.Image) -> Image.Image:
        """Ensure image is in RGB mode."""
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    def open_cached_image(self, x: int, y: int, zoom: int) -> Optional[Image.Image]:
        """Try to open a cached tile image and return it, or None."""
        if not self.use_cache:
            return None
        tile_path = self.get_tile_path(x, y, zoom)
        if not tile_path:
            return None
        try:
            with Image.open(tile_path) as img:
                img.load()
                return self.ensure_rgb(img.copy())
        except Exception as e:
            logger.warning(f"Failed to load cached tile: {e}")
            return None

    def cache_image(self, x: int, y: int, zoom: int, image: Image.Image) -> None:
        """Cache the given tile image if caching is enabled."""
        if self.use_cache and self.cache_dir:
            logger.info(f"Caching tile {x},{y} at zoom {zoom}")
            tile_path = os.path.join(self.cache_dir, f"{zoom}_{x}_{y}.png")
            image.save(tile_path)

    def build_tile_coords(self, min_x: int, max_x: int, min_y: int, max_y: int, zoom: int) -> List[tuple]:
        """Build list of (x,y,zoom) tile coordinates for the given tile bounds."""
        coords = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                coords.append((x, y, zoom))
        return coords

    def build_tile_coords_for_bounds(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> List[tuple]:
        """Build tile coords for geographic bounds."""
        min_x, max_y = self.deg2num(min_lat, min_lon, zoom)
        max_x, min_y = self.deg2num(max_lat, max_lon, zoom)
        return self.build_tile_coords(min_x, max_x, min_y, max_y, zoom)

    def paste_tiles_to_composite(
            self,
            composite: Image.Image,
            min_x: int,
            min_y: int,
            max_x: int,
            max_y: int,
            zoom: int,
            tile_lookup: dict,
    ) -> None:
        """Paste tiles from tile_lookup onto composite image, filling blanks if missing."""
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tile = tile_lookup.get((x, y))
                if tile and tile.image:
                    pos_x = (x - min_x) * 256
                    pos_y = (y - min_y) * 256
                    composite.paste(tile.image, (pos_x, pos_y))
                else:
                    logger.warning(f"Missing tile at {x},{y}, zoom {zoom}")
                    blank_tile = Image.new("RGB", (256, 256), (200, 200, 200))
                    pos_x = (x - min_x) * 256
                    pos_y = (y - min_y) * 256
                    composite.paste(blank_tile, (pos_x, pos_y))

    def compute_composite_geometry(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int):
        """Compute min/max tile bounds and composite dimensions for given geographic bounds.

        Returns:
            Tuple (min_tile, max_tile, dimensions) of Points.
        """
        min_geo = GeoPoint(lat=min_lat, lon=min_lon)
        max_geo = GeoPoint(lat=max_lat, lon=max_lon)

        min_tile_coords = self.deg2num(min_geo.lat, min_geo.lon, zoom)
        max_tile_coords = self.deg2num(max_geo.lat, max_geo.lon, zoom)

        min_tile = Point(x=min_tile_coords[0], y=max_tile_coords[1])
        max_tile = Point(x=max_tile_coords[0], y=min_tile_coords[1])

        dimensions = Point(x=(max_tile.x - min_tile.x + 1) * 256, y=(max_tile.y - min_tile.y + 1) * 256)

        return min_tile, max_tile, dimensions

    def set_composite_info(self, min_tile, max_tile, zoom: int, dimensions, composite: Image.Image) -> None:
        """Store the composite image and its metadata on this renderer instance."""
        self.composite_map = composite
        self.composite_map_info = {
            "min_x": min_tile.x,
            "min_y": min_tile.y,
            "max_x": max_tile.x,
            "max_y": max_tile.y,
            "zoom": zoom,
            "width": dimensions.x,
            "height": dimensions.y,
        }

    @classmethod
    def deg2num(cls, lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
        """Convert latitude and longitude to tile coordinates."""
        if zoom not in cls._zoom_cache:
            cls._zoom_cache[zoom] = 2.0 ** zoom
        n = cls._zoom_cache[zoom]

        lat_rad = math.radians(lat_deg)
        x = int((lon_deg + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    @classmethod
    def deg2point(cls, geo_point: GeoPoint, zoom: int) -> Point:
        """Convert a GeoPoint to a tile Point."""
        x, y = cls.deg2num(geo_point.lat, geo_point.lon, zoom)
        return Point(x=x, y=y)

    @classmethod
    def num2deg(cls, xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
        """Convert tile coordinates to latitude and longitude."""
        if zoom not in cls._zoom_cache:
            cls._zoom_cache[zoom] = 2.0 ** zoom
        n = cls._zoom_cache[zoom]

        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg

    @classmethod
    def point2geo(cls, tile_point: Point, zoom: int) -> GeoPoint:
        """Convert a tile Point to a GeoPoint."""
        lat, lon = cls.num2deg(tile_point.x, tile_point.y, zoom)
        return GeoPoint(lat=lat, lon=lon)

    def get_tile_path(self, x: int, y: int, zoom: int) -> Optional[str]:
        """Get the path to a cached tile if it exists."""
        if not self.use_cache or not self.cache_dir:
            return None

        tile_path = os.path.join(self.cache_dir, f"{zoom}_{x}_{y}.png")
        if os.path.exists(tile_path):
            return tile_path

        return None

    @abstractmethod
    def fetch_tile(self, x: int, y: int, zoom: int) -> Optional[MapTile]:
        """Fetch a map tile from the server or cache."""

    @abstractmethod
    def get_tiles_for_bounds(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> List[MapTile]:
        """Return all tiles needed to cover the geographic bounds."""

    @abstractmethod
    def create_composite_map(
            self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
    ) -> Image.Image:
        """Create a composite image from tiles for the bounding box."""

    def render_map_for_point(
            self,
            lat: float,
            lon: float,
            zoom: int,
            marker_color: Tuple[int, int, int] = (255, 0, 0),
            marker_size: int = 10,
    ) -> Optional[Image.Image]:
        """Render a map centered on the given coordinates with a marker."""
        geo_point = GeoPoint(lat=lat, lon=lon)

        tile_point = self.deg2point(geo_point, zoom)

        tile = self.fetch_tile(tile_point.x, tile_point.y, zoom)
        if not tile or not tile.image:
            logger.error(f"Failed to fetch tile for coordinates {geo_point.lat}, {geo_point.lon} at zoom {zoom}")
            return None

        result = tile.image.copy()
        draw = ImageDraw.Draw(result)

        n = 2.0 ** zoom
        lat_rad = math.radians(geo_point.lat)

        pixel_pos = Point(
            x=int((geo_point.lon + 180.0) / 360.0 * n * 256) % 256,
            y=int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * 256) % 256,
        )

        draw.ellipse(
            (
                pixel_pos.x - marker_size // 2,
                pixel_pos.y - marker_size // 2,
                pixel_pos.x + marker_size // 2,
                pixel_pos.y + marker_size // 2,
            ),
            fill=marker_color,
        )

        return result

    def render_from_composite(
            self,
            lat: float,
            lon: float,
            frame_width: int,
            frame_height: int,
            marker_color: Tuple[int, int, int] = (255, 0, 0),
            marker_size: int = 10,
    ) -> Optional[Image.Image]:
        """Render a map from the composite map centered on the given coordinates with a marker."""
        if self.composite_map is None or self.composite_map_info is None:
            logger.error("Composite map not created. Call create_composite_map first.")
            return None

        zoom = self.composite_map_info["zoom"]

        geo_point = GeoPoint(lat=lat, lon=lon)

        n = 2.0 ** zoom
        lat_rad = math.radians(geo_point.lat)
        global_pixel = Point(
            x=int((geo_point.lon + 180.0) / 360.0 * n * 256),
            y=int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * 256),
        )

        min_tile = Point(x=self.composite_map_info["min_x"], y=self.composite_map_info["min_y"])

        map_pixel = Point(x=global_pixel.x - min_tile.x * 256, y=global_pixel.y - min_tile.y * 256)

        left = max(0, map_pixel.x - frame_width // 2)
        top = max(0, map_pixel.y - frame_height // 2)
        right = min(self.composite_map_info["width"], left + frame_width)
        bottom = min(self.composite_map_info["height"], top + frame_height)

        if (right - left) < frame_width:
            if left == 0:
                right = min(self.composite_map_info["width"], frame_width)
            else:
                left = max(0, right - frame_width)

        if (bottom - top) < frame_height:
            if top == 0:
                bottom = min(self.composite_map_info["height"], frame_height)
            else:
                top = max(0, bottom - frame_height)

        crop_box = Rectangle(left=left, top=top, right=right, bottom=bottom)

        cropped = self.composite_map.crop((crop_box.left, crop_box.top, crop_box.right, crop_box.bottom))

        result = cropped.copy()
        draw = ImageDraw.Draw(result)

        marker_pos = Point(x=map_pixel.x - crop_box.left, y=map_pixel.y - crop_box.top)

        draw.ellipse(
            (
                marker_pos.x - marker_size // 2,
                marker_pos.y - marker_size // 2,
                marker_pos.x + marker_size // 2,
                marker_pos.y + marker_size // 2,
            ),
            fill=marker_color,
        )

        return result
