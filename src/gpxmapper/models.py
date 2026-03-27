"""Data representation classes for the gpxmapper package.

This module contains all the data classes used throughout the gpxmapper package.
These classes represent various entities like GPX track points, map tiles, and configuration objects.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

from PIL import Image


@dataclass(slots=True, frozen=True)
class Point:
    """Represents a 2D point with x and y coordinates."""
    x: int
    y: int


@dataclass(slots=True, frozen=True)
class GeoPoint:
    """Represents a geographic point with latitude and longitude."""
    lat: float
    lon: float


@dataclass(slots=True, frozen=True)
class Rectangle:
    """Represents a rectangle with left, top, right, bottom coordinates."""
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        """Get the width of the rectangle."""
        return self.right - self.left

    @property
    def height(self) -> int:
        """Get the height of the rectangle."""
        return self.bottom - self.top


class GPXTrackPoint:
    """Represents a single point in a GPX track with time and position."""

    def __init__(self, latitude: float, longitude: float, elevation: Optional[float], 
                 time: Optional[datetime], extensions: Optional[Dict[str, Any]] = None):
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.time = time
        self.extensions = extensions or {}

    def distance_to(self, other: 'GPXTrackPoint') -> float:
        """
        Calculates the great-circle distance between the current point and another GPX track point
        using the Haversine formula. The result is given in meters.

        :param other: An instance of `GPXTrackPoint` is representing the point to calculate the distance to.
        :type other: GPXTrackPoint
        :return: The great-circle distance between the two points in meters.
        :rtype: float
        """
        average_earth_radius: float = 6371000  # Earth radius in meters
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        return average_earth_radius * c

    def elevation_gain(self, other: 'GPXTrackPoint') -> float:
        """
        Calculates the elevation gain between the current track point and another.

        The elevation gain is determined by comparing the elevation of the current
        GPXTrackPoint object with the elevation of another GPXTrackPoint object. If
        either of the elevations is None, the method returns 0.0. Otherwise, if the
        elevation of the other point is higher than the current point, the difference
        is returned as the elevation gain; otherwise, the result is 0.0.

        :param other: Another GPXTrackPoint object to compare elevation with.
        :type other: GPXTrackPoint
        :return: The elevation gains as a float. If the elevation of the other point
                 is higher than this point, the function returns the positive difference.
                 Otherwise, it returns 0.0.
        :rtype: float
        """
        if self.elevation is None or other.elevation is None:
            return 0.0
        return max(0.0, other.elevation - self.elevation)

    def time_delta(self, other: 'GPXTrackPoint') -> Optional[float]:
        """Calculate the time difference in seconds."""
        if self.time is None or other.time is None:
            return None
        return (other.time - self.time).total_seconds()

    def speed_to(self, other: 'GPXTrackPoint') -> Optional[float]:
        """Calculate speed in m/s to another point."""
        distance = self.distance_to(other)
        time_diff = self.time_delta(other)

        if time_diff is None or math.isclose(time_diff, 0.0, rel_tol=1e-09, abs_tol=1e-09):
            return None

        return distance / time_diff

    @property
    def has_elevation(self) -> bool:
        """Check if a point has elevation data."""
        return self.elevation is not None

    @property
    def has_timestamp(self) -> bool:
        """Check if a point has timestamp data."""
        return self.time is not None

    @property
    def has_extensions(self) -> bool:
        """Check if the point has extension data."""
        return len(self.extensions) > 0

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'lat': self.latitude,
            'lon': self.longitude,
            'ele': self.elevation,
            'time': self.time.isoformat() if self.time else None,
            'extensions': self.extensions or {}
        }

    def __repr__(self) -> str:
        return f"GPXTrackPoint(lat={self.latitude}, lon={self.longitude}, time={self.time})"


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


@dataclass(slots=True, frozen=True)
class TextConfig:
    """Configuration for text rendering in the video."""
    font_scale: float = 0.7
    title_text: Optional[str] = None
    text_align: str = "left"
    timestamp_color: Tuple[int, int, int] = (0, 0, 0)
    font_file: Optional[str] = None
    show_timestamp: bool = True
    scrolling_text_file: Optional[str] = None
    scrolling_speed: Optional[float] = None
    timezone: Optional[str] = None


@dataclass(slots=True, frozen=True)
class VideoConfig:
    """Configuration for video generation."""
    fps: int
    width: int
    height: int
    duration: int


@dataclass(slots=True, frozen=True)
class MapConfig:
    """Configuration for map rendering."""
    zoom: int
    marker_size: int
    marker_color: Tuple[int, int, int]
