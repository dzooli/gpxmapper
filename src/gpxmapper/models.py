"""Data representation classes for the gpxmapper package.

This module contains all the data classes used throughout the gpxmapper package.
These classes represent various entities like GPX track points, map tiles, and configuration objects.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from PIL import Image


class GPXTrackPoint:
    """Represents a single point in a GPX track with time and position."""
    
    def __init__(self, latitude: float, longitude: float, elevation: Optional[float], 
                 time: Optional[datetime], extensions: Optional[Dict[str, Any]] = None):
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.time = time
        self.extensions = extensions or {}
    
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


@dataclass
class TextConfig:
    """Configuration for text rendering in the video."""
    font_scale: float
    title_text: Optional[str] = None
    text_align: str = "left"
    timestamp_color: Tuple[int, int, int] = (0, 0, 0)


@dataclass
class VideoConfig:
    """Configuration for video generation."""
    fps: int
    width: int
    height: int
    duration: int


@dataclass
class MapConfig:
    """Configuration for map rendering."""
    zoom: int
    marker_size: int
    marker_color: Tuple[int, int, int]