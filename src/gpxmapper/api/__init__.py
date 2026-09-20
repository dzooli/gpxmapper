"""Public programmatic API for GPXMapper.

Provides clean Python functions and data transfer objects for GPX parsing,
metadata inspection, video generation, cache management, and Nominatim status.
"""

from __future__ import annotations

from .cache import (
    clear_geolocation_cache,
    clear_tile_cache,
    get_geolocation_cache_info,
    get_tile_cache_info,
)
from .config import create_text_config, parse_color
from .info import get_gpx_info
from .nominatim import check_nominatim_status
from .video import generate_video

__all__ = [
    "generate_video",
    "get_gpx_info",
    "get_tile_cache_info",
    "clear_tile_cache",
    "get_geolocation_cache_info",
    "clear_geolocation_cache",
    "check_nominatim_status",
    "parse_color",
    "create_text_config",
]
