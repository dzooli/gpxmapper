"""GPX to video mapper - creates videos from GPX tracks."""

from __future__ import annotations

from .api import (
    clear_geolocation_cache,
    clear_tile_cache,
    check_nominatim_status,
    create_text_config,
    generate_video,
    get_geolocation_cache_info,
    get_gpx_info,
    get_tile_cache_info,
    parse_color,
)
from .exceptions import (
    ConfigurationError,
    GPXEmptyError,
    GPXMapperError,
    GPXMissingTimeError,
    GPXParseError,
    NominatimUnavailableError,
    VideoGenerationError,
)
from .models import (
    CacheClearResult,
    CacheInfo,
    GeoPoint,
    GPXInfo,
    GPXTrackPoint,
    MapConfig,
    Point,
    Rectangle,
    TextConfig,
    VideoConfig,
)

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "generate_video",
    "get_gpx_info",
    "get_tile_cache_info",
    "clear_tile_cache",
    "get_geolocation_cache_info",
    "clear_geolocation_cache",
    "check_nominatim_status",
    "parse_color",
    "create_text_config",
    "GPXMapperError",
    "GPXParseError",
    "GPXEmptyError",
    "GPXMissingTimeError",
    "ConfigurationError",
    "VideoGenerationError",
    "NominatimUnavailableError",
    "GPXTrackPoint",
    "GPXInfo",
    "CacheInfo",
    "CacheClearResult",
    "VideoConfig",
    "MapConfig",
    "TextConfig",
    "Point",
    "GeoPoint",
    "Rectangle",
]
