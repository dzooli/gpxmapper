"""Reverse-geocode label prefetch for video frames (Nominatim).

Frame timestamps must match ``VideoGenerator._write_video_frames`` scheduling so
labels align with rendered frames. See ``VideoGenerator._write_video_frames`` loop.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, List, Optional

from .geolocation_clients import GeolocationClientFactory
from .models import GPXTrackPoint
from .nominatim_config import (
    PUBLIC_NOMINATIM_MIN_INTERVAL_SEC,
    get_nominatim_base_url,
    is_public_osm_nominatim,
)

if TYPE_CHECKING:
    from .geolocation_clients.base import AbstractGeolocationClient
    from .reverse_geocode_cache import ReverseGeocodeCache
    from .video_generator import VideoGenerator

logger = logging.getLogger(__name__)


async def _reverse_label_from_cache_or_http(
    *,
    cache: ReverseGeocodeCache | None,
    client: AbstractGeolocationClient,
    base_url: str,
    lat: float,
    lon: float,
    throttle_public: bool,
) -> str:
    """Return display name from SQLite cache, or from Nominatim (then update cache)."""
    if cache is not None:
        cached = await cache.get(base_url, lat, lon)
        if cached is not None:
            return cached
    resp = await client.reverse_geocode(lat, lon)
    label = resp.display_name
    if cache is not None:
        await cache.put(base_url, lat, lon, label)
    if throttle_public:
        await asyncio.sleep(PUBLIC_NOMINATIM_MIN_INTERVAL_SEC)
    return label


async def prefetch_geolocation_labels(
    video_gen: VideoGenerator,
    points_with_time: List[GPXTrackPoint],
    duration_seconds: int,
    fps: int,
    start_time: datetime,
    total_track_seconds: float,
    *,
    user_agent: str = "gpxmapper/1.0",
    geocode_cache: Optional["ReverseGeocodeCache"] = None,
) -> List[str]:
    """Build one display string per video frame index using Nominatim reverse geocode.

    Uses a 10 m anchor: reuse the last label while the interpolated point is within
    10 m of the last *fetched* coordinates; otherwise fetch again. Public OSM host
    gets a polite delay after each successful reverse HTTP call (not after SQLite
    cache hits).

    Args:
        geocode_cache: Optional :class:`~gpxmapper.reverse_geocode_cache.ReverseGeocodeCache`
            instance (e.g. tests with ``tmp_path``). When ``None``, opens the default
            on-disk cache if possible, otherwise runs without caching.
    """
    from .reverse_geocode_cache import ReverseGeocodeCache as _ReverseGeocodeCache

    base = get_nominatim_base_url()
    throttle_public = is_public_osm_nominatim(base)
    cache = geocode_cache if geocode_cache is not None else _ReverseGeocodeCache.from_default_path()
    client = GeolocationClientFactory.create_client("nominatim", base_url=base, user_agent=user_agent)
    labels: List[str] = []
    anchor: GPXTrackPoint | None = None
    last_label = ""
    total_frames = duration_seconds * fps
    try:
        for frame_idx in range(total_frames):
            progress = (frame_idx / total_frames) if total_frames else 0.0
            track_seconds = progress * total_track_seconds
            frame_timestamp = start_time + timedelta(seconds=track_seconds)
            lat, lon = video_gen._interpolate_position(points_with_time, frame_timestamp)
            here = GPXTrackPoint(lat, lon, None, None)
            if anchor is None or anchor.distance_to(here) > 10.0:
                last_label = await _reverse_label_from_cache_or_http(
                    cache=cache,
                    client=client,
                    base_url=base,
                    lat=lat,
                    lon=lon,
                    throttle_public=throttle_public,
                )
                anchor = here
            labels.append(last_label)
        return labels
    finally:
        await client.aclose()
