"""SQLite cache for Nominatim reverse-geocode ``display_name`` results.

Stored next to the map tile cache directory (not inside it). Use
``gpxmapper clear-cache --geolocation`` to remove this file. See
:func:`resolve_reverse_geocode_cache_path`.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from .map_renderer.base import MapRendererBase

logger = logging.getLogger(__name__)

# ~1.1 m cell size; balances reuse vs. distinct nearby addresses.
GEOCODE_COORD_DECIMALS = 5


def quantize_coordinates(lat: float, lon: float) -> tuple[float, float]:
    """Round coordinates for stable cache keys (see :data:`GEOCODE_COORD_DECIMALS`)."""
    return (round(lat, GEOCODE_COORD_DECIMALS), round(lon, GEOCODE_COORD_DECIMALS))


def normalize_cache_base_url(url: str) -> str:
    """Normalize Nominatim base URL for cache keys."""
    return url.strip().rstrip("/").lower()


def resolve_reverse_geocode_cache_path() -> Path:
    """Return the default SQLite path (sibling of the tile cache root).

    Tile roots differ by OS (see ``MapRendererBase.resolve_default_cache_directory``).
    The database must not live inside the tile directory so tile ``clear-cache`` does
    not delete reverse-geocode data.
    """
    tile_cache = Path(MapRendererBase.resolve_default_cache_directory())
    parent = tile_cache.parent
    if tile_cache.name == "cache":
        return parent / "reverse_geocode.sqlite"
    return parent / "gpxmapper_reverse_geocode.sqlite"


class ReverseGeocodeCache:
    """SQLite cache for reverse geocode results.

    Use :meth:`get_sync` / :meth:`put_sync` from the asyncio event loop (e.g. prefetch
    coroutine). Do not use ``asyncio.to_thread`` with one connection: sqlite3 ties
    connections to the creating thread unless ``check_same_thread=False``.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._disabled = False

    @classmethod
    def from_default_path(cls) -> ReverseGeocodeCache | None:
        """Open the default cache file, or return ``None`` if the database cannot be used."""
        inst = cls(resolve_reverse_geocode_cache_path())
        inst._ensure_connection()
        return None if inst._disabled else inst

    def _ensure_connection(self) -> None:
        with self._lock:
            if self._disabled or self._conn is not None:
                return
            try:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(self._db_path), timeout=30.0)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reverse_geocode (
                        base_url TEXT NOT NULL,
                        lat_q REAL NOT NULL,
                        lon_q REAL NOT NULL,
                        display_name TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        PRIMARY KEY (base_url, lat_q, lon_q)
                    )
                    """
                )
                conn.commit()
                self._conn = conn
            except (OSError, sqlite3.Error) as exc:
                logger.warning("Reverse geocode cache init failed (%s): %s", self._db_path, exc)
                self._disabled = True
                self._conn = None

    def get_sync(self, base_url: str, lat: float, lon: float) -> str | None:
        """Return cached ``display_name`` or ``None``."""
        self._ensure_connection()
        if self._disabled or self._conn is None:
            return None
        norm = normalize_cache_base_url(base_url)
        lat_q, lon_q = quantize_coordinates(lat, lon)
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT display_name FROM reverse_geocode WHERE base_url = ? AND lat_q = ? AND lon_q = ?",
                    (norm, lat_q, lon_q),
                ).fetchone()
            return str(row[0]) if row else None
        except sqlite3.Error as exc:
            logger.warning("Reverse geocode cache read failed: %s", exc)
            return None

    def put_sync(self, base_url: str, lat: float, lon: float, display_name: str) -> None:
        """Store a row (best-effort; logs and ignores SQLite errors)."""
        self._ensure_connection()
        if self._disabled or self._conn is None:
            return
        norm = normalize_cache_base_url(base_url)
        lat_q, lon_q = quantize_coordinates(lat, lon)
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO reverse_geocode (base_url, lat_q, lon_q, display_name)
                    VALUES (?, ?, ?, ?)
                    """,
                    (norm, lat_q, lon_q, display_name),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning("Reverse geocode cache write failed: %s", exc)

    def close_sync(self) -> None:
        """Close the DB connection (e.g. in tests)."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None
