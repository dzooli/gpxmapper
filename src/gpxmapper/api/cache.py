"""Cache management services for map tiles and reverse-geocoding data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..map_renderer import MapRendererBase
from ..models import CacheClearResult, CacheInfo
from ..reverse_geocode_cache import resolve_reverse_geocode_cache_path

logger = logging.getLogger(__name__)


def get_tile_cache_info(cache_dir: Optional[Path | str] = None) -> CacheInfo:
    """Retrieve statistics about the map tile cache directory.

    Args:
        cache_dir: Optional path override. If omitted, uses default tile cache directory.

    Returns:
        CacheInfo instance with path, file count, and existence status.
    """
    path = Path(cache_dir) if cache_dir is not None else Path(MapRendererBase.resolve_default_cache_directory())
    if not path.exists():
        return CacheInfo(cache_path=path, file_count=0, exists=False)

    file_count = sum(1 for p in path.glob("*") if p.is_file())
    return CacheInfo(cache_path=path, file_count=file_count, exists=True)


def clear_tile_cache(cache_dir: Optional[Path | str] = None) -> CacheClearResult:
    """Clear cached map tile files from the cache directory.

    Args:
        cache_dir: Optional path override. If omitted, uses default tile cache directory.

    Returns:
        CacheClearResult indicating the number of deleted files and success.
    """
    path = Path(cache_dir) if cache_dir is not None else Path(MapRendererBase.resolve_default_cache_directory())
    if not path.exists():
        return CacheClearResult(cache_path=path, files_deleted=0, success=True)

    deleted = 0
    errors: list[str] = []
    for file_path in path.glob("*"):
        if file_path.is_file():
            try:
                file_path.unlink()
                deleted += 1
            except OSError as exc:
                logger.warning("Failed to delete %s: %s", file_path, exc)
                errors.append(f"{file_path.name}: {exc}")

    err_msg = "; ".join(errors) if errors else None
    return CacheClearResult(
        cache_path=path,
        files_deleted=deleted,
        success=len(errors) == 0,
        error_message=err_msg,
    )


def get_geolocation_cache_info(db_path: Optional[Path | str] = None) -> CacheInfo:
    """Retrieve status about the reverse-geocoding SQLite database cache.

    Args:
        db_path: Optional path override. If omitted, uses default reverse-geocode cache path.

    Returns:
        CacheInfo instance with database path and existence status.
    """
    path = Path(db_path) if db_path is not None else resolve_reverse_geocode_cache_path()
    exists = path.is_file()
    return CacheInfo(cache_path=path, file_count=1 if exists else 0, exists=exists)


def clear_geolocation_cache(db_path: Optional[Path | str] = None) -> CacheClearResult:
    """Delete the reverse-geocoding SQLite database cache file.

    Args:
        db_path: Optional path override. If omitted, uses default reverse-geocode cache path.

    Returns:
        CacheClearResult indicating deletion status.
    """
    path = Path(db_path) if db_path is not None else resolve_reverse_geocode_cache_path()
    if not path.is_file():
        return CacheClearResult(cache_path=path, files_deleted=0, success=True)

    try:
        path.unlink()
        return CacheClearResult(cache_path=path, files_deleted=1, success=True)
    except OSError as exc:
        logger.error("Failed to delete reverse geocode cache %s: %s", path, exc)
        return CacheClearResult(
            cache_path=path,
            files_deleted=0,
            success=False,
            error_message=str(exc),
        )
