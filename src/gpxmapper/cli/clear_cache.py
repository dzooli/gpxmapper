"""Command for clearing the map tiles cache and optional reverse-geocode cache."""

from __future__ import annotations

import logging
import os

import typer

from ..api import (
    clear_geolocation_cache,
    clear_tile_cache,
    get_geolocation_cache_info,
    get_tile_cache_info,
)
from ..map_renderer import MapRendererBase
from ..reverse_geocode_cache import resolve_reverse_geocode_cache_path
from . import app

logger = logging.getLogger(__name__)


def _clear_geolocation_cache_cli() -> None:
    """Delete the reverse-geocode SQLite file after user confirmation."""
    db_path = resolve_reverse_geocode_cache_path()
    info = get_geolocation_cache_info(db_path)
    if not info.exists:
        typer.echo(f"Reverse geocode cache file does not exist: {info.cache_path}")
        return
    if not typer.confirm(f"Delete reverse geocode cache file?\n  {info.cache_path}"):
        typer.echo("Operation cancelled.")
        return

    result = clear_geolocation_cache(info.cache_path)
    if result.success:
        typer.echo(f"Successfully deleted reverse geocode cache: {result.cache_path}")
    else:
        logger.error("Failed to delete reverse geocode cache: %s", result.error_message)
        raise typer.Abort()


@app.command("clear-cache")
def clear_cache(
    geolocation: bool = typer.Option(
        False,
        "--geolocation",
        help="Clear the reverse-geocode SQLite cache instead of map tiles.",
    ),
) -> None:
    """Clear cached map tiles, or with ``--geolocation`` clear the Nominatim address cache.

    By default, removes all files in the map tile cache directory. Use ``--geolocation``
    to delete only the SQLite database used for ``generate --geolocate`` (sibling of the
    tile cache directory, not inside it).
    """
    try:
        if geolocation:
            _clear_geolocation_cache_cli()
            return

        cache_dir = MapRendererBase.resolve_default_cache_directory()
        if not os.path.exists(cache_dir):
            typer.echo(f"Cache directory does not exist: {cache_dir}")
            return

        info = get_tile_cache_info(cache_dir)
        if info.file_count == 0:
            typer.echo(f"Cache directory is already empty: {cache_dir}")
            return

        # Confirm with user
        if not typer.confirm(f"Are you sure you want to delete {info.file_count} files from {cache_dir}?"):
            typer.echo("Operation cancelled.")
            return

        result = clear_tile_cache(cache_dir)
        typer.echo(f"Successfully cleared {result.files_deleted} files from cache directory: {result.cache_path}")

    except typer.Abort:
        raise
    except Exception as exc:
        logger.error("Error clearing cache: %s", exc)
        raise typer.Abort() from exc
