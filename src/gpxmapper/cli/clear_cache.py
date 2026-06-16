"""Command for clearing the map tiles cache and optional reverse-geocode cache."""

import logging
import os
from pathlib import Path

import typer

from . import app
from ..map_renderer import MapRendererBase
from ..reverse_geocode_cache import resolve_reverse_geocode_cache_path

logger = logging.getLogger(__name__)


def _clear_geolocation_cache() -> None:
    """Delete the reverse-geocode SQLite file after confirmation."""
    db_path = resolve_reverse_geocode_cache_path()
    if not db_path.is_file():
        typer.echo(f"Reverse geocode cache file does not exist: {db_path}")
        return
    if not typer.confirm(f"Delete reverse geocode cache file?\n  {db_path}"):
        typer.echo("Operation cancelled.")
        return
    try:
        db_path.unlink()
        typer.echo(f"Successfully deleted reverse geocode cache: {db_path}")
    except OSError as e:
        logger.error("Failed to delete reverse geocode cache: %s", e)
        raise typer.Abort() from e


@app.command("clear-cache")
def clear_cache(
    geolocation: bool = typer.Option(
        False,
        "--geolocation",
        help="Clear the reverse-geocode SQLite cache instead of map tiles.",
    ),
):
    """Clear cached map tiles, or with ``--geolocation`` clear the Nominatim address cache.

    By default, removes all files in the map tile cache directory. Use ``--geolocation``
    to delete only the SQLite database used for ``generate --geolocate`` (sibling of the
    tile cache directory, not inside it).
    """
    try:
        if geolocation:
            _clear_geolocation_cache()
            return

        cache_dir = MapRendererBase.resolve_default_cache_directory()

        if not os.path.exists(cache_dir):
            typer.echo(f"Cache directory does not exist: {cache_dir}")
            return

        # Count files before deletion
        file_count = sum(1 for _ in Path(cache_dir).glob("*"))

        if file_count == 0:
            typer.echo(f"Cache directory is already empty: {cache_dir}")
            return

        # Confirm with user
        if not typer.confirm(f"Are you sure you want to delete {file_count} files from {cache_dir}?"):
            typer.echo("Operation cancelled.")
            return

        # Remove all files in the cache directory
        for file_path in Path(cache_dir).glob("*"):
            try:
                file_path.unlink()
            except OSError as e:
                logger.warning("Failed to delete %s: %s", file_path, e)

        typer.echo(f"Successfully cleared {file_count} files from cache directory: {cache_dir}")

    except typer.Abort:
        raise
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise typer.Abort()
