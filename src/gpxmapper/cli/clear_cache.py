"""Command for clearing the map tiles cache."""

import logging
import os
from pathlib import Path

import typer

from ..map_renderer import MapRenderer
from . import app

logger = logging.getLogger(__name__)

@app.command("clear-cache")
def clear_cache():
    """Clear the map tiles cache directory.

    This command removes all cached map tiles to free up disk space.
    The cache directory is automatically determined based on the operating system.
    """
    try:
        # Initialize MapRenderer to get the default cache directory
        renderer = MapRenderer()
        cache_dir = renderer.cache_dir

        if not os.path.exists(cache_dir):
            typer.echo(f"Cache directory does not exist: {cache_dir}")
            return

        # Count files before deletion
        file_count = sum(1 for _ in Path(cache_dir).glob('*'))

        if file_count == 0:
            typer.echo(f"Cache directory is already empty: {cache_dir}")
            return

        # Confirm with user
        if not typer.confirm(f"Are you sure you want to delete {file_count} files from {cache_dir}?"):
            typer.echo("Operation cancelled.")
            return

        # Remove all files in the cache directory
        for file_path in Path(cache_dir).glob('*'):
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")

        typer.echo(f"Successfully cleared {file_count} files from cache directory: {cache_dir}")

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise typer.Abort()
