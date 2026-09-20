"""Command for displaying information about GPX files."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from ..api import get_gpx_info
from ..exceptions import GPXMapperError
from . import app

logger = logging.getLogger(__name__)


@app.command()
def info(
    gpx_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the GPX file",
    ),
) -> None:
    """Display information about a GPX file."""
    try:
        gpx_data = get_gpx_info(gpx_file)

        typer.echo(f"GPX File: {gpx_data.file_path}")
        typer.echo(f"Number of track points: {gpx_data.point_count}")

        if gpx_data.start_time and gpx_data.end_time:
            typer.echo(f"Time range: {gpx_data.start_time} to {gpx_data.end_time}")
            typer.echo(f"Duration: {gpx_data.duration}")
        else:
            typer.echo("No time data available")

        min_lat, min_lon, max_lat, max_lon = gpx_data.coordinate_bounds
        typer.echo(f"Coordinate bounds: {min_lat:.6f},{min_lon:.6f} to {max_lat:.6f},{max_lon:.6f}")

    except GPXMapperError as exc:
        logger.error("Error reading GPX file: %s", exc)
        raise typer.Abort() from exc
    except Exception as exc:
        logger.error("Unexpected error reading GPX file: %s", exc)
        raise typer.Abort() from exc
