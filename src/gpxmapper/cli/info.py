"""Command for displaying information about GPX files."""

import logging
from pathlib import Path

import typer

from ..gpx_parser import GPXParser
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
            help="Path to the GPX file"
        )
):
    """Display information about a GPX file."""
    try:
        # Parse GPX file
        logger.info(f"Parsing GPX file: {gpx_file}")
        parser = GPXParser(str(gpx_file))
        track_points = parser.parse()

        if not track_points:
            logger.error("No track points found in the GPX file")
            raise typer.Abort()

        # Get time bounds
        start_time, end_time = parser.get_time_bounds()

        # Get coordinate bounds
        min_lat, min_lon, max_lat, max_lon = parser.get_coordinate_bounds()

        # Print information
        typer.echo(f"GPX File: {gpx_file}")
        typer.echo(f"Number of track points: {len(track_points)}")

        if start_time and end_time:
            duration = end_time - start_time
            typer.echo(f"Time range: {start_time} to {end_time}")
            typer.echo(f"Duration: {duration}")
        else:
            typer.echo("No time data available")

        typer.echo(f"Coordinate bounds: {min_lat:.6f},{min_lon:.6f} to {max_lat:.6f},{max_lon:.6f}")

    except Exception as e:
        logger.error(f"Error reading GPX file: {e}")
        raise typer.Abort()
