"""Command-line interface for GPX to video mapper."""

import os
import sys
import logging
import typer
from typing import Optional
from pathlib import Path

from .gpx_parser import GPXParser
from .video_generator import VideoGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Create typer app
app = typer.Typer(help="GPX to video mapper - creates videos from GPX tracks")

@app.command()
def generate(
    gpx_file: Path = typer.Argument(
        ..., 
        exists=True, 
        file_okay=True, 
        dir_okay=False, 
        readable=True,
        help="Path to the input GPX file"
    ),
    output_file: Path = typer.Option(
        None,
        "--output", "-o",
        help="Path to the output video file (default: input filename with .mp4 extension)"
    ),
    duration: int = typer.Option(
        60,
        "--duration", "-d",
        min=1,
        help="Duration of the output video in seconds"
    ),
    fps: int = typer.Option(
        30,
        "--fps", "-f",
        min=1,
        max=60,
        help="Frames per second for the output video"
    ),
    width: int = typer.Option(
        320,
        "--width", "-w",
        min=128,
        help="Width of the output video in pixels"
    ),
    height: int = typer.Option(
        320,
        "--height", "-h",
        min=128,
        help="Height of the output video in pixels"
    ),
    zoom: int = typer.Option(
        15,
        "--zoom", "-z",
        min=1,
        max=19,
        help="Zoom level for the map (1-19, higher is more detailed)"
    ),
    marker_size: int = typer.Option(
        10,
        "--marker-size", "-m",
        min=1,
        help="Size of the position marker in pixels"
    ),
    marker_color: str = typer.Option(
        "255,0,0",
        "--marker-color", "-c",
        help="Color of the position marker as R,G,B (e.g., '255,0,0' for red)"
    ),
    timestamp_color: str = typer.Option(
        "0,0,0",
        "--timestamp-color", "-tc",
        help="Color of the timestamp text as R,G,B (e.g., '0,0,0' for black)"
    ),
    timestamp_font_scale: float = typer.Option(
        0.7,
        "--timestamp-font-scale", "-ts",
        min=0.1,
        max=5.0,
        help="Font scale for the timestamp text"
    ),
    title_text: Optional[str] = typer.Option(
        None,
        "--title",
        help="Optional text to display as a title on the video"
    ),
    title_align: str = typer.Option(
        "left",
        "--title-align",
        help="Alignment of the title text (left, center, right)"
    ),
):
    """Generate a video from a GPX track file.

    The video will show the track on a map with a marker indicating the current position.
    The GPX timeline will be mapped to the video duration.
    """
    try:
        # Parse marker color
        try:
            r, g, b = map(int, marker_color.split(","))
            if not all(0 <= c <= 255 for c in (r, g, b)):
                raise ValueError("Color values must be between 0 and 255")
            marker_color_tuple = (r, g, b)
        except Exception as e:
            logger.error(f"Invalid marker color format: {e}")
            raise typer.BadParameter("Marker color must be in format 'R,G,B' with values 0-255")

        # Parse timestamp color
        try:
            tr, tg, tb = map(int, timestamp_color.split(","))
            if not all(0 <= c <= 255 for c in (tr, tg, tb)):
                raise ValueError("Color values must be between 0 and 255")
            timestamp_color_tuple = (tr, tg, tb)
        except Exception as e:
            logger.error(f"Invalid timestamp color format: {e}")
            raise typer.BadParameter("Timestamp color must be in format 'R,G,B' with values 0-255")

        # Validate title alignment
        if title_align not in ["left", "center", "right"]:
            logger.error(f"Invalid title alignment: {title_align}")
            raise typer.BadParameter("Title alignment must be one of: left, center, right")

        # Set default output file if not provided
        if output_file is None:
            output_file = gpx_file.with_suffix(".mp4")

        # Create output directory if it doesn't exist
        output_dir = output_file.parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        # Parse GPX file
        logger.info(f"Parsing GPX file: {gpx_file}")
        parser = GPXParser(str(gpx_file))
        track_points = parser.parse()

        if not track_points:
            logger.error("No track points found in the GPX file")
            raise typer.Abort()

        # Check if track points have time data
        points_with_time = [p for p in track_points if p.time is not None]
        if not points_with_time:
            logger.error("GPX file doesn't contain time data, which is required for video generation")
            raise typer.Abort()

        # Get time bounds
        start_time, end_time = parser.get_time_bounds()
        logger.info(f"Track time range: {start_time} to {end_time}")

        # Generate video
        logger.info(f"Generating video: {output_file}")
        video_generator = VideoGenerator(
            output_path=str(output_file),
            fps=fps,
            resolution=(width, height),
            zoom_level=zoom,
            marker_color=marker_color_tuple,
            marker_size=marker_size,
            timestamp_color=timestamp_color_tuple,
            timestamp_font_scale=timestamp_font_scale,
            title_text=title_text,
            title_align=title_align
        )

        output_path = video_generator.generate_video(track_points, duration)

        logger.info(f"Video generated successfully: {output_path}")

    except Exception as e:
        logger.error(f"Error generating video: {e}")
        raise typer.Abort()

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

if __name__ == "__main__":
    app()
