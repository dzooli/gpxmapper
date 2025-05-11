"""Command-line interface for GPX to video mapper."""

import os
import sys
import logging
import typer
from typer.models import Context
from typing import Optional
from pathlib import Path

from .gpx_parser import GPXParser
from .video_generator import VideoGenerator
from .map_renderer import MapRenderer
from .models import TextConfig, VideoConfig, MapConfig

def create_text_config(
    font_scale: float,
    title_text: Optional[str] = None,
    text_align: str = "left",
    timestamp_color: str = "0,0,0",
    font_file: Optional[str] = None
) -> TextConfig:
    """Create a TextConfig object from the given parameters.

    Args:
        font_scale: Font scale for all text (timestamp, title, captions)
        title_text: Optional text to display as a title on the video
        text_align: Alignment of all text (title, captions) (left, center, right)
        timestamp_color: Color of the timestamp text as R,G,B (e.g., '0,0,0' for black)
        font_file: Optional path to a TrueType font file (.ttf) for text rendering

    Returns:
        TextConfig object
    """
    # Parse timestamp color if provided
    timestamp_color_tuple = (0, 0, 0)  # Default black
    if timestamp_color:
        try:
            tr, tg, tb = map(int, timestamp_color.split(","))
            if not all(0 <= c <= 255 for c in (tr, tg, tb)):
                raise ValueError("Color values must be between 0 and 255")
            timestamp_color_tuple = (tr, tg, tb)
        except Exception as e:
            logger.error(f"Invalid timestamp color format: {e}")
            raise typer.BadParameter("Timestamp color must be in format 'R,G,B' with values 0-255")

    # Validate text alignment
    if text_align not in ["left", "center", "right"]:
        logger.error(f"Invalid text alignment: {text_align}")
        raise typer.BadParameter("Text alignment must be one of: left, center, right")

    return TextConfig(
        font_scale=font_scale,
        title_text=title_text,
        text_align=text_align,
        timestamp_color=timestamp_color_tuple,
        font_file=font_file
    )

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Create typer app
app = typer.Typer(help="GPX to video mapper - creates videos from GPX tracks")

@app.callback(invoke_without_command=True)
def main(ctx: Context):
    """GPX to video mapper - creates videos from GPX tracks."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())

def generate_video(
    gpx_file: Path,
    output_file: Path,
    video_config: VideoConfig,
    map_config: MapConfig,
    text_config: TextConfig,
    captions: Optional[Path] = None
) -> str:
    """Generate a video from a GPX track file.

    Args:
        gpx_file: Path to the input GPX file
        output_file: Path to the output video file
        video_config: Configuration for video generation
        map_config: Configuration for map rendering
        text_config: Configuration for text rendering in the video
        captions: Optional path to a CSV file containing captions with timestamps

    Returns:
        Path to the generated video file
    """
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
        fps=video_config.fps,
        resolution=(video_config.width, video_config.height),
        zoom_level=map_config.zoom,
        marker_color=map_config.marker_color,
        marker_size=map_config.marker_size,
        text_config=text_config,
        captions_file=str(captions) if captions else None
    )

    output_path = video_generator.generate_video(track_points, video_config.duration)

    return output_path

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
    # Text rendering options
    font_scale: float = typer.Option(
        0.7,
        "--font-scale", "-fs",
        min=0.1,
        max=5.0,
        help="Font scale for all text (timestamp, title, captions)"
    ),
    title_text: Optional[str] = typer.Option(
        None,
        "--title",
        help="Optional text to display as a title on the video"
    ),
    text_align: str = typer.Option(
        "left",
        "--text-align", "-ta",
        help="Alignment of all text (title, captions) (left, center, right)"
    ),
    captions: Optional[Path] = typer.Option(
        None,
        "--captions",
        help="Path to a CSV file containing captions with timestamps in HH:MM:SS format (relative to the start of the video)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    font_file: Optional[Path] = typer.Option(
        None,
        "--font", "-ff",
        help="Path to a TrueType font file (.ttf) for text rendering",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
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

        # Set default output file if not provided
        if output_file is None:
            output_file = gpx_file.with_suffix(".mp4")  # Using .mp4 extension for H.264 codec

        # Create output directory if it doesn't exist
        output_dir = output_file.parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        # Create configurations
        text_config = create_text_config(
            font_scale=font_scale,
            title_text=title_text,
            text_align=text_align,
            font_file=str(font_file) if font_file else None
        )

        video_config = VideoConfig(
            fps=fps,
            width=width,
            height=height,
            duration=duration
        )

        map_config = MapConfig(
            zoom=zoom,
            marker_size=marker_size,
            marker_color=marker_color_tuple
        )

        # Generate video
        output_path = generate_video(
            gpx_file=gpx_file,
            output_file=output_file,
            video_config=video_config,
            map_config=map_config,
            text_config=text_config,
            captions=captions
        )

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

@app.command()
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

if __name__ == "__main__":
    app()
