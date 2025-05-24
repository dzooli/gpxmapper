"""Command for generating videos from GPX tracks."""

import logging
from pathlib import Path
from typing import Optional

import typer

from ..models import VideoConfig, MapConfig
from .utils import create_text_config, generate_video, parse_color
from . import app

logger = logging.getLogger(__name__)

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
    no_timestamp: bool = typer.Option(
        False,
        "--no-timestamp",
        help="Disable timestamp visualization in the video"
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
    scrolling_text: Optional[Path] = typer.Option(
        None,
        "--scrolling-text", "-st",
        help="Path to a text file containing content to be scrolled on the video",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    scrolling_speed: Optional[float] = typer.Option(
        None,
        "--scrolling-speed", "-ss",
        min=0.1,
        help="Speed at which the text scrolls across the video (pixels per frame). If not specified, speed will be calculated based on video duration."
    ),
    timezone: Optional[str] = typer.Option(
        None,
        "--timezone", "-tz",
        help="Timezone to convert timestamps to (e.g., 'Europe/London', 'US/Pacific'). "
             "If not specified, timestamps are not converted. "
             "Use 'local' to convert to the local timezone of the machine."
    ),
):
    """Generate a video from a GPX track file.

    The video will show the track on a map with a marker indicating the current position.
    The GPX timeline will be mapped to the video duration.
    """
    try:
        # Parse marker color
        marker_color_tuple = parse_color(marker_color)

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
            font_file=str(font_file) if font_file else None,
            no_timestamp=no_timestamp,
            scrolling_text_file=str(scrolling_text) if scrolling_text else None,
            scrolling_speed=scrolling_speed,
            timezone=timezone
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
