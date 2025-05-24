"""Utility functions for the CLI commands."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import typer

from ..gpx_parser import GPXParser
from ..map_renderer import MapRenderer
from ..models import TextConfig, VideoConfig, MapConfig
from ..video_generator import VideoGenerator

logger = logging.getLogger(__name__)

def create_text_config(
    font_scale: float,
    title_text: Optional[str] = None,
    text_align: str = "left",
    timestamp_color: str = "0,0,0",
    font_file: Optional[str] = None,
    no_timestamp: bool = False,
    scrolling_text_file: Optional[str] = None,
    scrolling_speed: Optional[float] = None,
    timezone: Optional[str] = None
) -> TextConfig:
    """Create a TextConfig object from the given parameters.

    Args:
        font_scale: Font scale for all text (timestamp, title, captions)
        title_text: Optional text to display as a title on the video
        text_align: Alignment of all text (title, captions) (left, center, right)
        timestamp_color: Color of the timestamp text as R,G,B (e.g., '0,0,0' for black)
        font_file: Optional path to a TrueType font file (.ttf) for text rendering
        no_timestamp: Whether to disable timestamp visualization
        scrolling_text_file: Optional path to a text file containing content to be scrolled on the video
        scrolling_speed: Optional speed at which the text scrolls across the video (pixels per frame)
        timezone: Optional timezone to convert timestamps to (e.g., 'Europe/London', 'US/Pacific')
                 If None, timestamps are not converted. Default is None.

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
        font_file=font_file,
        show_timestamp=not no_timestamp,
        scrolling_text_file=scrolling_text_file,
        scrolling_speed=scrolling_speed,
        timezone=timezone
    )

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

def parse_color(color_str: str) -> Tuple[int, int, int]:
    """Parse a color string in the format 'R,G,B' into a tuple of integers.

    Args:
        color_str: Color string in the format 'R,G,B'

    Returns:
        Tuple of (R, G, B) values as integers

    Raises:
        typer.BadParameter: If the color string is invalid
    """
    try:
        r, g, b = map(int, color_str.split(","))
        if not all(0 <= c <= 255 for c in (r, g, b)):
            raise ValueError("Color values must be between 0 and 255")
        return (r, g, b)
    except Exception as e:
        logger.error(f"Invalid color format: {e}")
        raise typer.BadParameter("Color must be in format 'R,G,B' with values 0-255")
