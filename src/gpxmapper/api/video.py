"""Video generation programmatic service for GPXMapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..exceptions import GPXEmptyError, GPXMissingTimeError, GPXParseError, VideoGenerationError
from ..gpx_parser import GPXParser
from ..models import MapConfig, TextConfig, VideoConfig
from ..video_generator import VideoGenerator

logger = logging.getLogger(__name__)


def generate_video(
    gpx_file: Path | str,
    output_file: Optional[Path | str] = None,
    video_config: Optional[VideoConfig] = None,
    map_config: Optional[MapConfig] = None,
    text_config: Optional[TextConfig] = None,
    captions: Optional[Path | str] = None,
) -> str:
    """Generate a video visualizing a GPX track on a map.

    Args:
        gpx_file: Path to the input GPX file.
        output_file: Path for the generated video file. If None, defaults to the GPX filename with .mp4 suffix.
        video_config: Configuration for video (fps, width, height, duration).
        map_config: Configuration for map rendering (zoom, marker size, marker color).
        text_config: Configuration for text overlays (title, captions, fonts, timestamps, geolocate).
        captions: Optional path to a CSV file with timestamped captions.

    Returns:
        String path to the generated output video file.

    Raises:
        GPXParseError: If parsing the GPX file fails.
        GPXEmptyError: If no track points are present in the GPX file.
        GPXMissingTimeError: If track points do not contain timestamp data.
        VideoGenerationError: If rendering or encoding the video fails.
    """
    gpx_path = Path(gpx_file)
    out_path = Path(output_file) if output_file is not None else gpx_path.with_suffix(".mp4")

    v_config = video_config or VideoConfig(fps=30, width=320, height=320, duration=60)
    m_config = map_config or MapConfig(zoom=15, marker_size=10, marker_color=(255, 0, 0))
    t_config = text_config or TextConfig()

    logger.info("Parsing GPX file: %s", gpx_path)
    try:
        parser = GPXParser(str(gpx_path))
        track_points = parser.parse()
    except Exception as exc:
        logger.error("Failed to parse GPX file %s: %s", gpx_path, exc)
        raise GPXParseError(f"Failed to parse GPX file {gpx_path}: {exc}") from exc

    if not track_points:
        logger.error("No track points found in GPX file: %s", gpx_path)
        raise GPXEmptyError(f"No track points found in GPX file: {gpx_path}")

    points_with_time = [p for p in track_points if p.time is not None]
    if not points_with_time:
        logger.error("GPX file %s lacks timestamps required for video generation", gpx_path)
        raise GPXMissingTimeError(
            f"GPX file {gpx_path} doesn't contain time data, which is required for video generation"
        )

    start_time, end_time = parser.get_time_bounds()
    logger.info("Track time range: %s to %s", start_time, end_time)

    out_dir = out_path.parent
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating video: %s", out_path)
    try:
        generator = VideoGenerator(
            output_path=str(out_path),
            fps=v_config.fps,
            resolution=(v_config.width, v_config.height),
            zoom_level=m_config.zoom,
            marker_color=m_config.marker_color,
            marker_size=m_config.marker_size,
            text_config=t_config,
            captions_file=str(captions) if captions else None,
        )
        return generator.generate_video(track_points, v_config.duration)
    except Exception as exc:
        logger.exception("Error generating video: %s", exc)
        raise VideoGenerationError(f"Error generating video {out_path}: {exc}") from exc
