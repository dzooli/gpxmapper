"""Video generation programmatic service for GPXMapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from .config import create_text_config, parse_color
from ..exceptions import GPXEmptyError, GPXMissingTimeError, GPXParseError, VideoGenerationError
from ..gpx_parser import GPXParser
from ..models import MapConfig, TextConfig, VideoConfig
from ..video_generator import VideoGenerator

logger = logging.getLogger(__name__)


def _resolve_video_config(
        video_config: Optional[VideoConfig] = None,
        duration: Optional[int] = None,
        fps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
) -> VideoConfig:
    """Resolve VideoConfig from an optional instance and convenience keyword arguments."""
    base = video_config or VideoConfig(fps=30, width=320, height=320, duration=60)
    return VideoConfig(
        fps=fps if fps is not None else base.fps,
        width=width if width is not None else base.width,
        height=height if height is not None else base.height,
        duration=duration if duration is not None else base.duration,
    )


def _resolve_map_config(
        map_config: Optional[MapConfig] = None,
        zoom: Optional[int] = None,
        marker_size: Optional[int] = None,
        marker_color: Optional[str | Tuple[int, int, int]] = None,
) -> MapConfig:
    """Resolve MapConfig from an optional instance and convenience keyword arguments."""
    base = map_config or MapConfig(zoom=15, marker_size=10, marker_color=(255, 0, 0))
    color = parse_color(marker_color) if marker_color is not None else base.marker_color
    return MapConfig(
        zoom=zoom if zoom is not None else base.zoom,
        marker_size=marker_size if marker_size is not None else base.marker_size,
        marker_color=color,
    )


def _resolve_text_config(
        text_config: Optional[TextConfig] = None,
        title: Optional[str] = None,
        title_text: Optional[str] = None,
        text_color: Optional[str | Tuple[int, int, int]] = None,
        timestamp_color: Optional[str | Tuple[int, int, int]] = None,
        font_scale: Optional[float] = None,
        text_align: Optional[str] = None,
        font_file: Optional[str] = None,
        no_timestamp: Optional[bool] = None,
        show_timestamp: Optional[bool] = None,
        scrolling_text_file: Optional[str] = None,
        scrolling_speed: Optional[float] = None,
        timezone: Optional[str] = None,
        geolocate: Optional[bool] = None,
) -> TextConfig:
    """Resolve TextConfig from an optional instance and convenience keyword arguments."""
    base = text_config or TextConfig()
    custom_title = title if title is not None else title_text
    custom_color = text_color if text_color is not None else timestamp_color

    if no_timestamp is not None:
        disabled_ts = no_timestamp
    elif show_timestamp is not None:
        disabled_ts = not show_timestamp
    else:
        disabled_ts = not base.show_timestamp

    return create_text_config(
        font_scale=font_scale if font_scale is not None else base.font_scale,
        title_text=custom_title if custom_title is not None else base.title_text,
        text_align=text_align if text_align is not None else base.text_align,
        timestamp_color=custom_color if custom_color is not None else base.timestamp_color,
        font_file=font_file if font_file is not None else base.font_file,
        no_timestamp=disabled_ts,
        scrolling_text_file=scrolling_text_file if scrolling_text_file is not None else base.scrolling_text_file,
        scrolling_speed=scrolling_speed if scrolling_speed is not None else base.scrolling_speed,
        timezone=timezone if timezone is not None else base.timezone,
        geolocate=geolocate if geolocate is not None else base.geolocate,
    )


def generate_video(
        gpx_file: Optional[Path | str] = None,
    output_file: Optional[Path | str] = None,
    video_config: Optional[VideoConfig] = None,
    map_config: Optional[MapConfig] = None,
    text_config: Optional[TextConfig] = None,
    captions: Optional[Path | str] = None,
        *,
        gpx_path: Optional[Path | str] = None,
        output_path: Optional[Path | str] = None,
        duration: Optional[int] = None,
        fps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        zoom: Optional[int] = None,
        marker_size: Optional[int] = None,
        marker_color: Optional[str | Tuple[int, int, int]] = None,
        title: Optional[str] = None,
        title_text: Optional[str] = None,
        text_color: Optional[str | Tuple[int, int, int]] = None,
        timestamp_color: Optional[str | Tuple[int, int, int]] = None,
        font_scale: Optional[float] = None,
        text_align: Optional[str] = None,
        font_file: Optional[str] = None,
        no_timestamp: Optional[bool] = None,
        show_timestamp: Optional[bool] = None,
        scrolling_text_file: Optional[str] = None,
        scrolling_speed: Optional[float] = None,
        timezone: Optional[str] = None,
        geolocate: Optional[bool] = None,
) -> str:
    """Generate a video visualizing a GPX track on a map.

    Supports either explicit configuration objects (`video_config`, `map_config`, `text_config`)
    or direct convenience keyword arguments (`duration`, `fps`, `width`, `height`, `zoom`,
    `marker_color`, `title`, `text_color`, etc.).

    Args:
        gpx_file: Path to the input GPX file (positional or keyword).
        output_file: Path for the generated video file. If None, defaults to the GPX filename with .mp4 suffix.
        video_config: Configuration for video (fps, width, height, duration).
        map_config: Configuration for map rendering (zoom, marker size, marker color).
        text_config: Configuration for text overlays (title, captions, fonts, timestamps, geolocate).
        captions: Optional path to a CSV file with timestamped captions.
        gpx_path: Alias for `gpx_file`.
        output_path: Alias for `output_file`.
        duration: Video duration in seconds (convenience argument for VideoConfig).
        fps: Frames per second (convenience argument for VideoConfig).
        width: Video width in pixels (convenience argument for VideoConfig).
        height: Video height in pixels (convenience argument for VideoConfig).
        zoom: Map zoom level 1-19 (convenience argument for MapConfig).
        marker_size: Marker size in pixels (convenience argument for MapConfig).
        marker_color: Marker color as 'R,G,B' or (R, G, B) tuple (convenience argument for MapConfig).
        title: Title overlay text (alias for `title_text`).
        title_text: Title overlay text (convenience argument for TextConfig).
        text_color: Text overlay color as 'R,G,B' or (R, G, B) tuple (alias for `timestamp_color`).
        timestamp_color: Text overlay color (convenience argument for TextConfig).
        font_scale: Font scaling factor (convenience argument for TextConfig).
        text_align: Text alignment 'left', 'center', or 'right' (convenience argument for TextConfig).
        font_file: Path to TrueType font file (convenience argument for TextConfig).
        no_timestamp: Disable timestamp display if True (convenience argument for TextConfig).
        show_timestamp: Show timestamp display if True (convenience argument for TextConfig).
        scrolling_text_file: Path to scrolling text file (convenience argument for TextConfig).
        scrolling_speed: Scrolling speed in pixels per frame (convenience argument for TextConfig).
        timezone: Timezone for timestamps (convenience argument for TextConfig).
        geolocate: Enable reverse-geocoded location line overlay if True.

    Returns:
        String path to the generated output video file.

    Raises:
        ValueError: If neither `gpx_file` nor `gpx_path` is provided.
        ConfigurationError: If any configuration option is invalid.
        GPXParseError: If parsing the GPX file fails.
        GPXEmptyError: If no track points are present in the GPX file.
        GPXMissingTimeError: If track points do not contain timestamp data.
        VideoGenerationError: If rendering or encoding the video fails.
    """
    target_gpx = gpx_file if gpx_file is not None else gpx_path
    if target_gpx is None:
        raise ValueError("A GPX file path must be provided (gpx_file or gpx_path)")

    target_out = output_file if output_file is not None else output_path
    gpx_path_obj = Path(target_gpx)
    out_path = Path(target_out) if target_out is not None else gpx_path_obj.with_suffix(".mp4")

    v_config = _resolve_video_config(
        video_config=video_config,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
    )
    m_config = _resolve_map_config(
        map_config=map_config,
        zoom=zoom,
        marker_size=marker_size,
        marker_color=marker_color,
    )
    t_config = _resolve_text_config(
        text_config=text_config,
        title=title,
        title_text=title_text,
        text_color=text_color,
        timestamp_color=timestamp_color,
        font_scale=font_scale,
        text_align=text_align,
        font_file=font_file,
        no_timestamp=no_timestamp,
        show_timestamp=show_timestamp,
        scrolling_text_file=scrolling_text_file,
        scrolling_speed=scrolling_speed,
        timezone=timezone,
        geolocate=geolocate,
    )

    logger.info("Parsing GPX file: %s", gpx_path_obj)
    try:
        parser = GPXParser(str(gpx_path_obj))
        track_points = parser.parse()
    except Exception as exc:
        logger.error("Failed to parse GPX file %s: %s", gpx_path_obj, exc)
        raise GPXParseError(f"Failed to parse GPX file {gpx_path_obj}: {exc}") from exc

    if not track_points:
        logger.error("No track points found in GPX file: %s", gpx_path_obj)
        raise GPXEmptyError(f"No track points found in GPX file: {gpx_path_obj}")

    points_with_time = [p for p in track_points if p.time is not None]
    if not points_with_time:
        logger.error("GPX file %s lacks timestamps required for video generation", gpx_path_obj)
        raise GPXMissingTimeError(
            f"GPX file {gpx_path_obj} doesn't contain time data, which is required for video generation"
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
