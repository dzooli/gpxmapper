"""Video generation programmatic service for GPXMapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .config import create_text_config, parse_color
from ..exceptions import GPXEmptyError, GPXMissingTimeError, GPXParseError, VideoGenerationError
from ..gpx_parser import GPXParser
from ..models import MapConfig, TextConfig, VideoConfig
from ..video_generator import VideoGenerator

logger = logging.getLogger(__name__)


def _resolve_configs(
        video_config: Optional[VideoConfig],
        map_config: Optional[MapConfig],
        text_config: Optional[TextConfig],
        options: dict,
) -> tuple[VideoConfig, MapConfig, TextConfig]:
    """Resolve VideoConfig, MapConfig, and TextConfig from instances and convenience kwargs."""
    v_base = video_config or VideoConfig(fps=30, width=320, height=320, duration=60)
    v_conf = VideoConfig(
        fps=options.get("fps") if options.get("fps") is not None else v_base.fps,
        width=options.get("width") if options.get("width") is not None else v_base.width,
        height=options.get("height") if options.get("height") is not None else v_base.height,
        duration=options.get("duration") if options.get("duration") is not None else v_base.duration,
    )

    m_base = map_config or MapConfig(zoom=15, marker_size=10, marker_color=(255, 0, 0))
    raw_color = options.get("marker_color")
    m_color = parse_color(raw_color) if raw_color is not None else m_base.marker_color
    m_conf = MapConfig(
        zoom=options.get("zoom") if options.get("zoom") is not None else m_base.zoom,
        marker_size=options.get("marker_size") if options.get("marker_size") is not None else m_base.marker_size,
        marker_color=m_color,
    )

    t_base = text_config or TextConfig()
    custom_title = options.get("title") if options.get("title") is not None else options.get("title_text")
    custom_color = (
        options.get("text_color") if options.get("text_color") is not None else options.get("timestamp_color")
    )

    if options.get("no_timestamp") is not None:
        disabled_ts = bool(options["no_timestamp"])
    elif options.get("show_timestamp") is not None:
        disabled_ts = not bool(options["show_timestamp"])
    else:
        disabled_ts = not t_base.show_timestamp

    t_conf = create_text_config(
        font_scale=options.get("font_scale") if options.get("font_scale") is not None else t_base.font_scale,
        title_text=custom_title if custom_title is not None else t_base.title_text,
        text_align=options.get("text_align") if options.get("text_align") is not None else t_base.text_align,
        timestamp_color=custom_color if custom_color is not None else t_base.timestamp_color,
        font_file=options.get("font_file") if options.get("font_file") is not None else t_base.font_file,
        no_timestamp=disabled_ts,
        scrolling_text_file=(
            options.get("scrolling_text_file")
            if options.get("scrolling_text_file") is not None
            else t_base.scrolling_text_file
        ),
        scrolling_speed=(
            options.get("scrolling_speed") if options.get("scrolling_speed") is not None else t_base.scrolling_speed
        ),
        timezone=options.get("timezone") if options.get("timezone") is not None else t_base.timezone,
        geolocate=options.get("geolocate") if options.get("geolocate") is not None else t_base.geolocate,
    )

    return v_conf, m_conf, t_conf


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
        **kwargs,
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
        **kwargs: Convenience keyword overrides for video, map, and text configurations.

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

    v_config, m_config, t_config = _resolve_configs(video_config, map_config, text_config, kwargs)

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
