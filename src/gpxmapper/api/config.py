"""Configuration and option parsing services for GPXMapper."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Optional, Tuple

from ..exceptions import ConfigurationError
from ..models import MapConfig, TextConfig, VideoConfig

logger = logging.getLogger(__name__)

_VALID_ALIGNMENTS = frozenset({"left", "center", "right"})


def parse_color(color_str: str | Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Parse a color string in format 'R,G,B' or (R, G, B) tuple into a tuple of 3 integers (0-255).

    Args:
        color_str: Color string in format 'R,G,B' or tuple (R, G, B).

    Returns:
        Tuple of (R, G, B) integer values.

    Raises:
        ConfigurationError: If format is invalid or values outside 0-255 range.
    """
    if isinstance(color_str, (tuple, list)) and len(color_str) == 3:
        if not all(isinstance(c, int) and 0 <= c <= 255 for c in color_str):
            raise ConfigurationError(f"RGB color channel values must be between 0 and 255, got: {color_str}")
        return int(color_str[0]), int(color_str[1]), int(color_str[2])
    if isinstance(color_str, str):
        try:
            parts = [int(p.strip()) for p in color_str.split(",")]
            if len(parts) != 3 or not all(0 <= c <= 255 for c in parts):
                raise ValueError("Values must be 3 integers between 0 and 255")
            return parts[0], parts[1], parts[2]
        except Exception as exc:
            raise ConfigurationError(
                f"Text overlay color must be in format 'R,G,B' with values 0-255 for each channel, got: {color_str!r}"
            ) from exc
    raise ConfigurationError(f"timestamp_color must be 'R,G,B' string or (R, G, B) tuple, got: {type(color_str)}")


def create_text_config(
    font_scale: float = 0.7,
    title_text: Optional[str] = None,
    text_align: str = "left",
    timestamp_color: str | Tuple[int, int, int] = "0,0,0",
    font_file: Optional[str] = None,
    no_timestamp: bool = False,
    scrolling_text_file: Optional[str] = None,
    scrolling_speed: Optional[float] = None,
    timezone: Optional[str] = None,
    geolocate: bool = False,
) -> TextConfig:
    """Create a validated TextConfig object.

    Args:
        font_scale: Font scale for overlay text.
        title_text: Optional title to display on the video.
        text_align: Alignment of text (left, center, right).
        timestamp_color: R,G,B color as string 'R,G,B' or tuple (R, G, B).
        font_file: Path to custom TrueType font file (.ttf).
        no_timestamp: If True, disable timestamp visualization.
        scrolling_text_file: Path to text file for scrolling content.
        scrolling_speed: Scroll speed in pixels per frame.
        timezone: Optional timezone for timestamp conversion.
        geolocate: If True, enable reverse-geocoded location line overlay.

    Returns:
        A validated TextConfig instance.

    Raises:
        ConfigurationError: If any configuration value is invalid.
    """
    color_tuple = parse_color(timestamp_color)

    norm_align = text_align.lower().strip()
    if norm_align not in _VALID_ALIGNMENTS:
        raise ConfigurationError(f"Text alignment must be one of: left, center, right (got: {text_align!r})")

    if geolocate and (scrolling_text_file is not None or scrolling_speed is not None):
        raise ConfigurationError("geolocate cannot be enabled together with scrolling_text_file or scrolling_speed")

    return TextConfig(
        font_scale=font_scale,
        title_text=title_text,
        text_align=norm_align,
        timestamp_color=color_tuple,
        font_file=font_file,
        show_timestamp=not no_timestamp,
        scrolling_text_file=scrolling_text_file,
        scrolling_speed=scrolling_speed,
        timezone=timezone,
        geolocate=geolocate,
    )


def _opt_value(options: dict, keys: tuple[str, ...], default: Any) -> Any:
    for key in keys:
        val = options.get(key)
        if val is not None:
            return val
    return default


def _resolve_no_timestamp(options: dict, default_show: bool) -> bool:
    if options.get("no_timestamp") is not None:
        return bool(options["no_timestamp"])
    if options.get("show_timestamp") is not None:
        return not bool(options["show_timestamp"])
    return not default_show


def resolve_video_config(config: Optional[VideoConfig] = None, options: Optional[dict] = None) -> VideoConfig:
    """Resolve VideoConfig from an optional instance and keyword arguments."""
    opts = options or {}
    base = config or VideoConfig(fps=30, width=320, height=320, duration=60)
    overrides = {k: opts[k] for k in ("fps", "width", "height", "duration") if opts.get(k) is not None}
    return replace(base, **overrides) if overrides else base


def resolve_map_config(config: Optional[MapConfig] = None, options: Optional[dict] = None) -> MapConfig:
    """Resolve MapConfig from an optional instance and keyword arguments."""
    opts = options or {}
    base = config or MapConfig(zoom=15, marker_size=10, marker_color=(255, 0, 0))
    zoom = _opt_value(opts, ("zoom",), base.zoom)
    marker_size = _opt_value(opts, ("marker_size",), base.marker_size)
    raw_color = opts.get("marker_color")
    marker_color = parse_color(raw_color) if raw_color is not None else base.marker_color
    return MapConfig(zoom=zoom, marker_size=marker_size, marker_color=marker_color)


def resolve_text_config(config: Optional[TextConfig] = None, options: Optional[dict] = None) -> TextConfig:
    """Resolve TextConfig from an optional instance and keyword arguments."""
    opts = options or {}
    base = config or TextConfig()
    return create_text_config(
        font_scale=_opt_value(opts, ("font_scale",), base.font_scale),
        title_text=_opt_value(opts, ("title", "title_text"), base.title_text),
        text_align=_opt_value(opts, ("text_align",), base.text_align),
        timestamp_color=_opt_value(opts, ("text_color", "timestamp_color"), base.timestamp_color),
        font_file=_opt_value(opts, ("font_file",), base.font_file),
        no_timestamp=_resolve_no_timestamp(opts, base.show_timestamp),
        scrolling_text_file=_opt_value(opts, ("scrolling_text_file",), base.scrolling_text_file),
        scrolling_speed=_opt_value(opts, ("scrolling_speed",), base.scrolling_speed),
        timezone=_opt_value(opts, ("timezone",), base.timezone),
        geolocate=_opt_value(opts, ("geolocate",), base.geolocate),
    )


def resolve_configs(
        video_config: Optional[VideoConfig] = None,
        map_config: Optional[MapConfig] = None,
        text_config: Optional[TextConfig] = None,
        options: Optional[dict] = None,
) -> tuple[VideoConfig, MapConfig, TextConfig]:
    """Resolve VideoConfig, MapConfig, and TextConfig from instances and convenience kwargs."""
    opts = options or {}
    return (
        resolve_video_config(video_config, opts),
        resolve_map_config(map_config, opts),
        resolve_text_config(text_config, opts),
    )
