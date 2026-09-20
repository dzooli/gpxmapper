"""Configuration and option parsing services for GPXMapper."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from ..exceptions import ConfigurationError
from ..models import TextConfig

logger = logging.getLogger(__name__)

_VALID_ALIGNMENTS = frozenset({"left", "center", "right"})


def parse_color(color_str: str) -> Tuple[int, int, int]:
    """Parse a color string in the format 'R,G,B' into a tuple of 3 integers (0-255).

    Args:
        color_str: Color string in format 'R,G,B'.

    Returns:
        Tuple of (R, G, B) integer values.

    Raises:
        ConfigurationError: If format is invalid or values outside 0-255 range.
    """
    try:
        parts = [int(p.strip()) for p in color_str.split(",")]
        if len(parts) != 3 or not all(0 <= c <= 255 for c in parts):
            raise ValueError("Values must be 3 integers between 0 and 255")
        return parts[0], parts[1], parts[2]
    except Exception as exc:
        raise ConfigurationError(
            f"Text overlay color must be in format 'R,G,B' with values 0-255 for each channel, got: {color_str!r}"
        ) from exc


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
    if isinstance(timestamp_color, str):
        color_tuple = parse_color(timestamp_color)
    elif isinstance(timestamp_color, tuple) and len(timestamp_color) == 3:
        if not all(0 <= c <= 255 for c in timestamp_color):
            raise ConfigurationError(f"RGB color channel values must be between 0 and 255, got: {timestamp_color}")
        color_tuple = timestamp_color
    else:
        raise ConfigurationError(
            f"timestamp_color must be 'R,G,B' string or (R, G, B) tuple, got: {type(timestamp_color)}"
        )

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
