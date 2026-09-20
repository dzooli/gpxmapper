"""Utility and presentation adapter functions for the CLI commands."""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple

import typer

from ..api import (
    create_text_config as api_create_text_config,
    generate_video as api_generate_video,
    parse_color as api_parse_color,
)
from ..exceptions import ConfigurationError, GPXMapperError
from ..models import MapConfig, TextConfig, VideoConfig
from ..nominatim_config import get_nominatim_base_url, probe_nominatim_status_sync

logger = logging.getLogger(__name__)


def _resolve_text_config_after_nominatim_probe(text_config: TextConfig) -> TextConfig:
    """If geolocate is requested, probe ``/status``; on failure optionally drop geolocation."""
    if not text_config.geolocate:
        return text_config
    ok, err = probe_nominatim_status_sync()
    if ok:
        return text_config
    base = get_nominatim_base_url()
    typer.secho(
        "\n".join(
            [
                "Nominatim server is unreachable.",
                f"  URL: {base}",
                "  After 3 attempts to GET /status, the last error was:",
                f"  {err}",
                "",
                "Hints:",
                "  - Start local Nominatim (e.g. Docker on port 8080), or",
                "  - Set NOMINATIM_SERVER=https://nominatim.openstreetmap.org (see OSM usage policy), or",
                "  - Check firewall / TLS / correct host and port.",
            ]
        ),
        fg=typer.colors.RED,
        err=True,
    )
    if not sys.stdin.isatty():
        typer.secho(
            "Not prompting because stdin is not a terminal; fix Nominatim or omit --geolocate.",
            err=True,
        )
        raise typer.Abort()
    if typer.confirm("Continue without reverse geolocation?", default=False):
        return replace(text_config, geolocate=False)
    raise typer.Abort()


def create_text_config(
    font_scale: float,
    title_text: Optional[str] = None,
    text_align: str = "left",
    timestamp_color: str = "0,0,0",
    font_file: Optional[str] = None,
    no_timestamp: bool = False,
    scrolling_text_file: Optional[str] = None,
    scrolling_speed: Optional[float] = None,
    timezone: Optional[str] = None,
    geolocate: bool = False,
) -> TextConfig:
    """Create a TextConfig object from the given parameters (CLI adapter).

    Raises:
        typer.BadParameter: If configuration values are invalid.
    """
    try:
        return api_create_text_config(
            font_scale=font_scale,
            title_text=title_text,
            text_align=text_align,
            timestamp_color=timestamp_color,
            font_file=font_file,
            no_timestamp=no_timestamp,
            scrolling_text_file=scrolling_text_file,
            scrolling_speed=scrolling_speed,
            timezone=timezone,
            geolocate=geolocate,
        )
    except ConfigurationError as exc:
        logger.error("Invalid text configuration: %s", exc)
        raise typer.BadParameter(str(exc)) from exc


def parse_color(color_str: str) -> Tuple[int, int, int]:
    """Parse a color string in format 'R,G,B' into RGB tuple (CLI adapter).

    Raises:
        typer.BadParameter: If the color string is invalid.
    """
    try:
        return api_parse_color(color_str)
    except ConfigurationError as exc:
        logger.error("Invalid color format: %s", exc)
        raise typer.BadParameter("Color must be in format 'R,G,B' with values 0-255") from exc


def generate_video(
    gpx_file: Path,
    output_file: Path,
    video_config: VideoConfig,
    map_config: MapConfig,
    text_config: TextConfig,
    captions: Optional[Path] = None,
) -> str:
    """Generate a video from a GPX track file (CLI adapter).

    Raises:
        typer.Abort: If video generation fails.
    """
    try:
        resolved_text_config = _resolve_text_config_after_nominatim_probe(text_config)
        return api_generate_video(
            gpx_file=gpx_file,
            output_file=output_file,
            video_config=video_config,
            map_config=map_config,
            text_config=resolved_text_config,
            captions=captions,
        )
    except typer.Abort:
        raise
    except GPXMapperError as exc:
        logger.error("Video generation failed: %s", exc)
        raise typer.Abort() from exc
    except Exception as exc:
        logger.exception("Unexpected error generating video: %s", exc)
        raise typer.Abort() from exc
