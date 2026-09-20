"""Domain exceptions for the GPXMapper package.

This hierarchy decouples core domain and API error handling from presentation
layers (e.g. Typer CLI, future GUI/TUI).
"""

from __future__ import annotations


class GPXMapperError(Exception):
    """Base exception class for all errors in GPXMapper."""


class GPXParseError(GPXMapperError, ValueError):
    """Raised when parsing a GPX track file fails or has invalid contents."""


class GPXEmptyError(GPXParseError):
    """Raised when a GPX file contains no track points."""


class GPXMissingTimeError(GPXParseError):
    """Raised when a GPX file points lack required timestamp data."""


class ConfigurationError(GPXMapperError, ValueError):
    """Raised when invalid configuration parameters or incompatible options are provided."""


class VideoGenerationError(GPXMapperError):
    """Raised when video rendering or encoding fails."""


class NominatimUnavailableError(GPXMapperError):
    """Raised when the configured Nominatim reverse-geocoding service is unavailable."""
