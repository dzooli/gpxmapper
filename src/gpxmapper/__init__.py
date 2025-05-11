"""GPX to video mapper - creates videos from GPX tracks."""

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("gpxmapper")
    except PackageNotFoundError:
        __version__ = "0.1.0"  # Default version if package is not installed
except ImportError:
    __version__ = "0.1.0"  # Fallback for Python < 3.8
