"""GPX inspection and metadata service for GPXMapper."""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

from ..exceptions import GPXEmptyError, GPXParseError
from ..gpx_parser import GPXParser
from ..models import GPXInfo

logger = logging.getLogger(__name__)


def get_gpx_info(gpx_file: Path | str) -> GPXInfo:
    """Parse a GPX file and return summary metadata.

    Args:
        gpx_file: Path to the GPX file.

    Returns:
        GPXInfo instance with track point count, time bounds, duration, and coordinate bounds.

    Raises:
        GPXEmptyError: If the GPX file contains no track points.
        GPXParseError: If parsing the GPX file fails.
    """
    path = Path(gpx_file)
    logger.info("Reading GPX file: %s", path)
    try:
        parser = GPXParser(str(path))
        track_points = parser.parse()
    except Exception as exc:
        logger.error("Error reading GPX file %s: %s", path, exc)
        raise GPXParseError(f"Failed to parse GPX file {path}: {exc}") from exc

    if not track_points:
        logger.error("No track points found in GPX file: %s", path)
        raise GPXEmptyError(f"No track points found in GPX file: {path}")

    start_time, end_time = parser.get_time_bounds()
    duration: Optional[timedelta] = None
    if start_time and end_time:
        duration = end_time - start_time

    bounds = parser.get_coordinate_bounds()

    return GPXInfo(
        file_path=path,
        point_count=len(track_points),
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        coordinate_bounds=bounds,
    )
