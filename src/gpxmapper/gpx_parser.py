"""GPX file parsing module for extracting track data."""

import gpxpy
from typing import List, Tuple, Optional
from datetime import datetime
import logging

from .models import GPXTrackPoint

logger = logging.getLogger(__name__)


class GPXParser:
    """Parser for GPX files that extracts track points with timestamps."""

    def __init__(self, gpx_file_path: str):
        """Initialize with the path to a GPX file.

        Args:
            gpx_file_path: Path to the GPX file to parse
        """
        self.gpx_file_path = gpx_file_path
        self.track_points: List[GPXTrackPoint] = []
        self._parsed = False

    def parse(self) -> List[GPXTrackPoint]:
        """Parse the GPX file and extract track points.

        Returns:
            List of GPXTrackPoint objects

        Raises:
            FileNotFoundError: If the GPX file doesn't exist
            ValueError: If the GPX file is invalid
        """
        try:
            with open(self.gpx_file_path, 'r') as gpx_file:
                gpx = gpxpy.parse(gpx_file)

                for track in gpx.tracks:
                    for segment in track.segments:
                        for point in segment.points:
                            track_point = GPXTrackPoint(
                                latitude=point.latitude,
                                longitude=point.longitude,
                                elevation=point.elevation,
                                time=point.time,
                                extensions=point.extensions
                            )
                            self.track_points.append(track_point)

                logger.info(f"Parsed {len(self.track_points)} track points from {self.gpx_file_path}")
                self._parsed = True
                return self.track_points

        except FileNotFoundError:
            logger.error(f"GPX file not found: {self.gpx_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error parsing GPX file: {e}")
            raise ValueError(f"Invalid GPX file: {e}")

    def get_time_bounds(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the start and end times of the track.

        Returns:
            Tuple of (start_time, end_time), both can be None if no time data
        """
        if not self._parsed:
            self.parse()

        if not self.track_points:
            return None, None

        # Filter out points without time data
        points_with_time = [p for p in self.track_points if p.time is not None]
        if not points_with_time:
            return None, None

        start_time = min(p.time for p in points_with_time)
        end_time = max(p.time for p in points_with_time)

        return start_time, end_time

    def get_coordinate_bounds(self) -> Tuple[float, float, float, float]:
        """Get the bounding box of the track.

        Returns:
            Tuple of (min_lat, min_lon, max_lat, max_lon)
        """
        if not self._parsed:
            self.parse()

        if not self.track_points:
            raise ValueError("No track points found")

        min_lat = min(p.latitude for p in self.track_points)
        max_lat = max(p.latitude for p in self.track_points)
        min_lon = min(p.longitude for p in self.track_points)
        max_lon = max(p.longitude for p in self.track_points)

        return min_lat, min_lon, max_lat, max_lon
