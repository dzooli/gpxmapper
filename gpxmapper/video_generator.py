"""Video generation module for creating videos from GPX tracks."""

import csv
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pprint import pprint
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .gpx_parser import GPXTrackPoint
from .map_renderer import MapRenderer

logger = logging.getLogger(__name__)


class VideoCaptioner:
    """Handles text rendering on video frames."""

    def __init__(self, width: int, height: int, timestamp_color: Tuple[int, int, int],
                 font_scale: float, title_text: str = None, text_align: str = "left",
                 captions_file: str = None):
        """Initialize the video captioner.

        Args:
            width: Width of the video frame
            height: Height of the video frame
            timestamp_color: RGB color tuple for the timestamp text
            font_scale: Font scale for all text (timestamp, title, captions)
            title_text: Optional text to display as a title on the video
            text_align: Alignment of all text (title, captions) ("left", "center", or "right")
            captions_file: Optional path to a CSV file containing captions with timestamps
        """
        self.width = width
        self.height = height
        self.timestamp_color = timestamp_color
        self.font_scale = font_scale
        self.title_text = title_text
        self.text_align = text_align
        self.captions = {}
        self.sorted_caption_timestamps = []
        self.video_start_time = None

        # Load captions if a file is provided
        if captions_file:
            self.load_captions(captions_file)

    def add_timestamp_to_frame(self, frame: np.ndarray, timestamp: datetime) -> None:
        """Add timestamp text to the frame.

        Args:
            frame: The frame to add the timestamp to
            timestamp: The timestamp to display
        """
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame, timestamp_str, (10, self.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.timestamp_color, 2
        )

    def set_video_start_time(self, start_time: datetime) -> None:
        """Set the start time of the video.

        This is used to calculate the relative timestamps for captions.

        Args:
            start_time: The start time of the video
        """
        self.video_start_time = start_time

    def load_captions(self, captions_file: str) -> None:
        """Load captions from a CSV file.

        The CSV file should have two columns:
        1. Timestamp in HH:MM:SS format (relative to the start of the video)
        2. Caption text

        Args:
            captions_file: Path to the CSV file containing captions
        """
        try:
            with open(captions_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Skip the header row
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        timestamp_str = row[0].strip()
                        caption_text = row[1].strip()

                        # Parse timestamp in HH:MM:SS format
                        try:
                            h, m, s = map(int, timestamp_str.split(':'))
                            # Store as total seconds for easy comparison
                            total_seconds = h * 3600 + m * 60 + s
                            self.captions[total_seconds] = caption_text
                        except ValueError:
                            logger.warning(f"Invalid timestamp format in captions file: {timestamp_str}")

            # Sort the caption timestamps for efficient lookup
            self.sorted_caption_timestamps = sorted(self.captions.keys())
        except Exception as e:
            logger.error(f"Error loading captions file: {e}")

    def add_caption_to_frame(self, frame: np.ndarray, seconds_since_start: float) -> None:
        """Add caption text to the frame if available for the current timestamp.

        The caption from the most recent timestamp that is less than or equal to the current
        timestamp will be displayed, and it will persist until a new caption is available.

        Args:
            frame: The frame to add the caption to
            seconds_since_start: Seconds elapsed since the start of the video
        """
        if not self.sorted_caption_timestamps:
            return

        if seconds_since_start < 0:
            logger.warning(f"Seconds since start {seconds_since_start} is negative")
            return

        # Find the most recent caption timestamp that is less than or equal to the current timestamp
        most_recent_timestamp = None

        # Use binary search to find the index of the first timestamp greater than seconds_since_start
        import bisect
        index = bisect.bisect_right(self.sorted_caption_timestamps, seconds_since_start)

        # If index is 0, there's no timestamp less than or equal to seconds_since_start
        # Otherwise, the most recent timestamp is at index - 1
        if index > 0:
            most_recent_timestamp = self.sorted_caption_timestamps[index - 1]

        # Debug logging
        logger.debug(f"Seconds since start: {seconds_since_start}, most_recent_timestamp: {most_recent_timestamp}")

        # Get the caption text for the most recent timestamp
        caption_text = self.captions.get(most_recent_timestamp) if most_recent_timestamp is not None else None

        if not caption_text:
            return

        # Calculate position based on alignment
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.font_scale
        thickness = 2

        # Get text size to calculate position
        (text_width, text_height), _ = cv2.getTextSize(
            caption_text, font, font_scale, thickness
        )

        # Calculate x position based on alignment
        margin = 10
        if self.text_align == "left":
            x_pos = margin
        elif self.text_align == "center":
            x_pos = (self.width - text_width) // 2
        elif self.text_align == "right":
            x_pos = self.width - text_width - margin
        else:  # Default to left if invalid alignment
            x_pos = margin

        # Calculate y position - below title with margin
        # First get title height if there is a title
        title_height = 0
        if self.title_text:
            (_, title_height), _ = cv2.getTextSize(
                self.title_text, font, self.font_scale, thickness
            )

        # Y position (below title with margin)
        y_pos = margin + title_height + text_height + (margin if title_height > 0 else 0)

        # Draw the caption
        cv2.putText(
            frame, caption_text, (x_pos, y_pos),
            font, font_scale, self.timestamp_color, thickness
        )

    def add_title_to_frame(self, frame: np.ndarray) -> None:
        """Add title text to the frame if provided.

        Args:
            frame: The frame to add the title to
        """
        if not self.title_text:
            return

        # Calculate position based on alignment
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.font_scale
        thickness = 2

        # Get text size to calculate position
        (text_width, text_height), _ = cv2.getTextSize(
            self.title_text, font, font_scale, thickness
        )

        # Calculate x position based on alignment
        margin = 10
        if self.text_align == "left":
            x_pos = margin
        elif self.text_align == "center":
            x_pos = (self.width - text_width) // 2
        elif self.text_align == "right":
            x_pos = self.width - text_width - margin
        else:  # Default to left if invalid alignment
            x_pos = margin

        # Y position (top of frame with margin)
        y_pos = margin + text_height

        # Draw the title
        cv2.putText(
            frame, self.title_text, (x_pos, y_pos),
            font, font_scale, self.timestamp_color, thickness
        )


class VideoGenerator:
    """Generates videos from GPX tracks with map visualization."""

    def __init__(self, output_path: str, fps: int = 30, resolution: Tuple[int, int] = (1280, 720),
                 zoom_level: int = 15, marker_color: Tuple[int, int, int] = (255, 0, 0),
                 marker_size: int = 10, text_config=None, captions_file: str = None):
        """Initialize the video generator.

        Args:
            output_path: Path where the output video will be saved
            fps: Frames per second for the output video
            resolution: Resolution of the output video as (width, height)
            zoom_level: Zoom level for the map tiles
            marker_color: RGB color tuple for the position marker
            marker_size: Size of the position marker in pixels
            text_config: Configuration for text rendering in the video
            captions_file: Optional path to a CSV file containing captions with timestamps
        """
        self.output_path = output_path
        self.fps = fps
        self.width, self.height = resolution
        self.zoom_level = zoom_level
        self.marker_color = marker_color
        self.marker_size = marker_size
        self.map_renderer = MapRenderer(cache_dir=os.path.join(tempfile.gettempdir(), "gpxmapper_tiles"))

        # Create a video captioner for text rendering
        self.captioner = VideoCaptioner(
            width=self.width,
            height=self.height,
            timestamp_color=text_config.timestamp_color,
            font_scale=text_config.font_scale,
            title_text=text_config.title_text,
            text_align=text_config.text_align,
            captions_file=captions_file
        )

    def _interpolate_position(self, track_points: List[GPXTrackPoint],
                              timestamp: datetime) -> Tuple[float, float]:
        """Interpolate position at a given timestamp between track points.

        Args:
            track_points: List of GPXTrackPoint objects
            timestamp: Timestamp to interpolate position for

        Returns:
            Tuple of (latitude, longitude)

        Raises:
            ValueError: If track points don't have time data or timestamp is out of range
        """
        # Filter points with time data
        points_with_time = [p for p in track_points if p.time is not None]
        if not points_with_time:
            raise ValueError("Track points don't have time data")

        # Sort points by time
        points_with_time.sort(key=lambda p: p.time)

        # Check if timestamp is within range
        if timestamp < points_with_time[0].time or timestamp > points_with_time[-1].time:
            raise ValueError(f"Timestamp {timestamp} is outside the track time range")

        # Find the two points to interpolate between
        for i in range(len(points_with_time) - 1):
            p1 = points_with_time[i]
            p2 = points_with_time[i + 1]

            if p1.time <= timestamp <= p2.time:
                # Calculate interpolation factor
                total_seconds = (p2.time - p1.time).total_seconds()
                if total_seconds == 0:
                    # Same timestamp, no need to interpolate
                    return p1.latitude, p1.longitude

                elapsed_seconds = (timestamp - p1.time).total_seconds()
                factor = elapsed_seconds / total_seconds

                # Interpolate latitude and longitude
                lat = p1.latitude + factor * (p2.latitude - p1.latitude)
                lon = p1.longitude + factor * (p2.longitude - p1.longitude)

                return lat, lon

        # This should not happen if the timestamp check above is correct
        raise ValueError(f"Failed to interpolate position for timestamp {timestamp}")

    def _generate_frame(self, frame_idx: int, frame_timestamp: datetime, frame_seconds: float,
                        points_with_time: List[GPXTrackPoint]) -> np.ndarray:
        """Generate a single video frame.

        Args:
            frame_idx: Index of the frame
            frame_timestamp: Timestamp for this frame
            frame_seconds: Seconds elapsed since the start of the video
            points_with_time: List of track points with time data

        Returns:
            Frame as numpy array in BGR format
        """
        # Interpolate position
        lat, lon = self._interpolate_position(points_with_time, frame_timestamp)

        # Render map for this position from the composite map
        map_image = self.map_renderer.render_from_composite(
            lat, lon, self.width, self.height, self.marker_color, self.marker_size
        )

        if map_image is None:
            logger.warning(f"Failed to render map for frame {frame_idx}, using blank frame")
            # Create a blank frame
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            # Convert PIL image to numpy array
            map_array = np.array(map_image)

            # Convert from RGB to BGR (OpenCV uses BGR)
            frame = cv2.cvtColor(map_array, cv2.COLOR_RGB2BGR)

        # Add timestamp, title, and caption using the captioner
        self.captioner.add_timestamp_to_frame(frame, frame_timestamp)
        self.captioner.add_title_to_frame(frame)
        self.captioner.add_caption_to_frame(frame, frame_seconds)

        return frame


    def _write_video_frames(self, video_writer: cv2.VideoWriter, points_with_time: List[GPXTrackPoint],
                            duration_seconds: int, start_time: datetime, total_track_seconds: float) -> None:
        """Write frames to video file.

        Args:
            video_writer: OpenCV VideoWriter object
            points_with_time: List of track points with time data
            duration_seconds: Duration of the video in seconds
            start_time: Start time of the track
            total_track_seconds: Total duration of the track in seconds
        """
        # Set the video start time in the captioner
        self.captioner.set_video_start_time(start_time)

        # Calculate total number of frames
        total_frames = duration_seconds * self.fps

        for frame_idx in range(total_frames):
            # Calculate seconds elapsed since the start of the video
            frame_seconds = frame_idx / self.fps
            # Calculate timestamp for this frame based on track progress
            progress = frame_idx / total_frames
            track_seconds = progress * total_track_seconds
            frame_timestamp = start_time + timedelta(seconds=track_seconds)

            frame = self._generate_frame(frame_idx, frame_timestamp, frame_seconds, points_with_time)
            video_writer.write(frame)

            # Log progress
            if frame_idx % (self.fps * 5) == 0:  # Log every 5 seconds of video
                logger.info(f"Generated {frame_idx}/{total_frames} frames ({frame_idx / total_frames:.1%})")

    def generate_video(self, track_points: List[GPXTrackPoint], duration_seconds: int) -> str:
        """Generate a video from GPX track points.

        Args:
            track_points: List of GPXTrackPoint objects
            duration_seconds: Duration of the output video in seconds

        Returns:
            Path to the generated video file

        Raises:
            ValueError: If track points don't have time data or other issues
        """
        # Filter points with time data
        points_with_time = [p for p in track_points if p.time is not None]
        if not points_with_time:
            raise ValueError("Track points don't have time data")

        # Sort points by time  
        points_with_time.sort(key=lambda p: p.time)

        # Get time range
        start_time = points_with_time[0].time
        end_time = points_with_time[-1].time
        total_track_seconds = (end_time - start_time).total_seconds()

        logger.info(f"Generating video with duration {duration_seconds}s from track spanning {total_track_seconds}s")

        # Calculate the bounding box of all track points
        min_lat = min(p.latitude for p in track_points)
        max_lat = max(p.latitude for p in track_points)
        min_lon = min(p.longitude for p in track_points)
        max_lon = max(p.longitude for p in track_points)

        # Add some padding to the bounding box (10%)
        lat_padding = (max_lat - min_lat) * 0.1
        lon_padding = (max_lon - min_lon) * 0.1
        min_lat -= lat_padding
        max_lat += lat_padding
        min_lon -= lon_padding
        max_lon += lon_padding

        logger.info(f"Track bounding box: {min_lat:.6f},{min_lon:.6f} to {max_lat:.6f},{max_lon:.6f}")

        # Create a composite map for the entire track
        logger.info(f"Creating composite map at zoom level {self.zoom_level}...")
        self.map_renderer.create_composite_map(min_lat, min_lon, max_lat, max_lon, self.zoom_level)
        logger.info(
            f"Composite map created with size {self.map_renderer.composite_map_info['width']}x{self.map_renderer.composite_map_info['height']} pixels")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        video_writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )

        if not video_writer.isOpened():
            raise ValueError(f"Failed to open video writer for {self.output_path}")

        try:
            self._write_video_frames(video_writer, points_with_time, duration_seconds, start_time, total_track_seconds)
            logger.info(f"Video generation complete: {self.output_path}")
            return self.output_path

        finally:
            # Release video writer
            video_writer.release()
