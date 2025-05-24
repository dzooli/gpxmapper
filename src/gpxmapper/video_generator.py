"""Video generation module for creating videos from GPX tracks.

This module has been optimized for performance with the following improvements:
1. Parallel frame generation using ThreadPoolExecutor
2. Efficient position interpolation with binary search
3. Caching of interpolated positions
4. Batch processing of frames for better memory management
"""

import csv
import logging
import os
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import concurrent.futures
import zoneinfo

import cv2
import numpy as np

from .models import GPXTrackPoint, TextConfig
from .map_renderer import MapRenderer
from .font_manager import FontManager

logger = logging.getLogger(__name__)


class VideoCaptioner:
    """Handles text rendering on video frames."""

    def __init__(self, width: int, height: int, timestamp_color: Tuple[int, int, int] = (0,0,0),
                 font_scale: float = 0.7, title_text: str = "", text_align: str = "left",
                 captions_file: str = "", font_file: str = "", show_timestamp: bool = True,
                 scrolling_text_file: str = None, scrolling_speed: float = None, timezone: str = None):
        """Initialize the video captioner.

        Args:
            width: Width of the video frame
            height: Height of the video frame
            timestamp_color: RGB color tuple for the timestamp text
            font_scale: Font scale for all text (timestamp, title, captions)
            title_text: Optional text to display as a title on the video
            text_align: Alignment of all text (title, captions) ("left", "center", or "right")
            captions_file: Optional path to a CSV file containing captions with timestamps
            font_file: Optional path to a TrueType font file (.ttf) for text rendering
            show_timestamp: Whether to display timestamps on the video (default: True)
            scrolling_text_file: Optional path to a text file containing content to be scrolled on the video
            scrolling_speed: Optional speed at which the text scrolls across the video (pixels per frame)
            timezone: Optional timezone to convert timestamps to. Must be a full timezone name (e.g., 'Europe/Budapest', 'US/Pacific')
                     If None, timestamps are not converted.
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
        self.show_timestamp = show_timestamp
        self.scrolling_text = ""
        self.scrolling_speed = scrolling_speed
        self.scrolling_position = self.width  # Start from the right edge of the frame
        self.scrolling_text_width = 0
        self.video_duration = None  # Will be set later
        self.timezone = timezone

        # Initialize font manager
        self.font_manager = FontManager(font_file, font_scale)

        # Load captions if a file is provided
        if captions_file:
            self.load_captions(captions_file)

        # Load scrolling text if a file is provided
        if scrolling_text_file:
            self.load_scrolling_text(scrolling_text_file)

    def add_timestamp_to_frame(self, frame: np.ndarray, timestamp: datetime) -> np.ndarray:
        """Add timestamp text to the frame.

        Args:
            frame: The frame to add the timestamp to
            timestamp: The timestamp to display (in UTC)

        Returns:
            The frame with timestamp text added (or unchanged if show_timestamp is False)
        """
        # Skip timestamp rendering if show_timestamp is False
        if not self.show_timestamp:
            return frame

        # Convert timestamp to the specified timezone if needed
        display_timestamp = timestamp
        if self.timezone:
            # Ensure the timestamp has UTC timezone info
            if timestamp.tzinfo is None:
                utc_timestamp = timestamp.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
            else:
                utc_timestamp = timestamp

            # Convert to the specified timezone
            try:
                # Use the specified timezone
                target_tz = zoneinfo.ZoneInfo(self.timezone)
                display_timestamp = utc_timestamp.astimezone(target_tz)
            except zoneinfo.ZoneInfoNotFoundError:
                logger.warning(f"Unknown timezone: {self.timezone}. Using UTC.")
                display_timestamp = utc_timestamp

        timestamp_str = display_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        # Use font manager to render text
        return self.font_manager.render_text(
            frame, timestamp_str, (10, self.height - 20),
            self.timestamp_color, 2
        )

    def set_video_start_time(self, start_time: datetime) -> None:
        """Set the start time of the video.

        This is used to calculate the relative timestamps for captions.

        Args:
            start_time: The start time of the video
        """
        self.video_start_time = start_time

    def set_video_duration(self, duration_seconds: int, fps: int) -> None:
        """Set the duration of the video and calculate scrolling speed if not provided.

        Args:
            duration_seconds: Duration of the video in seconds
            fps: Frames per second of the video
        """
        self.video_duration = duration_seconds

        # Calculate scrolling speed if not provided
        if self.scrolling_text and self.scrolling_speed is None:
            # Calculate a speed that allows the text to be read
            # The text should take about 75% of the video duration to scroll across the screen
            total_distance = self.width + self.scrolling_text_width
            total_frames = duration_seconds * fps * 0.75
            self.scrolling_speed = total_distance / total_frames
            logger.info(f"Calculated scrolling speed: {self.scrolling_speed:.2f} pixels per frame")

    def load_scrolling_text(self, scrolling_text_file: str) -> None:
        """Load scrolling text from a text file.

        The file content will be treated as one long string, even if it contains newlines.

        Args:
            scrolling_text_file: Path to the text file containing scrolling text
        """
        try:
            with open(scrolling_text_file, 'r', encoding='utf-8') as f:
                # Read the entire file content and replace newlines with spaces
                self.scrolling_text = ' '.join(line.strip() for line in f)

                # Calculate the width of the text
                thickness = 2
                (text_width, _), _ = self.font_manager.get_text_size(self.scrolling_text, thickness)
                self.scrolling_text_width = text_width

                logger.info(f"Loaded scrolling text from {scrolling_text_file} ({len(self.scrolling_text)} characters, {text_width} pixels wide)")
        except Exception as e:
            logger.error(f"Error loading scrolling text file: {e}")
            self.scrolling_text = ""

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

    def _calculate_text_x_position(self, text_width: int, margin:int = 10) -> int:
        """Calculate the x position for text based on alignment."""
        if self.text_align == "left":
            return margin
        elif self.text_align == "center":
            return (self.width - text_width) // 2
        elif self.text_align == "right":
            return self.width - text_width - margin
        return margin

    def add_caption_to_frame(self, frame: np.ndarray, seconds_since_start: float) -> np.ndarray:
        """Add caption text to the frame if available for the current timestamp.

        The caption from the most recent timestamp that is less than or equal to the current
        timestamp will be displayed, and it will persist until a new caption is available.

        Args:
            frame: The frame to add the caption to
            seconds_since_start: Seconds elapsed since the start of the video

        Returns:
            The frame with caption text added
        """
        if not self.sorted_caption_timestamps:
            return frame

        if seconds_since_start < 0:
            logger.warning(f"Seconds since start {seconds_since_start} is negative")
            return frame

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
            return frame

        thickness = 2

        # Get text size to calculate position using font manager
        (text_width, text_height), _ = self.font_manager.get_text_size(
            caption_text, thickness
        )

        # Calculate x position based on alignment
        margin = 10
        x_pos = self._calculate_text_x_position(text_width, margin)

        # Calculate y position - below title with margin
        # First get title height if there is a title
        title_height = 0
        if self.title_text:
            (_, title_height), _ = self.font_manager.get_text_size(
                self.title_text, thickness
            )

        # Y position (below title with margin)
        y_pos = margin + title_height + text_height + (margin if title_height > 0 else 0)

        # Draw the caption using font manager
        return self.font_manager.render_text(
            frame, caption_text, (x_pos, y_pos),
            self.timestamp_color, thickness
        )

    def add_scrolling_text_to_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Add scrolling text to the frame if provided.

        Args:
            frame: The frame to add the scrolling text to
            frame_idx: Index of the current frame

        Returns:
            The frame with scrolling text added
        """
        if not self.scrolling_text or self.scrolling_speed is None:
            return frame

        # Calculate current position of the text
        self.scrolling_position = self.width - (frame_idx * self.scrolling_speed)

        # If the text has completely scrolled off the left side, don't render it
        if self.scrolling_position < -self.scrolling_text_width:
            return frame

        thickness = 2
        (_, text_height), _ = self.font_manager.get_text_size(self.scrolling_text, thickness)

        # Calculate y position based on whether timestamp is shown
        margin = 10
        if self.show_timestamp:
            # Position above timestamp with margin
            y_pos = self.height - text_height - margin * 3
        else:
            # Position at bottom of frame where timestamp would be
            y_pos = self.height - 20  # Same position as timestamp

        # Draw the scrolling text
        return self.font_manager.render_text(
            frame, self.scrolling_text, (int(self.scrolling_position), int(y_pos)),
            self.timestamp_color, thickness
        )

    def add_title_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """Add title text to the frame if provided.

        Args:
            frame: The frame to add the title to

        Returns:
            The frame with title text added
        """
        if not self.title_text:
            return frame

        thickness = 2

        # Get text size to calculate position
        (text_width, text_height), _ = self.font_manager.get_text_size(
            self.title_text, thickness
        )

        margin = 10
        x_pos = self._calculate_text_x_position(text_width, margin)

        # Y position (top of frame with margin)
        y_pos = margin + text_height

        # Draw the title using font manager
        return self.font_manager.render_text(
            frame, self.title_text, (x_pos, y_pos),
            self.timestamp_color, thickness
        )


class VideoGenerator:
    """Generates videos from GPX tracks with map visualization."""

    def __init__(self, output_path: str, fps: int = 30, resolution: Tuple[int, int] = (1280, 720),
                 zoom_level: int = 15, marker_color: Tuple[int, int, int] = (255, 0, 0),
                 marker_size: int = 10, text_config=None, captions_file: str = ""):
        """Initialize the video generator.

        Args:
            output_path: Path to the output video file
            fps: Frames per second
            resolution: Tuple of (width, height) in pixels
            zoom_level: Zoom level for the map (1-19)
            marker_color: RGB color tuple for the position marker
            marker_size: Size of the position marker in pixels
            text_config: Configuration for text overlays (includes font_file for custom TrueType font)
            captions_file: Path to a CSV file with captions
        """
        self.output_path = output_path
        self.fps = fps
        self.width, self.height = resolution
        self.zoom_level = zoom_level
        self.marker_color = marker_color
        self.marker_size = marker_size

        # Initialize map renderer
        self.map_renderer = MapRenderer()

        # Initialize text config with defaults if not provided
        default_text_config = TextConfig(
            timestamp_color=(255, 255, 255),
            font_scale=1.0,
            title_text="",
            text_align="left",
            font_file=None,
            show_timestamp=True,
            scrolling_text_file=None,
            scrolling_speed=None,
            timezone=None
        )

        # Initialize captioner
        self.captioner = VideoCaptioner(
            width=self.width,
            height=self.height,
            timestamp_color=default_text_config.timestamp_color if text_config is None else text_config.timestamp_color,
            font_scale=default_text_config.font_scale if text_config is None else text_config.font_scale,
            title_text=default_text_config.title_text if text_config is None else text_config.title_text,
            text_align=default_text_config.text_align if text_config is None else text_config.text_align,
            captions_file=captions_file,
            font_file=default_text_config.font_file if text_config is None else text_config.font_file,
            show_timestamp=default_text_config.show_timestamp if text_config is None else text_config.show_timestamp,
            scrolling_text_file=default_text_config.scrolling_text_file if text_config is None else text_config.scrolling_text_file,
            scrolling_speed=default_text_config.scrolling_speed if text_config is None else text_config.scrolling_speed,
            timezone=default_text_config.timezone if text_config is None else text_config.timezone
        )

        # Cache for interpolated positions
        self._position_cache = {}
        # Last found index for binary search optimization
        self._last_index = 0

    def _prepare_track_points(self, track_points: List[GPXTrackPoint]) -> List[GPXTrackPoint]:
        """Prepare track points for interpolation by filtering and sorting.

        Args:
            track_points: List of GPXTrackPoint objects

        Returns:
            Filtered and sorted list of track points with time data

        Raises:
            ValueError: If track points don't have time data
        """
        # Filter points with time data
        points_with_time = [p for p in track_points if p.time is not None]
        if not points_with_time:
            raise ValueError("Track points don't have time data")

        # Sort points by time
        points_with_time.sort(key=lambda p: p.time)

        return points_with_time

    def _find_interpolation_points(self, points_with_time: List[GPXTrackPoint], 
                                timestamp: datetime) -> Tuple[int, GPXTrackPoint]:
        """Find the index and point to use for interpolation using binary search.

        Args:
            points_with_time: List of GPXTrackPoint objects with time data
            timestamp: Timestamp to find points for

        Returns:
            Tuple of (index, point) where index is the lower bound for interpolation
        """
        # Start from last found index as an optimization for sequential access
        left = self._last_index
        right = len(points_with_time) - 1

        # If timestamp is before the last found point, search from beginning
        if timestamp < points_with_time[left].time:
            left = 0

        # Binary search
        while left < right:
            mid = (left + right) // 2
            if points_with_time[mid].time < timestamp:
                left = mid + 1
            else:
                right = mid

        # Adjust index if needed
        if left > 0 and points_with_time[left].time > timestamp:
            left -= 1

        # Save index for next search
        self._last_index = left

        return left, points_with_time[left]

    def _interpolate_position(self, points_with_time: List[GPXTrackPoint],
                              timestamp: datetime) -> Tuple[float, float]:
        """Interpolate position at a given timestamp between track points.

        Args:
            points_with_time: List of GPXTrackPoint objects with time data, already filtered and sorted
            timestamp: Timestamp to interpolate position for

        Returns:
            Tuple of (latitude, longitude)

        Raises:
            ValueError: If timestamp is out of range
        """
        # Check if position is already in cache
        cache_key = timestamp.isoformat()
        if cache_key in self._position_cache:
            return self._position_cache[cache_key]

        # Check if timestamp is within range
        if timestamp < points_with_time[0].time or timestamp > points_with_time[-1].time:
            raise ValueError(f"Timestamp {timestamp} is outside the track time range")

        # Find the point to interpolate from
        left, p1 = self._find_interpolation_points(points_with_time, timestamp)

        # Initialize result with the first point's coordinates
        result = (p1.latitude, p1.longitude)

        # Check if we need to interpolate
        needs_interpolation = (
            left < len(points_with_time) - 1 and  # Not the last point
            p1.time != timestamp                   # Not an exact match
        )

        if needs_interpolation:
            p2 = points_with_time[left + 1]
            total_seconds = (p2.time - p1.time).total_seconds()

            # Only interpolate if the points have different timestamps
            if total_seconds > 0:
                elapsed_seconds = (timestamp - p1.time).total_seconds()
                factor = elapsed_seconds / total_seconds

                # Interpolate latitude and longitude
                lat = p1.latitude + factor * (p2.latitude - p1.latitude)
                lon = p1.longitude + factor * (p2.longitude - p1.longitude)
                result = (lat, lon)

        # Cache and return result
        self._position_cache[cache_key] = result
        return result

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

        # Add timestamp, title, caption, and scrolling text using the captioner
        frame = self.captioner.add_timestamp_to_frame(frame, frame_timestamp)
        frame = self.captioner.add_title_to_frame(frame)
        frame = self.captioner.add_caption_to_frame(frame, frame_seconds)
        frame = self.captioner.add_scrolling_text_to_frame(frame, frame_idx)

        return frame


    def _generate_frame_data(self, frame_info):
        """Generate frame data for a single frame.

        Args:
            frame_info: Tuple of (frame_idx, frame_timestamp, frame_seconds, points_with_time)

        Returns:
            Tuple of (frame_idx, frame) where frame is a numpy array in BGR format
        """
        frame_idx, frame_timestamp, frame_seconds, points_with_time = frame_info
        frame = self._generate_frame(frame_idx, frame_timestamp, frame_seconds, points_with_time)
        return frame_idx, frame

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
        # Set the video start time and duration in the captioner
        self.captioner.set_video_start_time(start_time)
        self.captioner.set_video_duration(duration_seconds, self.fps)

        # Calculate total number of frames
        total_frames = duration_seconds * self.fps

        # Prepare frame information for all frames
        frame_infos = []
        for frame_idx in range(total_frames):
            # Calculate seconds elapsed since the start of the video
            frame_seconds = frame_idx / self.fps
            # Calculate timestamp for this frame based on track progress
            progress = frame_idx / total_frames
            track_seconds = progress * total_track_seconds
            frame_timestamp = start_time + timedelta(seconds=track_seconds)

            frame_infos.append((frame_idx, frame_timestamp, frame_seconds, points_with_time))

        # Generate frames in parallel
        frames_dict = {}  # Dictionary to store frames by index

        # Determine the number of workers based on CPU cores (use at most 4 workers)
        max_workers = min(4, os.cpu_count() or 4)

        # Use a smaller batch size for better progress reporting
        batch_size = self.fps * 2  # Process 2 seconds of video at a time

        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            batch_infos = frame_infos[batch_start:batch_end]

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Generate frames in parallel
                results = list(executor.map(self._generate_frame_data, batch_infos))

                # Store frames in dictionary
                for frame_idx, frame in results:
                    frames_dict[frame_idx] = frame

            # Write frames to video in correct order
            for frame_idx in range(batch_start, batch_end):
                video_writer.write(frames_dict[frame_idx])

            # Log progress
            logger.info(f"Generated {batch_end}/{total_frames} frames ({batch_end / total_frames:.1%})")

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
        # Prepare track points (filter and sort)
        points_with_time = self._prepare_track_points(track_points)

        # Reset position cache and last index for a new video
        self._position_cache = {}
        self._last_index = 0

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
        # Use MPEG4 codec with mp4v FOURCC for MP4 format
        # Define VideoWriter_fourcc directly to avoid undefined reference
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
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
