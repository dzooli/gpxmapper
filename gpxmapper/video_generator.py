"""Video generation module for creating videos from GPX tracks."""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import tempfile

from .gpx_parser import GPXTrackPoint
from .map_renderer import MapRenderer

logger = logging.getLogger(__name__)

class VideoGenerator:
    """Generates videos from GPX tracks with map visualization."""
    
    def __init__(self, output_path: str, fps: int = 30, resolution: Tuple[int, int] = (1280, 720),
                 zoom_level: int = 15, marker_color: Tuple[int, int, int] = (255, 0, 0),
                 marker_size: int = 10):
        """Initialize the video generator.
        
        Args:
            output_path: Path where the output video will be saved
            fps: Frames per second for the output video
            resolution: Resolution of the output video as (width, height)
            zoom_level: Zoom level for the map tiles
            marker_color: RGB color tuple for the position marker
            marker_size: Size of the position marker in pixels
        """
        self.output_path = output_path
        self.fps = fps
        self.width, self.height = resolution
        self.zoom_level = zoom_level
        self.marker_color = marker_color
        self.marker_size = marker_size
        self.map_renderer = MapRenderer(cache_dir=os.path.join(tempfile.gettempdir(), "gpxmapper_tiles"))
    
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
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        video_writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )
        
        if not video_writer.isOpened():
            raise ValueError(f"Failed to open video writer for {self.output_path}")
        
        try:
            # Calculate total number of frames
            total_frames = duration_seconds * self.fps
            
            for frame_idx in range(total_frames):
                # Calculate timestamp for this frame
                progress = frame_idx / total_frames
                frame_seconds = progress * total_track_seconds
                frame_timestamp = start_time + timedelta(seconds=frame_seconds)
                
                # Interpolate position
                lat, lon = self._interpolate_position(points_with_time, frame_timestamp)
                
                # Render map for this position
                map_image = self.map_renderer.render_map_for_point(
                    lat, lon, self.zoom_level, self.marker_color, self.marker_size
                )
                
                if map_image is None:
                    logger.warning(f"Failed to render map for frame {frame_idx}, using blank frame")
                    # Create a blank frame
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                else:
                    # Resize map image to video resolution
                    map_array = np.array(map_image)
                    frame = cv2.resize(map_array, (self.width, self.height))
                    
                    # Convert from RGB to BGR (OpenCV uses BGR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Add timestamp text
                timestamp_str = frame_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    frame, timestamp_str, (10, self.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                
                # Write frame to video
                video_writer.write(frame)
                
                # Log progress
                if frame_idx % (self.fps * 5) == 0:  # Log every 5 seconds of video
                    logger.info(f"Generated {frame_idx}/{total_frames} frames ({frame_idx/total_frames:.1%})")
            
            logger.info(f"Video generation complete: {self.output_path}")
            return self.output_path
            
        finally:
            # Release video writer
            video_writer.release()