"""Example script demonstrating how to use GPXMapper programmatically."""

import os
import logging
from pathlib import Path

from gpxmapper.gpx_parser import GPXParser
from gpxmapper.video_generator import VideoGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def generate_video_from_gpx(gpx_file_path, output_file_path=None, duration_seconds=60):
    """Generate a video from a GPX file.
    
    Args:
        gpx_file_path: Path to the GPX file
        output_file_path: Path to the output video file (default: GPX filename with .mp4 extension)
        duration_seconds: Duration of the output video in seconds
        
    Returns:
        Path to the generated video file
    """
    # Convert to Path objects
    gpx_path = Path(gpx_file_path)
    
    # Set default output path if not provided
    if output_file_path is None:
        output_file_path = gpx_path.with_suffix(".mp4")
    else:
        output_file_path = Path(output_file_path)
    
    # Create output directory if it doesn't exist
    output_dir = output_file_path.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # Parse GPX file
    print(f"Parsing GPX file: {gpx_path}")
    parser = GPXParser(str(gpx_path))
    track_points = parser.parse()
    
    if not track_points:
        raise ValueError("No track points found in the GPX file")
    
    # Check if track points have time data
    points_with_time = [p for p in track_points if p.time is not None]
    if not points_with_time:
        raise ValueError("GPX file doesn't contain time data, which is required for video generation")
    
    # Get time bounds
    start_time, end_time = parser.get_time_bounds()
    print(f"Track time range: {start_time} to {end_time}")
    
    # Generate video
    print(f"Generating video: {output_file_path}")
    video_generator = VideoGenerator(
        output_path=str(output_file_path),
        fps=30,
        resolution=(1280, 720),
        zoom_level=15,
        marker_color=(255, 0, 0),
        marker_size=10
    )
    
    output_path = video_generator.generate_video(track_points, duration_seconds)
    
    print(f"Video generated successfully: {output_path}")
    return output_path

if __name__ == "__main__":
    # Example usage
    # Replace with the path to your GPX file
    gpx_file = "path/to/your/gpx_file.gpx"
    
    # Uncomment the following line to run the example
    # generate_video_from_gpx(gpx_file, duration_seconds=120)
    
    print("This is an example script. Please modify the gpx_file variable with your own GPX file path.")
    print("Then uncomment the generate_video_from_gpx() call to run the example.")