"""Example script demonstrating how to use GPXMapper programmatically.

This script shows how to use the GPXMapper module to generate videos from GPX tracks.
It includes examples of basic usage as well as advanced features like adding titles and captions.
"""

import os
import logging
from pathlib import Path
import sys

# Import from the installed package
from gpxmapper.gpx_parser import GPXParser
from gpxmapper.video_generator import VideoGenerator
from gpxmapper.cli import TextConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def generate_basic_video(gpx_file_path, output_file_path=None, duration_seconds=60):
    """Generate a basic video from a GPX file with default settings.

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

    # Create a basic text configuration
    text_config = TextConfig(
        font_scale=0.7,
        timestamp_color=(0, 0, 0)  # Black color for timestamp
    )

    # Generate video
    print(f"Generating video: {output_file_path}")
    video_generator = VideoGenerator(
        output_path=str(output_file_path),
        fps=30,
        resolution=(1280, 720),
        zoom_level=15,
        marker_color=(255, 0, 0),  # Red marker
        marker_size=10,
        text_config=text_config
    )

    output_path = video_generator.generate_video(track_points, duration_seconds)

    print(f"Video generated successfully: {output_path}")
    return output_path

def generate_advanced_video(gpx_file_path, output_file_path=None, duration_seconds=60, 
                           title_text="My GPX Track", captions_file=None):
    """Generate a video from a GPX file with advanced features like title and captions.

    Args:
        gpx_file_path: Path to the GPX file
        output_file_path: Path to the output video file (default: GPX filename with _advanced.mp4 extension)
        duration_seconds: Duration of the output video in seconds
        title_text: Text to display as a title on the video
        captions_file: Path to a CSV file containing captions with timestamps

    Returns:
        Path to the generated video file
    """
    # Convert to Path objects
    gpx_path = Path(gpx_file_path)

    # Set default output path if not provided
    if output_file_path is None:
        output_file_path = gpx_path.with_stem(f"{gpx_path.stem}_advanced").with_suffix(".mp4")
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

    # Create an advanced text configuration with title and centered alignment
    text_config = TextConfig(
        font_scale=1.0,
        title_text=title_text,
        text_align="center",
        timestamp_color=(255, 255, 255)  # White color for timestamp
    )

    # Generate video
    print(f"Generating video: {output_file_path}")
    video_generator = VideoGenerator(
        output_path=str(output_file_path),
        fps=30,
        resolution=(1920, 1080),  # Full HD resolution
        zoom_level=14,  # Slightly zoomed out
        marker_color=(0, 0, 255),  # Blue marker
        marker_size=15,  # Larger marker
        text_config=text_config,
        captions_file=captions_file
    )

    output_path = video_generator.generate_video(track_points, duration_seconds)

    print(f"Video generated successfully: {output_path}")
    return output_path

if __name__ == "__main__":
    # Example usage with a sample GPX file included in the repository
    # This example will work without modification if run from the repository root

    # Find the repository root directory
    repo_root = Path(__file__).parent.parent

    # Path to the sample GPX file
    gpx_file = repo_root / "K1-fel.gpx"

    # Path to the sample captions file
    captions_file = repo_root / "captions-example.csv"

    if not gpx_file.exists():
        print(f"Sample GPX file not found: {gpx_file}")
        print("Please modify the gpx_file variable with your own GPX file path.")
        sys.exit(1)

    # Example 1: Generate a basic video
    print("\n=== Example 1: Basic Video ===")
    basic_output = generate_basic_video(
        gpx_file_path=gpx_file,
        duration_seconds=30  # Short duration for quick example
    )

    # Example 2: Generate an advanced video with title and captions
    print("\n=== Example 2: Advanced Video with Title and Captions ===")
    if captions_file.exists():
        advanced_output = generate_advanced_video(
            gpx_file_path=gpx_file,
            duration_seconds=30,  # Short duration for quick example
            title_text="My GPX Adventure",
            captions_file=str(captions_file)
        )
    else:
        print(f"Sample captions file not found: {captions_file}")
        print("Generating advanced video without captions...")
        advanced_output = generate_advanced_video(
            gpx_file_path=gpx_file,
            duration_seconds=30,  # Short duration for quick example
            title_text="My GPX Adventure"
        )

    print("\n=== Summary ===")
    print(f"Basic video: {basic_output}")
    print(f"Advanced video: {advanced_output}")
    print("\nVideos generated successfully! Check the output files to see the results.")
