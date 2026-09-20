"""Example script demonstrating how to use GPXMapper programmatically.

This script shows how to use the GPXMapper module to inspect tracks and
generate videos from GPX files using both the high-level API and custom configurations.
"""

import logging
import sys
from pathlib import Path

# Import from the installed package
import gpxmapper
from gpxmapper.models import MapConfig, TextConfig, VideoConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def inspect_track_info(gpx_file_path):
    """Inspect GPX track information using the high-level API."""
    print(f"\n=== Inspecting GPX Track: {gpx_file_path} ===")
    info = gpxmapper.get_gpx_info(gpx_file_path)
    print(f"Point count: {info.point_count}")
    print(f"Start time:  {info.start_time}")
    print(f"End time:    {info.end_time}")
    print(f"Duration:    {info.duration}")
    print(f"Bounds:      {info.coordinate_bounds}")
    return info


def generate_basic_video(gpx_file_path, output_file_path=None, duration_seconds=60):
    """Generate a basic video from a GPX file using the high-level API.

    Args:
        gpx_file_path: Path to the GPX file
        output_file_path: Path to the output video file (default: GPX filename with .mp4 extension)
        duration_seconds: Duration of the output video in seconds

    Returns:
        Path to the generated video file
    """
    print(f"\nGenerating basic video for {gpx_file_path}...")
    output_path = gpxmapper.generate_video(
        gpx_file=gpx_file_path,
        output_file=output_file_path,
        video_config=VideoConfig(fps=30, width=1280, height=720, duration=duration_seconds),
        map_config=MapConfig(zoom=15, marker_size=10, marker_color=(255, 0, 0)),
        text_config=TextConfig(font_scale=0.7, timestamp_color=(0, 0, 0)),
    )
    print(f"Video generated successfully: {output_path}")
    return output_path


def generate_advanced_video(
    gpx_file_path, output_file_path=None, duration_seconds=60, title_text="My GPX Track", captions_filename=None
):
    """Generate a video from a GPX file with advanced features like title and captions.

    Args:
        gpx_file_path: Path to the GPX file
        output_file_path: Path to the output video file
        duration_seconds: Duration of the output video in seconds
        title_text: Text to display as a title on the video
        captions_filename: Path to a CSV file containing captions with timestamps

    Returns:
        Path to the generated video file
    """
    gpx_path = Path(gpx_file_path)
    if output_file_path is None:
        output_file_path = gpx_path.with_stem(f"{gpx_path.stem}_advanced").with_suffix(".mp4")

    print(f"\nGenerating advanced video for {gpx_path}...")
    output_path = gpxmapper.generate_video(
        gpx_file=gpx_path,
        output_file=output_file_path,
        video_config=VideoConfig(fps=30, width=1920, height=1080, duration=duration_seconds),
        map_config=MapConfig(zoom=14, marker_size=15, marker_color=(0, 0, 255)),
        text_config=gpxmapper.create_text_config(
            font_scale=1.0,
            title_text=title_text,
            text_align="center",
            timestamp_color=(255, 255, 255),
        ),
        captions=captions_filename,
    )
    print(f"Video generated successfully: {output_path}")
    return output_path


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    gpx_file = repo_root / "K1-fel.gpx"
    captions_file = repo_root / "captions-example.csv"

    if not gpx_file.exists():
        print(f"Sample GPX file not found: {gpx_file}")
        print("Please modify the gpx_file variable with your own GPX file path.")
        sys.exit(1)

    # Example 1: Inspect track info
    inspect_track_info(gpx_file)

    # Example 2: Generate basic video
    basic_output = generate_basic_video(gpx_file_path=gpx_file, duration_seconds=30)

    # Example 3: Generate advanced video with title and captions
    captions = str(captions_file) if captions_file.exists() else None
    advanced_output = generate_advanced_video(
        gpx_file_path=gpx_file,
        duration_seconds=30,
        title_text="My GPX Adventure",
        captions_filename=captions,
    )

    print("\n=== Summary ===")
    print(f"Basic video: {basic_output}")
    print(f"Advanced video: {advanced_output}")
