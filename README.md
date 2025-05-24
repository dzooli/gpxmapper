# GPX Mapper

[![Build and Release](https://github.com/dzooli/gpxmapper/actions/workflows/release.yml/badge.svg)](https://github.com/dzooli/gpxmapper/actions/workflows/release.yml) [![Quality gate](https://sonarcloud.io/api/project_badges/quality_gate?project=dzooli_gpxmapper)](https://sonarcloud.io/summary/new_code?id=dzooli_gpxmapper)

A command-line tool that generates videos from GPX tracks, showing the route on a map with a marker indicating the current position.

## Features

- Parse GPX files and extract track data with timestamps
- Fetch map tiles from OpenStreetMap
- Generate videos with customizable duration, resolution, and FPS
- Show position marker on the map with customizable color and size
- Display timestamp overlay with customizable color
- Match GPX timeline to video duration
- Add custom title text to videos
- Include timed captions from CSV files
- Add scrolling text from a text file with customizable speed
- Customize text alignment and font scale
- Customize the font of text overlays (TTF only)
- Cache map tiles for faster rendering
- Clear cache to free up disk space
- Performance optimizations:
  - Parallel frame generation using multiple threads
  - Efficient position interpolation with binary search
  - Caching of interpolated positions
  - Batch processing of frames for better memory management

## Installation

### Option 1: Windows Standalone Executable

For Windows users who don't want to install Python or any dependencies:

1. Download the latest `gpxmapper.exe` from the [releases page](https://github.com/dzooli/gpxmapper/releases)
2. Place the executable in a directory of your choice
3. Run the executable from the command line:

```cmd
gpxmapper.exe generate path\to\your\file.gpx
```

### Option 2: Install from source

#### Requirements

- Python 3.12 or higher
- uv (for package management)

```bash
# Clone the repository
git clone https://github.com/zoltan-dzooli-fabian/gpxmapper.git
cd gpxmapper

# Install with uv
uv pip install -e .
```

### Building the Windows Executable

If you want to build the executable yourself, there are several methods available:

#### Option 1: Using the build script (Recommended)

This is the simplest method that handles all dependencies automatically:

1. Clone the repository and navigate to the project directory:
   ```cmd
   git clone https://github.com/zoltan-dzooli-fabian/gpxmapper.git
   cd gpxmapper
   ```

2. Make sure you have either `pip` or `uv` installed for package management:
   ```cmd
   # Check if pip is installed
   python -m pip --version

   # OR check if uv is installed
   uv --version

   # Install uv if needed
   python -m pip install uv
   ```

3. Run the build script:
   ```cmd
   python build_exe.py
   ```

4. The script will automatically:
   - Install PyInstaller if not already installed
   - Read configuration from pyproject.toml
   - Create a spec file
   - Build the executable
   - Clean up temporary files

5. The executable will be created in the `dist` directory as `gpxmapper.exe`

6. You can test the executable by running:
   ```cmd
   .\dist\gpxmapper.exe --help
   ```

## Usage

### Generate a video from a GPX file

For Python installation:
```bash
gpxmapper generate path/to/your/file.gpx
```

For Windows executable:
```cmd
gpxmapper.exe generate path\to\your\file.gpx
```

This will create a video file with the same name as the GPX file but with a `.mp4` extension.

### Customize the video

For Python installation:
```bash
gpxmapper generate path/to/your/file.gpx --output output.mp4 --duration 120 --fps 30 --width 1920 --height 1080 --zoom 14
```

For Windows executable:
```cmd
gpxmapper.exe generate path\to\your\file.gpx --output output.mp4 --duration 120 --fps 30 --width 1920 --height 1080 --zoom 14
```

### Get information about a GPX file

For Python installation:
```bash
gpxmapper info path/to/your/file.gpx
```

For Windows executable:
```cmd
gpxmapper.exe info path\to\your\file.gpx
```

## Command-line options

### `generate` command

- `gpx_file`: Path to the input GPX file (required)
- `--output`, `-o`: Path to the output video file (default: input filename with .mp4 extension)
- `--duration`, `-d`: Duration of the output video in seconds (default: 60)
- `--fps`, `-f`: Frames per second for the output video (default: 30)
- `--width`, `-w`: Width of the output video in pixels (default: 320)
- `--height`, `-h`: Height of the output video in pixels (default: 320)
- `--zoom`, `-z`: Zoom level for the map (1-19, higher is more detailed) (default: 15)
- `--marker-size`, `-m`: Size of the position marker in pixels (default: 10)
- `--marker-color`, `-c`: Color of the position marker as R,G,B (default: 255,0,0)
- `--font-scale`, `-fs`: Font scale for all text (timestamp, title, captions) (default: 0.7)
- `--title`: Optional text to display as a title on the video
- `--text-align`, `-ta`: Alignment of all text (title, captions) (left, center, right) (default: left)
- `--captions`: Path to a CSV file containing captions with timestamps in HH:MM:SS format (relative to the start of the video)
- `--font`, `-ff`: Path to a TrueType font file (.ttf) for text rendering
- `--no-timestamp`: Disable timestamp visualization in the video
- `--scrolling-text`, `-st`: Path to a text file containing content to be scrolled on the video
- `--scrolling-speed`, `-ss`: Speed at which the text scrolls across the video (pixels per frame). If not specified, speed will be calculated based on video duration.
- `--timezone`, `-tz`: Timezone to convert timestamps to. Must be a full timezone name (e.g., 'Europe/Budapest', 'US/Pacific'). If not specified, timestamps are not converted.

Note: The timestamp color is fixed to black (0,0,0) in the command-line interface but can be customized when using the library programmatically.

### `info` command

- `gpx_file`: Path to the GPX file (required)

### `clear-cache` command

Clears the map tiles cache directory to free up disk space. The cache directory is automatically determined based on the operating system.

## Programmatic Usage

GPXMapper can also be used programmatically in your Python code. Here's how to use the library directly:

### Basic Video Generation

```python
from gpxmapper.gpx_parser import GPXParser
from gpxmapper.video_generator import VideoGenerator
from gpxmapper.cli import TextConfig

# Parse GPX file
gpx_path = "my_bike_ride.gpx"
parser = GPXParser(gpx_path)
track_points = parser.parse()

# Create a basic text configuration
text_config = TextConfig(
    font_scale=0.7,
    timestamp_color=(0, 0, 0)  # Black color for timestamp
)

# Generate video
output_path = "output.mp4"
video_generator = VideoGenerator(
    output_path=output_path,
    fps=30,
    resolution=(1280, 720),
    zoom_level=15,
    marker_color=(255, 0, 0),  # Red marker
    marker_size=10,
    text_config=text_config
)

# Generate a 60-second video
output_path = video_generator.generate_video(track_points, 60)
print(f"Video generated successfully: {output_path}")
```

### Advanced Video Generation

```python
from gpxmapper.gpx_parser import GPXParser
from gpxmapper.video_generator import VideoGenerator
from gpxmapper.cli import TextConfig

# Parse GPX file
gpx_path = "my_hike.gpx"
parser = GPXParser(gpx_path)
track_points = parser.parse()

# Create an advanced text configuration with title and centered alignment
text_config = TextConfig(
    font_scale=1.0,
    title_text="My Hiking Adventure",
    text_align="center",
    timestamp_color=(255, 255, 255),  # White color for timestamp
    font_file="path/to/custom_font.ttf",  # Custom TrueType font
    show_timestamp=True,  # Set to False to disable timestamp display
    scrolling_text_file="path/to/scrolling.txt",  # Text file with scrolling content
    scrolling_speed=2.5,  # Speed in pixels per frame (optional)
    timezone="Europe/London"  # Convert timestamps to London time (optional)
)

# Generate video
output_path = "advanced_output.mp4"
video_generator = VideoGenerator(
    output_path=output_path,
    fps=30,
    resolution=(1920, 1080),  # Full HD resolution
    zoom_level=14,  # Slightly zoomed out
    marker_color=(0, 0, 255),  # Blue marker
    marker_size=15,  # Larger marker
    text_config=text_config,
    captions_file="captions.csv"  # Optional captions file
)

# Generate a 120-second video
output_path = video_generator.generate_video(track_points, 120)
print(f"Video generated successfully: {output_path}")
```

## Command-line Examples

### Basic usage

For Python installation:
```bash
gpxmapper generate my_bike_ride.gpx
```

For Windows executable:
```cmd
gpxmapper.exe generate my_bike_ride.gpx
```
Note: Use backslashes for paths on Windows (e.g., `C:\path\to\file.gpx`)

### Create a 2-minute video with higher resolution

For Python installation:
```bash
gpxmapper generate my_hike.gpx --duration 120 --width 1920 --height 1080
```

For Windows executable:
```cmd
gpxmapper.exe generate my_hike.gpx --duration 120 --width 1920 --height 1080
```

### Use a different marker color and size

For Python installation:
```bash
gpxmapper generate my_run.gpx --marker-color 0,0,255 --marker-size 15
```

For Windows executable:
```cmd
gpxmapper.exe generate my_run.gpx --marker-color 0,0,255 --marker-size 15
```

### Customize the text appearance

For Python installation:
```bash
gpxmapper generate my_run.gpx --font-scale 1.0 --text-align center
```

For Windows executable:
```cmd
gpxmapper.exe generate my_run.gpx --font-scale 1.0 --text-align center
```

### Disable timestamp display

For Python installation:
```bash
gpxmapper generate my_run.gpx --no-timestamp
```

For Windows executable:
```cmd
gpxmapper.exe generate my_run.gpx --no-timestamp
```

### Add a title to the video

For Python installation:
```bash
gpxmapper generate my_run.gpx --title "My Morning Run" --text-align center
```

For Windows executable:
```cmd
gpxmapper.exe generate my_run.gpx --title "My Morning Run" --text-align center
```

### Use a custom font for text rendering

For Python installation:
```bash
gpxmapper generate my_run.gpx --font path/to/custom_font.ttf
```

For Windows executable:
```cmd
gpxmapper.exe generate my_run.gpx --font path\to\custom_font.ttf
```

### Add captions to the video

First, create a CSV file with your captions (e.g., `captions.csv`):
```
TIME,CAPTION
00:00:01,Starting the journey
00:00:30,Halfway point
00:01:00,Finishing up
```

Then, use the `--captions` option:

For Python installation:
```bash
gpxmapper generate my_run.gpx --captions captions.csv
```

For Windows executable:
```cmd
gpxmapper.exe generate my_run.gpx --captions captions.csv
```

### Add scrolling text to the video

First, create a text file with your scrolling content (e.g., `scrolling.txt`):
```
This is a scrolling text that will appear at the bottom of the video.
```

Then, use the `--scrolling-text` option:

For Python installation:
```bash
gpxmapper generate my_run.gpx --scrolling-text scrolling.txt
```

For Windows executable:
```cmd
gpxmapper.exe generate my_run.gpx --scrolling-text scrolling.txt
```

You can also specify the scrolling speed:

For Python installation:
```bash
gpxmapper generate my_run.gpx --scrolling-text scrolling.txt --scrolling-speed 2.5
```

For Windows executable:
```cmd
gpxmapper.exe generate my_run.gpx --scrolling-text scrolling.txt --scrolling-speed 2.5
```

### Convert timestamps to a specific timezone

For Python installation:
```bash
gpxmapper generate my_run.gpx --timezone Europe/London
```

For Windows executable:
```cmd
gpxmapper.exe generate my_run.gpx --timezone Europe/London
```


### Clear the map tiles cache

To free up disk space by removing cached map tiles:

For Python installation:
```bash
gpxmapper clear-cache
```

For Windows executable:
```cmd
gpxmapper.exe clear-cache
```

## License

MIT
