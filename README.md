# GPX Mapper

A command-line tool that generates videos from GPX tracks, showing the route on a map with a marker indicating the current position.

## Features

- Parse GPX files and extract track data with timestamps
- Fetch map tiles from OpenStreetMap
- Generate videos with customizable duration, resolution, and FPS
- Show position marker on the map
- Display timestamp overlay
- Match GPX timeline to video duration
- Add custom title text to videos
- Include timed captions from CSV files
- Customize text alignment and font scale
- Cache map tiles for faster rendering
- Clear cache to free up disk space

## Installation

### Option 1: Windows Standalone Executable

For Windows users who don't want to install Python or any dependencies:

1. Download the latest `gpxmapper.exe` from the [releases page](https://github.com/yourusername/gpxmapper/releases)
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
git clone https://github.com/yourusername/gpxmapper.git
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
   git clone https://github.com/yourusername/gpxmapper.git
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

### `info` command

- `gpx_file`: Path to the GPX file (required)

### `clear_cache` command

Clears the map tiles cache directory to free up disk space. The cache directory is automatically determined based on the operating system.

## Examples

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

### Add a title to the video

For Python installation:
```bash
gpxmapper generate my_run.gpx --title "My Morning Run" --text-align center
```

For Windows executable:
```cmd
gpxmapper.exe generate my_run.gpx --title "My Morning Run" --text-align center
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

### Clear the map tiles cache

To free up disk space by removing cached map tiles:

For Python installation:
```bash
gpxmapper clear_cache
```

For Windows executable:
```cmd
gpxmapper.exe clear_cache
```

## License

MIT
