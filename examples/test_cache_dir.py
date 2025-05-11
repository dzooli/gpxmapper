"""Test script to verify the default cache directory functionality.

This script initializes a MapRenderer without specifying a cache directory
and prints the default directory that was automatically selected based on
the operating system.
"""

import os
import sys
import platform

# Import from the installed package
from gpxmapper.map_renderer import MapRenderer

def main():
    # Initialize MapRenderer without specifying a cache directory
    renderer = MapRenderer()

    # Print information about the environment and the selected cache directory
    print(f"Operating System: {platform.system()}")
    print(f"Default cache directory: {renderer.cache_dir}")

    # Verify the directory exists
    if os.path.exists(renderer.cache_dir):
        print("Cache directory exists: Yes")
    else:
        print("Cache directory exists: No (This is unexpected, check permissions)")

if __name__ == "__main__":
    main()
