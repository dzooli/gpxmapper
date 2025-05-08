#!/usr/bin/env python
"""Entry point script for the gpxmapper executable."""

import sys
from gpxmapper.cli import app

if __name__ == "__main__":
    sys.exit(app())