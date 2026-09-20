"""TUI (Terminal User Interface) command powered by Trogon and Textual."""

from __future__ import annotations

from trogon.typer import init_tui

from . import app

# Attach trogon interactive TUI to the Typer app as `gpxmapper tui`
init_tui(app, name="gpxmapper")
