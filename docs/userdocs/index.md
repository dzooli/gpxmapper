# GPXMapper

GPXMapper creates short map videos from GPX tracks.

Usable for demonstrating the track you have taken or for sharing with others.

## Features

- Creates video map visualizations from GPX track data
- Supports reverse geocoding to add location names to tracks
- Generates videos with customizable styles and overlays

## Nominatim and reverse geolocation

Planned CLI behavior (see the implementation plan under `docs/superpowers/plans/2026-06-07-nominatim-geolocate-cli.md`):

- **`NOMINATIM_SERVER`** — Base URL for Nominatim. Default when unset: **`http://localhost:8080`**. Public option: **`https://nominatim.openstreetmap.org`** (respect [OSM usage policy](https://operations.osmfoundation.org/policies/nominatim/)); local instances are preferred for performance.
- **`gpxmapper generate … --geolocate`** — Before heavy work, the tool checks **`/status`** up to **three times**. If the server is still unreachable, it shows a **clear error** and asks to **continue without reverse geolocation** or **abort** (default: abort). Non-interactive runs (no TTY) **abort** without prompting.
- **Reverse-geocode cache** — Successful lookups are written to a small **SQLite** file next to the map tile cache directory (not inside it), keyed by Nominatim base URL and rounded coordinates. Re-running `generate --geolocate` reuses cached labels and skips HTTP for those cells. **`gpxmapper clear-cache`** only removes map tiles, not this database.

## Usage

- See the CLI help: `gpxmapper --help`.
- For generated API reference, use the [Reference](reference/gpxmapper/cli) section.

## Contributing

Contributions are welcome! Please see the [Contributing Guide](contributing.md) for details on how to contribute to the
project.

## License

MIT License

Copyright (c) 2025 Zoltan Fabian

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


