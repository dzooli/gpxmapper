# GPXMapper

> **Windows release ZIP:** unzip to get `gpxmapper.exe`, **`start_server.bat`**, **`start_server.sh`**, **`verify_local_hungary.sh`**, **`README.md`**, **`doc/USER_GUIDE.md`**, and **`doc/THIRD_PARTY_NOTICES.md`** (installation layout). **From a Git clone**, all of those helpers (except the copied `USER_GUIDE`) live under **`nominatim/`**; the bundle user guide source is **`install/doc/USER_GUIDE.md`**; third-party notices source is **`install/doc/THIRD_PARTY_NOTICES.md`**.

GPXMapper turns GPX tracks into short **map videos**: your route on OpenStreetMap tiles with a moving position marker and optional text overlays. Handy for demos, social posts, or showing someone exactly where you went.

## Features

### Video and map

- **Output control** — Length (`--duration`), frame rate (`--fps`), resolution (`--width` / `--height`), and map **zoom** so you can match social formats or archive quality.
- **Position marker** — Size (`--marker-size`, `-m`) and **RGB color** (`--marker-color`, `-c`) in `R,G,B` form (e.g. red `255,0,0`).
- **OSM tiles** — Fetches map imagery as needed and **caches** tiles locally for faster re-renders; use `gpxmapper clear-cache` when you want to reclaim disk space (tiles only; geolocation cache is separate — see below).

### Text and overlays

- **One text color for all overlays** — **`--text-color` / `-tc`** uses the same `R,G,B` rules as the marker and applies to the **timestamp**, **title**, **CSV captions**, **scrolling text**, and **geolocation** labels (default black `0,0,0`). Use white or another color when the map background would hide dark text.
- **Typography** — Global **font scale** (`--font-scale`), optional **TTF** (`--font`), alignment (`--text-align`), and optional **title** (`--title`).
- **Timestamp** — Shown by default; **`--no-timestamp`** hides it. **`--timezone`** converts displayed times when you need a local wall-clock view.
- **Captions** — **`--captions`** loads a CSV with `HH:MM:SS` times relative to video start.
- **Scrolling text** — **`--scrolling-text`** and **`--scrolling-speed`** (mutually exclusive with **`--geolocate`** — pick one overlay style).

### CLI beyond `generate`

- **`gpxmapper info`** — Summarizes a GPX file (points, time range, bounds) without rendering video.
- **`gpxmapper check-nominatim`** — Runs the same **`GET …/status`** probe as **`generate --geolocate`** (handy to verify `NOMINATIM_SERVER` before a long render).
- **Logging** — Global **`--log-level`** (or **`GPXMAPPER_LOG_LEVEL`**) for quieter default runs vs. **DEBUG** for tile and HTTP detail.

### Library use

- You can drive the same pipeline from Python (see the project **README** and the [Reference](reference/gpxmapper/cli) for types and entry points).

## Nominatim and reverse geolocation

When you pass **`--geolocate`** to **`gpxmapper generate`**, GPXMapper calls a **Nominatim-compatible** server for reverse geocoding and draws short location labels on the video (instead of scrolling text). Behavior in code today:

- **Local Nominatim (Docker)** — **Repository clone:** use **`nominatim/start_server.sh`** on Linux, macOS, or WSL (Docker, checks, default **Hungary** `PBF_URL`), or **`nominatim/start_server.bat`** on Windows with Docker Desktop. **`nominatim/verify_local_hungary.sh`** checks **`/status`** and sample reverse results (optional **`NOMINATIM_URL`**). First startup **imports** the PBF (can take a long time). **Windows release ZIP:** the same three files sit next to **`gpxmapper.exe`**; read **`doc/USER_GUIDE.md`** and **`doc/THIRD_PARTY_NOTICES.md`** in the unzip folder. From the ZIP folder you can also run **`bash verify_local_hungary.sh`** or **`gpxmapper.exe check-nominatim`**. Matches GPXMapper’s default **`http://localhost:8080`** when **`NOMINATIM_SERVER`** is unset.
- **`NOMINATIM_SERVER`** — Base URL (no trailing slash required). If unset, the default is **`http://localhost:8080`** (typical local Docker). For the public instance set **`https://nominatim.openstreetmap.org`** and follow the [OSM Nominatim usage policy](https://operations.osmfoundation.org/policies/nominatim/); a **local** server is faster and avoids strict rate limits (the tool adds a minimum delay between reverse requests when the public OSM host is detected).
- **Health check** — Before heavy work, **`GET {base}/status`** is tried up to **three times** with short backoff. If it still fails, the CLI prints the URL, last error, and hints; on a **TTY** it asks whether to **continue without reverse geolocation** or **abort** (pressing Enter keeps **abort**). If **stdin is not a terminal**, it **aborts** without prompting (non-interactive/CI-safe).
- **Reverse-geocode cache** — Successful lookups are stored in a small **SQLite** file **next to** the map tile cache directory (not inside it), keyed by server base URL and rounded coordinates. Re-running **`generate --geolocate`** reuses hits and skips HTTP for those cells. **`gpxmapper clear-cache`** removes **tiles only**; **`gpxmapper clear-cache --geolocation`** deletes the geocode database (after confirmation).

For historical design notes, see `docs/superpowers/plans/2026-06-07-nominatim-geolocate-cli.md`.

## Usage

- **Help** — `gpxmapper --help` and `gpxmapper generate --help` list every flag; global **`--log-level`** / **`GPXMAPPER_LOG_LEVEL`** default to **INFO** (use **DEBUG** for per-request HTTP and tile-cache detail).
- **Typical render** — `gpxmapper generate path/to/track.gpx` writes `track.mp4` next to the GPX (override with **`--output` / `-o`**).
- **Quick customization** — e.g. longer HD clip with a blue marker and light text:

  ```bash
  gpxmapper generate ride.gpx -o ride.mp4 -d 120 --width 1280 --height 720 \
    --marker-color 0,0,255 --text-color 255,255,255
  ```

- **Full option list and examples** — See the repository **README** (`Command-line options` and `Command-line Examples`).
- **Windows release ZIP** — Same content as `doc\USER_GUIDE.md` in the download: [Windows ZIP guide](windows-bundle.md) (single source: `install/doc/USER_GUIDE.md`).
- **API reference** — [Reference](reference/gpxmapper/cli) (generated docs for the CLI package).

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


