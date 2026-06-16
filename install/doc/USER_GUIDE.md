# GPXMapper — Windows bundle user guide

This file is shipped inside the **release ZIP** as `doc\USER_GUIDE.md`, next to `gpxmapper.exe`. Unzip the archive into a folder of your choice; that folder is your **installation directory**.

## Layout after unzip

| Item | Purpose |
|------|---------|
| `gpxmapper.exe` | Command-line program (run from a terminal in this folder, or add the folder to `PATH`). |
| `README.md` | Full project readme: all CLI flags, examples, and programmatic usage. |
| `LICENSE` | MIT license text. |
| `CHANGELOG.md` | Version history. |
| `start_server.bat` | **Windows** helper to start local **Docker** Nominatim on port **8080** (same role as `start_server.sh`). |
| `start_server.sh` | **Linux / macOS / Git Bash / WSL** — same Docker flow as the repository’s `nominatim/start_server.sh`. |
| `verify_local_hungary.sh` | Optional health checks for a **Hungary** extract (needs **Git Bash**, **WSL**, or another Unix shell plus `curl` and `python`). |
| `doc\USER_GUIDE.md` | This guide. |
| `doc\THIRD_PARTY_NOTICES.md` | Third-party Python packages in the executable (license summary + Apache 2.0 / MPL 2.0 full texts). |

Run the executable from a **Command Prompt** or **PowerShell** after `cd` into the installation directory, for example:

```cmd
gpxmapper.exe --help
gpxmapper.exe generate C:\path\to\track.gpx
```

## Features (short)

- Map video from GPX: moving marker, timestamps, optional title, captions CSV, scrolling text, or **reverse-geolocation labels** (`--geolocate`).
- **`--text-color` / `-tc`** — same `R,G,B` format as **`--marker-color`** for all on-screen text.
- Tile cache and **`clear-cache`** / **`clear-cache --geolocation`** (see `README.md`).

## Local Nominatim for `--geolocate`

GPXMapper’s default **`NOMINATIM_SERVER`** is **`http://localhost:8080`**. To run your own server on Windows:

1. Install **Docker Desktop** and ensure it is running.
2. In the installation directory, run **`start_server.bat`** (double-click or from `cmd`).  
   - Optional: set **`PBF_URL`** before starting to use another Geofabrik `.osm.pbf` extract (default is **Hungary**).
3. Wait until the container finishes **importing** the first time (can take hours for large regions).
4. Check connectivity: **`gpxmapper.exe check-nominatim`** or, in Git Bash from this folder: **`bash verify_local_hungary.sh`**.

For the **public** Nominatim service instead, set `NOMINATIM_SERVER=https://nominatim.openstreetmap.org` and follow [OpenStreetMap’s usage policy](https://operations.osmfoundation.org/policies/nominatim/). Local Docker is faster and avoids strict client-side throttling in GPXMapper.

## More documentation

- **Full CLI reference and examples:** `README.md` in this folder.  
- **Repository clone** (developers): **`start_server.sh`**, **`start_server.bat`**, and **`verify_local_hungary.sh`** live under **`nominatim/`** in the [source tree](https://github.com/dzooli/gpxmapper); the ZIP copies them beside the executable. The Markdown source for this guide is **`install/doc/USER_GUIDE.md`** (also embedded on the project docs site).

## Contributing and license

Contributing workflow and full legal text are in the main repository; see **`LICENSE`** in this folder and the project **README** on GitHub.
