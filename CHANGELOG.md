## 0.3.0 (2026-06-16)

### Feat

- **geolocation**: Reverse geocode using Nominatim with local cache

### Fix

- **ci**: keep gpxmapper-v{version} root in artifacts and release zip

## 0.2.0 (2025-05-24)

### Feat

- Add timezone support for timestamp conversion in videos
- Add support for scrolling text in videos
- Add `--no-timestamp` parameter to `generate` command to control timestamp display

### Refactor

- Remove support for 'local' timezone and improve timezone handling
- CLI structure into modular commands
- cli.py in one file
- Refactor default values and text configuration handling.

## v0.1.0 (2025-05-11)

### Feat

- Add TTF font support for text rendering in videos
- Add custom font support for text rendering in videos

### Fix

- Update Commitizen action to latest version in release workflow

### Refactor

- Improve CLI structure and formatting in gpxmapper/cli.py
- Refactor and enhance GPXMapper models, map rendering, and parsing.
- Clean up unused imports across multiple modules
- Refactor data models into a dedicated `models` module
