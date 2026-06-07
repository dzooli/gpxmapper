## Unreleased

### Feat

- Add `--text-color` / `-tc` to `gpxmapper generate` for R,G,B text overlay color (same format as `--marker-color`).
- Windows build/release ZIPs include `start_server.bat`, `start_server.sh`, `verify_local_hungary.sh`, and `doc/USER_GUIDE.md` next to `gpxmapper.exe`.
- Add `scripts/package-windows-release.ps1` to build the same ZIP locally after `build_exe.py`.
- Pin MkDocs to 1.x (`mkdocs>=1.6,<2`); add `scripts/build-docs.{ps1,sh}` with `NO_MKDOCS_2_WARNING` / `DISABLE_MKDOCS_2_WARNING` to silence Material / pymdownx MkDocs 2.0 notices.

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
