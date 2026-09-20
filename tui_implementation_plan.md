# GPXM-28: Add TUI (Terminal User Interface) Implementation Plan

## Overview
GPXM-28 introduces a Terminal User Interface (TUI) for GPXMapper to provide an interactive, visual terminal interface for generating videos, inspecting GPX tracks, and managing caches without needing to remember CLI flags.

Following the direction in the issue description, we use [**Trogon**](https://github.com/Textualize/trogon) (built on [**Textual**](https://github.com/Textualize/textual) by Textualize) to provide an interactive form-based TUI seamlessly integrated with our Typer application and underlying `gpxmapper.api`.

---

## User Review Required

> [!NOTE]
> - **Dependency**: `trogon>=0.6.0` (which brings `textual>=8.2.8`) added to `dependencies` in `pyproject.toml`.
> - **Command Access**: The TUI is accessible via `gpxmapper tui` (as well as `uv run poe tui`).
> - **PyInstaller**: PyInstaller options updated with `--collect-all=trogon` and `--collect-all=textual` so `./dist/gpxmapper.exe tui` works out of the box in standalone builds.

---

## Architecture & Integration

```mermaid
flowchart TD
    subgraph Users ["User Entry Points"]
        CLI_RUN["CLI (gpxmapper generate/info/...)"]
        TUI_RUN["Interactive TUI (gpxmapper tui)"]
        POE_RUN["Poe Task Runner (poe tui)"]
        EXE_RUN["Standalone Executable (gpxmapper.exe tui)"]
        SCRIPT_RUN["Python Code (import gpxmapper.api)"]
    end

    subgraph Presentation ["Presentation Layer (gpxmapper.cli & trogon)"]
        TYPER["Typer App (src/gpxmapper/cli)"]
        TROGON["Trogon TUI Engine (trogon.typer.init_tui)"]
        TEXTUAL["Textual Widgets & CSS"]
    end

    subgraph Core ["Programmatic Layer (gpxmapper.api)"]
        API["gpxmapper.api (generate_video, get_gpx_info, etc.)"]
        MODELS["gpxmapper.models (GPXInfo, Configs)"]
    end

    CLI_RUN --> TYPER
    TUI_RUN --> TROGON
    POE_RUN --> TYPER
    EXE_RUN --> TROGON
    TROGON --> TEXTUAL
    TROGON --> TYPER
    TYPER --> API
    SCRIPT_RUN --> API
    API --> Core
```

---

## Implementation Details

### 1. Dependencies and Configuration
- Added `trogon>=0.6.0` to `project.dependencies` in `pyproject.toml`.
- Added task alias `tui = "gpxmapper tui"` to `[tool.poe.tasks]` in `pyproject.toml`.
- Configured PyInstaller options in `pyproject.toml`:
  ```toml
  [tool.pyinstaller]
  entry_point = "exe_entry_point.py"
  options = [
      "--name=gpxmapper",
      "--onefile",
      "--console",
      "--hidden-import=gpxpy",
      "--hidden-import=typer",
      "--hidden-import=cv2",
      "--hidden-import=requests",
      "--hidden-import=PIL",
      "--hidden-import=numpy",
      "--hidden-import=zoneinfo",
      "--collect-all=trogon",
      "--collect-all=textual",
      "--specpath=.",
  ]
  ```

### 2. TUI Command and Advanced Widgets Integration
- Implemented `src/gpxmapper/cli/widgets.py` with:
  - `RangeSlider`: Visual Textual widget for bounded numerical options (`--zoom`, `--fps`, `--font-scale`) with decrement/increment buttons (`◀`/`▶`), keyboard arrow support (`left`/`right`/`h`/`l`), unicode track bar (`[━━━━●────────]`), and automatic live command preview sync.
  - Monkeypatches applied via `apply_trogon_patches()`:
    - Bounded `click.IntRange` and `click.FloatRange` render as `RangeSlider`.
    - Boolean options cleanly render as native `Checkbox` controls instead of text boxes.
    - Enum / Choice options cleanly render as `Select` dropdowns with properly resolved default values.
- Implemented `src/gpxmapper/cli/tui.py`:
  - Patched upstream Trogon bug where `action_show_command_info` queried the wrong screen instance on `ctrl+o`.
  - Initialized Typer TUI via `init_tui(app, name="gpxmapper")`.
- Registered subcommand in `src/gpxmapper/cli/__init__.py`.

### 3. Automated Test Suite
- Added `tests/test_tui.py`:
  - `test_tui_command_in_cli_help`: Verifies `tui` is present in `gpxmapper --help`.
  - `test_tui_help`: Verifies `gpxmapper tui --help` exits 0 with descriptive help text.
  - `test_tui_invocation_runs_trogon`: Verifies invoking `tui` initiates `Trogon.run`.
  - `test_trogon_action_show_command_info_does_not_crash`: Verifies `action_show_command_info` (`ctrl+o`) pushes the CommandInfo screen safely.
  - `test_trogon_renders_checkboxes_and_select_controls`: Verifies boolean flags render as `Checkbox` and enum choices render as `Select`.
  - `test_trogon_renders_range_sliders_for_bounded_numbers`: Verifies bounded numerical flags render as `RangeSlider` with valid min/max and defaults.
  - `test_range_slider_widget_interaction`: Verifies button clicks and keyboard navigation on `RangeSlider`.

### 4. Documentation
- Updated `README.md`:
  - Added TUI feature to the features list.
  - Added interactive TUI launch section (`gpxmapper tui`, `poe tui`, `gpxmapper.exe tui`).
  - Added `tui` subcommand documentation under Command-line options.
  - Added `## Development and Tasks (Poe the Poet)` task runner guide.
  - Added `gpxmapper.api` high-level programmatic API code examples.

---

## Verification Plan

### Automated Tests
- `uv run poe check` (runs `ruff check .`, `ruff format --check .`, `pytest --cov=gpxmapper`).

### Binary & Standalone Packaging
- `uv run poe build-exe`
- `./dist/gpxmapper.exe --help`
- `./dist/gpxmapper.exe tui --help`

### Documentation Site
- `uv run poe docs`
