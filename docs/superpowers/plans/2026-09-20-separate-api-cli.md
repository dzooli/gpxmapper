# GPXM-27: Separate API and CLI Interface Implementation Plan

**Goal:** Decouple the core GPXMapper business logic and services into a clean, reusable Python programmatic API (`gpxmapper.api`) with domain-specific exceptions (`gpxmapper.exceptions`). Refactor the CLI layer (`gpxmapper.cli`) to act as a thin presentation and interactive wrapper over this API, enabling future GUI (GPXM-13) and TUI (GPXM-28) frontends.

---

## Architecture & Design

```mermaid
flowchart TD
    subgraph Packaging ["Packaging & Entry Points"]
        CLI_MAIN["gpxmapper/__main__.py"]
        EXE_EP["exe_entry_point.py (PyInstaller)"]
    end

    subgraph Tooling ["Poe Task Runner & Quality (pyproject.toml)"]
        POE_TEST["poe test / poe test-cov"]
        POE_LINT["poe lint / poe format"]
        POE_CHECK["poe check"]
        POE_EXE["poe build-exe"]
    end

    subgraph Interfaces ["Interface Layer"]
        CLI["gpxmapper.cli (Typer CLI)"]
        GUI["Future GUI (GPXM-13)"]
        TUI["Future TUI (GPXM-28)"]
        SCRIPT["Python API / Client Scripts"]
    end

    subgraph API ["Programmatic API Layer (gpxmapper.api)"]
        API_GEN["generate_video()"]
        API_INFO["get_gpx_info() -> GPXInfo"]
        API_CACHE["clear_tile_cache(), clear_geolocation_cache()"]
        API_NOM["check_nominatim_status()"]
        API_CFG["create_text_config(), parse_color()"]
        API_EXC["gpxmapper.exceptions.*"]
    end

    subgraph Core ["Core Domain & Services"]
        PARSER["GPXParser"]
        VIDGEN["VideoGenerator"]
        RENDERER["MapRendererBase / Factory"]
        MODELS["gpxmapper.models (GPXInfo, CacheInfo, Configs)"]
    end

    CLI_MAIN --> CLI
    EXE_EP --> CLI
    CLI --> API
    GUI --> API
    TUI --> API
    SCRIPT --> API
    API --> Core
```

---

## Task List

- [x] **Git Branch Setup**: Work on dedicated branch `feature/GPXM-27-separate-api-and-cli`.
- [x] **Task Execution Tool (`poethepoet`)**:
  - Add `poethepoet` to dev dependencies in `pyproject.toml`.
  - Configure tasks: `test`, `test-cov`, `lint`, `lint-fix`, `format`, `format-check`, `check`, `build-exe`, `docs`.
- [x] **Exceptions Hierarchy (`src/gpxmapper/exceptions.py`)**:
  - `GPXMapperError(Exception)` - base exception
  - `GPXParseError(GPXMapperError, ValueError)` - parsing failures
  - `GPXEmptyError(GPXParseError)` - file has no track points
  - `GPXMissingTimeError(GPXParseError)` - track points lack timestamps
  - `ConfigurationError(GPXMapperError, ValueError)` - invalid parameters/colors/alignments
  - `VideoGenerationError(GPXMapperError)` - video generation failure
  - `NominatimUnavailableError(GPXMapperError)` - service unreachable error
- [x] **Models & DTOs (`src/gpxmapper/models.py`)**:
  - `GPXInfo`: slotted dataclass with `file_path`, `point_count`, `start_time`, `end_time`, `duration`, `coordinate_bounds`.
  - `CacheInfo`: dataclass with `cache_path`, `file_count`, `exists`.
  - `CacheClearResult`: dataclass with `cache_path`, `files_deleted`, `success`, `error_message`.
- [x] **Programmatic API Layer (`src/gpxmapper/api/`)**:
  - `src/gpxmapper/api/video.py`: Programmatic `generate_video(...)` function.
  - `src/gpxmapper/api/info.py`: `get_gpx_info(...)` returning `GPXInfo`.
  - `src/gpxmapper/api/cache.py`: `get_tile_cache_info`, `clear_tile_cache`, `get_geolocation_cache_info`, `clear_geolocation_cache`.
  - `src/gpxmapper/api/config.py`: `parse_color`, `create_text_config`.
  - `src/gpxmapper/api/nominatim.py`: `check_nominatim_status`.
  - `src/gpxmapper/api/__init__.py`: Public API exports.
- [x] **Package Root Exports (`src/gpxmapper/__init__.py`)**:
  - Re-export `__version__ = "0.3.0"`.
  - Re-export public API functions, exceptions, and models.
- [x] **CLI Presentation Layer (`src/gpxmapper/cli/`)**:
  - `src/gpxmapper/cli/generate.py`: Flag parsing and delegation to `api.generate_video`.
  - `src/gpxmapper/cli/info.py`: Output formatting using `api.get_gpx_info`.
  - `src/gpxmapper/cli/clear_cache.py`: Interactive confirmation and delegation to `api.clear_tile_cache` / `api.clear_geolocation_cache`.
  - `src/gpxmapper/cli/check_nominatim.py`: Status presentation using `api.check_nominatim_status`.
  - `src/gpxmapper/cli/utils.py`: Presentation helpers and exception adapters.
- [x] **Testing & Verification**:
  - `tests/test_api.py`: Full unit tests for all `gpxmapper.api` functions.
  - `tests/test_cli.py`: Verified CLI tests against refactored layer.
  - Offline Nominatim testing: All tests run without live network endpoints.
  - PyInstaller verification: `poe build-exe` builds standalone binary and `./dist/gpxmapper.exe --help` runs cleanly.
  - Code quality: `poe check` passes all 171 tests (85% coverage), lint, and formatting.
