# Nominatim `--geolocate` CLI Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional `--geolocate` to `gpxmapper generate` that shows reverse-geocoded addresses (from configurable Nominatim) in place of scrolling text, with 10 m fetch throttling, safe interaction with parallel frame rendering, and updated user docs.

**Architecture:** Resolve Nominatim base URL from `NOMINATIM_SERVER` (fallback: **`http://localhost:8080`** — local Docker / dev default). **Before** any reverse-geocode prefetch or map composite work for a geolocated run, perform an explicit **`GET /status`** health check with **up to 3 attempts total** (no fourth try). On repeated failure, show a **clear error** and **prompt** the user to **either** continue **without** reverse geolocation **or** abort entirely (see Task 1). Validate `--geolocate` against scrolling options in the CLI. Before any `ThreadPoolExecutor` frame work, **precompute** a `list[str]` of overlay text per frame index in the **main thread** using `asyncio.run()` + existing `AsyncNominatimClient` (single client, sequential awaits). Reuse `GPXTrackPoint.distance_to` for the 10 m gate against the last **fetched** anchor. Pass the list into `VideoCaptioner`, which draws **static** text at the **same vertical band and font** as scrolling text, with horizontal layout from existing `text_align` / `_calculate_text_x_position`.

**Tech stack:** Python 3.12+, Typer, `httpx` async client (`gpxmapper.geolocation_clients`), `asyncio`, existing `VideoGenerator` / `VideoCaptioner` / `TextConfig`.

**Default URL & rate limiting:** When `NOMINATIM_SERVER` is unset, default to **`http://localhost:8080`** (fast, no policy throttle in code). **User docs** must state that the **public** instance **`https://nominatim.openstreetmap.org`** is available for users without a local server: set `NOMINATIM_SERVER` to that URL. When the resolved base URL is the **public OSM Nominatim** (normalize host + scheme for comparison), **after each successful reverse request** apply a **safe minimum delay** before the next HTTP call (e.g. `await asyncio.sleep(1.1)` — OSM asks for max ~1 req/s; pick one constant, document it). **Local / other hosts:** do **not** insert that delay (more performant; local Nominatim has no such global limit). Implement via `nominatim_config.should_rate_limit_public(base_url: str) -> bool` or equivalent next to `get_nominatim_base_url()`.

---

## File map

| Area | Files |
|------|--------|
| Config / URL / **health check** | New `src/gpxmapper/nominatim_config.py` — base URL, public-host detection, **`verify_nominatim_server`** (`/status` + retries) |
| Models | `src/gpxmapper/models.py` — extend `TextConfig` |
| Geocode orchestration | New `src/gpxmapper/geolocation_overlay.py` — build frame schedule + prefetch labels (keeps `video_generator.py` slimmer) |
| Video / captions | `src/gpxmapper/video_generator.py` — prefetch hook, captioner static geolocate draw |
| CLI | `src/gpxmapper/cli/generate.py`, `src/gpxmapper/cli/utils.py` |
| CLI package init | `src/gpxmapper/cli/__init__.py` — **fix import order** (`app` before command modules) to remove circular import |
| Tests | `tests/test_cli.py`, new `tests/test_geolocation_overlay.py` (or similar), extend `tests/test_video_generator.py` if needed |
| Docs | `README.md`, `docs/userdocs/index.md` |

---

## Task 1: Nominatim base URL helper

**Files:**
- Create: `src/gpxmapper/nominatim_config.py` (or `geolocation_config.py`)
- Test: `tests/test_nominatim_config.py`

- [ ] **Implement** `get_nominatim_base_url() -> str`: `os.environ.get("NOMINATIM_SERVER", "http://localhost:8080").rstrip("/")`.
- [ ] **Implement** `is_public_osm_nominatim(base_url: str) -> bool` (or `should_apply_public_usage_throttle`): return true only when the URL targets **`nominatim.openstreetmap.org`** with **https** (normalize trailing slash, lowercase host). Used by prefetch to decide on `asyncio.sleep` after each geocode.
- [ ] **Document** in module docstring: default is local `8080`; set `NOMINATIM_SERVER=https://nominatim.openstreetmap.org` for the public service; public path applies inter-request delay; local does not. Mention identifying **User-Agent** (reuse client default).

- [ ] **Preliminary server verification (`/status`):** Implement `async def verify_nominatim_server(...)` (and `verify_nominatim_server_sync` for CLI / sync callers) in the same module. Behavior:
  - **Endpoint:** `GET {base_url}/status` with the same **User-Agent** header policy as reverse geocoding.
  - **Attempts:** **Exactly 3 HTTP tries total** (not 4): attempt 1, on failure backoff, attempt 2, on failure backoff, attempt 3. On connection errors, non-2xx HTTP status, or timeouts, wait with **exponential backoff between attempts only** (e.g. `0.5s` after attempt 1 fails, `1.0s` after attempt 2 fails — tune to match `RobustExternalCalls` spirit without a fourth call).
  - **Success:** any **2xx** response from `/status` on any attempt → return normally (no prompt).
  - **Failure after 3 attempts:** Do **not** raise immediately to the top level. Return a **structured failure** (e.g. `NominatimStatusCheckFailed` dataclass or `(False, url, last_error)`) **or** raise a dedicated exception caught only at the CLI boundary — the important part is the **next step** below.
- [ ] **User prompt on total `/status` failure:** When **`--geolocate`** is set and all **3** `/status` attempts fail, **`cli/utils.generate_video`** (or `generate()` before calling it) must:
  1. Print a **clear, multi-line error** to stderr (via `typer.secho(..., err=True)`): configured **base URL**, **that 3 attempts failed**, **last error** (exception message), and short hints (start local Nominatim on port 8080, or set `NOMINATIM_SERVER` to `https://nominatim.openstreetmap.org`, check firewall/TLS).
  2. Ask: **Continue without reverse geolocation?** vs **Abort** — use **`typer.confirm(..., default=False)`** so **[Enter] / No aborts** and the user must explicitly confirm to drop geolocation. If **Yes**: rebuild **`TextConfig`** with **`geolocate=False`** (e.g. `dataclasses.replace(text_config, geolocate=False)` if frozen) and continue with normal video generation (no prefetch, no address overlay). If **No** / default: **`raise typer.Abort()`**.
  3. **Non-interactive / no TTY:** If **`sys.stdin.isatty()`** is false (CI, pipes), **skip confirm** and **abort** with the same error text plus one line: *"Not prompting because stdin is not a terminal; fix Nominatim or omit --geolocate."* then **`typer.Abort()`**.
- [ ] **Optional (YAGNI unless quick):** `gpxmapper check-nominatim` subcommand that only runs the same verify + exits 0/1 for CI or ops smoke tests.
- [ ] **Tests:** `tests/test_nominatim_config.py` — success on first try (1 HTTP call); failures then success uses **≤3** total calls; all 3 fail → no success; **prompt path**: mock `typer.confirm` returning True → `geolocate` cleared and video path proceeds (mock `generate_video` internals); False → abort. Mock non-TTY → abort without confirm.

---

## Task 2: `TextConfig` and CLI wiring

**Files:**
- Modify: `src/gpxmapper/models.py` (`TextConfig`: add `geolocate: bool = False`)
- Modify: `src/gpxmapper/cli/generate.py` — `--geolocate` flag (`is_flag=True`, help text)
- Modify: `src/gpxmapper/cli/utils.py` — `create_text_config(..., geolocate: bool = False)`; pass into `TextConfig`

- [ ] **Mutual exclusion:** At start of `generate()` (after parsing options), if `geolocate` and (`scrolling_text is not None` or `scrolling_speed is not None`): `typer.echo(...)` clear message + `raise typer.BadParameter(...)` or `typer.BadParameter` with message listing conflicting flags (Typer shows this cleanly).
- [ ] **Tests:** CLI test: `--geolocate` + `--scrolling-text` fails with expected substring; `--geolocate` alone succeeds path mocked at video layer.

---

## Task 3: Geolocation overlay prefetch (10 m + HTTP)

**Files:**
- Create: `src/gpxmapper/geolocation_overlay.py`
- Test: `tests/test_geolocation_overlay.py`

- [ ] **Frame schedule:** Reuse the same time/progress math as `_write_video_frames` (extract a small iterator or duplicate minimally with a comment pointing to the twin in `video_generator` — prefer **one shared helper** on `VideoGenerator` like `_iter_frame_timestamps(duration_seconds, fps, start_time, total_track_seconds)` returning `(frame_idx, frame_timestamp)` to avoid drift between prefetch and render).

- [ ] **Interpolate lat/lon:** Call existing `VideoGenerator._interpolate_position(points_with_time, ts)` — requires `VideoGenerator` instance with cleared caches same as `generate_video` start, or move interpolation to a **stateless function** taking `(points_with_time, ts, cache_dict)` to avoid tight coupling; **KISS:** use a short-lived `VideoGenerator` only for interpolation is heavy (map renderer). **Better:** extract `_interpolate_position` logic to module-level function in `video_generator.py` or `models` helper used by both — only if refactor stays small; otherwise pass `VideoGenerator` into prefetch after `__init__` but **before** `create_composite_map` and use its `_interpolate_position` (reset caches first).

- [ ] **Async prefetch coroutine:** `async def prefetch_geolocation_labels(...)-> list[str]`:
  - Instantiate `GeolocationClientFactory.create_client("nominatim", base_url=get_nominatim_base_url(), user_agent=...)`.
  - `throttle_public = is_public_osm_nominatim(base_url)` (or equivalent).
  - Maintain `_anchor: GPXTrackPoint | None`, `_label: str`.
  - For each frame’s `(lat, lon)`: build temporary `GPXTrackPoint` with time=None; if anchor is None or `anchor.distance_to(current) > 10`: `await client.reverse_geocode(lat, lon)`, set `_label = response.display_name`, set anchor to current point; if `throttle_public`: `await asyncio.sleep(PUBLIC_NOMINATIM_MIN_INTERVAL_SEC)` (constant ~1.1, documented). Else keep `_label` (no HTTP, no sleep).
  - Append `_label` to list (same length as `total_frames`).
  - `try/finally: await client.aclose()`.

- [ ] **Optional micro-cache:** If the same rounded coordinate repeats, skip HTTP (anchor logic already covers <10 m). No `lru_cache` required for correctness; add **`functools.lru_cache`** on a private pure `(_round(lat), _round(lon)) -> None` only if we split HTTP — **YAGNI:** skip unless profiling shows duplicate calls at identical coords.

- [ ] **Tests (httpx mock):** Three points in a line: first fetch, second <10 m no new request, third >10 m second request; use `pytest-httpx` or mock client.
- [ ] **Tests (rate limit):** With base URL public OSM, assert **sleep** called between distinct HTTP fetches (mock `asyncio.sleep`); with `http://localhost:8080`, assert **no** sleep between fetches for the same sequence.

---

## Task 4: `VideoCaptioner` static geolocate line

**Files:**
- Modify: `src/gpxmapper/video_generator.py` (`VideoCaptioner`)

- [ ] **State:** `self._geolocation_labels: Optional[list[str]] = None` set via `set_geolocation_labels(self, labels: list[str])` or constructor arg from `VideoGenerator`.

- [ ] **Draw:** `add_geolocation_text_to_frame(self, frame, frame_idx) -> ndarray`:
  - If labels is None or `frame_idx >= len`: return frame.
  - `text = labels[frame_idx]`; reuse **same `y_pos` and thickness** as `add_scrolling_text_to_frame` (timestamp branch vs no-timestamp branch).
  - **Horizontal:** use `_calculate_text_x_position` with measured width of full string (like captions/title), not scrolling x.

- [ ] **`add_scrolling_text_to_frame`:** Early return if geolocation labels active (so scrolling path never runs when geolocate mode).

---

## Task 5: `VideoGenerator.generate_video` integration

**Files:**
- Modify: `src/gpxmapper/video_generator.py`

- [ ] After `_prepare_track_points` and cache reset, if `text_config.geolocate`:
  - **Precondition:** `/status` flow (Task 1) must have either succeeded **or** the user chose **continue without geolocation** (in which case `text_config.geolocate` is false and this block is skipped). Do not duplicate `/status` in `VideoGenerator`.
  - `labels = asyncio.run(prefetch_geolocation_labels(...))` **or** `asyncio.run(main())` wrapper from overlay module.
  - `self.captioner.set_geolocation_labels(labels)`.
- [ ] Pass `text_config` into `VideoGenerator` (already partially via captioner); ensure `VideoGenerator` can read `text_config.geolocate` — may need to store `self._text_config` reference.

- [ ] In `_generate_frame`, after captions: call `add_geolocation_text_to_frame` when geolocate; else keep scrolling call as today.

---

## Task 6: Fix `cli/__init__.py` circular import

**Files:**
- Modify: `src/gpxmapper/cli/__init__.py`

- [ ] Order: logging setup → `app = typer.Typer(...)` → **then** `from . import generate, info, clear_cache` (side-effect registration) with single `# noqa: E402` block if needed.

---

## Task 7: Documentation

**Files:**
- `README.md` — env var, `--geolocate`, conflict with scrolling.
- `docs/userdocs/index.md` — same, aligned with existing CLI docs style.

**Required doc content:**
- **Default:** With no `NOMINATIM_SERVER`, GPXMapper uses **`http://localhost:8080`** (typical local Docker Nominatim); no inter-request throttle in code for that default.
- **Public option:** Users without a local server may set **`NOMINATIM_SERVER=https://nominatim.openstreetmap.org`** (spell full URL). Note OSM’s **usage policy** (rate limits, attribution); GPXMapper applies a **minimum delay between reverse requests** when this public instance is detected, which is **slower** but safer than hammering the service.
- **Performance:** Local Nominatim is recommended for **`--geolocate`** on long videos (many distinct fetches).
- **Health check & prompt:** With **`--geolocate`**, the tool probes **`GET /status`** up to **3 times** (with short backoff between failures). If the server never responds successfully, it prints a **clear error** (URL, last error, hints) and asks whether to **continue without reverse geolocation** or **abort** (`typer.confirm`, default **No** = abort). In **non-interactive** runs (stdin not a TTY), it **aborts** without prompting and explains why.

---

## Task 8: QA (ruff, pytest, manual)

- [ ] `uv run ruff check src tests`
- [ ] `uv run pytest` full suite
- [ ] Manual: `gpxmapper generate ... --geolocate` — with Nominatim up, `/status` succeeds; with it down, confirm **error + prompt**; answer **Yes** → video completes without geolocation; **No** → abort. Pipe stdin → abort without prompt.

---

## Risks / decisions

1. **Parallel frames:** Geocode **must not** run inside worker threads; prefetch list is mandatory.
2. **Prefetch vs map:** Running Nominatim before composite map is fine; alternatively after composite — same ordering relative to threads matters only: **before** `ThreadPoolExecutor`.
3. **Rate limits:** **Public** OSM Nominatim only — enforced delay between HTTP reverse calls; **local default** — no artificial delay (YAGNI: no configurable throttle env unless requested later).
4. **Frozen `TextConfig`:** Adding `geolocate` field is a **breaking** dataclass change for any external callers constructing `TextConfig` positionally — unlikely; document in changelog if you keep one.

5. **Default localhost:** Users without Nominatim on `8080` will get connection errors until they start local Docker or set `NOMINATIM_SERVER` to the public URL — call this out prominently in docs.
6. **`/status` vs reverse:** Health check uses **`/status`** only; a server that answers `/status` but breaks `/reverse` is still a partial failure — acceptable for “preliminary” scope; document if needed.
7. **Interactive default:** Prompt defaults to **abort** so CI/scripts that forget `--geolocate` implications still fail safe when piped; only explicit **Yes** continues without geolocation.

## Out of scope (YAGNI)

- Forward geocoding, POI search, offline tiles.
- `NOMINATIM_USER_AGENT` / email env (reuse hardcoded sensible default).
- Replacing `ThreadPoolExecutor` with async video pipeline.
