# Reverse geolocation SQLite cache

> **For agentic workers:** Implement task-by-task; steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist successful Nominatim reverse lookups in a **SQLite** database on disk so repeated runs (and nearby coordinates) avoid redundant HTTP calls. The database file lives **next to** the existing map tile cache directory (same app data layout as `MapRendererBase.resolve_default_cache_directory()`), not inside the tile folder (so `clear-cache` does not delete it accidentally).

**Non-goals (v1):** Caching `GET /status`; sharing cache across machines; LRU eviction / max size (optional follow-up); caching geopy client path if it diverges from httpx (treat both as `base_url`-scoped rows if both hit same server).

---

## Context: tile cache layout (must mirror)

`MapRendererBase.resolve_default_cache_directory()` returns:

| OS | Resolved tile cache directory |
|----|-------------------------------|
| Windows | `%LOCALAPPDATA%\gpxmapper\cache` (fallback `~\AppData\Local\gpxmapper\cache`) |
| Linux | `~/.cache/gpxmapper` |
| Other (e.g. macOS) | `~/.gpxmapper/cache` |

**DB placement rule (sibling of tile root, not inside tile dir):**

- If the resolved path’s **basename** is `cache` (Windows, macOS): `Path(tile_cache).parent / "reverse_geocode.sqlite"`.
- Else (Linux, tile root is `.../gpxmapper`): `Path(tile_cache).parent / "gpxmapper_reverse_geocode.sqlite"` so the file is under `~/.cache/` with a **gpxmapper-specific** name (avoid a generic `~/.cache/reverse_geocode.sqlite`).

Centralize this in one helper, e.g. `resolve_reverse_geocode_cache_path() -> Path`, implemented in a **new** module (see below) that imports `MapRendererBase` only for path resolution (same pattern as `clear_cache` CLI).

---

## Architecture

1. **Lookup key:** Scope by **normalized Nominatim base URL** (strip trailing slash, optional lowercase host) **+ quantized coordinates**. Quantize `lat`/`lon` to a fixed precision (e.g. **5 decimal places**, ~1.1 m) so the 10 m anchor in prefetch still benefits from cache while not exploding row count for jitter. Store both **raw** request coords (or quantized only) consistently—**recommend:** store quantized values as the unique key so one row per cell.

2. **Value:** At minimum **`display_name`** (what the overlay shows). Optionally persist `place_id` / `fetched_at` for debugging and future TTL; v1 can keep schema minimal: `base_url`, `lat_q`, `lon_q`, `display_name`, `created_at`.

3. **Integration point:** **`prefetch_geolocation_labels`** in `geolocation_overlay.py` — before `await client.reverse_geocode(lat, lon)`, check cache; on miss, call Nominatim, then **insert** row. This keeps HTTP clients unchanged and avoids coupling cache to httpx/geopy internals.

4. **Public OSM throttle:** Apply **`asyncio.sleep(PUBLIC_NOMINATIM_MIN_INTERVAL_SEC)`** only after a **real HTTP** reverse call, **not** after a cache hit (preserves current policy and speeds local + cached runs).

5. **Concurrency / SQLite:** Prefetch runs in `asyncio` on the main event loop. Use **`sqlite3`** in the stdlib with a **process-wide `threading.Lock`** around all DB operations **or** delegate sync DB work to `asyncio.to_thread` under that lock. WAL mode (`PRAGMA journal_mode=WAL`) is recommended on open for readers/writers if you later extend usage. Single writer in v1 is enough.

6. **Initialization:** Open DB on first use; `CREATE TABLE IF NOT EXISTS` + `CREATE UNIQUE INDEX IF NOT EXISTS` on `(base_url, lat_q, lon_q)` (or primary key on those three). Parent directory must exist: `mkdir(parents=True)` for the DB file’s parent (tile parent already exists after map use, but geocode-only tests should still succeed).

7. **Errors:** If SQLite open/write fails, **log a warning** and **fall back** to uncached HTTP (do not fail video generation).

---

## File map

| Area | Files |
|------|--------|
| Path + DB API | **New** `src/gpxmapper/reverse_geocode_cache.py` — `resolve_reverse_geocode_cache_path()`, `ReverseGeocodeCache` class (get/put, context manager optional) |
| Prefetch | `src/gpxmapper/geolocation_overlay.py` — consult + populate cache around `reverse_geocode` |
| Docs / memory | `docs/solutions/2026-06-nominatim-geolocate.md` — short “SQLite cache” bullet + path rules; optional README note |
| Tests | **New** `tests/test_reverse_geocode_cache.py` — path logic (monkeypatch `resolve_default_cache_directory`), get/put roundtrip with `tmp_path` DB file |
| CLI (optional) | `clear-cache` or new flag — document in plan; implement only if quick: e.g. `clear-cache --include-geocode` or document manual delete path in v1 |

---

## Tasks

### Task 1: Path helper

- [ ] Add `resolve_reverse_geocode_cache_path() -> Path` with the basename rule above; unit-test Windows/Linux/macOS branches by forcing `resolve_default_cache_directory` return values (monkeypatch on `MapRendererBase`).

### Task 2: `ReverseGeocodeCache` (SQLite)

- [ ] Schema: unique `(base_url, lat_q, lon_q)`; columns sufficient for overlay (`display_name`); `created_at` default `CURRENT_TIMESTAMP`.
- [ ] `quantize(lat, lon) -> tuple[float, float]` (5 dp) as module-level or static helper; document why 5 dp.
- [ ] Methods: `get(base_url, lat, lon) -> str | None`, `put(base_url, lat, lon, display_name) -> None` (quantize inside).
- [ ] Thread-safe; enable WAL on connect; catch `sqlite3.Error`, log, no raise from `put` if best-effort.
- [ ] Optional: `from_default_path()` classmethod constructing path via `resolve_reverse_geocode_cache_path()`.

### Task 3: Wire into `prefetch_geolocation_labels`

- [ ] Instantiate cache once per prefetch (or module-level lazy singleton with lock—prefer **one instance per prefetch** for testability).
- [ ] On cache hit: set `last_label`, update anchor, **do not** HTTP; **do not** public throttle sleep.
- [ ] On miss: existing HTTP flow; on success `put(...)` before or after assigning `last_label`.

### Task 4: Tests

- [ ] Path tests (Task 1).
- [ ] Cache hit skips HTTP: mock `AsyncNominatimClient.reverse_geocode` in a small async test calling a thin wrapper or `prefetch_geolocation_labels` with mocked `VideoGenerator._interpolate_position` returning fixed coords twice—assert `reverse_geocode` call count 1 when second point within 10 m… **Note:** 10 m anchor already reduces calls; cache test should use **distinct** anchors (>10 m apart) with **same quantized cell** (tricky) **or** simpler: **unit-test** `ReverseGeocodeCache` get/put and separately test prefetch with **two frames** same coordinate → one HTTP (existing anchor behavior); add **process restart** simulation by only testing cache class. **Stronger test:** two prefetch passes with mocked client: first pass 1 call, second pass 0 calls if cache file reused—use `tmp_path` and inject cache path via env or constructor injection.
- [ ] **Injection:** Prefer passing optional `cache: ReverseGeocodeCache | None = None` into `prefetch_geolocation_labels` defaulting to `ReverseGeocodeCache.from_default_path()` so tests use `tmp_path` without touching real home.

### Task 5: Docs

- [ ] Update solution doc + one sentence in user-facing docs about cache location and that `clear-cache` does not remove it.

### Task 6 (optional): CLI

- [ ] Extend `clear-cache` with `--geocode` to delete the SQLite file with confirm, or defer to a follow-up PR.

---

## Verification

- [ ] `uv run ruff check src tests`
- [ ] `uv run pytest`
- [ ] Manual: run `generate --geolocate` twice on same GPX; second run should log fewer HTTP reverse requests (optional: temporary debug log line behind `logger.debug`—not required if tests cover cache).

---

## Open questions (resolved for v1)

| Question | Decision |
|----------|----------|
| Cache per Nominatim host? | Yes — `base_url` column (normalized string). |
| Quantization | 5 decimal places; document. |
| Inside tile directory? | **No** — sibling file to avoid `clear-cache` wiping DB. |
| Fail if DB corrupt? | Fall back to HTTP + warn. |
