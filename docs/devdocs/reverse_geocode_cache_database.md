# Reverse geocode location cache (SQLite)

Developer reference for the on-disk cache used when **`generate --geolocate`** prefetches Nominatim reverse-geocode labels. Implementation: `src/gpxmapper/reverse_geocode_cache.py`.

## Purpose

- Avoid repeated HTTP reverse requests for the **same Nominatim server** and **same quantized map cell**.
- Persist the **overlay string** actually drawn on each video frame (produced by `geolocation_label_format.format_geolocation_overlay_label`), not necessarily the raw Nominatim `display_name` field from JSON.

## File location and naming

The database file is always a **sibling** of the map tile cache directory (never inside the tile folder), so `gpxmapper clear-cache` does not delete it. Resolution: `resolve_reverse_geocode_cache_path()`.

| Tile cache root (from `MapRendererBase.resolve_default_cache_directory()`) | Geocode SQLite filename |
|------------------------------------------------------------------------------|-------------------------|
| `…/gpxmapper/cache` (typical Windows / macOS layout; basename is `cache`)     | `reverse_geocode.sqlite` in the **parent** of `cache` |
| `…/.cache/gpxmapper` (typical Linux; basename is `gpxmapper`)                | `gpxmapper_reverse_geocode.sqlite` in **`.cache/`** |

Rationale for two names: on Linux the tile parent is often a shared `.cache` directory; the `gpxmapper_` prefix avoids a generic `reverse_geocode.sqlite` colliding with other apps.

## Connection and pragmas

- Opened with `sqlite3.connect(path, timeout=30.0)`.
- **`PRAGMA journal_mode=WAL`** is set on open so readers and a single writer behave reasonably if usage grows.
- Parent directories are created with `mkdir(parents=True, exist_ok=True)` when possible.
- On init failure (permissions, corrupt file, etc.), the cache class **disables itself** and geolocation continues without persistence (see module logging).

## Threading and asyncio

- A **process-wide `threading.Lock`** wraps connection creation and all SQL.
- **`get_sync` / `put_sync`** are intended to run on the **same thread as the asyncio event loop** that runs prefetch (see `geolocation_overlay.prefetch_geolocation_labels`). Do **not** call these from `asyncio.to_thread` with the same connection: SQLite’s default is `check_same_thread=True`.

## Table: `reverse_geocode`

Created lazily with `CREATE TABLE IF NOT EXISTS` when the first connection succeeds.

| Column | Type | Role |
|--------|------|------|
| **`base_url`** | `TEXT NOT NULL` | Normalized Nominatim base URL (see below). Part of the primary key so caches from different servers (e.g. local Docker vs `nominatim.openstreetmap.org`) do not collide. |
| **`lat_q`** | `REAL NOT NULL` | Quantized latitude (see **Coordinate quantization**). |
| **`lon_q`** | `REAL NOT NULL` | Quantized longitude. |
| **`display_name`** | `TEXT NOT NULL` | **Historical column name.** Stores the **formatted overlay label** (e.g. street + neighbourhood + city when Nominatim provides them), not the raw JSON `display_name`. |
| **`created_at`** | `TEXT NOT NULL DEFAULT (datetime('now'))` | SQLite UTC timestamp when the row was inserted or replaced (ISO-like string from `datetime('now')`). |

**Primary key:** `(base_url, lat_q, lon_q)` — one row per server per cell.

**Upsert semantics:** Writes use `INSERT OR REPLACE`. A new reverse lookup for the same key overwrites `display_name` and refreshes `created_at` via the default on insert (replaced rows get a new row payload as SQLite handles `OR REPLACE`).

## Key derivation details

### `base_url` normalization

`normalize_cache_base_url(url)`:

- Strips leading/trailing whitespace.
- Strips a trailing `/`.
- Lowercases the entire string.

So `HTTPS://Localhost:8080/` and `https://localhost:8080` map to the same key bucket.

### Coordinate quantization

`quantize_coordinates(lat, lon)` rounds each coordinate to **`GEOCODE_COORD_DECIMALS` (5)** decimal places (~1.1 m). Prefetch uses the **interpolated** frame position before quantizing, so nearby frames often share one cell and one cached label (in addition to the separate **10 m anchor** logic in `geolocation_overlay` that reduces HTTP calls).

## Operational notes

- **Clear cache:** `gpxmapper clear-cache --geolocation` deletes this SQLite file after confirmation.
- **Stale or verbose rows:** If overlay formatting rules change, existing rows are not auto-invalidated; operators can clear the geolocation cache or wait until coordinates fall into new cells / servers.
- **Inspection:** Any SQLite client can open the file; e.g. `SELECT base_url, lat_q, lon_q, length(display_name), created_at FROM reverse_geocode LIMIT 20;`

## Related documentation

- **Local Nominatim:** `nominatim/start_server.sh`, **`nominatim/start_server.bat`** (Windows), `nominatim/verify_local_hungary.sh`. **Release ZIP:** those files plus **`doc/USER_GUIDE.md`** beside `gpxmapper.exe` — user guide source **`install/doc/USER_GUIDE.md`**. Operators: README + `docs/userdocs/index.md` + `docs/solutions/2026-06-nominatim-geolocate.md`.
- Overlay string formatting: `docs/superpowers/plans/2026-06-07-geolocate-label-verbosity.md`
- Original cache design notes: `docs/superpowers/plans/2026-06-07-reverse-geocode-sqlite-cache.md`
