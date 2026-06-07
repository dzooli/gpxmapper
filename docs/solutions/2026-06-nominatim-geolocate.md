---
title: Nominatim geolocate CLI (context anchor)
date: 2026-06-07
tags: [nominatim, cli, geolocation, video]
---

## Canonical plan

Full task list and decisions: [`docs/superpowers/plans/2026-06-07-nominatim-geolocate-cli.md`](../superpowers/plans/2026-06-07-nominatim-geolocate-cli.md).

## Compressed behavior (for agents)

- **`NOMINATIM_SERVER`**: default `http://localhost:8080`; public `https://nominatim.openstreetmap.org` documented.
- **`--geolocate`**: conflicts with `--scrolling-text` / `--scrolling-speed`.
- **`/status`**: up to **3** HTTP attempts; on total failure → stderr error + **`typer.confirm`** (default abort) to continue **without** geolocation; no TTY → abort.
- **Prefetch**: main thread only, before `ThreadPoolExecutor`; **10 m** anchor + **`GPXTrackPoint.distance_to`**; public host → **`asyncio.sleep(1.1)`** after each reverse HTTP.
- **SQLite cache**: Successful reverse lookups are stored under the same app-data parent as map tiles (not inside the tile folder); see [`docs/superpowers/plans/2026-06-07-reverse-geocode-sqlite-cache.md`](../superpowers/plans/2026-06-07-reverse-geocode-sqlite-cache.md). `clear-cache` does **not** delete this database.
- **Overlay**: static address line, same vertical band as scrolling text, `text_align` for X.

## Key files (when implemented)

`nominatim_config.py`, `geolocation_overlay.py`, `reverse_geocode_cache.py`, `video_generator.py` (`VideoCaptioner` + `VideoGenerator`), `cli/generate.py`, `cli/utils.py`, `cli/__init__.py` (import order), `models.py` (`TextConfig.geolocate`).

## Verification (2026-06-07)

- `uv run ruff check src tests` — clean.
- `uv run pytest` — 128 passed (2 skipped).
- Follow-up fixes in-session: `generate` re-raises `typer.BadParameter` / `typer.Abort` (was swallowed by broad `except Exception`); `_invoke(..., standalone_mode=True)` for conflict CLI test; `_DummyCaptioner.add_geolocation_text_to_frame`; ruff `F401` on side-effect CLI imports; removed unused `frame_seconds` in prefetch loop.
