# Geolocate overlay label verbosity

> **Status:** Baseline shipped — compact overlay via `geolocation_label_format`; extended i18n is a **follow-up**.

## Problem

Nominatim’s full `display_name` is often too long for a single-line map overlay. The SQLite cache and video frames used that string verbatim.

## Decision (product)

- **Overlay string** is a **short, traveler-oriented line** built from structured `address` fields when available, with safe fallbacks to trimmed `display_name`.
- **Internationalization (baseline):** Tolerate **missing `city`** and other gaps; pick the **most helpful** known fields for locating the traveler (village, town, suburb, road, region, etc.). No locale-specific typography in this phase.
- **Cache:** Still one string per quantized cell; stored value is the **formatted overlay text** (not necessarily raw Nominatim `display_name`). Existing rows stay as-is until re-fetch or `gpxmapper clear-cache --geolocation`.

## Implementation (done / in repo)

| Item | Notes |
|------|--------|
| `format_geolocation_overlay_label()` | `src/gpxmapper/geolocation_label_format.py` — pure function, unit-tested |
| Prefetch wiring | `geolocation_overlay._reverse_label_from_cache_or_http` formats after HTTP, before `put_sync` |
| Truncation | Hard cap on overlay length to respect single-line rendering |

## Follow-up: deeper internationalization

**Goal:** Go beyond heuristic field priority with evidence from multiple countries and scripts.

- [ ] Curate **sample payloads** per region (EU urban, US suburban, rural Asia, trail-only, admin-boundary-only) and lock **expected overlay strings** in tests or a golden file.
- [ ] Optional **CLI** (`--geolocate-format`, `--geolocate-max-chars`) if users need full `display_name` or tighter caps.
- [ ] **RTL / non-Latin** display: verify font rendering and field ordering with real locales (may need UI/font follow-up, not only string building).
- [ ] **Per-country field priority** tweaks only when samples show systematic wrong picks (avoid premature abstraction).

## References

- `docs/solutions/2026-06-nominatim-geolocate.md`
- `docs/superpowers/plans/2026-06-07-reverse-geocode-sqlite-cache.md`
