"""Build short reverse-geocode strings for the video overlay from Nominatim data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .geolocation_clients.base import NominatimReverseResponse

# Single-line overlay: keep conservative; font scaling is separate.
DEFAULT_MAX_OVERLAY_CHARS = 88

# Street-like linear features (first non-empty wins after house_number + road).
_STREET_LINE_KEYS = (
    "road",
    "pedestrian",
    "path",
    "footway",
    "cycleway",
    "bridleway",
    "track",
)

# Finer-grained area (optional first); settlement is appended by default when present.
_LOCAL_PLACE_KEYS = (
    "neighbourhood",
    "suburb",
    "quarter",
)

# City / town / village / … — appended when available so the traveler sees “where”
# even if a street or neighbourhood was already added.
_SETTLEMENT_KEYS = (
    "village",
    "hamlet",
    "town",
    "city",
    "municipality",
    "city_district",
    "locality",
    "county",
)

# Broader disambiguation when finer place names are missing.
_REGION_KEYS = (
    "state",
    "region",
    "state_district",
)

# Named POI / landuse when there is no street line (trail, peak, parking, etc.).
_NAMED_FEATURE_KEYS = (
    "amenity",
    "tourism",
    "leisure",
    "historic",
    "natural",
    "man_made",
    "shop",
)


def _address_dict(resp: NominatimReverseResponse) -> dict[str, Any]:
    addr = getattr(resp, "address", None)
    data = getattr(addr, "data", None) if addr is not None else None
    if isinstance(data, Mapping):
        return dict(data)
    return {}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return s


def _first_field(data: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        s = _clean(data.get(key))
        if s:
            return s
    return ""


def _street_line(data: Mapping[str, Any]) -> str:
    hn = _clean(data.get("house_number"))
    road = _clean(data.get("road"))
    if road:
        return f"{hn} {road}".strip() if hn else road
    for key in _STREET_LINE_KEYS:
        if key == "road":
            continue
        s = _clean(data.get(key))
        if s:
            return f"{hn} {s}".strip() if hn else s
    return ""


def _poi_line(data: Mapping[str, Any]) -> str:
    return _first_field(data, _NAMED_FEATURE_KEYS)


def _truncate(text: str, max_len: int) -> str:
    if max_len < 4:
        max_len = 4
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _fallback_from_display_name(display_name: str, max_len: int) -> str:
    if not display_name.strip():
        return ""
    parts = [p.strip() for p in display_name.split(",") if p.strip()]
    if len(parts) >= 2:
        joined = ", ".join(parts[:3])
        return _truncate(joined, max_len)
    return _truncate(display_name.strip(), max_len)


def _append_unique_segment(segments: list[str], part: str) -> None:
    """Append ``part`` if non-empty and not already equal to an existing segment (case-insensitive)."""
    p = _clean(part)
    if not p:
        return
    low = p.lower()
    if any(s.lower() == low for s in segments):
        return
    segments.append(p)


def _append_region_for_disambiguation(
    segments: list[str],
    place_label: str,
    data: Mapping[str, Any],
) -> None:
    """Append state/region when the line would otherwise be too vague."""
    if not segments:
        return
    region = _first_field(data, _REGION_KEYS)
    if not region:
        return
    if not place_label:
        if all(region.lower() not in s.lower() for s in segments):
            segments.append(region)
        return
    if (
        len(segments) == 1
        and place_label == segments[0]
        and region.lower() != place_label.lower()
    ):
        segments.append(region)


def format_geolocation_overlay_label(
    resp: NominatimReverseResponse,
    *,
    max_chars: int = DEFAULT_MAX_OVERLAY_CHARS,
) -> str:
    """Return a short line for map overlay from a reverse-geocode response.

    Uses structured ``address`` fields when present (tolerates missing ``city`` and
    other gaps). After street or POI text, appends neighbourhood/suburb (if any) and
    then village/town/city/etc. when present so the settlement is shown by default.
    Falls back to the first segments of ``display_name``, then truncation. See
    ``docs/superpowers/plans/2026-06-07-geolocate-label-verbosity.md``.
    """
    data = _address_dict(resp)
    display_name = _clean(getattr(resp, "display_name", None))

    segments: list[str] = []

    street = _street_line(data)
    poi = _poi_line(data)
    local = _first_field(data, _LOCAL_PLACE_KEYS)
    settlement = _first_field(data, _SETTLEMENT_KEYS)

    if street:
        segments.append(street)
    elif poi:
        segments.append(poi)

    if street or poi:
        _append_unique_segment(segments, local)
        _append_unique_segment(segments, settlement)
    else:
        if local:
            segments.append(local)
            _append_unique_segment(segments, settlement)
        elif settlement:
            segments.append(settlement)

    place_for_region = settlement or local
    _append_region_for_disambiguation(segments, place_for_region, data)

    if not segments:
        return _fallback_from_display_name(display_name, max_chars)

    out = ", ".join(segments)
    return _truncate(out, max_chars)
