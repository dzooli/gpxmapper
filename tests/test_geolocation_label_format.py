"""Tests for :mod:`gpxmapper.geolocation_label_format`."""

from __future__ import annotations

import pytest

from gpxmapper.geolocation_clients.base import NominatimAddress, NominatimReverseResponse
from gpxmapper.geolocation_label_format import (
    DEFAULT_MAX_OVERLAY_CHARS,
    format_geolocation_overlay_label,
)


def _resp(*, display_name: str, address: dict) -> NominatimReverseResponse:
    return NominatimReverseResponse(
        place_id=1,
        lat=47.0,
        lon=19.0,
        display_name=display_name,
        address=NominatimAddress(data=address),
    )


def test_street_and_city_no_duplicate_when_names_differ() -> None:
    r = _resp(
        display_name="Very Long Official Name, District, Big City, Region, Country",
        address={
            "road": "Rue de Paris",
            "city": "Paris",
            "country": "France",
        },
    )
    out = format_geolocation_overlay_label(r)
    assert "Rue de Paris" in out
    assert "Paris" in out
    assert "France" not in out


def test_tolerates_missing_city_uses_village_and_state() -> None:
    r = _resp(
        display_name="Haus, 12345, Bayern, Deutschland",
        address={
            "village": "Oberhof",
            "state": "Bayern",
            "postcode": "12345",
            "country": "Germany",
        },
    )
    out = format_geolocation_overlay_label(r)
    assert "Oberhof" in out
    assert "Bayern" in out


def test_poi_when_no_street() -> None:
    r = _resp(
        display_name="Something, Somewhere, Earth",
        address={
            "tourism": "alpine_hut",
            "village": "Grindelwald",
            "state": "Bern",
        },
    )
    out = format_geolocation_overlay_label(r)
    assert "alpine_hut" in out
    assert "Grindelwald" in out


def test_fallback_first_comma_segments_when_no_address() -> None:
    long_dn = (
        "Building A, Example Street 99, Neighbourhood, "
        "Big City, Large Region, Country, Earth"
    )
    r = _resp(display_name=long_dn, address={})
    out = format_geolocation_overlay_label(r)
    assert "Building A" in out
    assert "Example Street 99" in out
    assert "Neighbourhood" in out
    assert "Earth" not in out


def test_truncation_respects_max_chars() -> None:
    long_dn = "Part0, " + ", ".join(f"Segment{i:02d}WithExtraChars" for i in range(1, 30))
    r = _resp(display_name=long_dn, address={})
    out = format_geolocation_overlay_label(r, max_chars=40)
    assert len(out) <= 40
    assert out.endswith("…")


def test_house_number_and_road() -> None:
    r = _resp(
        display_name="ignored for structured",
        address={"house_number": "42", "road": "Main Street", "town": "Springfield"},
    )
    out = format_geolocation_overlay_label(r)
    assert "42" in out
    assert "Main Street" in out
    assert "Springfield" in out


def test_empty_returns_empty() -> None:
    r = NominatimReverseResponse(
        place_id=0,
        lat=0.0,
        lon=0.0,
        display_name="",
        address=NominatimAddress(data={}),
    )
    assert format_geolocation_overlay_label(r) == ""


@pytest.mark.parametrize(
    ("max_chars", "expected_max_len"),
    [(12, 12), (DEFAULT_MAX_OVERLAY_CHARS, DEFAULT_MAX_OVERLAY_CHARS)],
)
def test_max_chars_boundaries(max_chars: int, expected_max_len: int) -> None:
    r = _resp(display_name="x" * 500, address={"city": "y" * 500})
    out = format_geolocation_overlay_label(r, max_chars=max_chars)
    assert len(out) <= expected_max_len
