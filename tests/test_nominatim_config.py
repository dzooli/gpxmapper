"""Tests for ``gpxmapper.nominatim_config``."""

from __future__ import annotations

import pytest

from gpxmapper.nominatim_config import (
    STATUS_PROBE_ATTEMPTS,
    get_nominatim_base_url,
    is_public_osm_nominatim,
    probe_nominatim_status_sync,
)


def test_get_nominatim_base_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOMINATIM_SERVER", raising=False)
    assert get_nominatim_base_url() == "http://localhost:8080"


def test_get_nominatim_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOMINATIM_SERVER", "https://example.com/nominatim/")
    assert get_nominatim_base_url() == "https://example.com/nominatim"


def test_is_public_osm_nominatim() -> None:
    assert is_public_osm_nominatim("https://nominatim.openstreetmap.org")
    assert not is_public_osm_nominatim("http://localhost:8080")
    assert not is_public_osm_nominatim("http://nominatim.openstreetmap.org")


def test_probe_status_success_first_try(httpx_mock) -> None:
    httpx_mock.add_response(url="http://localhost:8080/status", text="OK")
    ok, err = probe_nominatim_status_sync(base_url="http://localhost:8080")
    assert ok is True
    assert err is None
    assert len(httpx_mock.get_requests()) == 1


def test_probe_status_three_failures(httpx_mock) -> None:
    for _ in range(STATUS_PROBE_ATTEMPTS):
        httpx_mock.add_response(url="http://localhost:8080/status", status_code=503)
    ok, err = probe_nominatim_status_sync(base_url="http://localhost:8080")
    assert ok is False
    assert err is not None
    assert len(httpx_mock.get_requests()) == STATUS_PROBE_ATTEMPTS


def test_probe_status_success_on_third_try(httpx_mock) -> None:
    httpx_mock.add_response(url="http://localhost:8080/status", status_code=503)
    httpx_mock.add_response(url="http://localhost:8080/status", status_code=503)
    httpx_mock.add_response(url="http://localhost:8080/status", text="OK")
    ok, err = probe_nominatim_status_sync(base_url="http://localhost:8080")
    assert ok is True
    assert err is None
    assert len(httpx_mock.get_requests()) == 3
