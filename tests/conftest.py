"""Shared pytest fixtures."""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _quiet_http_library_logs() -> None:
    """Default CLI behavior: httpx/httpcore stay at WARNING so tests do not INFO-spam."""
    httpx_log = logging.getLogger("httpx")
    httpcore_log = logging.getLogger("httpcore")
    saved = (httpx_log.level, httpcore_log.level)
    httpx_log.setLevel(logging.WARNING)
    httpcore_log.setLevel(logging.WARNING)
    yield
    httpx_log.setLevel(saved[0])
    httpcore_log.setLevel(saved[1])
