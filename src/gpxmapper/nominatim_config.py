"""Nominatim base URL, public-host detection, and `/status` health probing.

Environment:

- ``NOMINATIM_SERVER`` — Base URL (no trailing slash required). If unset, defaults to
  ``http://localhost:8080`` (typical local Docker). For the public OSM instance set
  ``https://nominatim.openstreetmap.org`` and follow OSM Nominatim usage policy.

Reverse geocode requests must send a identifying ``User-Agent`` (see OSM policy).
"""

from __future__ import annotations

import asyncio
import logging
import os
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

DEFAULT_NOMINATIM_SERVER = "http://localhost:8080"
"""Default when ``NOMINATIM_SERVER`` is not set (local Nominatim)."""

PUBLIC_NOMINATIM_MIN_INTERVAL_SEC = 1.1
"""Delay after each reverse HTTP call when using the public OSM Nominatim host."""

STATUS_PROBE_ATTEMPTS = 3
"""Total HTTP GET ``/status`` attempts (no fourth try)."""

STATUS_BACKOFF_BASE_SEC = 0.5


def get_nominatim_base_url() -> str:
    return os.environ.get("NOMINATIM_SERVER", DEFAULT_NOMINATIM_SERVER).rstrip("/")


def is_public_osm_nominatim(base_url: str) -> bool:
    """True when ``base_url`` targets the public https Nominatim instance (rate-limit reverse calls)."""
    parsed = urlparse(base_url)
    scheme = (parsed.scheme or "http").lower()
    host = (parsed.netloc or "").lower()
    return scheme == "https" and host == "nominatim.openstreetmap.org"


async def probe_nominatim_status(
    base_url: str | None = None,
    *,
    user_agent: str = "gpxmapper/1.0",
    timeout: float = 10.0,
) -> tuple[bool, str | None]:
    """Try ``GET {base}/status`` up to :data:`STATUS_PROBE_ATTEMPTS` times.

    Returns:
        ``(True, None)`` on first successful 2xx response, or ``(False, last_error_text)``
        after all attempts fail.
    """
    base = (base_url or get_nominatim_base_url()).rstrip("/")
    url = f"{base}/status"
    last_err: str | None = None
    for attempt in range(STATUS_PROBE_ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers={"User-Agent": user_agent})
                response.raise_for_status()
            logger.debug(
                "Nominatim status probe OK (attempt %s/%s): GET %s -> %s",
                attempt + 1,
                STATUS_PROBE_ATTEMPTS,
                url,
                response.status_code,
            )
            return True, None
        except Exception as exc:  # noqa: BLE001 — aggregate failures for operator message
            last_err = str(exc)
            logger.debug(
                "Nominatim status probe attempt %s/%s failed: %s",
                attempt + 1,
                STATUS_PROBE_ATTEMPTS,
                last_err,
            )
            if attempt < STATUS_PROBE_ATTEMPTS - 1:
                await asyncio.sleep(STATUS_BACKOFF_BASE_SEC * (2**attempt))
    return False, last_err


def probe_nominatim_status_sync(
    base_url: str | None = None,
    *,
    user_agent: str = "gpxmapper/1.0",
    timeout: float = 10.0,
) -> tuple[bool, str | None]:
    """Sync wrapper for :func:`probe_nominatim_status` (CLI / non-async callers)."""
    return asyncio.run(probe_nominatim_status(base_url=base_url, user_agent=user_agent, timeout=timeout))
