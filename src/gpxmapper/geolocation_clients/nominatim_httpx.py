from __future__ import annotations

import logging
from typing import Dict, Optional

import httpx

from .base import (
    NominatimHttpClientBase,
    NominatimReverseResponse,
)


class AsyncNominatimClient(NominatimHttpClientBase):
    """
    Async client for Nominatim reverse geocoding and status queries using httpx.
    """

    def __init__(
        self,
        base_url: str = "https://nominatim.openstreetmap.org",
        user_agent: str = "gpxmapper/1.0",
        email: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(
            base_url=base_url,
            user_agent=user_agent,
            email=email,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            logger=logger,
        )
        # Eagerly create the HTTP client so tests can inspect is_closed without prior requests
        self._httpx_client = httpx.AsyncClient(timeout=self.timeout)

    async def reverse_geocode(
        self, lat: float, lon: float, extra_params: Optional[Dict[str, Any]] = None
    ) -> NominatimReverseResponse:
        params = {"lat": lat, "lon": lon, "format": "json", **(extra_params or {})}
        url = self._build_reverse_url()
        resp = await self.request("GET", url, params=params)
        data = resp.json()
        return self._build_reverse_response(data)
