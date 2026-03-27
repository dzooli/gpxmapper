from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from geopy.geocoders import Nominatim as GeopyNominatim

from .base import (
    GeolocationServiceUnavailable,
    NominatimHttpClientBase,
    NominatimReverseResponse,
)


class AsyncGeopyNominatimClient(NominatimHttpClientBase):
    """Async Nominatim client powered by geopy; async via asyncio.to_thread."""

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
        domain, scheme = self._extract_base_domain_and_scheme(self.base_url)

        # geopy Nominatim supports domain and scheme
        self._geocoder = GeopyNominatim(
            user_agent=self.user_agent,
            domain=domain,
            scheme=scheme,
            timeout=int(self.timeout),
        )

    async def reverse_geocode(
        self, lat: float, lon: float, extra_params: Optional[Dict[str, Any]] = None
    ) -> NominatimReverseResponse:
        kwargs = {
            "exactly_one": True,
            "namedetails": True,
            "addressdetails": True,
            **(extra_params or {}),
        }

        async def _op():
            return await asyncio.to_thread(self._geocoder.reverse, (lat, lon), **kwargs)

        loc = await self.retry(_op)
        data: Dict[str, Any] = getattr(loc, "raw", {}) if loc is not None else {}
        if not data:
            raise GeolocationServiceUnavailable("Empty response from geocoder")
        return self._build_reverse_response(data, fallback_lat=lat, fallback_lon=lon)


# get_status and aclose inherited from NominatimHttpClientBase
