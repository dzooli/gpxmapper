from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
from typing import Optional, Dict, Any
import asyncio
import logging

import httpx


@dataclass(slots=True)
class NominatimAddress:
    # This is a flexible mapping for address components (e.g., road, city, country, etc.)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NominatimReverseResponse:
    place_id: int
    lat: float
    lon: float
    display_name: str
    address: NominatimAddress
    boundingbox: Optional[List[float]] = None
    osm_type: Optional[str] = None
    osm_id: Optional[int] = None
    # Add more fields as needed


@dataclass(slots=True)
class NominatimStatusResponse:
    raw_html: str


class AbstractGeolocationClient(ABC):
    """
    Abstract base class for geolocation clients.
    """

    @abstractmethod
    async def reverse_geocode(
        self, lat: float, lon: float, extra_params: Optional[Dict[str, Any]] = None
    ) -> NominatimReverseResponse:
        pass

    @abstractmethod
    async def get_status(self) -> NominatimStatusResponse:
        pass

    @abstractmethod
    async def aclose(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()


class GeolocationServiceUnavailable(Exception):
    """Raised when the geolocation service is unavailable after retries."""

    pass


class AsyncNominatimClient(AbstractGeolocationClient):
    """
    Async client for Nominatim reverse geocoding and status queries.
    Follows SOLID, DRY, and KISS principles for maintainability and clarity.
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
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.email = email
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logger or logging.getLogger(__name__)
        self._client = httpx.AsyncClient(timeout=self.timeout)

    def _build_headers(self) -> Dict[str, str]:
        headers = {"User-Agent": self.user_agent}
        if self.email:
            headers["From"] = self.email
        return headers

    async def _request_with_retries(
        self, method: str, url: str, **kwargs
    ) -> httpx.Response:
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self._client.request(
                    method, url, headers=self._build_headers(), **kwargs
                )
                resp.raise_for_status()
                return resp
            except Exception as exc:  # Broaden to retry on any exception
                self.logger.warning(f"Attempt {attempt} failed: {exc}")
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff_factor * (2 ** (attempt - 1)))
        raise GeolocationServiceUnavailable(
            f"Service unavailable after {self.max_retries} attempts"
        ) from last_exc

    async def reverse_geocode(
        self, lat: float, lon: float, extra_params: Optional[Dict[str, Any]] = None
    ) -> NominatimReverseResponse:
        """
        Perform reverse geocoding for the given latitude and longitude.

        Args:
            lat (float): Latitude.
            lon (float): Longitude.
            extra_params (dict, optional): Additional query parameters.

        Returns:
            dict: Reverse geocoding result.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        params = {"lat": lat, "lon": lon, "format": "json", **(extra_params or {})}
        url = f"{self.base_url}/reverse"
        resp = await self._request_with_retries("GET", url, params=params)
        data = resp.json()
        boundingbox = None
        if "boundingbox" in data:
            try:
                boundingbox = [float(x) for x in data["boundingbox"]]
            except ValueError:
                pass
        return NominatimReverseResponse(
            place_id=int(data["place_id"]),
            lat=float(data["lat"]),
            lon=float(data["lon"]),
            display_name=data.get("display_name", ""),
            address=NominatimAddress(data=data.get("address", {})),
            boundingbox=boundingbox,
            osm_type=data.get("osm_type"),
            osm_id=int(data["osm_id"]) if "osm_id" in data else None,
        )

    async def get_status(self) -> NominatimStatusResponse:
        """
        Get status information from the Nominatim server.

        Returns:
            str: Status page HTML or text.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        url = f"{self.base_url}/status"
        resp = await self._request_with_retries("GET", url)
        return NominatimStatusResponse(raw_html=resp.text)

    async def aclose(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()


class GeolocationClientFactory:
    """
    Factory for creating geolocation client instances.
    Supports registration of new client types.
    """

    _registry = {}

    @classmethod
    def register_client(cls, name: str, client_cls):
        cls._registry[name] = client_cls

    @classmethod
    def create_client(cls, name: str, **kwargs) -> AbstractGeolocationClient:
        if name not in cls._registry:
            raise ValueError(f"Unknown geolocation client: {name}")
        return cls._registry[name](**kwargs)


# Singleton pattern for a geolocation client (optional usage)
class GeolocationClientSingleton:
    _instance: Optional[AbstractGeolocationClient] = None

    @classmethod
    def get_instance(
        cls, client: AbstractGeolocationClient = None
    ) -> AbstractGeolocationClient:
        if client is not None:
            cls._instance = client
        if cls._instance is None:
            raise RuntimeError("No geolocation client instance set.")
        return cls._instance

    @classmethod
    def clear_instance(cls):
        cls._instance = None


# Register the default Nominatim client
GeolocationClientFactory.register_client("nominatim", AsyncNominatimClient)
