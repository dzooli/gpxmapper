from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar
from urllib.parse import urlparse

import httpx


@dataclass(slots=True)
class NominatimAddress:
    # Flexible mapping for address components (e.g., road, city, country, etc.)
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


@dataclass(slots=True)
class NominatimStatusResponse:
    raw_html: str


class AbstractGeolocationClient(ABC):
    """Abstract base class for geolocation clients."""

    @abstractmethod
    async def reverse_geocode(
        self, lat: float, lon: float, extra_params: Optional[Dict[str, Any]] = None
    ) -> NominatimReverseResponse:
        raise NotImplementedError

    @abstractmethod
    async def get_status(self) -> NominatimStatusResponse:
        raise NotImplementedError

    @abstractmethod
    async def aclose(self):
        raise NotImplementedError

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()


class GeolocationServiceUnavailable(Exception):
    """Raised when the geolocation service is unavailable after retries."""

    pass


class GeolocationClientFactory:
    """Factory for creating geolocation client instances. Supports registration of new client types."""

    _registry: Dict[str, type[AbstractGeolocationClient]] = {}

    @classmethod
    def register_client(cls, name: str, client_cls: type[AbstractGeolocationClient]):
        cls._registry[name] = client_cls

    @classmethod
    def create_client(cls, name: str, **kwargs) -> AbstractGeolocationClient:
        client_cls = cls._registry.get(name)
        if client_cls is None:
            raise ValueError(f"Unknown geolocation client: {name}")
        return client_cls(**kwargs)


class GeolocationClientSingleton:
    """Process-wide geolocation client holder.

    Prefer explicit dependency injection at call sites when possible. This helper
    exists for legacy wiring and should be used sparingly to avoid hidden global state.
    """

    _instance: Optional[AbstractGeolocationClient] = None

    @classmethod
    def get_instance(
        cls, client: Optional[AbstractGeolocationClient] = None
    ) -> AbstractGeolocationClient:
        """Return the singleton instance, optionally setting it first."""
        if client is not None:
            cls._instance = client
        if cls._instance is None:
            raise RuntimeError("No geolocation client instance set.")
        return cls._instance

    @classmethod
    def clear_instance(cls):
        """Clear singleton state (primarily for tests and controlled lifecycle resets)."""
        cls._instance = None


# ------------- Shared helpers for robust external calls -------------
T = TypeVar("T")


class RobustExternalCalls:
    """Mixin that provides header building and retry helpers as instance methods."""

    user_agent: str
    email: Optional[str]
    max_retries: int
    backoff_factor: float
    logger: Optional[logging.Logger]

    def build_headers(self) -> Dict[str, str]:
        headers = {"User-Agent": self.user_agent}
        if self.email:
            headers["From"] = self.email
        return headers

    async def retry(self, op: Callable[[], Awaitable[T]]) -> T:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await op()
            except Exception as exc:  # noqa: BLE001 catch broad for test/mocks and transient errors
                if self.logger:
                    self.logger.warning(f"Attempt {attempt} failed: {exc}")
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff_factor * (2 ** (attempt - 1)))
        raise GeolocationServiceUnavailable(
            f"Service unavailable after {self.max_retries} attempts"
        ) from last_exc


class HttpxClientMixin(RobustExternalCalls):
    """Mixin that manages a client httpx.AsyncClient and provides a robust request method."""

    timeout: float
    _httpx_client: Optional[httpx.AsyncClient] = None

    @property
    def _client(self):  # backward-compat for tests
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(timeout=self.timeout)
        return self._httpx_client

    def _ensure_httpx_client(self) -> httpx.AsyncClient:
        if self._httpx_client is None or getattr(
            self._httpx_client, "is_closed", False
        ):
            self._httpx_client = httpx.AsyncClient(timeout=self.timeout)
        return self._httpx_client

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        client = self._ensure_httpx_client()
        headers = kwargs.pop("headers", {})
        merged_headers = self.build_headers()
        if headers is not None:
            merged_headers = {**merged_headers, **headers}

        async def _do() -> httpx.Response:
            resp = await client.request(method, url, headers=merged_headers, **kwargs)
            resp.raise_for_status()
            return resp

        return await self.retry(_do)

    async def aclose(self):
        if self._httpx_client is not None and not self._httpx_client.is_closed:
            await self._httpx_client.aclose()


# ------------- Concrete common base for Nominatim HTTP clients -------------
class NominatimHttpClientBase(HttpxClientMixin, AbstractGeolocationClient):
    """Common base for Nominatim HTTP clients.

    Expects subclasses to set base_url, user_agent, email, timeout, max_retries, backoff_factor, logger.
    Provides shared get_status behavior.
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

    def _build_status_url(self) -> str:
        """Build the Nominatim status URL."""
        return f"{self.base_url}/status"

    def _build_reverse_url(self) -> str:
        """Build the Nominatim reverse-geocoding URL."""
        return f"{self.base_url}/reverse"

    @staticmethod
    def _extract_base_domain_and_scheme(base_url: str) -> tuple[str, str]:
        """Extract geopy-compatible ``(domain, scheme)`` from a base URL."""
        parsed = urlparse(base_url)
        domain = parsed.netloc or base_url
        scheme = parsed.scheme or "https"
        return domain, scheme

    @staticmethod
    def _parse_boundingbox(raw_bbox: Any) -> Optional[List[float]]:
        """Parse Nominatim ``boundingbox`` field into float coordinates."""
        if raw_bbox is None:
            return None
        try:
            return [float(x) for x in raw_bbox]
        except (ValueError, TypeError):
            return None

    def _build_reverse_response(
            self,
            data: Dict[str, Any],
            *,
            fallback_lat: Optional[float] = None,
            fallback_lon: Optional[float] = None,
    ) -> NominatimReverseResponse:
        """Convert raw Nominatim payload to a typed reverse-geocode response."""
        lat_value = data.get("lat", fallback_lat)
        lon_value = data.get("lon", fallback_lon)
        if lat_value is None or lon_value is None:
            raise GeolocationServiceUnavailable("Missing latitude/longitude in geocoder response")

        return NominatimReverseResponse(
            place_id=int(data.get("place_id", 0)),
            lat=float(lat_value),
            lon=float(lon_value),
            display_name=data.get("display_name", ""),
            address=NominatimAddress(data=data.get("address", {})),
            boundingbox=self._parse_boundingbox(data.get("boundingbox")),
            osm_type=data.get("osm_type"),
            osm_id=int(data["osm_id"]) if "osm_id" in data else None,
        )

    async def get_status(self) -> NominatimStatusResponse:  # type: ignore[override]
        url = self._build_status_url()
        resp = await self.request("GET", url)
        return NominatimStatusResponse(raw_html=resp.text)

    async def reverse_geocode(
        self, lat: float, lon: float, extra_params: Optional[Dict[str, Any]] = None
    ) -> NominatimReverseResponse:
        raise NotImplementedError
