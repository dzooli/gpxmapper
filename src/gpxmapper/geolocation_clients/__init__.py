from .base import (
    AbstractGeolocationClient,
    GeolocationClientFactory,
    GeolocationClientSingleton,
    GeolocationServiceUnavailable,
    NominatimAddress,
    NominatimReverseResponse,
    NominatimStatusResponse,
)
from .nominatim_httpx import AsyncNominatimClient

__all__ = [
    "AbstractGeolocationClient",
    "NominatimAddress",
    "NominatimReverseResponse",
    "NominatimStatusResponse",
    "GeolocationServiceUnavailable",
    "GeolocationClientFactory",
    "GeolocationClientSingleton",
    "AsyncNominatimClient",
]

# Optional: geopy-based client if geopy is installed
try:
    from .nominatim_geopy import AsyncGeopyNominatimClient  # type: ignore
except Exception:  # pragma: no cover
    AsyncGeopyNominatimClient = None  # type: ignore
else:  # pragma: no cover
    __all__.append("AsyncGeopyNominatimClient")
    GeolocationClientFactory.register_client(
        "geopy-nominatim", AsyncGeopyNominatimClient
    )

# Register default httpx-based client
GeolocationClientFactory.register_client("nominatim", AsyncNominatimClient)
