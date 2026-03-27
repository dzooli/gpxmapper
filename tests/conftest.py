import pytest_asyncio

from gpxmapper.geolocation_clients import AsyncNominatimClient


@pytest_asyncio.fixture
def client():
    async def _client():
        async with AsyncNominatimClient(
            base_url="https://test-nominatim.org", user_agent="test-agent"
        ) as c:
            yield c

    return _client
