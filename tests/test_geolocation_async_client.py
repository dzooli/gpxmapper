import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock
from src.gpxmapper.geolocation_clients import (
    AsyncNominatimClient,
    NominatimAddress,
    NominatimReverseResponse,
    NominatimStatusResponse,
    GeolocationServiceUnavailable,
)


@pytest_asyncio.fixture
async def client():
    async with AsyncNominatimClient(
        base_url="https://test-nominatim.org", user_agent="test-agent"
    ) as c:
        yield c


class TestAsyncNominatimClient:
    @pytest.mark.asyncio
    async def test_reverse_geocode_success(self, client):
        mock_response = Mock()
        mock_response.json = Mock(
            return_value={
                "place_id": 123,
                "lat": "47.4979",
                "lon": "19.0402",
                "display_name": "Budapest, Hungary",
                "address": {"city": "Budapest", "country": "Hungary"},
                "boundingbox": ["47.4", "47.5", "19.0", "19.1"],
                "osm_type": "node",
                "osm_id": 456,
            }
        )
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        client._client.request = AsyncMock(return_value=mock_response)

        result = await client.reverse_geocode(47.4979, 19.0402)
        assert isinstance(result, NominatimReverseResponse)
        assert result.place_id == 123
        assert result.lat == pytest.approx(47.4979)
        assert result.lon == pytest.approx(19.0402)
        assert result.display_name == "Budapest, Hungary"
        assert isinstance(result.address, NominatimAddress)
        assert result.address.data["city"] == "Budapest"
        assert result.address.data["country"] == "Hungary"
        assert result.boundingbox == [47.4, 47.5, 19.0, 19.1]
        assert result.osm_type == "node"
        assert result.osm_id == 456

    @pytest.mark.asyncio
    async def test_get_status_success(self, client):
        mock_response = Mock()
        mock_response.text = "<html>Status OK</html>"
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        client._client.request = AsyncMock(return_value=mock_response)

        result = await client.get_status()
        assert isinstance(result, NominatimStatusResponse)
        assert result.raw_html == "<html>Status OK</html>"

    @pytest.mark.asyncio
    async def test_reverse_geocode_retries_and_fails(self, client):
        # Simulate all attempts raising an exception (max_retries=3 by default)
        call_count = {"count": 0}

        async def side_effect(*args, **kwargs):
            call_count["count"] += 1
            raise Exception("Network down")

        client._client.request = AsyncMock(side_effect=side_effect)
        with pytest.raises(GeolocationServiceUnavailable):
            await client.reverse_geocode(47.4979, 19.0402)
        assert call_count["count"] == client.max_retries

    @pytest.mark.asyncio
    async def test_get_status_retries_and_fails(self, client):
        call_count = {"count": 0}

        async def side_effect(*args, **kwargs):
            call_count["count"] += 1
            raise Exception("Network down")

        client._client.request = AsyncMock(side_effect=side_effect)
        with pytest.raises(GeolocationServiceUnavailable):
            await client.get_status()
        assert call_count["count"] == client.max_retries

    @pytest.mark.asyncio
    async def test_reverse_geocode_succeeds_after_retry(self, client):
        # Raise for (max_retries-1) times, then succeed
        mock_response = Mock()
        mock_response.json = Mock(
            return_value={
                "place_id": 1,
                "lat": "1.0",
                "lon": "2.0",
                "display_name": "Test Place",
                "address": {"city": "Test City"},
                "boundingbox": ["1.0", "2.0", "3.0", "4.0"],
                "osm_type": "node",
                "osm_id": 2,
            }
        )
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        call_count = {"count": 0}

        async def side_effect(*args, **kwargs):
            if call_count["count"] < client.max_retries - 1:
                call_count["count"] += 1
                raise Exception("Temporary error")
            return mock_response

        client._client.request = AsyncMock(side_effect=side_effect)
        result = await client.reverse_geocode(1.0, 2.0)
        assert isinstance(result, NominatimReverseResponse)
        assert result.place_id == 1
        assert result.display_name == "Test Place"
        assert call_count["count"] == client.max_retries - 1

    @pytest.mark.asyncio
    async def test_get_status_succeeds_after_retry(self, client):
        mock_response = Mock()
        mock_response.text = "<html>Status OK</html>"
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        call_count = {"count": 0}

        async def side_effect(*args, **kwargs):
            if call_count["count"] < client.max_retries - 1:
                call_count["count"] += 1
                raise Exception("Temporary error")
            return mock_response

        client._client.request = AsyncMock(side_effect=side_effect)
        result = await client.get_status()
        assert isinstance(result, NominatimStatusResponse)
        assert result.raw_html == "<html>Status OK</html>"
        assert call_count["count"] == client.max_retries - 1

    @pytest.mark.asyncio
    async def test_aclose(self):
        client = AsyncNominatimClient()
        await client.aclose()
        assert client._client.is_closed

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with AsyncNominatimClient() as client:
            assert not client._client.is_closed
        assert client._client.is_closed
