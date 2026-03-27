import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from gpxmapper.geolocation_clients import (
    AsyncGeopyNominatimClient,
    GeolocationServiceUnavailable,
    NominatimAddress,
    NominatimReverseResponse,
    NominatimStatusResponse,
)


@pytest_asyncio.fixture
async def geopy_client():
    async with AsyncGeopyNominatimClient(
        base_url="https://test-nominatim.org",
        user_agent="test-agent",
        max_retries=3,
        backoff_factor=0.0,
    ) as c:
        yield c


class TestAsyncGeopyNominatimClient:
    @pytest.mark.asyncio
    async def test_reverse_geocode_success(self, geopy_client, monkeypatch):
        # Patch asyncio.to_thread to synchronously call the function
        async def fake_to_thread(func, /, *args, **kwargs):
            await asyncio.sleep(0)
            return func(*args, **kwargs)

        monkeypatch.setattr("asyncio.to_thread", fake_to_thread)

        # Prepare a fake geopy response object with .raw
        raw = {
            "place_id": 123,
            "lat": "47.4979",
            "lon": "19.0402",
            "display_name": "Budapest, Hungary",
            "address": {"city": "Budapest", "country": "Hungary"},
            "boundingbox": ["47.4", "47.5", "19.0", "19.1"],
            "osm_type": "node",
            "osm_id": 456,
        }
        monkeypatch.setattr(
            geopy_client._geocoder,  # type: ignore[attr-defined]
            "reverse",
            lambda *a, **k: SimpleNamespace(raw=raw),
        )

        result = await geopy_client.reverse_geocode(47.4979, 19.0402)
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
    async def test_reverse_geocode_empty_response_raises(
        self, geopy_client, monkeypatch
    ):
        async def fake_to_thread(func, /, *args, **kwargs):
            await asyncio.sleep(0)
            return func(*args, **kwargs)

        monkeypatch.setattr("asyncio.to_thread", fake_to_thread)
        # Return an object with empty raw -> should raise
        monkeypatch.setattr(
            geopy_client._geocoder,  # type: ignore[attr-defined]
            "reverse",
            lambda *a, **k: SimpleNamespace(raw={}),
        )
        with pytest.raises(GeolocationServiceUnavailable):
            await geopy_client.reverse_geocode(1.0, 2.0)

        # Also test None location -> should raise
        monkeypatch.setattr(
            geopy_client._geocoder,  # type: ignore[attr-defined]
            "reverse",
            lambda *a, **k: None,
        )
        with pytest.raises(GeolocationServiceUnavailable):
            await geopy_client.reverse_geocode(1.0, 2.0)

    @pytest.mark.asyncio
    async def test_reverse_geocode_invalid_boundingbox(self, geopy_client, monkeypatch):
        async def fake_to_thread(func, /, *args, **kwargs):
            await asyncio.sleep(0)
            return func(*args, **kwargs)

        monkeypatch.setattr("asyncio.to_thread", fake_to_thread)
        raw = {
            "place_id": 1,
            "lat": "1.0",
            "lon": "2.0",
            "display_name": "Place",
            "address": {},
            "boundingbox": ["a", "b", "c", "d"],
        }
        monkeypatch.setattr(
            geopy_client._geocoder,  # type: ignore[attr-defined]
            "reverse",
            lambda *a, **k: SimpleNamespace(raw=raw),
        )
        result = await geopy_client.reverse_geocode(1.0, 2.0)
        assert result.boundingbox is None

    @pytest.mark.asyncio
    async def test_reverse_geocode_missing_lat_lon_fallback(
        self, geopy_client, monkeypatch
    ):
        async def fake_to_thread(func, /, *args, **kwargs):
            await asyncio.sleep(0)
            return func(*args, **kwargs)

        monkeypatch.setattr("asyncio.to_thread", fake_to_thread)
        raw = {
            "place_id": 99,
            # 'lat' and 'lon' intentionally missing
            "display_name": "Unknown",
            "address": {},
        }
        monkeypatch.setattr(
            geopy_client._geocoder,  # type: ignore[attr-defined]
            "reverse",
            lambda *a, **k: SimpleNamespace(raw=raw),
        )
        result = await geopy_client.reverse_geocode(10.5, -20.25)
        assert result.lat == pytest.approx(10.5)
        assert result.lon == pytest.approx(-20.25)
        assert result.osm_id is None

    @pytest.mark.asyncio
    async def test_reverse_geocode_merges_extra_params(self, geopy_client, monkeypatch):
        # Capture kwargs passed to geopy's reverse
        captured = {}

        def fake_reverse(arg_tuple, **kwargs):
            captured["arg"] = arg_tuple
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                raw={
                    "place_id": 1,
                    "lat": "1.0",
                    "lon": "2.0",
                    "display_name": "x",
                    "address": {},
                }
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            await asyncio.sleep(0)
            return func(*args, **kwargs)

        monkeypatch.setattr("asyncio.to_thread", fake_to_thread)
        monkeypatch.setattr(
            geopy_client._geocoder,  # type: ignore[attr-defined]
            "reverse",
            fake_reverse,
        )

        await geopy_client.reverse_geocode(
            1.0, 2.0, extra_params={"namedetails": False, "zoom": 10}
        )
        assert captured["arg"] == (1.0, 2.0)
        assert captured["kwargs"]["exactly_one"] is True
        assert captured["kwargs"]["addressdetails"] is True
        # overridden by extra_params
        assert captured["kwargs"]["namedetails"] is False
        assert captured["kwargs"]["zoom"] == 10

    @pytest.mark.asyncio
    async def test_reverse_geocode_retries_and_fails(self, geopy_client, monkeypatch):
        call_count = {"count": 0}

        async def failing_to_thread(func, /, *args, **kwargs):
            call_count["count"] += 1
            await asyncio.sleep(0)
            raise RuntimeError("Network down")

        monkeypatch.setattr("asyncio.to_thread", failing_to_thread)

        with pytest.raises(GeolocationServiceUnavailable):
            await geopy_client.reverse_geocode(1.0, 2.0)
        assert call_count["count"] == geopy_client.max_retries

    @pytest.mark.asyncio
    async def test_reverse_geocode_succeeds_after_retry(
        self, geopy_client, monkeypatch
    ):
        attempt = {"i": 0}
        result_raw = {
            "place_id": 7,
            "lat": "1.0",
            "lon": "2.0",
            "display_name": "ok",
            "address": {},
        }

        async def flaky_to_thread(func, /, *args, **kwargs):
            if attempt["i"] < geopy_client.max_retries - 1:
                attempt["i"] += 1
                await asyncio.sleep(0)
                raise RuntimeError("Temporary")
            await asyncio.sleep(0)
            return func(*args, **kwargs)

        monkeypatch.setattr("asyncio.to_thread", flaky_to_thread)
        monkeypatch.setattr(
            geopy_client._geocoder,  # type: ignore[attr-defined]
            "reverse",
            lambda *a, **k: SimpleNamespace(raw=result_raw),
        )

        result = await geopy_client.reverse_geocode(1.0, 2.0)
        assert isinstance(result, NominatimReverseResponse)
        assert result.place_id == 7
        assert attempt["i"] == geopy_client.max_retries - 1

    @pytest.mark.asyncio
    async def test_get_status_success(self, geopy_client, monkeypatch):
        # Ensure httpx client exists so we can patch request
        httpx_client = geopy_client._client
        mock_response = Mock()
        mock_response.text = "<html>Status OK</html>"
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        monkeypatch.setattr(
            httpx_client, "request", AsyncMock(return_value=mock_response)
        )

        result = await geopy_client.get_status()
        assert isinstance(result, NominatimStatusResponse)
        assert result.raw_html == "<html>Status OK</html>"

    @pytest.mark.asyncio
    async def test_get_status_retries_and_fails(self, geopy_client, monkeypatch):
        httpx_client = geopy_client._client
        call_count = {"count": 0}

        async def side_effect(*args, **kwargs):
            call_count["count"] += 1
            await asyncio.sleep(0)
            raise RuntimeError("Network down")

        monkeypatch.setattr(httpx_client, "request", AsyncMock(side_effect=side_effect))
        with pytest.raises(GeolocationServiceUnavailable):
            await geopy_client.get_status()
        assert call_count["count"] == geopy_client.max_retries

    @pytest.mark.asyncio
    async def test_get_status_succeeds_after_retry(self, geopy_client, monkeypatch):
        httpx_client = geopy_client._client
        mock_response = Mock()
        mock_response.text = "<html>Status OK</html>"
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        attempts = {"i": 0}

        async def side_effect(*args, **kwargs):
            if attempts["i"] < geopy_client.max_retries - 1:
                attempts["i"] += 1
                await asyncio.sleep(0)
                raise RuntimeError("Temporary error")
            await asyncio.sleep(0)
            return mock_response

        monkeypatch.setattr(httpx_client, "request", AsyncMock(side_effect=side_effect))
        result = await geopy_client.get_status()
        assert isinstance(result, NominatimStatusResponse)
        assert result.raw_html == "<html>Status OK</html>"
        assert attempts["i"] == geopy_client.max_retries - 1

    @pytest.mark.asyncio
    async def test_aclose(self):
        geoclient = AsyncGeopyNominatimClient(backoff_factor=0.0)
        # force-create httpx client so we can inspect it without re-creating via property
        _ = geoclient._client  # noqa: F841
        await geoclient.aclose()
        # access the internal field directly to avoid lazy re-creation
        assert geoclient._httpx_client is not None
        assert geoclient._httpx_client.is_closed  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with AsyncGeopyNominatimClient(backoff_factor=0.0) as geoclient:
            # Create the httpx client within context
            _ = geoclient._client  # noqa: F841
            assert geoclient._httpx_client is not None
            assert not geoclient._httpx_client.is_closed  # type: ignore[union-attr]
        assert geoclient._httpx_client is not None
        assert geoclient._httpx_client.is_closed  # type: ignore[union-attr]
