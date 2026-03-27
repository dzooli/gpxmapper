"""Map tile fetching and composite rendering."""

from .async_renderer import MapRendererAsync
from .base import MapRendererBase
from .constants import DEFAULT_TILE_SERVER
from .factory import MapRendererFactory, MapRendererKind
from .sync import MapRenderer

MapRendererFactory.register_renderer(MapRendererKind.SYNC.value, MapRenderer)
MapRendererFactory.register_renderer(MapRendererKind.ASYNC.value, MapRendererAsync)

__all__ = [
    "DEFAULT_TILE_SERVER",
    "MapRenderer",
    "MapRendererAsync",
    "MapRendererBase",
    "MapRendererFactory",
    "MapRendererKind",
]
