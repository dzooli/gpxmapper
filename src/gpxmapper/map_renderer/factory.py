"""Factory for creating map renderer implementations."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Union

from .base import MapRendererBase
from .constants import DEFAULT_TILE_SERVER


class MapRendererKind(str, Enum):
    """Selectable map renderer implementation."""

    SYNC = "sync"
    ASYNC = "async"


class MapRendererFactory:
    """Factory for map renderer instances. Supports registration of new renderer types."""

    _registry: Dict[str, type[MapRendererBase]] = {}

    @classmethod
    def register_renderer(cls, name: str, renderer_cls: type[MapRendererBase]) -> None:
        cls._registry[name] = renderer_cls

    @classmethod
    def create_renderer(cls, name: str, **kwargs) -> MapRendererBase:
        renderer_cls = cls._registry.get(name)
        if renderer_cls is None:
            raise ValueError(f"Unknown map renderer: {name}")
        if kwargs.get("tile_server") is None and name == MapRendererKind.SYNC.value:
            kwargs = {**kwargs, "tile_server": DEFAULT_TILE_SERVER}
        return renderer_cls(**kwargs)

    @classmethod
    def create(
            cls,
            kind: Union[MapRendererKind, str] = MapRendererKind.SYNC,
            *,
            tile_server: Optional[str] = None,
            cache_dir: Optional[str] = None,
            use_cache: bool = True,
    ) -> MapRendererBase:
        """Instantiate the requested renderer (convenience wrapper around :meth:`create_renderer`).

        Args:
            kind: ``sync`` (default) or ``async``, as enum or string.
            tile_server: Tile URL template; OSM default when omitted for sync; async resolves ``None`` internally.
            cache_dir: Tile cache directory; OS-dependent default when ``None``.
            use_cache: Whether disk caching is enabled.

        Returns:
            A concrete map renderer.
        """
        name = kind.value if isinstance(kind, MapRendererKind) else str(kind).lower()
        return cls.create_renderer(
            name,
            tile_server=tile_server,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )
