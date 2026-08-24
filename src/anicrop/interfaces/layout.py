from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
from anicrop.spatial import Region


@runtime_checkable
class LayoutStrategy(Protocol):
    """Protocolo estrutural para estratégias de layout de elementos."""

    @classmethod
    def _fit(cls, target: Any, ref_region: Region) -> bool:
        ...

    @classmethod
    def _align(
        cls,
        target: Any,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ...

    @classmethod
    def _resize_bounds(
        cls,
        target: Any,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ...

    @classmethod
    def _fit_content(cls, target: Any, *args: Any, **kwargs: Any) -> bool:
        ...
