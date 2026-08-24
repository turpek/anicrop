from __future__ import annotations
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ContentStrategy(Protocol):
    """Protocolo estrutural para estratégias de manipulação de conteúdo/pixels."""

    @classmethod
    def _crop(cls, target: Any, ref: Any) -> bool:
        ...

    @classmethod
    def _resize(cls, target: Any, width: int, height: int) -> bool:
        ...

    @classmethod
    def _fit(cls, target: Any, ref: Any) -> bool:
        ...

    @classmethod
    def _flip_x(cls, target: Any) -> bool:
        ...

    @classmethod
    def _flip_y(cls, target: Any) -> bool:
        ...
