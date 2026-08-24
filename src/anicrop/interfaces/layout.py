from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from anicrop.spatial import Region


class LayoutStrategy(ABC):
    """Classe base abstrata para estratégias de layout e enquadramento de moldura."""

    @abstractmethod
    def fit(self, ref: Any) -> bool:
        """Enquadra a moldura no retângulo de referência."""
        ...

    @abstractmethod
    def align(
        self,
        ref: Any,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        """Alinha a posição global em relação ao retângulo de referência."""
        ...

    @abstractmethod
    def resize_bounds(
        self,
        new_width: int,
        new_height: int,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        """Redimensiona os limites da moldura."""
        ...

    @abstractmethod
    def fit_content(self, *args: Any, **kwargs: Any) -> bool:
        """Ajusta a moldura aos limites do conteúdo visível."""
        ...

    @classmethod
    @abstractmethod
    def _fit(cls, target: Any, ref_region: Region) -> bool:
        ...

    @classmethod
    @abstractmethod
    def _align(
        cls,
        target: Any,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ...

    @classmethod
    @abstractmethod
    def _resize_bounds(
        cls,
        target: Any,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ...

    @classmethod
    @abstractmethod
    def _fit_content(cls, target: Any, *args: Any, **kwargs: Any) -> bool:
        ...
