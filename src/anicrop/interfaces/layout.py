from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anicrop.spatial import Point


class LayoutStrategy(ABC):
    """Classe base abstrata para estratégias de layout e enquadramento de moldura."""

    @abstractmethod
    def fit(self, ref: Any) -> bool:
        """Enquadra a moldura no retângulo de referência."""
        pass

    @abstractmethod
    def align(
        self,
        ref: Any,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        """Alinha a posição global em relação ao retângulo de referência."""
        pass

    @abstractmethod
    def pin(
        self,
        point: tuple[float, float] | Point,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        """Posiciona a moldura de modo que sua âncora coincida com o ponto especificado."""
        pass

    @abstractmethod
    def resize_bounds(
        self,
        new_width: float,
        new_height: float,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        """Redimensiona os limites da moldura."""
        pass

    @abstractmethod
    def fit_content(self, *args: Any, **kwargs: Any) -> bool:
        """Ajusta a moldura aos limites do conteúdo visível."""
        pass
