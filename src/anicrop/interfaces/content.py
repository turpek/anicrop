from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class ContentStrategy(ABC):
    """Classe base abstrata para manipulação, transformação e ajuste de pixels/conteúdo."""

    @abstractmethod
    def crop(self, ref: Any) -> bool:
        """Recorta o conteúdo para a região de referência via máscara de corte."""
        ...

    @abstractmethod
    def resize(self, width: int, height: int) -> bool:
        """Redimensiona o conteúdo por fator de escala."""
        ...

    @abstractmethod
    def fit(self, ref: Any) -> bool:
        """Ajusta o conteúdo à região de referência."""
        ...

    @abstractmethod
    def flip_x(self) -> bool:
        """Espelha o conteúdo horizontalmente."""
        ...

    @abstractmethod
    def flip_y(self) -> bool:
        """Espelha o conteúdo verticalmente."""
        ...
