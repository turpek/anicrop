from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from anicrop.container import BaseLayer, Container
    from anicrop.layer import Layer


class AbstractLayerCache(ABC):
    """Classe base abstrata para gerenciadores de cache de renderização de camadas."""

    @abstractmethod
    def __call__(
        self, container: Sequence[BaseLayer] | Container
    ) -> AbstractContextManager[Any]:
        """Cria um escopo de contexto para ativação de cache durante a renderização."""
        pass

    @abstractmethod
    def is_dirty(self, layer: Layer) -> bool:
        """Verifica se a camada precisa ser renderizada do zero."""
        pass

    @abstractmethod
    def register(self, layer: Layer) -> None:
        """Registra a camada para gerenciamento de cache."""
        pass

    @abstractmethod
    def unregister(self, layer: Layer) -> None:
        """Remove a camada do gerenciamento de cache e restaura seu estado original."""
        pass
