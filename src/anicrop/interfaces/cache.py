from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from anicrop.container import BaseLayer, Container
    from anicrop.layer import Layer
    from anicrop.spatial import Region


class AbstractLayerCache(ABC):
    """Classe base abstrata para gerenciadores de cache de renderização de camadas."""

    @abstractmethod
    def __call__(
        self,
        container: Sequence[BaseLayer] | Container,
        effective_region: Region | None = None,
    ) -> AbstractContextManager[Any]:
        """Cria um escopo de contexto para ativação de cache durante a renderização."""
        pass

    @abstractmethod
    def is_dirty(self, layer: Layer) -> bool:
        """Verifica se a camada precisa ser renderizada do zero."""
        pass

    @abstractmethod
    def register(self, item: Layer | Container) -> None:
        """Registra a camada ou contêiner (recursivo) para gerenciamento de cache."""
        pass

    @abstractmethod
    def unregister(self, item: Layer | Container) -> None:
        """Remove a camada ou contêiner do gerenciamento de cache e restaura seu estado original."""
        pass
