from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from anicrop.container import BaseLayer, Container
    from anicrop.layer import Layer
    from anicrop.spatial import Region


@runtime_checkable
class Cacheable(Protocol):
    """Protocolo estrutural para elementos ou efeitos que expõem estado observável para cache."""

    def cache_state(self) -> dict[str, Any]:
        """Retorna um dicionário contendo os atributos que afetam o resultado visual.

        Valores compostos como matrizes devem ser convertidos para bytes ou tuplas
        para suportar comparação direta por igualdade.
        """
        ...


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
