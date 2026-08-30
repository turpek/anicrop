from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, TYPE_CHECKING
import numpy as np
from anicrop.interfaces.layer import AbstractBaseLayer

if TYPE_CHECKING:
    from anicrop.container import NullContainer


class AbstractContainer(ABC):
    """Classe base abstrata para contêineres hierárquicos na árvore de camadas."""

    parent: AbstractContainer | NullContainer
    _parent_inverse: np.ndarray

    @property
    @abstractmethod
    def _children(self) -> list[Any]:
        """Lista interna de nós filhos."""
        ...

    @_children.setter
    @abstractmethod
    def _children(self, value: list[Any]) -> None:
        ...

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Matriz de transformação do contêiner."""
        ...

    @abstractmethod
    def append(self, item: Any) -> None:
        """Adiciona um item ao contêiner."""
        ...

    @abstractmethod
    def insert(self, index: int, item: Any) -> None:
        """Insere um item em uma posição específica."""
        ...

    @abstractmethod
    def remove(self, item: Any) -> None:
        """Remove um item do contêiner."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove todos os itens do contêiner."""
        ...

    @abstractmethod
    def pop(self, index: int = -1) -> Any:
        """Remove e retorna o item no índice especificado."""
        ...

    @abstractmethod
    def move(self, item: Any, new_index: int) -> None:
        """Move um item existente para um novo índice."""
        ...

    @abstractmethod
    def move_relative(self, item: Any, steps: int) -> None:
        """Move um item relativamente na pilha."""
        ...

    @abstractmethod
    def move_to_front(self, item: Any) -> None:
        """Move o item para o topo da pilha."""
        ...

    @abstractmethod
    def move_to_back(self, item: Any) -> None:
        """Move o item para a base da pilha."""
        ...

    @abstractmethod
    def swap(self, item_a: Any, item_b: Any) -> None:
        """Troca a posição de dois itens na pilha."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        ...

    @abstractmethod
    def __reversed__(self) -> Iterator[Any]:
        ...


class AbstractGroupLayer(AbstractBaseLayer, AbstractContainer):
    """Classe base abstrata para grupos de camadas (combina container e camada espacial)."""
    ...
