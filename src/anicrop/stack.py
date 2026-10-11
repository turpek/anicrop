from __future__ import annotations

from abc import ABC
from typing import Any, Generic, Iterator, Protocol, Sequence, TypeVar, overload


class NamedItem(Protocol):
    """Protocolo estrutural para elementos de pilhas que possuem nome identificador."""

    name: str


T = TypeVar("T", bound=NamedItem)


class NamedStack(Generic[T], ABC):
    """Contêiner genérico para pilhas sequenciais ordenadas de itens com nome."""

    _item_type_name: str = "Item"

    def __init__(self, items: Sequence[T] | None = None) -> None:
        self._items: list[T] = []
        if items is not None:
            self.extend(items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __reversed__(self) -> Iterator[T]:
        return reversed(self._items)

    def _contains_item(self, item: Any) -> bool:
        """Hook para checagem de pertencimento do item. Pode ser customizado por subclasses."""
        return any(x is item for x in self._items) or item in self._items

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str):
            return any(x.name == item for x in self._items)
        return self._contains_item(item)

    @overload
    def __getitem__(self, key: int) -> T:
        pass

    @overload
    def __getitem__(self, key: str) -> T:
        pass

    @overload
    def __getitem__(self, key: slice) -> list[T]:
        pass

    def __getitem__(self, key: int | slice | str) -> T | list[T]:
        if isinstance(key, (int, slice)):
            return self._items[key]
        if isinstance(key, str):
            for x in self._items:
                if x.name == key:
                    return x
            raise KeyError(
                f"{self._item_type_name} '{key}' not found in {self.__class__.__name__}"
            )
        raise TypeError(
            f"{self.__class__.__name__} indices must be integers, slices or strings, not {type(key).__name__}"
        )

    def __setitem__(self, key: int, item: T) -> None:
        if not isinstance(key, int):
            raise TypeError(
                f"{self.__class__.__name__} indices must be integers, not {type(key).__name__}"
            )
        self._validate_item(item)
        self._items[key] = item

    def __delitem__(self, key: int | str) -> None:
        if isinstance(key, int):
            del self._items[key]
        elif isinstance(key, str):
            target = self[key]
            self._items.remove(target)
        else:
            raise TypeError(
                f"{self.__class__.__name__} indices must be integers or strings, not {type(key).__name__}"
            )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NamedStack):
            return self._items == other._items
        if isinstance(other, (list, tuple)):
            return self._items == list(other)
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._items!r})"

    def _validate_item(self, item: T) -> None:
        """Hook opcional para validação de tipo nas subclasses."""
        pass

    def _matches_item(self, candidate: T, target: Any) -> bool:
        """Hook para correspondência de itens em remove e index."""
        return candidate is target or candidate == target

    def add(self, item: T) -> T:
        """Adiciona um item ao topo da pilha."""
        self._validate_item(item)
        self._items.append(item)
        return item

    def append(self, item: T) -> T:
        """Alias para add()."""
        return self.add(item)

    def extend(self, items: Sequence[T] | NamedStack[T]) -> None:
        """Adiciona múltiplos itens à pilha."""
        for x in items:
            self.add(x)

    def remove(self, item: T | str) -> None:
        """Remove um item da pilha por instância ou por nome."""
        if isinstance(item, str):
            try:
                target = self[item]
            except KeyError as err:
                raise ValueError(
                    f"{self._item_type_name} '{item}' not found in {self.__class__.__name__}"
                ) from err
            self._items.remove(target)
            return

        for x in self._items:
            if self._matches_item(x, item):
                self._items.remove(x)
                return
        raise ValueError(f"{self._item_type_name} {item} not found in {self.__class__.__name__}")

    def pop(self, index: int = -1) -> T:
        """Remove e retorna o item no índice especificado (padrão: topo)."""
        return self._items.pop(index)

    def clear(self) -> None:
        """Remove todos os itens da pilha."""
        self._items.clear()

    def insert(self, index: int, item: T) -> T:
        """Insere um item em uma posição específica da pilha."""
        self._validate_item(item)
        self._items.insert(index, item)
        return item

    def move(self, item: T | str, new_index: int) -> None:
        """Move um item para uma nova posição na pilha."""
        idx = self.index(item)
        it = self._items.pop(idx)
        self._items.insert(new_index, it)

    def swap(self, a: T | int, b: T | int) -> None:
        """Troca a posição de dois itens na pilha."""
        idx_a = a if isinstance(a, int) else self.index(a)
        idx_b = b if isinstance(b, int) else self.index(b)
        self._items[idx_a], self._items[idx_b] = (
            self._items[idx_b],
            self._items[idx_a],
        )

    def index(self, item: T | str) -> int:
        """Retorna o índice de um item por instância ou nome."""
        if isinstance(item, str):
            for i, x in enumerate(self._items):
                if x.name == item:
                    return i
            raise ValueError(
                f"{self._item_type_name} '{item}' not found in {self.__class__.__name__}"
            )

        for i, x in enumerate(self._items):
            if self._matches_item(x, item):
                return i
        raise ValueError(f"{self._item_type_name} {item} not found in {self.__class__.__name__}")
