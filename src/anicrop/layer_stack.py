from anicrop.layer import Layer
from typing import Optional


class LayerStack:

    def __init__(self):
        self._stack = []

    def __len__(self) -> int:
        return len(self._stack)

    def __iter__(self) -> iter:
        return iter(self._stack)

    def __normalize_index(self, index: int) -> int:
        index = index if index >= 0 else len(self) + index
        if index >= len(self) or index < 0:
            raise IndexError('stack index out of range')
        return index

    def add(self, layer: Layer, index: Optional[int] = None) -> None:
        if layer in self._stack:
            raise ValueError(
                "Layer instance already exists in the LayerStack. "
                "Use layer.clone() to duplicate."
            )

        elif index is None:
            self._stack.append(layer)

        else:
            self._stack.insert(index, layer)

    def get(self, index: int) -> Layer:
        return self._stack[index]

    def remove(self, index: int | Layer) -> Layer:
        layer = self.get(index) if isinstance(index, int) else index
        self._stack.remove(layer)
        return layer

    def __swap(self, a: int, b: int) -> None:
        self._stack[a], self._stack[b] = self._stack[b], self._stack[a]

    def swap(self, index_a: int, index_b: int) -> None:
        a, b = index_a, index_b
        self.__swap(a, b)

    def move_up(self, index: int) -> None:
        index = self.__normalize_index(index)
        if len(self) > (index + 1):
            self.__swap(index, index + 1)

    def move_down(self, index: int) -> None:
        index = self.__normalize_index(index)
        if index != 0:
            self.__swap(index, index - 1)

    def move_to_front(self, index: int) -> None:
        layer = self.remove(index)
        self.add(layer)

    def move_to_back(self, index: int) -> None:
        layer = self.remove(index)
        self.add(layer, 0)
