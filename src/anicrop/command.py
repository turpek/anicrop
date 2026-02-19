from __future__ import annotations
from abc import ABC, abstractmethod
from anicrop.layer import Layer
from typing import Any


class Command(ABC):

    def __init__(self, name: str, layer: Layer, value: Any):
        self._sealed = False
        self._layer = layer
        self._new_state = value
        self._name = name

    @abstractmethod
    def execute(self) -> None:
        ...

    @abstractmethod
    def undo(self) -> None:
        ...

    @abstractmethod
    def update_value(self, value: Any) -> None:
        ...

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    def seal(self) -> None:
        self._sealed = True

    def can_merge(self, name: str, layer: object) -> bool:
        if self.is_sealed:
            return False
        return self._name == name and self._layer == layer
