from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class Command(ABC):
    @abstractmethod
    def execute(self) -> None:
        ...

    @abstractmethod
    def undo(self) -> None:
        ...

    @abstractmethod
    def update_value(self, value: Any) -> bool:
        ...

    def can_merge(self, command_cls: Command, layer: object) -> bool:
        return isinstance(self, command_cls) and self._layer == layer
