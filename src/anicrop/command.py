from __future__ import annotations
from abc import ABC, abstractmethod
from anicrop.layer import Layer
from anicrop.spatial import Span, Region
from anicrop.transform import calculate_new_bbox_from_layer
from anicrop.type import RotationInput, ScaleInput, Transform
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


class SetAttributeCommand(Command):

    def __init__(
            self,
            name: str,
            layer: Layer,
            value: RotationInput | ScaleInput | Transform
    ):
        super().__init__(name, layer, value)
        self._old_state = getattr(layer, name)

    def execute(self) -> None:
        setattr(self._layer, self._name, self._new_state)

    def undo(self) -> None:
        setattr(self._layer, self._name, self._old_state)

    def update_value(self, value: Any) -> None:
        self._new_state = value

    def __repr__(self):
        return f'{type(self).__name__}(name="{self._name}")'
