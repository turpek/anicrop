from __future__ import annotations
from typing import TYPE_CHECKING
from anicrop.spatial import Region
from anicrop.type import Rotation, Vector


if TYPE_CHECKING:
    from anicrop.proxy import ProxyLayer




class RotationHandler:
    def __init__(self, proxy: ProxyLayer, value: Rotation | float | int):
        self._proxy = proxy
        self._value = self._normalize_rotation(value)
        self._state = proxy.rotate

    def _normalize_rotation(self, value: Rotation | float | int) -> Rotation:
        if isinstance(value, (float, int)):
            rotate = self._proxy.rotate
            delta = value - rotate.value
            return rotate + delta
        return value

    def rotate(self) -> None:
        operation = self._value.operation
        origin = self._value.origin
        self._proxy._layer.rotate = self._value
        for edit in self._proxy._edits:
            edit.rotate = operation(edit.rotate, origin)
