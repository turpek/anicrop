from anicrop.layer import EditLayer, Layer
from anicrop.type import Rotation


class RotationHandler:
    def __init__(self, layer: Layer, edits: list[EditLayer], value: Rotation):
        self._layer = layer
        self._edits = edits
        self._value = value
        self._state = layer.rotate

    def rotate(self) -> None:
        operation = self._value.operation
        origin = self._value.origin
        self._layer.rotate = self._value
        for edit in self._edits:
            edit.rotate = operation(edit.rotate, origin)
