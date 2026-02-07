from anicrop.layer import EditLayer, Layer
from anicrop.type import OperationFloat


class RotationHandler:
    def __init__(self, layer: Layer, edits: list[EditLayer], value: OperationFloat):
        self.__layer = layer
        self.__edits = edits
        self.__value = value
        self.__state = layer.rotate

    def rotate(self) -> None:
        operation = self.__value.operation
        origin = self.__value.origin_value
        self.__layer.rotate = self.__value
        for edit in self.__edits:
            edit.rotate = operation(edit.rotate, origin)
