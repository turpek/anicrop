from __future__ import annotations
from operator import add, sub, mul, truediv
from typing import Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class Vector:
    x: float = 0
    y: float = 0

    def __abs__(self) -> Vector:
        return Vector(abs(self.x), abs(self.y))


class OperationFloat(float):
    """
    Um float que lembra qual operação matemática o criou.
    """
    def __new__(cls, value, operation=None, origin_value=None):
        return super().__new__(cls, value)

    def __init__(self, value, operation=None, origin_value=None):
        self.operation = operation
        self.origin_value = origin_value

    def __add__(self, other):
        result = super().__add__(other)
        return OperationFloat(result, operation=add, origin_value=float(other))

    def __radd__(self, other):
        result = super().__radd__(other)
        return OperationFloat(result, operation=add, origin_value=float(other))

    def __sub__(self, other):
        result = super().__sub__(other)
        return OperationFloat(result, operation=sub, origin_value=float(other))

    def __rsub__(self, other):
        result = super().__rsub__(other)
        return OperationFloat(result, operation=sub, origin_value=float(other))

    def __mul__(self, other):
        result = super().__mul__(other)
        return OperationFloat(result, operation=mul, origin_value=float(other))

    def __rmul__(self, other):
        result = super().__rmul__(other)
        return OperationFloat(result, operation=mul, origin_value=float(other))

    def __truediv__(self, other):
        res = super().__truediv__(other)
        return OperationFloat(res, operation=truediv, origin_value=float(other))

    def __rtruediv__(self, other):
        raise NotImplementedError
        # res = super().__rtruediv__(other)
        # return OperationFloat(res, operation=truediv, origin_value=float(other))

    def __repr__(self):
        return f"{float(self)}"


class Rotation:
    def __init__(self, value: float = 0.0, pivo: Vector = Vector(0, 0), operation=None, origin: Optional[float] = None):
        self._value = float(value)
        self._operation = operation
        self._origin = origin
        self._pivo = pivo

    def _normalize(self, other: float | tuple[float, float, float]):
        if isinstance(other, tuple):
            origin = float(other[0])
            pivo = Vector(other[1], other[2])
            return origin, pivo
        return float(other), Vector(0, 0)

    def __eq__(self, other):
        return self._value == float(other)

    def __add__(self, other: float | tuple[float, float, float]):
        origin, pivo = self._normalize(other)
        return Rotation(self._value + origin, pivo=pivo, operation=add, origin=origin)

    def __sub__(self, other: float):
        origin, pivo = self._normalize(other)
        return Rotation(self._value - origin, pivo=pivo, operation=sub, origin=origin)

    def __mul__(self, other: float):
        origin, pivo = self._normalize(other)
        return Rotation(self._value * origin, pivo=pivo, operation=mul, origin=origin)

    def __truediv__(self, other: float):
        origin, pivo = self._normalize(other)
        return Rotation(self._value / origin, pivo=pivo, operation=truediv, origin=origin)

    def __float__(self) -> float:
        return self._value

    @property
    def pivo(self) -> Vector:
        return self._pivo

    @property
    def operation(self):
        return self._operation

    @property
    def origin(self):
        return self._origin

    @property
    def value(self):
        return self._value

    def __repr__(self):
        return f'{self.__class__.__name__}(Angle=({self._value}), Pivo=({self.pivo.x},{self.pivo.y}))'
