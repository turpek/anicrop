from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Vector:
    x: float = 0
    y: float = 0

    def __abs__(self) -> Vector:
        return Vector(abs(self.x), abs(self.y))

    def __iter__(self) -> tuple[float, float]:
        return iter((self.x, self.y))


@dataclass
class Rotation:
    def __init__(self, value: float = 0.0, pivot: Vector = Vector(0.5, 0.5), operation=None, origin: Optional[float] = None):
        self._value = float(value)
        self._operation = operation
        self._origin = origin
        self._pivot = pivot

    def _normalize(self, other: float | tuple[float, float, float]):
        if isinstance(other, tuple):
            origin = float(other[0])
            pivot = Vector(other[1], other[2])
            return origin, pivot
        return float(other), Vector(0.5, 0.5)

    def __eq__(self, other):
        return self._value == float(other)

    def __add__(self, other: float | tuple[float, float, float]):
        origin, pivot = self._normalize(other)
        return Rotation(self._value + origin, pivot=pivot, operation=add, origin=origin)

    def __sub__(self, other: float):
        origin, pivot = self._normalize(other)
        return Rotation(self._value - origin, pivot=pivot, operation=sub, origin=origin)

    def __mul__(self, other: float):
        origin, pivot = self._normalize(other)
        return Rotation(self._value * origin, pivot=pivot, operation=mul, origin=origin)

    def __truediv__(self, other: float):
        origin, pivot = self._normalize(other)
        return Rotation(self._value / origin, pivot=pivot, operation=truediv, origin=origin)

    def __float__(self) -> float:
        return self._value

    @property
    def pivot(self) -> Vector:
        return self._pivot

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
        return f'{self.__class__.__name__}(Angle=({self._value}), pivot=({self.pivot.x},{self.pivot.y}))'


class TransDelta:
    def __init__(self, value: int):
        self._v = value

    def __add__(self, other: int) -> None:
        return self.__class__(self._v + int(other))

    def __sub__(self, other: int) -> None:
        return self.__class__(self._v - int(other))

    def __int__(self):
        return self._v

    def __eq__(self, other: int) -> bool:
        return self._v == int(other)

    def __repr__(self) -> str:
        return f'{self._v}'


class Translation:

    def __init__(self, x: int = 0, y: int = 0):
        self._x = TransDelta(x)
        self._y = TransDelta(y)

    def _normalize(self, value: float | tuple[float, float, float]) -> tuple[float, float]:
        if isinstance(value, int):
            return (value, value)
        elif isinstance(value, tuple):
            return value
        elif isinstance(value, Vector):
            return value.x, value.y
        raise NotImplementedError(f"Unsupported type: {type(value).__name__}")

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(x={self._x}, y={self._y})'

    def __add__(self, value: int | tuple[int, int]) -> None:
        dx, dy = self._normalize(value)
        return Translation(self.x + dx, self.y + dy)

    def __sub__(self, value: int | tuple[int, int]) -> None:
        dx, dy = self._normalize(value)
        return Translation(self.x - dx, self.y - dy)

    def __eq__(self, other: Translation | Vector) -> bool:
        if not isinstance(other, (self.__class__, Vector)):
            raise NotImplementedError
        return self.x == other.x and self.y == other.y

    @property
    def x(self) -> TransDelta:
        return self._x

    @x.setter
    def x(self, value: int) -> None:
        self._x = value

    @property
    def y(self) -> TransDelta:
        return self._y

    @y.setter
    def y(self, value: int) -> None:
        self._y = value
