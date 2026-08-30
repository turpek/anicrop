from __future__ import annotations
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import count
from operator import add, sub
from typing import Protocol

import numpy as np

RotationInput = float | tuple[float, float, float]
ScaleInput = float | tuple[float, float] | tuple[float, float, float, float]


_count = count()


class Id:
    def __init__(self):
        self.__id = next(_count)

    def __hash__(self):
        return hash(self.__id)

    def __eq__(self, other):
        return isinstance(other, Id) and self.__id == other.__id


class TransformState(Protocol):
    @property
    def matrix(self) -> np.ndarray:
        pass

    @property
    def pivot(self) -> tuple[float, float]:
        pass


@dataclass(frozen=True)
class Vector:
    x: float = 0
    y: float = 0

    def __abs__(self) -> Vector:
        return Vector(abs(self.x), abs(self.y))

    def __iter__(self) -> Iterator:
        return iter((self.x, self.y))


@dataclass
class Rotation:
    angle: float = 0.0
    pivot_x: float = 0.5
    pivot_y: float = 0.5

    def _apply_operation(
        self, value: RotationInput, op: Callable
    ) -> tuple[float, float, float]:
        """
        Centraliza a matemática:
        - Se for número: Aplica operação no valor, MANTÉM pivô.
        - Se for tupla: Aplica operação no valor (índice 0), SUBSTITUI pivô
        - (índices 1,2).
        """
        if isinstance(value, (float, int)):
            return (op(self.angle, value), self.pivot_x, self.pivot_y)

        elif isinstance(value, tuple):
            return (op(self.angle, value[0]), float(value[1]), float(value[2]))

        return NotImplemented

    def __add__(self, value: RotationInput) -> Rotation:
        return Rotation(*self._apply_operation(value, add))

    def __sub__(self, value: RotationInput) -> Rotation:
        return Rotation(*self._apply_operation(value, sub))

    @property
    def pivot(self) -> tuple[float, float]:
        return (self.pivot_x, self.pivot_y)

    def from_input(self, value: Rotation | RotationInput) -> Rotation:
        if isinstance(value, Rotation):
            return value
        elif isinstance(value, (float, int)):
            return Rotation(value, *self.pivot)
        elif isinstance(value, tuple):
            return Rotation(*value)
        raise NotImplementedError

    @property
    def matrix(self) -> np.ndarray:
        theta = np.radians(self.angle)
        c, s = np.cos(theta), np.sin(theta)

        m = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        return m


@dataclass
class Scale:
    sx: float
    sy: float
    pivot_x: float = 0.5
    pivot_y: float = 0.5

    def _apply_operation(
        self, value: ScaleInput, op: Callable
    ) -> tuple[float, float, float, float]:
        """
        Centraliza a matemática:
        - Se for número: Aplica operação no valor, MANTÉM pivô.
        - Se for tupla: Aplica operação no valor (índice 0), SUBSTITUI pivô
        - (índices 1,2).
        """
        if isinstance(value, (float, int)):
            return (op(self.sx, value), op(self.sy, value), self.pivot_x, self.pivot_y)

        elif isinstance(value, tuple) and len(value) == 4:
            return (
                op(self.sx, value[0]),
                op(self.sy, value[1]),
                float(value[2]),
                float(value[3]),
            )

        elif isinstance(value, tuple) and len(value) == 2:
            return (
                op(self.sx, value[0]),
                op(self.sy, value[1]),
                self.pivot_x,
                self.pivot_y,
            )

        return NotImplemented

    def __add__(self, value: ScaleInput) -> Scale:
        return Scale(*self._apply_operation(value, add))

    def __sub__(self, value: ScaleInput) -> Scale:
        return Scale(*self._apply_operation(value, sub))

    @property
    def pivot(self) -> tuple[float, float]:
        return (self.pivot_x, self.pivot_y)

    def from_input(self, value: Scale | ScaleInput):
        if isinstance(value, Scale):
            return value
        elif isinstance(value, tuple) and len(value) == 2:
            return Scale(value[0], value[1], *self.pivot)
        elif isinstance(value, tuple):
            return Scale(*value)
        elif isinstance(value, (float, int)):
            return Scale(value, value, *self.pivot)
        raise NotImplementedError

    @property
    def matrix(self) -> np.ndarray:
        """Retorna a matriz de escala PURA (em torno de 0,0)"""
        return np.array([[self.sx, 0, 0], [0, self.sy, 0], [0, 0, 1]], dtype=np.float32)
