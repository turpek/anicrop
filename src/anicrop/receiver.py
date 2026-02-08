from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from anicrop.spatial import Region
from anicrop.type import Rotation, Vector


if TYPE_CHECKING:
    from anicrop.proxy import ProxyLayer


def calculate_new_top_left(base_pos: tuple[int, int], region: Region, rotation: Rotation) -> Vector:
    x, y = base_pos
    width, height = region.width, region.height

    # 1. Definir os 4 cantos originais (shape 2x4 para vetorização)
    # top-left, top-right, bottom-right, bottom-left
    corners = np.array([
        [x, x + width, x + width, x],
        [y, y, y + height, y + height]
    ], dtype=float)

    # 2. Calcular Pivô Absoluto
    pivo_vec = rotation.pivot
    # Pivot relativo e absoluto (shape 2x1 para broadcasting)
    pivot_rel = np.array([[pivo_vec.x], [pivo_vec.y]], dtype=float)
    base_tl = np.array([[x], [y]], dtype=float)
    size_arr = np.array([[width], [height]], dtype=float)

    pivot_abs = base_tl + (size_arr * pivot_rel)

    # 3. Matriz de Rotação
    theta = np.radians(rotation.value)
    c, s = np.cos(theta), np.sin(theta)

    rotation_matrix = np.array([
        [c, -s],
        [s, c]
    ])

    # 4. Rotacionar todos os cantos
    vectors_to_corners = corners - pivot_abs
    rotated_vectors = rotation_matrix @ vectors_to_corners
    new_corners = pivot_abs + rotated_vectors

    # 5. Encontrar o novo AABB (min_x, min_y)
    min_x = np.min(new_corners[0, :])
    min_y = np.min(new_corners[1, :])

    return Vector(float(min_x), float(min_y))


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
