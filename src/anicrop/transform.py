from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Self

import numpy as np

from anicrop.spatial import Point, Region

if TYPE_CHECKING:
    from anicrop.container import BaseLayer
    from anicrop.layer import EditLayer, Layer
    from anicrop.type import TransformState

Size2D = tuple[float, float]
Point2D = tuple[float, float]


def corners_to_rect(
    min_x: float, min_y: float, max_x: float, max_y: float
) -> tuple[float, float, float, float]:
    new_w = max(1e-6, max_x - min_x)
    new_h = max(1e-6, max_y - min_y)
    return min_x, min_y, new_w, new_h


def has_distortion(matrix: np.ndarray) -> bool:
    """Retorna True se a matriz contiver distorção afim (rotação, escala != 1 ou cisalhamento)."""
    return not (
        math.isclose(float(matrix[0, 0]), 1.0, abs_tol=1e-5)
        and math.isclose(float(matrix[1, 1]), 1.0, abs_tol=1e-5)
        and math.isclose(float(matrix[0, 1]), 0.0, abs_tol=1e-5)
        and math.isclose(float(matrix[1, 0]), 0.0, abs_tol=1e-5)
        and math.isclose(float(matrix[2, 0]), 0.0, abs_tol=1e-5)
        and math.isclose(float(matrix[2, 1]), 0.0, abs_tol=1e-5)
        and math.isclose(float(matrix[2, 2]), 1.0, abs_tol=1e-5)
    )


def calculate_new_corners(
    matrix: np.ndarray,
    size: tuple[float, float],
    top_left: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float, float, float]:
    """retorna min_x, min_y, max_x, max_y"""
    x, y = top_left
    w, h = size

    corners = np.array(
        [[x, y, 1.0], [x + w, y, 1.0], [x + w, y + h, 1.0], [x, y + h, 1.0]],
        dtype=np.float32,
    ).T

    transformed_corners = matrix @ corners
    transformed_corners[0, :] /= transformed_corners[2, :]
    transformed_corners[1, :] /= transformed_corners[2, :]

    min_x = float(np.min(transformed_corners[0, :]))
    min_y = float(np.min(transformed_corners[1, :]))

    max_x = float(np.max(transformed_corners[0, :]))
    max_y = float(np.max(transformed_corners[1, :]))

    return min_x, min_y, max_x, max_y


def calculate_new_rect_smart(
    matrix: np.ndarray,
    size: tuple[float, float],
    top_left: tuple[float, float] = (0.0, 0.0),
    eps: float = 1e-5,
) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = calculate_new_corners(
        matrix,
        size,
        top_left,
    )
    new_w = max(1e-6, max_x - min_x)
    new_h = max(1e-6, max_y - min_y)
    return min_x, min_y, new_w, new_h


def calculate_new_rect(
    matrix: np.ndarray,
    size: tuple[float, float],
    eps: float = 1e-5,
) -> tuple[float, float, float, float]:
    return calculate_new_rect_smart(matrix, size, (0.0, 0.0), eps)


def calculate_region_rect(
    matrix: np.ndarray, region: Region, eps: float = 1e-5
) -> tuple[float, float, float, float]:
    return calculate_new_rect_smart(matrix, region.size, region.top_left, eps)


def calculate_new_rect_from_layer(layer) -> tuple[float, float, float, float]:
    return calculate_new_rect(mat_global(layer), layer.base.region.size)


def create_pivot_transform_abs(
    matrix_pure: np.ndarray,
    w: float,
    h: float,
    px: float = 0,
    py: float = 0,
    *args,
) -> np.ndarray:
    """Gera o Sanduíche: Ida ao Pivô -> Transformação -> Volta do Pivô"""

    # 2. Matrizes de Ida e Volta
    T_neg = mat_translation(-px, -py)
    T_pos = mat_translation(px, py)

    # 3. O Sanduíche
    return T_pos @ matrix_pure @ T_neg


def create_pivot_transform_rel(
    matrix_pure: np.ndarray,
    w: float,
    h: float,
    px_rel: float,
    py_rel: float,
    x: float = 0,
    y: float = 0,
) -> np.ndarray:
    """Gera o Sanduíche: Ida ao Pivô -> Transformação -> Volta do Pivô"""

    px, py = x + w * px_rel, y + h * py_rel
    return create_pivot_transform_abs(matrix_pure, w, h, px, py)


def mat_translation(x: float, y: float) -> np.ndarray:
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=np.float32)


def mat_position(region: Region) -> np.ndarray:
    return mat_translation(region.x.start, region.y.start)


def mat_rotation(angle: float) -> np.ndarray:
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)

    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def mat_scale(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)


def mat_pivot(transform: TransformState, size: Size2D) -> np.ndarray:
    return create_pivot_transform_rel(
        transform.matrix, size[0], size[1], *transform.pivot
    )


def mat_global(layer: BaseLayer) -> np.ndarray:
    return layer.matrix


def mat_final(layer: Layer, x: float, y: float) -> np.ndarray:
    """
    Gera a matriz de renderização final e o tamanho do buffer de destino.
    Utiliza calculate_new_rect para obter a compensação necessária.
    """
    # 2. Obtemos a matriz global
    m_glob = mat_global(layer)

    # 3. Criamos a matriz de compensação para evitar o clipping (corte)
    # Movemos o mundo de volta para a origem (0,0) do novo Rect
    m_compensation = mat_translation(-x, -y)

    # 4. A matriz final é a global 'puxada' para o topo-esquerdo do buffer
    m_render = m_compensation @ m_glob

    # Retornamos a matriz e o tamanho (arredondado para cima para evitar bordas cortadas)
    return m_render


def mat_edit_global(edit_layer: EditLayer, matrix_global: np.ndarray):
    return matrix_global @ edit_layer.local_matrix


def mat_edit_local(edit_layer: EditLayer, matrix_final: np.ndarray) -> np.ndarray:
    return matrix_final @ edit_layer.local_matrix


def mat_edit_final(edit_layer: EditLayer, matrix_final: np.ndarray):
    """matrix_final é a matriz gerada pela função `mat_final."""
    local_matrix = mat_edit_local(edit_layer, matrix_final)
    x, y, *_ = calculate_new_rect(local_matrix, edit_layer.image.size)
    m_compensation = mat_translation(-x, -y)
    return m_compensation @ local_matrix


def mat_inverse(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(matrix)


def transform_vector(
    matrix: np.ndarray,
    start_region: Region,
    target_region: Region,
) -> tuple[float, float]:
    """Calcula o vetor offset entre duas regiões e o projeta através de uma matriz 3x3 (sem translação)."""
    dx, dy = (target_region - start_region).top_left

    m_rot_scale = matrix.copy()
    m_rot_scale[:2, 2] = 0.0

    vec = m_rot_scale @ np.array([dx, dy, 1.0], dtype=np.float32)
    return float(vec[0]), float(vec[1])


class TransformBase(ABC):
    @abstractmethod
    def matrix(
        self, size: Size2D = (0.0, 0.0), top_left: Point2D = (0.0, 0.0)
    ) -> np.ndarray:
        pass


class TRotate(TransformBase):
    def __init__(
        self,
        angle: float,
        pivot_x: float = 0.5,
        pivot_y: float = 0.5,
        pivot_fn: Callable = create_pivot_transform_rel,
    ):
        self._angle = angle
        self._pivots = pivot_x, pivot_y
        self._pivot_fn = pivot_fn

    def matrix(
        self, size: Size2D = (0.0, 0.0), top_left: Point2D = (0.0, 0.0)
    ) -> np.ndarray:
        w, h = size
        x, y = top_left
        return self._pivot_fn(mat_rotation(self._angle), w, h, *self._pivots, x, y)


class TScale(TransformBase):
    def __init__(
        self,
        sx: float,
        sy: float,
        pivot_x: float = 0.5,
        pivot_y: float = 0.5,
        pivot_fn: Callable = create_pivot_transform_rel,
    ):
        if sx == 0 or sy == 0:
            raise ValueError("Scale factors cannot be zero.")
        self._sx = sx
        self._sy = sy
        self._pivots = pivot_x, pivot_y
        self._pivot_fn = pivot_fn

    def matrix(
        self, size: Size2D = (0.0, 0.0), top_left: Point2D = (0.0, 0.0)
    ) -> np.ndarray:
        w, h = size
        x, y = top_left
        return self._pivot_fn(mat_scale(self._sx, self._sy), w, h, *self._pivots, x, y)


class TTranslate(TransformBase):
    def __init__(
        self,
        x: float,
        y: float,
    ):
        self._x = x
        self._y = y

    def matrix(
        self, size: Size2D = (0.0, 0.0), top_left: Point2D = (0.0, 0.0)
    ) -> np.ndarray:
        return mat_translation(self._x, self._y)


class Composer(ABC):
    def __init__(self, size: Size2D):
        self._distortion = np.identity(3, dtype=np.float32)
        self._region = Region.from_size(size[0], size[1])
        self._translation = np.identity(3, dtype=np.float32)

    @property
    def matrix(self) -> np.ndarray:
        return self._translation @ self._distortion

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Composer):
            return False
        return (
            np.allclose(self._distortion, other._distortion, atol=1e-5)
            and np.allclose(self._translation, other._translation, atol=1e-5)
            and self._region == other._region
        )

    def copy(self) -> Self:
        """Cria uma cópia independente e profunda deste Composer."""
        new_composer = self.__class__(self.size)
        new_composer._distortion = np.copy(self._distortion)
        new_composer._translation = np.copy(self._translation)
        new_composer._region = self._region
        return new_composer

    def copy_from(self, other: Self) -> None:
        """Copia o estado de outro Composer para este objeto in-place."""
        self._distortion = np.copy(other._distortion)
        self._translation = np.copy(other._translation)
        self._region = other._region

    @property
    def size(self) -> Point:
        return self.region.size

    @property
    def region(self) -> Region:
        return self._region

    def sync_region(self, region: Region) -> None:
        """Sincroniza a região de referência deste Composer com a região ativa da camada."""
        self._region = region

    @abstractmethod
    def rotate(self, angle: float, px: float = 0.5, py: float = 0.5) -> Self:
        pass

    @abstractmethod
    def scale(self, sx: float, sy: float, px: float = 0.5, py: float = 0.5) -> Self:
        pass

    def translate(self, x: float = 0.0, y: float = 0.0) -> Self:
        M_trans = TTranslate(x, y).matrix(self.size)
        self._translation = M_trans @ self._translation
        return self

    def add_transform(
        self, transf: Transform, reference_size: Size2D | None = None
    ) -> Self:

        ref_size = reference_size or self.size
        self._distortion = transf._get_distortion(ref_size) @ self._distortion
        self._translation = transf._get_translate(ref_size) @ self._translation
        return self


class ComposerRel(Composer):
    def __init__(self, size: Size2D):
        super().__init__(size)

    def __get_rect(self) -> tuple[tuple[float, float], tuple[float, float]]:
        x, y, w, h = corners_to_rect(*calculate_new_corners(self._distortion, self.size))
        return (x, y), (w, h)

    def rotate(
        self, angle: float = 0, pivot_x: float = 0.5, pivot_y: float = 0.5
    ) -> ComposerRel:

        top_left, size = self.__get_rect()
        M_rot = TRotate(angle, pivot_x, pivot_y).matrix(size, top_left)
        self._distortion = M_rot @ self._distortion
        return self

    def scale(
        self,
        sx: float = 1,
        sy: float = 1,
        pivot_x: float = 0.5,
        pivot_y: float = 0.5,
    ) -> ComposerRel:

        top_left, size = self.__get_rect()
        M_scale = TScale(sx, sy, pivot_x, pivot_y).matrix(size, top_left)
        self._distortion = M_scale @ self._distortion
        return self


class ComposerAbs(Composer):
    def __init__(self, size: tuple[int, int]):
        super().__init__(size)

    def rotate(
        self,
        angle: float = 0,
        px: float = 0.0,
        py: float = 0.0,
    ) -> ComposerAbs:

        M_rot = TRotate(angle, px, py, pivot_fn=create_pivot_transform_abs).matrix()
        self._distortion = M_rot @ self._distortion
        return self

    def scale(
        self,
        sx: float = 1,
        sy: float = 1,
        px: float = 0.0,
        py: float = 0.0,
    ) -> ComposerAbs:

        M_scale = TScale(sx, sy, px, py, pivot_fn=create_pivot_transform_abs).matrix()
        self._distortion = M_scale @ self._distortion
        return self


class Transform(ABC):
    COMPOSER_CLS: type[Composer]

    @classmethod
    def relative(cls) -> TransformRel:
        """Fábrica para criar uma cadeia de transformações de pivô relativo."""
        return TransformRel()

    @classmethod
    def absolute(cls) -> TransformAbs:
        """Fábrica para criar uma cadeia de transformações de pivô absoluto."""
        return TransformAbs()

    def create_composer(self, size: Size2D) -> Composer:
        return self.COMPOSER_CLS(size)

    def __init__(
        self, intentions: list[TRotate | TScale] = [], translate: list[TTranslate] = []
    ):
        if self._validate_list(intentions, TTranslate):
            raise TypeError("intentions list can only contain TRotate or TScale")
        elif self._validate_list(translate, (TRotate, TScale)):
            raise TypeError("translate list can only contain TTranslate")

        self._intentions = intentions
        self._translate = translate

    def _validate_list(self, transf: list, cls_types: type | tuple[type, ...]) -> bool:
        for op in transf:
            if isinstance(op, cls_types):
                return True
        return False

    def _check_transform_list(self, transf: Sequence[TransformBase]) -> bool:
        return len(transf) == 0 or len(transf) == 1

    def translate(self, x: float = 0.0, y: float = 0.0) -> Self:
        new_translate = self._translate + [TTranslate(x, y)]
        return self.__class__(self._intentions, new_translate)

    @property
    def has_distortion(self) -> bool:
        return self._validate_list(self._intentions, (TRotate, TScale))

    @abstractmethod
    def _list_to_matrix(
        self,
        size: Size2D,
        matrix_list: Sequence[TransformBase],
    ) -> np.ndarray:
        pass

    def _get_first_transform(
        self,
        size: Size2D,
        transf: Sequence[TransformBase],
    ) -> np.ndarray:
        if len(transf) == 0:
            return np.identity(3, dtype=np.float32)
        return transf[0].matrix(size)

    def _get_translate(self, size: Size2D = (0, 0)) -> np.ndarray:
        if self._check_transform_list(self._translate):
            return self._get_first_transform(size, self._translate)
        return self._list_to_matrix(size, self._translate)

    def _get_distortion(self, size: Size2D = (0, 0)) -> np.ndarray:
        if self._check_transform_list(self._intentions):
            return self._get_first_transform(size, self._intentions)
        return self._list_to_matrix(size, self._intentions)

    def get_matrix(self, size: Size2D = (0, 0)) -> np.ndarray:
        M_trans = self._get_translate(size)
        M_intent = self._get_distortion(size)
        return M_trans @ M_intent

    @abstractmethod
    def rotate(self, *args, **kwargs) -> Self:
        pass

    @abstractmethod
    def scale(self, *args, **kwargs) -> Self:
        pass


class TransformRel(Transform):
    COMPOSER_CLS = ComposerRel

    def __init__(
        self,
        intentions: list[TRotate | TScale] = [],
        translate: list[TTranslate] = [],
    ):
        super().__init__(intentions, translate)

    def __get_rect(self, matrix: np.ndarray, size: Size2D):
        x, y, w, h = corners_to_rect(*calculate_new_corners(matrix, size))
        return (x, y), (w, h)

    def _list_to_matrix(
        self,
        size: Size2D,
        matrix_list: Sequence[TransformBase],
    ) -> np.ndarray:

        top_left: Point2D = (0.0, 0.0)
        m_total = np.identity(3, dtype=np.float32)
        current_size: Size2D = size
        for op in matrix_list:
            m_total = op.matrix(current_size, top_left) @ m_total
            top_left, current_size = self.__get_rect(m_total, size)
        return m_total

    def rotate(
        self,
        angle: float = 0,
        pivot_x: float = 0.5,
        pivot_y: float = 0.5,
    ) -> TransformRel:

        new_intentions = self._intentions + [TRotate(angle, pivot_x, pivot_y)]
        return TransformRel(new_intentions, self._translate)

    def scale(
        self, sx: float = 1, sy: float = 1, pivot_x: float = 0.5, pivot_y: float = 0.5
    ) -> TransformRel:

        new_intentions = self._intentions + [TScale(sx, sy, pivot_x, pivot_y)]
        return TransformRel(new_intentions, self._translate)


class TransformAbs(Transform):
    COMPOSER_CLS = ComposerAbs

    def __init__(
        self, intentions: list[TRotate | TScale] = [], translate: list[TTranslate] = []
    ):
        super().__init__(intentions, translate)

    def _list_to_matrix(
        self,
        size: Size2D,
        matrix_list: Sequence[TransformBase],
    ) -> np.ndarray:
        m_total = np.identity(3, dtype=np.float32)
        for op in matrix_list:
            m_total = op.matrix(size) @ m_total
        return m_total

    def rotate(
        self,
        angle: float = 0,
        px: float = 0.0,
        py: float = 0.0,
    ) -> TransformAbs:

        new_intentions = self._intentions + [
            TRotate(angle, px, py, pivot_fn=create_pivot_transform_abs)
        ]
        return TransformAbs(new_intentions, self._translate)

    def scale(
        self,
        sx: float = 1,
        sy: float = 1,
        px: float = 0.0,
        py: float = 0.0,
    ) -> TransformAbs:

        new_intentions = self._intentions + [
            TScale(sx, sy, px, py, pivot_fn=create_pivot_transform_abs)
        ]
        return TransformAbs(new_intentions, self._translate)
