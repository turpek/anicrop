from __future__ import annotations
from anicrop.spatial import Region, bbox_to_region
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anicrop.layer import EditLayer, Layer
    from anicrop.type import TransformState


def calculate_new_bbox(matrix: np.ndarray, size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size

    corners = np.array([
        [0, 0, 1.0],
        [w, 0, 1.0],
        [w, h, 1.0],
        [0, h, 1.0]
    ], dtype=np.float32).T

    # Multiplica os cantos pela matriz de transformação
    transformed_corners = matrix @ corners

    # Normaliza (divide por Z, se houver projeção 3D/perspectiva)
    transformed_corners[0, :] /= transformed_corners[2, :]
    transformed_corners[1, :] /= transformed_corners[2, :]

    # breakpoint()
    # Acha os novos limites (usando o round para o vizinho mais próximo)
    min_x = int(np.floor(np.min(transformed_corners[0, :])))
    min_y = int(np.floor(np.min(transformed_corners[1, :])))
    max_x = int(np.ceil(np.max(transformed_corners[0, :])))
    max_y = int(np.ceil(np.max(transformed_corners[1, :])))

    # A nova largura/altura soma +1 porque (max - min) de índices conta os intervalos.
    # Ex: (99 - 0) = 99 intervalos, o que significa 100 pixels de largura real!
    new_w = max(1, max_x - min_x)
    new_h = max(1, max_y - min_y)

    return min_x, min_y, new_w, new_h


def calculate_region_bbox(matrix: np.ndarray, region: Region) -> tuple[int, int, int, int]:
    x, y = region.top_left
    w, h = region.size

    # Agora os cantos consideram a posição inicial (x, y) exata da Região,
    # e não apenas a largura e altura partindo do zero.
    corners = np.array([
        [x, y, 1.0],
        [x + w, y, 1.0],
        [x + w, y + h, 1.0],
        [x, y + h, 1.0]
    ], dtype=np.float32).T

    transformed_corners = matrix @ corners

    # Normaliza (divide por Z, se houver projeção 3D/perspectiva)
    transformed_corners[0, :] /= transformed_corners[2, :]
    transformed_corners[1, :] /= transformed_corners[2, :]

    # Acha os novos limites (usando o floor/ceil para garantir cobertura completa)
    min_x = int(np.floor(np.min(transformed_corners[0, :])))
    min_y = int(np.floor(np.min(transformed_corners[1, :])))
    max_x = int(np.ceil(np.max(transformed_corners[0, :])))
    max_y = int(np.ceil(np.max(transformed_corners[1, :])))

    # A nova largura/altura soma +1 porque (max - min) de índices conta os intervalos.
    # Ex: (99 - 0) = 99 intervalos, o que significa 100 pixels de largura real!
    new_w = max(1, max_x - min_x)
    new_h = max(1, max_y - min_y)

    return min_x, min_y, new_w, new_h


def calculate_new_bbox_from_layer(layer) -> tuple[float, float, float, float]:
    return calculate_new_bbox(mat_global(layer), layer.region.size)


def create_pivot_transform(matrix_pure: np.ndarray, w: float, h: float, px_rel: float, py_rel: float) -> np.ndarray:
    """Gera o Sanduíche: Ida ao Pivô -> Transformação -> Volta do Pivô"""

    # 1. Calcula pivô em pixels
    px, py = w * px_rel, h * py_rel

    # 2. Matrizes de Ida e Volta
    T_neg = np.array([[1, 0, -px], [0, 1, -py], [0, 0, 1]], dtype=np.float32)
    T_pos = np.array([[1, 0, px], [0, 1, py], [0, 0, 1]], dtype=np.float32)

    # 3. O Sanduíche
    return T_pos @ matrix_pure @ T_neg


def mat_translation(x: float, y: float) -> np.ndarray:
    return np.array([
        [1, 0, x],
        [0, 1, y],
        [0, 0, 1]
    ], dtype=np.float32)


def mat_position(region: Region) -> np.ndarray:
    return mat_translation(region.x.start, region.y.start)


def mat_rotation(angle: float) -> np.ndarray:
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=np.float32)


def mat_scale(sx: float, sy: float) -> np.ndarray:
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ], dtype=np.float32)


def mat_pivot(transform: TransformState, size: tuple[int, int]) -> np.ndarray:
    return create_pivot_transform(transform.matrix, *size, *transform.pivot)


def mat_global_state(layer: Layer) -> np.ndarray:
    m_translation = mat_position(layer.region)
    m_rotation = mat_pivot(layer.rotation, layer.region.size)
    m_scale = mat_pivot(layer.scale, layer.region.size)
    return m_translation @ m_rotation @ m_scale


def mat_global_transform(layer: Layer) -> np.ndarray:
    return mat_position(layer.region) @ layer.transform.matrix


def mat_global(layer: Layer) -> np.ndarray:
    if layer.transform_used:
        return mat_global_transform(layer)
    return mat_global_state(layer)


def mat_final(layer: Layer, x: float, y: float) -> np.ndarray:
    """
    Gera a matriz de renderização final e o tamanho do buffer de destino.
    Utiliza calculate_new_bbox para obter a compensação necessária.
    """
    # 2. Obtemos a matriz global
    m_glob = mat_global(layer)

    # 3. Criamos a matriz de compensação para evitar o clipping (corte)
    # Movemos o mundo de volta para a origem (0,0) do novo BBox
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
    x, y, *_ = calculate_new_bbox(local_matrix, edit_layer.image.size)
    m_compensation = mat_translation(-x, -y)
    return m_compensation @ local_matrix


def mat_inverse(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(matrix)


class Transform:

    def __init__(self, size: tuple[int, int]):
        self._matrix = np.identity(3, dtype=np.float32)
        self._region = Region.from_size(*size)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def size(self) -> tuple[int, int]:
        return self.region.size

    @property
    def region(self) -> Region:
        return self._region

    def rotation(
        self,
        angle: float = 0,
        pivot_x: float = 0.5, pivot_y: float = 0.5
    ) -> Transform:

        w, h = self.size
        M_rot = create_pivot_transform(
            mat_rotation(angle), w, h, pivot_x, pivot_y
        )
        self._matrix = M_rot @ self.matrix
        return self

    def scale(
        self,
        sx: float = 1, sy: float = 1,
        pivot_x: float = 0.5, pivot_y: float = 0.5
    ) -> Transform:

        if sx == 0 or sy == 0:
            raise ValueError("Scale factors cannot be zero.")

        w, h = self.size
        M_scale = create_pivot_transform(
            mat_scale(sx, sy), w, h, pivot_x, pivot_y
        )
        self._matrix = M_scale @ self.matrix
        return self

    def translate(self, x: int = 0, y: int = 0) -> Transform:
        M_trans = mat_translation(x, y)
        self._matrix = M_trans @ self.matrix
        return self
