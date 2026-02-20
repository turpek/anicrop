from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from anicrop.spatial import Region
    from anicrop.layer import Layer
    from anicrop.type import Transform


def calculate_new_bbox(m_global: np.ndarray, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """
    Calcula o Axis-Aligned Bounding Box (AABB) projetando os 4 cantos
    locais da imagem através da matriz global.
    """
    w, h = size
    # 1. Definir os 4 cantos originais (Locais) [x, y, 1]
    corners = np.array([
        [0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]
    ], dtype=np.float32)

    # 2. Projetar cantos para o Espaço Global (Mundo)
    projected = corners @ m_global.T

    # 3. Encontrar os limites Min/Max com arredondamento conservador
    min_xy = np.min(projected[:, :2], axis=0)
    max_xy = np.max(projected[:, :2], axis=0)

    # Arredondamento Enveloping: expande para fora para garantir que caiba na grade de pixels
    x = int(np.floor(min_xy[0]))
    y = int(np.floor(min_xy[1]))
    width = int(np.ceil(max_xy[0])) - x
    height = int(np.ceil(max_xy[1])) - y

    return (x, y, width, height)


def calculate_new_bbox_from_layer(layer) -> tuple[float, float, float, float]:
    return calculate_new_bbox(mat_global(layer), layer.image.size)


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


def mat_pivot(transform: Transform, size: tuple[int, int]) -> np.ndarray:
    return create_pivot_transform(transform.matrix, *size, *transform.pivot)


def mat_global(layer: Layer) -> np.ndarray:
    m_translation = mat_position(layer.region)
    m_rotation = mat_pivot(layer.rotation, layer.image.size)
    m_scale = mat_pivot(layer.scale, layer.image.size)
    return m_translation @ m_rotation @ m_scale


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


def mat_inverse(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(matrix)
