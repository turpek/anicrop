from unittest.mock import MagicMock
import numpy as np
import pytest

from anicrop.container import GroupLayer
from anicrop.geometry import FitGroupGeometry, GeometryController, GroupGeometry, LayerGeometry
from anicrop.layer import Layer
from anicrop.spatial import Region
from anicrop.transform import mat_translation


def make_layer_mock(
    x: int = 50,
    y: int = 50,
    w: int = 100,
    h: int = 100,
    parent_matrix: np.ndarray = np.array([
        [1.0, 0.0, 200.0],
        [0.0, 1.0, 200.0],
        [0.0, 0.0, 1.0]
    ]),
    transform_matrix: np.ndarray = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0]
    ]),
) -> MagicMock:
    """Cria um mock de Layer sem atributo region solto, usando geometry e native_geometry."""
    layer_mock = MagicMock(spec=Layer)

    layer_mock.parent = MagicMock()
    layer_mock.parent.matrix = parent_matrix

    layer_mock.transform = MagicMock()
    layer_mock.transform.matrix = transform_matrix

    geom_mock = MagicMock()
    geom_mock.region = Region.from_rect(x, y, w, h)

    # Arbitrary mock for global_region to test union behavior
    geom_mock.global_region = Region.from_rect(x * 2, y * 2, w * 2, h * 2)

    layer_mock.layout = geom_mock
    layer_mock.base = geom_mock
    layer_mock.region = geom_mock.region
    layer_mock.global_region = geom_mock.global_region

    return layer_mock


def make_group_mock() -> MagicMock:
    """Cria um mock de GroupLayer contendo um subgrupo e layers para testar hierarquia."""
    group_mock = MagicMock(spec=GroupLayer)

    group_mock.parent = MagicMock()
    group_mock.parent.matrix = np.array([
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 10.0],
        [0.0, 0.0, 1.0]
    ])

    group_mock.transform = MagicMock()
    group_mock.transform.matrix = np.identity(3)
    group_mock._parent_inverse = np.identity(3)

    # Sub group
    sub_group = MagicMock(spec=GroupLayer)

    # Layers (Mock region: (10, 10, 50, 50) and (100, 100, 50, 50))
    layer1 = make_layer_mock(10, 10, 50, 50)
    layer2 = make_layer_mock(100, 100, 50, 50)

    sub_group.__iter__.return_value = [layer2]
    sub_group.__len__.return_value = 1

    sub_geom_mock = MagicMock()
    sub_geom_mock.region = layer2.layout.region
    sub_geom_mock.global_region = layer2.layout.global_region
    sub_group.layout = sub_geom_mock
    sub_group.base = sub_geom_mock
    sub_group.region = sub_geom_mock.region
    sub_group.global_region = sub_geom_mock.global_region

    group_mock.__iter__.return_value = [layer1, sub_group]
    group_mock.__len__.return_value = 2

    return group_mock


LAYER_EXPECTED_MATRIX = np.array([
    [2.0, 0.0, 250.0],
    [0.0, 2.0, 250.0],
    [0.0, 0.0, 1.0]
])

GROUP_EXPECTED_MATRIX = np.array([
    [1.0, 0.0, 10.0],
    [0.0, 1.0, 10.0],
    [0.0, 0.0, 1.0]
])


@pytest.mark.parametrize("geometry_cls, make_mock, init_region, expected_matrix", [
    (LayerGeometry, make_layer_mock, (50, 50, 100, 100), LAYER_EXPECTED_MATRIX),
    (GroupGeometry, make_group_mock, (0, 0, 1, 1), GROUP_EXPECTED_MATRIX),
], ids=["LayerGeometry", "GroupGeometry"])
def test_geometry_matrix(geometry_cls, make_mock, init_region, expected_matrix):
    """A propriedade matrix retorna a matriz global correta."""
    base_mock = make_mock()
    region = Region.from_rect(*init_region)

    strategy = geometry_cls(base_mock, region)

    np.testing.assert_array_almost_equal(strategy.matrix, expected_matrix)


@pytest.mark.parametrize("geometry_cls, make_mock, init_region, expected_bbox", [
    (LayerGeometry, make_layer_mock, (50, 50, 100, 100), (250, 250, 200, 200)),
    (GroupGeometry, make_group_mock, (0, 0, 1, 1), (20, 20, 280, 280)),
], ids=["LayerGeometry", "GroupGeometry"])
def test_geometry_global_region(geometry_cls, make_mock, init_region, expected_bbox):
    """Garante que a property global_region retorna a bbox correta."""
    base_mock = make_mock()
    region = Region.from_rect(*init_region)

    strategy = geometry_cls(base_mock, region)

    assert strategy.global_region == Region.from_rect(*expected_bbox)


@pytest.mark.parametrize("geometry_cls, make_mock, init_region, expected_bbox", [
    (LayerGeometry, make_layer_mock, (50, 50, 100, 100), (50, 50, 100, 100)),
    (GroupGeometry, make_group_mock, (0, 0, 1, 1), (10, 10, 140, 140)),
], ids=["LayerGeometry", "GroupGeometry"])
def test_geometry_region(geometry_cls, make_mock, init_region, expected_bbox):
    """Garante que a property region retorna a região local ou união das regiões dos filhos."""
    base_mock = make_mock()
    region = Region.from_rect(*init_region)

    strategy = geometry_cls(base_mock, region)

    assert strategy.region == Region.from_rect(*expected_bbox)


def test_geometry_controller_sync_on_coordinate_mutation():
    """Valida se o GeometryController sincroniza as geometrias base e layout ao mutar coordenadas."""
    mock_layer = MagicMock(spec=Layer)
    base_geom = LayerGeometry(mock_layer, Region.from_rect(0, 0, 100, 100))
    layout_geom = LayerGeometry(mock_layer, Region.from_rect(5, 5, 100, 100))

    controller = GeometryController(base_geom, layout_geom)

    # Mutação de coordenadas da região de layout via controller.sync
    new_layout_region = Region.from_rect(10, 0, 100, 100)
    controller.sync(new_layout_region)

    # Verifica se ambas as estratégias (layout e base) foram sincronizadas proporcionalmente
    assert controller.layout.region == Region.from_rect(10, 0, 100, 100)
    assert controller.base.region == Region.from_rect(5, -5, 100, 100)


def test_fit_group_geometry_region_and_global_region():
    """Valida se FitGroupGeometry armazena a região local no pai e projeta a global_region no Canvas."""
    group_mock = MagicMock(spec=GroupLayer)
    group_mock.parent = MagicMock()
    # Pai com translação de (20, 30)
    group_mock.parent.matrix = np.array([
        [1.0, 0.0, 20.0],
        [0.0, 1.0, 30.0],
        [0.0, 0.0, 1.0]
    ])
    group_mock.transform = MagicMock()
    group_mock.transform.matrix = np.identity(3)

    # Moldura passada no Canvas: (50, 50, 150, 100)
    ref_region = Region.from_rect(50, 50, 150, 100)
    strategy = FitGroupGeometry(group_mock, ref_region)

    # 1. Região local no espaço do grupo pai (subtrai 20, 30): Region(30, 20, 150, 100)
    assert strategy.region == Region.from_rect(30, 20, 150, 100)

    # 2. Região global no Canvas (re-projeta através do pai): Region(50, 50, 150, 100)
    assert strategy.global_region == Region.from_rect(50, 50, 150, 100)


def test_fit_group_geometry_matrix():
    """Valida se a matriz da FitGroupGeometry é composta pela matriz do pai e transformação do grupo."""
    group_mock = MagicMock(spec=GroupLayer)
    group_mock.parent = MagicMock()
    group_mock.parent.matrix = np.array([
        [1.0, 0.0, 20.0],
        [0.0, 1.0, 30.0],
        [0.0, 0.0, 1.0]
    ])
    group_mock.transform = MagicMock()
    group_mock.transform.matrix = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    ref_region = Region.from_rect(0, 0, 100, 100)
    strategy = FitGroupGeometry(group_mock, ref_region)

    expected_matrix = np.array([
        [2.0, 0.0, 20.0],
        [0.0, 2.0, 30.0],
        [0.0, 0.0, 1.0]
    ])
    np.testing.assert_array_almost_equal(strategy.matrix, expected_matrix)
