from unittest.mock import MagicMock

import numpy as np
import pytest

from anicrop.container import GroupLayer, freeze_geometry
from anicrop.enums import ImageFormat
from anicrop.geometry import (
    FitGroupGeometry,
    GeometryController,
    GroupGeometry,
    LayerGeometry,
)
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.spatial import Region


def make_image_mock(
    w: int = 50, h: int = 50, fmt: ImageFormat = ImageFormat.RGBA
) -> Image:
    mock_array = MagicMock(spec=np.ndarray)
    mock_array.ndim = 3
    mock_array.shape = (h, w, fmt.channels)
    return Image(mock_array, fmt)


def make_layer_mock(
    x: int = 50,
    y: int = 50,
    w: int = 100,
    h: int = 100,
    parent_matrix: np.ndarray = np.array(
        [[1.0, 0.0, 200.0], [0.0, 1.0, 200.0], [0.0, 0.0, 1.0]]
    ),
    transform_matrix: np.ndarray = np.array(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
    ),
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

    layer_mock.frame = geom_mock
    layer_mock.base = geom_mock
    layer_mock.region = geom_mock.region
    layer_mock.global_region = geom_mock.global_region

    return layer_mock


def make_group_mock() -> MagicMock:
    """Cria um mock de GroupLayer contendo um subgrupo e layers para testar hierarquia."""
    group_mock = MagicMock(spec=GroupLayer)

    group_mock.parent = MagicMock()
    group_mock.parent.matrix = np.array(
        [[1.0, 0.0, 10.0], [0.0, 1.0, 10.0], [0.0, 0.0, 1.0]]
    )

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
    sub_geom_mock.region = layer2.frame.region
    sub_geom_mock.global_region = layer2.frame.global_region
    sub_group.frame = sub_geom_mock
    sub_group.base = sub_geom_mock
    sub_group.region = sub_geom_mock.region
    sub_group.global_region = sub_geom_mock.global_region

    group_mock.__iter__.return_value = [layer1, sub_group]
    group_mock.__len__.return_value = 2

    return group_mock


LAYER_EXPECTED_MATRIX = np.array([[2.0, 0.0, 250.0], [0.0, 2.0, 250.0], [0.0, 0.0, 1.0]])

GROUP_EXPECTED_MATRIX = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 10.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize(
    "geometry_cls, make_mock, init_region, expected_matrix",
    [
        (LayerGeometry, make_layer_mock, (50, 50, 100, 100), LAYER_EXPECTED_MATRIX),
        (GroupGeometry, make_group_mock, (0, 0, 1, 1), GROUP_EXPECTED_MATRIX),
    ],
    ids=["LayerGeometry", "GroupGeometry"],
)
def test_geometry_matrix(geometry_cls, make_mock, init_region, expected_matrix):
    """A propriedade matrix retorna a matriz global correta."""
    base_mock = make_mock()
    region = Region.from_rect(*init_region)

    strategy = geometry_cls(base_mock, region)

    np.testing.assert_array_almost_equal(strategy.matrix, expected_matrix)


@pytest.mark.parametrize(
    "geometry_cls, make_mock, init_region, expected_bbox",
    [
        (LayerGeometry, make_layer_mock, (50, 50, 100, 100), (250, 250, 200, 200)),
        (GroupGeometry, make_group_mock, (0, 0, 1, 1), (20, 20, 280, 280)),
    ],
    ids=["LayerGeometry", "GroupGeometry"],
)
def test_geometry_global_region(geometry_cls, make_mock, init_region, expected_bbox):
    """Garante que a property global_region retorna a bbox correta."""
    base_mock = make_mock()
    region = Region.from_rect(*init_region)

    strategy = geometry_cls(base_mock, region)

    assert strategy.global_region == Region.from_rect(*expected_bbox)


@pytest.mark.parametrize(
    "geometry_cls, make_mock, init_region, expected_bbox",
    [
        (LayerGeometry, make_layer_mock, (50, 50, 100, 100), (50, 50, 100, 100)),
        (GroupGeometry, make_group_mock, (0, 0, 1, 1), (10, 10, 140, 140)),
    ],
    ids=["LayerGeometry", "GroupGeometry"],
)
def test_geometry_region(geometry_cls, make_mock, init_region, expected_bbox):
    """Garante que a property region retorna a região local ou união das regiões dos filhos."""
    base_mock = make_mock()
    region = Region.from_rect(*init_region)

    strategy = geometry_cls(base_mock, region)

    assert strategy.region == Region.from_rect(*expected_bbox)


def test_geometry_controller_sync_on_coordinate_mutation():
    """Valida se o GeometryController sincroniza as geometrias base e frame ao mutar coordenadas."""
    mock_layer = make_layer_mock(transform_matrix=np.identity(3))
    base_geom = LayerGeometry(mock_layer, Region.from_rect(0, 0, 100, 100))
    frame_geom = LayerGeometry(mock_layer, Region.from_rect(5, 5, 100, 100))

    controller = GeometryController(base_geom, frame_geom)

    # Mutação de coordenadas da região de frame via controller.sync
    new_frame_region = Region.from_rect(10, 0, 100, 100)
    controller.sync(new_frame_region)

    # Verifica se ambas as estratégias (frame e base) foram sincronizadas proporcionalmente
    assert controller.frame.region == Region.from_rect(10, 0, 100, 100)
    assert controller.base.region == Region.from_rect(5, -5, 100, 100)


def test_fit_group_geometry_region_and_global_region():
    """Valida se FitGroupGeometry armazena a região local na base e projeta a global_region no Canvas."""
    group_mock = MagicMock(spec=GroupLayer)
    # Matriz global da base com translação de (20, 30)
    group_mock.matrix = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 30.0], [0.0, 0.0, 1.0]])

    # Moldura passada no Canvas: (50, 50, 150, 100)
    ref_region = Region.from_rect(50, 50, 150, 100)
    strategy = FitGroupGeometry(group_mock, ref_region)

    # 1. Região local no espaço da própria base (subtrai 20, 30): Region(30, 20, 150, 100)
    assert strategy.region == Region.from_rect(30, 20, 150, 100)

    # 2. Região global no Canvas (re-projeta através de base.matrix): Region(50, 50, 150, 100)
    assert strategy.global_region == Region.from_rect(50, 50, 150, 100)


def test_fit_group_geometry_matrix():
    """Valida se a matriz da FitGroupGeometry retorna a matriz global da base do grupo."""
    group_mock = MagicMock(spec=GroupLayer)
    group_mock.matrix = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 30.0], [0.0, 0.0, 1.0]])

    ref_region = Region.from_rect(0, 0, 100, 100)
    strategy = FitGroupGeometry(group_mock, ref_region)

    # 1. No momento inicial, a matriz é a matriz global da base
    np.testing.assert_array_almost_equal(strategy.matrix, group_mock.matrix)

    # 2. Se a base for transformada posteriormente (ex: move mais +10, +15)
    group_mock.matrix = np.array([[1.0, 0.0, 30.0], [0.0, 1.0, 45.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_almost_equal(strategy.matrix, group_mock.matrix)


def test_fit_group_geometry_com_transformacao_propria():
    """
    Testa se FitGroupGeometry projeta a global_region no Canvas com precisão
    e acompanha transformações posteriores da própria base como uma unidade rígida.
    """
    group_mock = MagicMock(spec=GroupLayer)
    # Matriz composta total do grupo (50 + 30 = 80)
    group_mock.matrix = np.array([[1.0, 0.0, 80.0], [0.0, 1.0, 80.0], [0.0, 0.0, 1.0]])

    # Passamos uma moldura de (80, 80, 80, 80) no Canvas
    ref_region = Region.from_rect(80, 80, 80, 80)
    strategy = FitGroupGeometry(group_mock, ref_region)

    # Região local na base fica em (0, 0, 80, 80)
    assert strategy.region == Region.from_rect(0, 0, 80, 80)

    # A global_region resultante DEVE coincidir exatamente com a moldura solicitada no Canvas
    assert strategy.global_region == Region.from_rect(80, 80, 80, 80)

    # Se a base for movida posteriormente para (90, 90):
    group_mock.matrix = np.array([[1.0, 0.0, 90.0], [0.0, 1.0, 90.0], [0.0, 0.0, 1.0]])
    assert strategy.global_region == Region.from_rect(90, 90, 80, 80)


def test_freeze_geometry_congelamento_e_restauracao(mocker):
    """Valida se freeze_geometry congela matrizes sob demanda e restaura o modo dinâmico ao sair."""
    group = GroupLayer()
    layer1 = Layer(make_image_mock(50, 50))
    layer2 = Layer(make_image_mock(50, 50))
    group.append(layer1)
    group.append(layer2)

    spy_calc = mocker.spy(layer1.frame, "_compute_matrix")

    # Fora do contexto: acessos dinâmicos
    _ = layer1.matrix
    _ = layer1.matrix
    assert spy_calc.call_count == 2

    spy_calc.reset_mock()

    # Dentro do contexto: calcula apenas no 1º acesso e reutiliza
    with freeze_geometry(group):
        for _ in range(5):
            _ = layer1.matrix
        assert spy_calc.call_count == 1

    spy_calc.reset_mock()

    # Fora do contexto: volta a calcular dinamicamente
    _ = layer1.matrix
    _ = layer1.matrix
    assert spy_calc.call_count == 2


def test_freeze_geometry_restaura_em_caso_de_excecao():
    """Valida se freeze_geometry restaura o modo dinâmico mesmo quando ocorre exceção dentro do bloco."""
    group = GroupLayer()
    layer = Layer(make_image_mock(50, 50))
    group.append(layer)

    with pytest.raises(RuntimeError):
        with freeze_geometry(group):
            _ = layer.matrix
            raise RuntimeError("Erro forçado")

    assert layer.base._cached_matrix is None


def test_freeze_geometry_congelamento_region_e_global_region(mocker):
    """Valida se freeze_geometry congela region e global_region de grupos e layers sob demanda."""
    group = GroupLayer()
    layer = Layer(make_image_mock(50, 50))
    group.append(layer)

    spy_region = mocker.spy(group.frame, "_compute_region")
    spy_global = mocker.spy(group.frame, "_compute_global_region")

    with freeze_geometry(group):
        for _ in range(5):
            _ = group.region
            _ = group.global_region

        assert spy_region.call_count == 1
        assert spy_global.call_count == 1

    assert group.frame._cached_region is None
    assert group.frame._cached_global_region is None


def test_geometry_controller_sync_preserves_base_region_size():
    """Valida se GeometryController.sync desloca a base preservando a largura e altura originais."""
    layer_mock = make_layer_mock(transform_matrix=np.identity(3))
    base_geom = LayerGeometry(layer_mock, Region.from_rect(0, 0, 736, 1104))
    frame_geom = LayerGeometry(layer_mock, Region.from_rect(100, 50, 400, 400))

    controller = GeometryController(base_geom, frame_geom)
    controller.sync(Region.from_rect(168, 352, 400, 400))

    assert controller.frame.region == Region.from_rect(168, 352, 400, 400)
    assert controller.base.region == Region.from_rect(68, 302, 736, 1104)
    assert controller.base.region.size == (736, 1104)
