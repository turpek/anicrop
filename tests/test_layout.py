from unittest.mock import MagicMock
import numpy as np
import pytest

from anicrop.canvas import Canvas
from anicrop.container import GroupLayer
from anicrop.geometry import FitGeometry
from anicrop.image import Image
from anicrop.layer import EditLayer, Layer
from anicrop.layout import Layout, resolve_region, content_region
from anicrop.spatial import Region
from anicrop.transform import TransformRel


def make_transformed_layer(
    x: int = 10,
    y: int = 20,
    w: int = 100,
    h: int = 50,
    transform: TransformRel | None = None,
) -> Layer:
    """Cria uma camada com mock de Image, translação e transformações opcionais."""
    mock_img = MagicMock(spec=Image)
    mock_img.size = (w, h)
    layer = Layer(mock_img)
    layer.region += (x, y)
    if transform is not None:
        layer.set_transform(transform)
    return layer


@pytest.mark.parametrize(
    'ref, expected_rect',
    [
        pytest.param((10, 20, 100, 50), (10, 20, 100, 50), id='tuple_rect'),
        pytest.param(Region.from_rect(10, 20, 100, 50), (10, 20, 100, 50), id='region_object'),
        pytest.param(
            make_transformed_layer(x=30, y=40, w=100, h=50),
            (30, 40, 100, 50),
            id='layer_no_rotation',
        ),
        pytest.param(
            make_transformed_layer(
                x=0, y=0, w=100, h=50, transform=TransformRel().rotate(90)
            ),
            (25, -25, 50, 100),
            id='layer_rotated_90',
        ),
        pytest.param(Canvas.from_size(200, 150), (0, 0, 200, 150), id='canvas_object'),
    ],
)
def test_resolve_region(ref, expected_rect):
    resolved = resolve_region(ref)
    assert resolved == Region.from_rect(*expected_rect)


@pytest.mark.parametrize(
    'ref, expect_global_rect',
    [
        pytest.param((10, 20, 100, 50), None, id='fit_equal_no_op'),
        pytest.param((0, 0, 200, 200), (0, 0, 200, 200), id='fit_expansion'),
        pytest.param((20, 30, 50, 30), (20, 30, 50, 30), id='fit_contraction'),
    ],
)
def test_layout_fit(ref, expect_global_rect):
    target = make_transformed_layer(x=10, y=20, w=100, h=50)
    original_base_region = target.base.region

    layout = Layout()
    result = layout.fit(target, ref)

    if expect_global_rect is None:
        assert result is False
    else:
        assert result is True
        assert isinstance(target.layout, FitGeometry)
        assert target.global_region == Region.from_rect(*expect_global_rect)
        assert target.base.region == original_base_region


@pytest.mark.parametrize(
    'ref, anchor_x, anchor_y, expect_rect',
    [
        pytest.param((0, 0, 200, 200), 0.0, 0.0, (0, 0, 100, 50), id='align_top_left'),
        pytest.param((0, 0, 200, 200), 0.5, 0.5, (50, 75, 100, 50), id='align_center'),
        pytest.param((0, 0, 200, 200), 1.0, 1.0, (100, 150, 100, 50), id='align_bottom_right'),
        pytest.param((10, 20, 200, 200), 0.0, 0.0, None, id='align_already_aligned_no_op'),
    ],
)
def test_layout_align(ref, anchor_x, anchor_y, expect_rect):
    target = make_transformed_layer(x=10, y=20, w=100, h=50)
    layout = Layout()

    result = layout.align(target, ref, anchor_x=anchor_x, anchor_y=anchor_y)

    if expect_rect is None:
        assert result is False
    else:
        assert result is True
        assert target.global_region == Region.from_rect(*expect_rect)


def test_layout_align_com_rotacao_90_deg():
    target = make_transformed_layer(
        x=50, y=50, w=100, h=50, transform=TransformRel().rotate(90)
    )
    ref = (0, 0, 200, 200)

    layout = Layout()
    result = layout.align(target, ref, anchor_x=0.0, anchor_y=0.0)

    assert result is True
    assert target.global_region == Region.from_rect(0, 0, 50, 100)


def test_layout_align_layer_dentro_de_grupo_rotacionado():
    group = GroupLayer()
    group.set_transform(TransformRel().rotate(45, 0.5, 0.5))

    target = make_transformed_layer(x=50, y=50, w=100, h=50)
    group.append(target)

    layout = Layout()
    result = layout.align(target, (0, 0, 200, 200), anchor_x=0.0, anchor_y=0.0)

    assert result is True


@pytest.mark.parametrize(
    'ref, anchor_x, anchor_y, expected_global_rect',
    [
        pytest.param((0, 0, 400, 400), 0.0, 0.0, (0, 0, 200, 100), id='align_group_top_left'),
        pytest.param((0, 0, 400, 400), 0.5, 0.5, (100, 150, 200, 100), id='align_group_center'),
        pytest.param((0, 0, 400, 400), 1.0, 1.0, (200, 300, 200, 100), id='align_group_bottom_right'),
        pytest.param((10, 20, 400, 400), 0.0, 0.0, (10, 20, 200, 100), id='align_group_no_op'),
    ],
)
def test_group_layout_align(ref, anchor_x, anchor_y, expected_global_rect):
    group = GroupLayer()
    layer1 = make_transformed_layer(x=10, y=20, w=100, h=50)
    layer2 = make_transformed_layer(x=110, y=70, w=100, h=50)
    group.append(layer1)
    group.append(layer2)

    layout = Layout()
    layout.align(group, ref, anchor_x=anchor_x, anchor_y=anchor_y)

    assert group.global_region == Region.from_rect(*expected_global_rect)
    assert layer1.region == Region.from_rect(10, 20, 100, 50)
    assert layer2.region == Region.from_rect(110, 70, 100, 50)


@pytest.mark.parametrize(
    'new_w, new_h, anchor_x, anchor_y, expect_rect',
    [
        pytest.param(200, 100, 0.5, 0.5, (-40, -5, 200, 100), id='resize_center_anchored'),
        pytest.param(200, 100, 0.0, 0.0, (10, 20, 200, 100), id='resize_top_left_anchored'),
        pytest.param(200, 100, 1.0, 1.0, (-90, -30, 200, 100), id='resize_bottom_right_anchored'),
        pytest.param(200, 100, 0.25, 0.75, (-15, -18, 200, 100), id='resize_asymmetric_anchored'),

        pytest.param(100, 50, 0.5, 0.5, None, id='resize_no_op'),
    ],
)
def test_layout_resize_bounds(new_w, new_h, anchor_x, anchor_y, expect_rect):
    target = make_transformed_layer(x=10, y=20, w=100, h=50)
    layout = Layout()

    result = layout.resize_bounds(target, new_w, new_h, anchor_x=anchor_x, anchor_y=anchor_y)

    if expect_rect is None:
        assert result is False
    else:
        assert result is True
        assert target.global_region == Region.from_rect(*expect_rect)


def test_layout_resize_bounds_com_rotacao_90_deg():
    target = make_transformed_layer(
        x=50, y=50, w=100, h=50, transform=TransformRel().rotate(90)
    )
    layout = Layout()

    result = layout.resize_bounds(target, 100, 200, anchor_x=0.0, anchor_y=0.0)
    assert result is True


@pytest.mark.parametrize(
    'edits_rect, expect_global_rect',
    [
        pytest.param((10, 10, 40, 20), (20, 30, 40, 20), id='fit_content_contraction'),
        pytest.param((-20, -10, 200, 150), (-10, 10, 200, 150), id='fit_content_expansion'),
        pytest.param(None, None, id='fit_content_empty_no_edits'),
        pytest.param((0, 0, 100, 50), None, id='fit_content_already_fitted_no_op'),
    ],
)
def test_layout_fit_content(mocker, edits_rect, expect_global_rect):
    target = make_transformed_layer(x=10, y=20, w=100, h=50)

    if edits_rect is not None:
        mock_edit_img = MagicMock(spec=Image)
        mock_edit_img.size = (edits_rect[2], edits_rect[3])
        edit_layer = EditLayer(
            mock_edit_img, Region.from_rect(*edits_rect), np.identity(3)
        )
        target._edits.clear()
        target._edits.append(edit_layer)
        mocker.patch(
            'anicrop.layout.calculate_content_rect',
            return_value=Region.from_rect(0, 0, edits_rect[2], edits_rect[3]),
        )
    else:
        target._edits.clear()

    layout = Layout()
    result = layout.fit_content(target)

    if expect_global_rect is None:
        assert result is False
    else:
        assert result is True
        assert target.global_region == Region.from_rect(*expect_global_rect)


def test_layout_fit_content_apos_fit_preserva_tamanho_das_edicoes(mocker):
    layer = make_transformed_layer(x=50, y=50, w=100, h=100)

    mock_edit_img = MagicMock(spec=Image)
    mock_edit_img.size = (100, 100)
    edit_layer = EditLayer(
        mock_edit_img, Region.from_rect(0, 0, 100, 100), np.identity(3)
    )
    layer._edits.clear()
    layer._edits.append(edit_layer)

    layout = Layout()
    layout.fit(layer, (0, 0, 200, 200))

    mocker.patch(
        'anicrop.layout.calculate_content_rect',
        return_value=Region.from_rect(0, 0, 100, 100),
    )

    roi = content_region(layer)
    assert roi == Region.from_rect(50, 50, 100, 100)


def test_layout_fit_content_em_camada_dentro_de_group_layer_transformado(mocker):
    """Testa se fit_content em camada dentro de um GroupLayer transformado projeta target.global_region no Espaço Global (Canvas) correto."""
    group = GroupLayer()
    group.set_transform(TransformRel().translate(20, 30))

    layer = make_transformed_layer(x=10, y=10, w=200, h=200)
    group.append(layer)

    mocker.patch(
        "anicrop.layout.calculate_content_rect",
        return_value=Region.from_rect(25, 25, 150, 150),
    )

    layout = Layout()
    result = layout.fit_content(layer)

    assert result is True
    assert layer.global_region == Region.from_rect(55, 65, 150, 150)


def test_layout_group_layer_fit_usa_fit_group_geometry():
    """Valida se layout.fit aplicado sobre um GroupLayer ativa FitGroupGeometry sem alterar as camadas filhas."""
    group = GroupLayer()
    layer1 = make_transformed_layer(x=0, y=0, w=100, h=50)
    layer2 = make_transformed_layer(x=100, y=50, w=100, h=50)
    group.append(layer1)
    group.append(layer2)

    layout = Layout()
    result = layout.fit(group, Region.from_rect(0, 0, 150, 80))

    assert result is True
    assert group.global_region == Region.from_rect(0, 0, 150, 80)
    # Filhas permanecem 100% intactas
    assert layer1.region == Region.from_rect(0, 0, 100, 50)
    assert layer2.region == Region.from_rect(100, 50, 100, 50)
