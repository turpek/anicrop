from unittest.mock import MagicMock, PropertyMock
import numpy as np
import pytest

from anicrop.canvas import Canvas
from anicrop.container import GroupLayer
from anicrop.edit_layer import CropEditLayer
from anicrop.geometry import FitGeometry, FitGroupGeometry
from anicrop.image import Image, ImageFormat
from anicrop.layer import EditLayer, Layer
from anicrop.layout import (
    Layout,
    resolve_region,
    _compute_layer_local_roi,
    content_region,
    global_content_region,
)
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
        pytest.param(Region.from_rect(10, 20, 100, 50),
                     (10, 20, 100, 50), id='region_object'),
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
        assert isinstance(target.frame, FitGeometry)
        assert target.global_region == Region.from_rect(*expect_global_rect)
        assert target.base.region == original_base_region


@pytest.mark.parametrize(
    'ref, anchor_x, anchor_y, expect_rect',
    [
        pytest.param((0, 0, 200, 200), 0.0, 0.0, (0, 0, 100, 50), id='align_top_left'),
        pytest.param((0, 0, 200, 200), 0.5, 0.5, (50, 75, 100, 50), id='align_center'),
        pytest.param((0, 0, 200, 200), 1.0, 1.0,
                     (100, 150, 100, 50), id='align_bottom_right'),
        pytest.param((10, 20, 200, 200), 0.0, 0.0, None,
                     id='align_already_aligned_no_op'),
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
        pytest.param((0, 0, 400, 400), 0.0, 0.0,
                     (0, 0, 200, 100), id='align_group_top_left'),
        pytest.param((0, 0, 400, 400), 0.5, 0.5,
                     (100, 150, 200, 100), id='align_group_center'),
        pytest.param((0, 0, 400, 400), 1.0, 1.0, (200, 300, 200, 100),
                     id='align_group_bottom_right'),
        pytest.param((10, 20, 400, 400), 0.0, 0.0,
                     (10, 20, 200, 100), id='align_group_no_op'),
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
        pytest.param(200, 100, 0.5, 0.5, (-40, -5, 200, 100),
                     id='resize_center_anchored'),
        pytest.param(200, 100, 0.0, 0.0, (10, 20, 200, 100),
                     id='resize_top_left_anchored'),
        pytest.param(200, 100, 1.0, 1.0, (-90, -30, 200, 100),
                     id='resize_bottom_right_anchored'),
        pytest.param(200, 100, 0.25, 0.75, (-15, -18, 200, 100),
                     id='resize_asymmetric_anchored'),

        pytest.param(100, 50, 0.5, 0.5, None, id='resize_no_op'),
    ],
)
def test_layout_resize_bounds(new_w, new_h, anchor_x, anchor_y, expect_rect):
    target = make_transformed_layer(x=10, y=20, w=100, h=50)
    layout = Layout()

    result = layout.resize_bounds(
        target, new_w, new_h, anchor_x=anchor_x, anchor_y=anchor_y)

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
        pytest.param((-20, -10, 200, 150), (-10, 10, 200, 150),
                     id='fit_content_expansion'),
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


@pytest.mark.parametrize("root_transform, sub_transform, ref_rect", [
    (
        TransformRel().translate(50, 50),
        TransformRel().translate(30, 30),
        (80, 80, 80, 80),
    ),
    (
        TransformRel().translate(40, 60).scale(2.0, 2.0),
        TransformRel().translate(20, 10).scale(1.0, 0.5),
        (100, 120, 200, 150),
    ),
    (
        TransformRel().translate(50, 50),
        TransformRel().translate(30, 30).rotate(90),
        (50, 50, 120, 120),
    ),
    (
        TransformRel().translate(30, 40).scale(2.0, 2.0).rotate(180),
        TransformRel().translate(20, 10).rotate(180),
        (60, 70, 240, 180),
    ),
], ids=["translation_only", "translation_scale", "translation_rotation_90", "composite_transforms_180"])
def test_layout_fit_group_layer_com_transformacao_propria(root_transform, sub_transform, ref_rect):
    """
    Valida se layout.fit em GroupLayer com transformações arbitrárias (translação, escala, rotação)
    e contido em um pai também transformado posiciona a global_region no Canvas com precisão exata.
    """
    root_group = GroupLayer()
    sub_group = GroupLayer()
    root_group.append(sub_group)

    child = make_transformed_layer(x=20, y=20, w=40, h=40)
    sub_group.append(child)

    root_group.set_transform(root_transform)
    sub_group.set_transform(sub_transform)

    target_region = Region.from_rect(*ref_rect)

    layout = Layout()
    result = layout.fit(sub_group, target_region)

    assert result is True
    assert sub_group.global_region == target_region


def test_layout_fit_group_layer_rigid_unit_rotation_45():
    """
    Testa se após o layout.fit limitar o GroupLayer em (0, 0, 50, 50),
    uma rotação posterior de 45° no grupo gira o grupo como uma unidade rígida,
    fazendo a global_region no Canvas se expandir para a AABB envolvente do quadrado girado (~72x72).
    """
    group = GroupLayer()
    layer = make_transformed_layer(x=0, y=0, w=100, h=100)
    group.append(layer)

    layout = Layout()
    layout.fit(group, Region.from_rect(0, 0, 50, 50))
    assert group.global_region == Region.from_rect(0, 0, 50, 50)

    # Aplica rotação de 45° no grupo após o Fit
    group.transform.rotate(45)

    # O quadrado de 50x50 girado em 45° tem AABB perfeitamente simétrica de (72, 72)
    assert group.global_region.size == (72, 72)


@pytest.mark.parametrize(
    "new_w, new_h, anchor_x, anchor_y, expected_rect",
    [
        pytest.param(300, 200, 0.0, 0.0, (0, 0, 300, 200), id="anchor_top_left"),
        pytest.param(300, 200, 0.5, 0.5, (-50, -50, 300, 200), id="anchor_center"),
        pytest.param(300, 200, 1.0, 1.0, (-100, -100, 300, 200),
                     id="anchor_bottom_right"),
    ],
)
def test_group_layout_resize_bounds(new_w, new_h, anchor_x, anchor_y, expected_rect):
    """Valida se layout.resize_bounds em GroupLayer redimensiona a moldura com as ancoras corretas sem alterar os filhos."""
    group = GroupLayer()
    layer1 = make_transformed_layer(x=0, y=0, w=100, h=50)
    layer2 = make_transformed_layer(x=100, y=50, w=100, h=50)
    group.append(layer1)
    group.append(layer2)

    layout = Layout()
    result = layout.resize_bounds(
        group, new_w, new_h, anchor_x=anchor_x, anchor_y=anchor_y)

    assert result is True
    assert group.global_region == Region.from_rect(*expected_rect)
    assert layer1.region == Region.from_rect(0, 0, 100, 50)
    assert layer2.region == Region.from_rect(100, 50, 100, 50)


def test_group_layout_fit_content(mocker):
    """Valida se layout.fit_content em GroupLayer consolida o conteudo de todas as camadas filhas e ajusta a moldura do grupo."""
    group = GroupLayer()
    layer1 = make_transformed_layer(x=10, y=20, w=100, h=50)
    layer2 = make_transformed_layer(x=100, y=50, w=100, h=50)

    mock_img1 = MagicMock(spec=Image)
    mock_img1.size = (40, 20)
    edit1 = EditLayer(mock_img1, Region.from_rect(10, 10, 40, 20), np.identity(3))
    layer1._edits.clear()
    layer1._edits.append(edit1)

    mock_img2 = MagicMock(spec=Image)
    mock_img2.size = (80, 40)
    edit2 = EditLayer(mock_img2, Region.from_rect(0, 0, 80, 40), np.identity(3))
    layer2._edits.clear()
    layer2._edits.append(edit2)

    group.append(layer1)
    group.append(layer2)

    def fake_content_rect(img):
        if img is mock_img1:
            return Region.from_rect(0, 0, 40, 20)
        return Region.from_rect(0, 0, 80, 40)

    mocker.patch("anicrop.layout.calculate_content_rect", side_effect=fake_content_rect)

    layout = Layout()
    result = layout.fit_content(group)

    assert result is True
    assert group.global_region == Region.from_rect(20, 30, 160, 60)
    assert layer1.region == Region.from_rect(10, 20, 100, 50)
    assert layer2.region == Region.from_rect(100, 50, 100, 50)


def test_group_layout_fit_content_com_camada_filha_rotacionada(mocker):
    """Valida se layout.fit_content em GroupLayer projeta corretamente o conteudo de camada filha com rotacao de 90°."""
    group = GroupLayer()
    layer = make_transformed_layer(
        x=50, y=50, w=100, h=100, transform=TransformRel().rotate(90))

    mock_img = MagicMock(spec=Image)
    mock_img.size = (40, 20)
    edit = EditLayer(mock_img, Region.from_rect(20, 30, 40, 20), np.identity(3))
    layer._edits.clear()
    layer._edits.append(edit)
    group.append(layer)

    mocker.patch(
        "anicrop.layout.calculate_content_rect",
        return_value=Region.from_rect(0, 0, 40, 20),
    )

    layout = Layout()
    result = layout.fit_content(group)

    assert result is True
    assert group.global_region.size == (20, 40)


def test_canvas_fit_content_com_group_layer_aninhado(mocker):
    """Valida se layout.fit_content em Canvas calcula a bounding box global através de GroupLayers recursivos."""
    group = GroupLayer()
    layer = make_transformed_layer(x=20, y=30, w=100, h=50)

    mock_img = MagicMock(spec=Image)
    mock_img.size = (40, 20)
    edit = EditLayer(mock_img, Region.from_rect(10, 10, 40, 20), np.identity(3))
    layer._edits.clear()
    layer._edits.append(edit)
    group.append(layer)

    mocker.patch(
        "anicrop.layout.calculate_content_rect",
        return_value=Region.from_rect(0, 0, 40, 20),
    )

    canvas = Canvas.from_size(500, 500)
    layout = Layout()
    result = layout.fit_content(canvas, [group])

    assert result is True
    assert canvas.region == Region.from_rect(30, 40, 40, 20)


def test_group_layout_fit_content_empty_group():
    """Valida se layout.fit_content em GroupLayer vazio retorna False sem alterar a moldura."""
    group = GroupLayer()
    layout = Layout()

    result = layout.fit_content(group)

    assert result is False


def test_group_layout_fit_content_already_fitted(mocker):
    """Valida se layout.fit_content em GroupLayer cujo conteudo ja coincide com a moldura retorna False."""
    group = GroupLayer()
    layer = make_transformed_layer(x=0, y=0, w=100, h=50)
    group.append(layer)

    mocker.patch(
        "anicrop.layout.calculate_content_rect",
        return_value=Region.from_rect(0, 0, 100, 50),
    )

    layout = Layout()
    result = layout.fit_content(group)

    assert result is False


def test_global_content_region_mariachi_cropped_rotated_and_uncropped(mocker):
    """Valida o calculo do global_content_region para camada com crop e rotacao de 45 graus, antes e depois do uncrop."""
    layer = make_transformed_layer(x=68, y=302, w=736, h=1104)

    cos45 = float(np.cos(np.radians(45)))
    sin45 = float(np.sin(np.radians(45)))
    matrix = np.array([
        [cos45, -sin45, 368.0 - 300.0 * cos45 + 250.0 * sin45],
        [sin45, cos45, 552.0 - 300.0 * sin45 - 250.0 * cos45],
        [0.0, 0.0, 1.0],
    ])
    mocker.patch.object(Layer, "matrix", new_callable=PropertyMock, return_value=matrix)

    mock_base_img = MagicMock(spec=Image)
    mock_base_img.size = (736, 1104)
    mock_base_img.is_zarr = False

    mock_crop_img = MagicMock(spec=Image)
    mock_crop_img.size = (400, 400)
    mock_crop_img.is_zarr = False

    base_edit = EditLayer(mock_base_img, Region.from_rect(0, 0, 736, 1104), np.identity(3))
    crop_edit = CropEditLayer(mock_crop_img, Region.from_rect(100, 50, 400, 400), np.identity(3))

    layer._edits.clear()
    layer._edits.extend([base_edit, crop_edit])

    def fake_calculate_content_rect(img):
        return Region.from_size(400, 400) if img is mock_crop_img else Region.from_size(736, 1104)

    mocker.patch("anicrop.layout.calculate_content_rect", side_effect=fake_calculate_content_rect)

    roi_cropped = global_content_region(layer)

    crop_edit.visible = False
    roi_uncropped = global_content_region(layer)

    assert roi_cropped == Region.from_rect(85, 269, 566, 566)
    assert roi_uncropped == Region.from_rect(-449, 163, 1303, 1302)


def test_layout_fit_content_com_edicao_desativada_retorna_false():
    """Valida que fit_content retorna False e não altera o layout quando todas as edições estão desativadas."""
    layer = make_transformed_layer(x=10, y=20, w=100, h=50)

    mock_img = MagicMock(spec=Image)
    mock_img.size = (40, 20)
    edit = EditLayer(mock_img, Region.from_rect(10, 10, 40, 20), np.identity(3))
    edit.visible = False
    layer._edits.clear()
    layer._edits.append(edit)

    initial_region = layer.region
    initial_global_region = layer.global_region

    layout = Layout()
    result = layout.fit_content(layer)

    assert result is False
    assert layer.region == initial_region
    assert layer.global_region == initial_global_region


def test_compute_layer_local_roi_com_crop_edit_layer_isolado(mocker):
    """Valida o cálculo do local_roi quando um CropEditLayer é o único edit da camada."""
    mock_layer = MagicMock(spec=Layer)
    mock_crop_img = MagicMock(spec=Image)
    mock_crop_img.size = (400, 400)

    crop_edit = CropEditLayer(mock_crop_img, Region.from_rect(100, 50, 400, 400), np.identity(3))
    type(mock_layer).edits = PropertyMock(return_value=(crop_edit,))

    mocker.patch(
        "anicrop.layout.calculate_content_rect",
        return_value=Region.from_rect(0, 0, 400, 400),
    )

    roi = _compute_layer_local_roi(mock_layer)
    assert roi == Region.from_rect(100, 50, 400, 400)


def test_group_layout_fit_content_enquadra_conteudo_global_sem_alterar_filhos(mocker):
    """Valida que fit_content em GroupLayer enquadra a moldura do grupo no ROI de conteúdo dos filhos."""
    group = GroupLayer()
    img_data = np.full((100, 100, 4), 255, dtype=np.uint8)
    img = Image(img_data, ImageFormat.RGBA)
    layer = Layer(img)
    layer.region = Region.from_rect(0, 0, 100, 100)
    group.append(layer)

    # Adiciona um edit de recorte no filho (50x50 em 20, 20)
    mock_edit_img = MagicMock(spec=Image)
    mock_edit_img.size = (50, 50)
    mock_edit_img.has_alpha = False
    edit = CropEditLayer(mock_edit_img, Region.from_rect(20, 20, 50, 50), np.identity(3))
    layer._edits.append(edit)

    def fake_calculate_content_rect(img):
        return Region.from_size(50, 50) if img is mock_edit_img else Region.from_size(100, 100)

    mocker.patch(
        "anicrop.layout.calculate_content_rect",
        side_effect=fake_calculate_content_rect,
    )

    layout = Layout()
    assert group.global_region == Region.from_rect(0, 0, 100, 100)

    # 1. Executa fit_content no grupo (deve ajustar a moldura do grupo para 20, 20, 50, 50)
    result = layout.fit_content(group)
    assert result is True
    assert group.global_region == Region.from_rect(20, 20, 50, 50)
    assert isinstance(group.frame, FitGroupGeometry)

    # 2. O filho individual permanece intacto
    assert layer.region == Region.from_rect(0, 0, 100, 100)


def test_fit_group_geometry_rotacionado_mantem_global_region_solicitada():
    """Valida que fit em GroupLayer rotacionado projeta exatamente a moldura solicitada no Canvas sem dupla inflação de AABB."""
    group = GroupLayer()
    img = Image.new((100, 100), ImageFormat.RGBA)
    layer = Layer(img)
    group.append(layer)

    group.transform.rotate(45)

    layout = Layout()
    ref_frame = Region.from_rect(100, 100, 200, 200)
    layout.fit(group, ref_frame)

    # A moldura global no Canvas deve ser exatamente o ref_frame solicitado
    assert group.global_region == ref_frame
