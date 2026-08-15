from anicrop.render import BaseRenderer, CanvasRender
from unittest.mock import MagicMock
import numpy as np
import pytest

import anicrop.render
from anicrop.canvas import Canvas
from anicrop.container import GroupLayer
from anicrop.frame import CanvasFrame
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.layout import Layout
from anicrop.render import CanvasRender
from anicrop.spatial import Region
from anicrop.transform import TransformRel


def make_test_layer(top_left: tuple[int, int], transform: TransformRel) -> Layer:
    """Cria uma camada de teste de 100x50 com Image real e aplica a transformação."""
    img = Image.new((100, 50), ImageFormat.RGBA)
    layer = Layer(img)
    layer.region += top_left
    layer.set_transform(transform)
    return layer


# ==============================================================================
# 1. Testes do Pipeline de Renderização em Camada Padrão (Sem FitGeometry)
# ==============================================================================

@pytest.mark.parametrize(
    "top_left, transform, expect_plan_dst_rect, expect_rel_rect",
    [
        # 1. Posição na origem (0, 0), sem rotação/escala extra
        pytest.param(
            (0, 0),
            TransformRel(),
            (0, 0, 100, 50),
            (0, 0, 100, 50),
            id="origin_no_transform",
        ),
        # 2. Posição qualquer (50, 50), sem rotação/escala extra
        pytest.param(
            (50, 50),
            TransformRel(),
            (50, 50, 100, 50),
            (0, 0, 100, 50),
            id="shifted_no_transform",
        ),
        # 3. Posição na origem (0, 0), com rotação de 90° em torno do centro (0.5, 0.5)
        pytest.param(
            (0, 0),
            TransformRel().rotate(90),
            (25, -25, 50, 100),
            (0, 0, 50, 100),
            id="origin_with_rotation",
        ),
        # 4. Posição qualquer (50, 50), com rotação de 90° em torno do centro (0.5, 0.5)
        pytest.param(
            (50, 50),
            TransformRel().rotate(90),
            (75, 25, 50, 100),
            (0, 0, 50, 100),
            id="shifted_with_rotation",
        ),
    ],
)
def test_render_pipeline_integration(
    mocker, top_left, transform, expect_plan_dst_rect, expect_rel_rect
):
    """Testa o pipeline real de renderização acompanhando plan.dst_region e rel_region (view)."""
    def fake_render_patch(image, m_render, dest_region, *args, **kwargs):
        return np.zeros((dest_region.height, dest_region.width, 4), dtype=np.uint8)

    mocker.patch("anicrop.render.render_patch", side_effect=fake_render_patch)

    layer = make_test_layer(top_left, transform)
    spy_view = mocker.spy(Image, "view")

    renderer = CanvasRender()
    rendered_image = renderer.render_layer(layer)

    plan = CanvasFrame(layer, Canvas(layer.global_region))
    assert plan.dst_region == Region.from_rect(*expect_plan_dst_rect)

    assert rendered_image is not None
    assert rendered_image.size == plan.dst_region.size

    assert spy_view.call_count >= 1
    view_region_arg = spy_view.call_args_list[-1][0][1]
    assert view_region_arg == Region.from_rect(*expect_rel_rect)


def test_render_pipeline_integration_with_parent_group_transforms(mocker):
    """Testa o pipeline real de renderização quando o GroupLayer pai possui translação, rotação e escala."""
    def fake_render_patch(image, m_render, dest_region, *args, **kwargs):
        return np.zeros((dest_region.height, dest_region.width, 4), dtype=np.uint8)

    mocker.patch("anicrop.render.render_patch", side_effect=fake_render_patch)

    group = GroupLayer()
    group.set_transform(TransformRel().translate(20, 30).rotate(45).scale(2.0, 2.0))

    layer = make_test_layer((10, 10), TransformRel().rotate(90))
    group.append(layer)

    spy_view = mocker.spy(Image, "view")

    renderer = CanvasRender()
    rendered_image = renderer.render_layer(layer)

    plan = CanvasFrame(layer, Canvas(layer.global_region))
    assert plan.dst_region == Region.from_rect(-51, 57, 213, 213)

    assert rendered_image is not None
    assert rendered_image.size == plan.dst_region.size

    assert spy_view.call_count >= 1
    view_region_arg = spy_view.call_args_list[-1][0][1]
    assert view_region_arg == Region.from_rect(0, 0, 213, 213)


# ==============================================================================
# 2. Testes do Pipeline de Renderização em Camada com FitGeometry (layout.fit)
# ==============================================================================

@pytest.mark.parametrize(
    "top_left, transform, fit_rect, expect_plan_dst_rect, expect_rel_rect",
    [
        # 1. Fit de Expansão (200x100) em camada na origem (0, 0)
        pytest.param(
            (0, 0),
            TransformRel(),
            (0, 0, 200, 100),
            (0, 0, 200, 100),
            (0, 0, 100, 50),
            id="origin_fit_expansion",
        ),
        # 2. Fit de Retração / Crop (200x100) em camada deslocada para (50, 50) com escala (2.0, 2.0)
        pytest.param(
            (50, 50),
            TransformRel().scale(2.0, 2.0),
            (0, 25, 200, 100),
            (0, 25, 200, 100),
            (0, 0, 200, 100),
            id="shifted_fit_contraction_with_scale",
        ),


        # 3. Fit em camada rotacionada 90°
        pytest.param(
            (0, 0),
            TransformRel().rotate(90),
            (25, -25, 50, 100),
            (25, -25, 50, 100),
            (0, 0, 50, 100),
            id="rotated_fit",
        ),
    ],
)
def test_fit_geometry_render_pipeline(
    mocker,
    top_left,
    transform,
    fit_rect,
    expect_plan_dst_rect,
    expect_rel_rect,
):
    """Testa o pipeline real de renderização com FitGeometry ativo (layout.fit)."""
    def fake_render_patch(image, m_render, dest_region, *args, **kwargs):
        return np.zeros((dest_region.height, dest_region.width, 4), dtype=np.uint8)

    mocker.patch("anicrop.render.render_patch", side_effect=fake_render_patch)

    layer = make_test_layer(top_left, transform)

    layout = Layout()
    layout.fit(layer, Region.from_rect(*fit_rect))

    spy_view = mocker.spy(Image, "view")

    renderer = CanvasRender()
    rendered_image = renderer.render_layer(layer)

    plan = CanvasFrame(layer, Canvas(layer.global_region))
    assert plan.dst_region == Region.from_rect(*expect_plan_dst_rect)

    assert rendered_image is not None
    assert rendered_image.size == plan.dst_region.size

    assert spy_view.call_count >= 1
    view_region_arg = spy_view.call_args_list[-1][0][1]
    assert view_region_arg == Region.from_rect(*expect_rel_rect)


def test_fit_geometry_render_pipeline_with_parent_group_transforms(mocker):
    """Testa o pipeline real com FitGeometry quando a camada pertence a um GroupLayer pai transformado."""
    def fake_render_patch(image, m_render, dest_region, *args, **kwargs):
        return np.zeros((dest_region.height, dest_region.width, 4), dtype=np.uint8)

    mocker.patch("anicrop.render.render_patch", side_effect=fake_render_patch)

    group = GroupLayer()
    group.set_transform(TransformRel().translate(20, 30).rotate(45).scale(2.0, 2.0))

    layer = make_test_layer((10, 10), TransformRel().rotate(90))
    group.append(layer)

    layout = Layout()
    layout.fit(layer, Region.from_rect(-51, 57, 213, 213))

    spy_view = mocker.spy(Image, "view")

    renderer = CanvasRender()
    rendered_image = renderer.render_layer(layer)

    plan = CanvasFrame(layer, Canvas(layer.global_region))
    assert plan.dst_region == Region.from_rect(-51, 57, 213, 213)

    assert rendered_image is not None
    assert rendered_image.size == plan.dst_region.size

    assert spy_view.call_count >= 1
    view_region_arg = spy_view.call_args_list[-1][0][1]
    assert view_region_arg == Region.from_rect(0, 0, 213, 213)


# ==============================================================================
# 3. Testes do Pipeline de Renderização com Align (layout.align)
# ==============================================================================

@pytest.mark.parametrize(
    "top_left, transform, align_anchors, expect_plan_dst_rect, expect_rel_rect",
    [
        # 1. Align no canto inferior direito do Canvas (1.0, 1.0)
        pytest.param(
            (0, 0),
            TransformRel(),
            (1.0, 1.0),
            (100, 150, 100, 50),
            (0, 0, 100, 50),
            id="align_bottom_right_simple_layer",
        ),
        # 2. Align no centro do Canvas (0.5, 0.5)
        pytest.param(
            (0, 0),
            TransformRel(),
            (0.5, 0.5),
            (50, 75, 100, 50),
            (0, 0, 100, 50),
            id="align_center_simple_layer",
        ),
    ],
)
def test_align_geometry_render_pipeline(
    mocker,
    top_left,
    transform,
    align_anchors,
    expect_plan_dst_rect,
    expect_rel_rect,
):
    """Testa o pipeline real de renderização quando layout.align é executado."""
    def fake_render_patch(image, m_render, dest_region, *args, **kwargs):
        return np.zeros((dest_region.height, dest_region.width, 4), dtype=np.uint8)

    mocker.patch("anicrop.render.render_patch", side_effect=fake_render_patch)

    layer = make_test_layer(top_left, transform)
    canvas = Canvas.from_size(200, 200)

    layout = Layout()
    layout.align(layer, canvas, anchor_x=align_anchors[0], anchor_y=align_anchors[1])

    spy_view = mocker.spy(Image, "view")

    renderer = CanvasRender()
    plan = CanvasFrame(layer, canvas)
    rendered_image = renderer.render_area(layer, plan)

    assert plan.dst_region == Region.from_rect(*expect_plan_dst_rect)

    assert rendered_image is not None
    assert rendered_image.size == plan.dst_region.size

    assert spy_view.call_count >= 1
    view_region_arg = spy_view.call_args_list[-1][0][1]
    assert view_region_arg == Region.from_rect(*expect_rel_rect)


def test_fit_and_align_geometry_render_pipeline(mocker):
    """Testa o pipeline real combinando FitGeometry (layout.fit) seguido de layout.align."""
    def fake_render_patch(image, m_render, dest_region, *args, **kwargs):
        return np.zeros((dest_region.height, dest_region.width, 4), dtype=np.uint8)

    mocker.patch("anicrop.render.render_patch", side_effect=fake_render_patch)

    layer = make_test_layer((0, 0), TransformRel())
    canvas = Canvas.from_size(200, 200)

    layout = Layout()
    # Step 1: fit expandindo para (0, 0, 150, 80)
    layout.fit(layer, Region.from_rect(0, 0, 150, 80))

    # Step 2: align no canto inferior direito do Canvas (1.0, 1.0)
    layout.align(layer, canvas, anchor_x=1.0, anchor_y=1.0)

    spy_view = mocker.spy(Image, "view")

    renderer = CanvasRender()
    plan = CanvasFrame(layer, canvas)
    rendered_image = renderer.render_area(layer, plan)

    # Moldura de 150x80 encostada no canto (200, 200) -> (50, 120, 150, 80)
    assert plan.dst_region == Region.from_rect(50, 120, 150, 80)

    assert rendered_image is not None
    assert rendered_image.size == plan.dst_region.size

    # Imagem base de 100x50 posicionada na origem (0, 0) do buffer de 150x80
    assert spy_view.call_count >= 1
    view_region_arg = spy_view.call_args_list[-1][0][1]
    assert view_region_arg == Region.from_rect(0, 0, 100, 50)


# ==============================================================================
# 4. Testes do Pipeline de Renderização com Fit Content (layout.fit_content)
# ==============================================================================

def test_fit_content_geometry_render_pipeline(mocker):
    """Testa o pipeline real de renderização quando layout.fit_content é executado em uma imagem de 200x200 com conteúdo ativo de (25, 25) até (175, 175)."""
    def fake_render_patch(image, m_render, dest_region, *args, **kwargs):
        return np.zeros((dest_region.height, dest_region.width, 4), dtype=np.uint8)

    mocker.patch("anicrop.render.render_patch", side_effect=fake_render_patch)

    # Imagem base de 200x200 cujo conteúdo visível fica entre (25, 25) e (175, 175) -> Region(25, 25, 150, 150)
    mocker.patch(
        "anicrop.layout.calculate_content_rect",
        return_value=Region.from_rect(25, 25, 150, 150),
    )

    img = Image.new((200, 200), ImageFormat.RGBA)
    layer = Layer(img)

    layout = Layout()
    layout.fit_content(layer)

    spy_view = mocker.spy(Image, "view")

    renderer = CanvasRender()
    rendered_image = renderer.render_layer(layer)

    # A moldura do fit_content envolve o conteúdo visível de (25, 25, 150, 150)
    plan = CanvasFrame(layer, Canvas(layer.global_region))
    assert plan.dst_region == Region.from_rect(25, 25, 150, 150)

    assert rendered_image is not None
    assert rendered_image.size == plan.dst_region.size

    # Valida o fatiamento relativo no Image.view
    assert spy_view.call_count >= 1
    view_region_arg = spy_view.call_args_list[-1][0][1]
    assert view_region_arg == Region.from_rect(0, 0, 150, 150)


def test_fit_content_and_align_geometry_render_pipeline(mocker):
    """Testa o pipeline real combinando fit_content seguido de layout.align."""
    def fake_render_patch(image, m_render, dest_region, *args, **kwargs):
        return np.zeros((dest_region.height, dest_region.width, 4), dtype=np.uint8)

    mocker.patch("anicrop.render.render_patch", side_effect=fake_render_patch)
    mocker.patch(
        "anicrop.layout.calculate_content_rect",
        return_value=Region.from_rect(25, 25, 150, 150),
    )

    img = Image.new((200, 200), ImageFormat.RGBA)
    layer = Layer(img)
    canvas = Canvas.from_size(200, 200)
    layout = Layout()

    # Step 1: fit_content -> moldura (25, 25, 150, 150)
    layout.fit_content(layer)

    # Step 2: align no canto inferior direito do Canvas (1.0, 1.0)
    layout.align(layer, canvas, anchor_x=1.0, anchor_y=1.0)

    spy_view = mocker.spy(Image, "view")

    renderer = CanvasRender()
    plan = CanvasFrame(layer, canvas)
    rendered_image = renderer.render_area(layer, plan)

    # Moldura de 150x150 encostada no canto (200, 200) -> (50, 50, 150, 150)
    assert plan.dst_region == Region.from_rect(50, 50, 150, 150)

    assert rendered_image is not None
    assert rendered_image.size == plan.dst_region.size

    assert spy_view.call_count >= 1
    view_region_arg = spy_view.call_args_list[-1][0][1]
    assert view_region_arg == Region.from_rect(0, 0, 150, 150)


def test_fit_content_with_layer_transform_render_pipeline(mocker):
    """Testa o pipeline real de renderização quando layout.fit_content é executado em uma camada rotacionada 90°."""
    def fake_render_patch(image, m_render, dest_region, *args, **kwargs):
        return np.zeros((dest_region.height, dest_region.width, 4), dtype=np.uint8)

    mocker.patch("anicrop.render.render_patch", side_effect=fake_render_patch)
    mocker.patch(
        "anicrop.layout.calculate_content_rect",
        return_value=Region.from_rect(25, 25, 150, 150),
    )

    img = Image.new((200, 200), ImageFormat.RGBA)
    layer = Layer(img)
    layer.set_transform(TransformRel().rotate(90))

    layout = Layout()
    layout.fit_content(layer)

    spy_view = mocker.spy(Image, "view")

    renderer = CanvasRender()
    rendered_image = renderer.render_layer(layer)

    plan = CanvasFrame(layer, Canvas(layer.global_region))
    assert plan.dst_region == Region.from_rect(25, 25, 150, 150)

    assert rendered_image is not None
    assert rendered_image.size == plan.dst_region.size

    assert spy_view.call_count >= 1
    view_region_arg = spy_view.call_args_list[-1][0][1]
    assert view_region_arg == Region.from_rect(0, 0, 150, 150)


def test_fit_content_in_parent_group_render_pipeline(mocker):
    """Testa o pipeline real de renderização quando layout.fit_content é executado em uma camada dentro de um GroupLayer pai transformado."""
    def fake_render_patch(image, m_render, dest_region, *args, **kwargs):
        return np.zeros((dest_region.height, dest_region.width, 4), dtype=np.uint8)

    mocker.patch("anicrop.render.render_patch", side_effect=fake_render_patch)
    mocker.patch(
        "anicrop.layout.calculate_content_rect",
        return_value=Region.from_rect(25, 25, 150, 150),
    )

    group = GroupLayer()
    group.set_transform(TransformRel().translate(20, 30))

    img = Image.new((200, 200), ImageFormat.RGBA)
    layer = Layer(img)
    layer.region += (10, 10)
    group.append(layer)

    layout = Layout()
    layout.fit_content(layer)

    spy_view = mocker.spy(Image, "view")

    renderer = CanvasRender()
    rendered_image = renderer.render_layer(layer)

    # ROI local na camada: (25, 25, 150, 150) + offset base (10, 10) -> no grupo: (35, 35, 150, 150)
    # No Espaço Global (Canvas) com translação do grupo (20, 30): (55, 65, 150, 150)
    plan = CanvasFrame(layer, Canvas(layer.global_region))
    assert plan.dst_region == Region.from_rect(55, 65, 150, 150)

    assert rendered_image is not None
    assert rendered_image.size == plan.dst_region.size

    assert spy_view.call_count >= 1
    view_region_arg = spy_view.call_args_list[-1][0][1]
    assert view_region_arg == Region.from_rect(0, 0, 150, 150)


# ==============================================================================
# 5. Testes do Pipeline de Renderização com Resize Bounds (layout.resize_bounds)
# ==============================================================================

@pytest.mark.parametrize(
    "anchor_x, anchor_y, expect_plan_dst_rect, expect_rel_rect",
    [
        # 1. Topo-Esquerda (0.0, 0.0) fixo em (50, 50)
        pytest.param(
            0.0,
            0.0,
            (50, 50, 150, 80),
            (0, 0, 100, 50),
            id="resize_top_left_anchored",
        ),
        # 2. Centro (0.5, 0.5) fixo em (100, 75)
        pytest.param(
            0.5,
            0.5,
            (25, 35, 150, 80),
            (25, 15, 100, 50),
            id="resize_center_anchored",
        ),
        # 3. Canto Inferior Direito (1.0, 1.0) fixo em (150, 100)
        pytest.param(
            1.0,
            1.0,
            (0, 20, 150, 80),
            (50, 30, 100, 50),
            id="resize_bottom_right_anchored",
        ),
    ],
)
def test_resize_bounds_geometry_render_pipeline(
    mocker,
    anchor_x,
    anchor_y,
    expect_plan_dst_rect,
    expect_rel_rect,
):
    """Testa o pipeline real de renderização quando layout.resize_bounds é executado com diferentes âncoras na camada."""
    def fake_render_patch(image, m_render, dest_region, *args, **kwargs):
        return np.zeros((dest_region.height, dest_region.width, 4), dtype=np.uint8)

    mocker.patch("anicrop.render.render_patch", side_effect=fake_render_patch)

    layer = make_test_layer((50, 50), TransformRel())

    layout = Layout()
    layout.resize_bounds(layer, 150, 80, anchor_x=anchor_x, anchor_y=anchor_y)

    spy_view = mocker.spy(Image, "view")

    renderer = CanvasRender()
    rendered_image = renderer.render_layer(layer)

    plan = CanvasFrame(layer, Canvas(layer.global_region))
    assert plan.dst_region == Region.from_rect(*expect_plan_dst_rect)

    assert rendered_image is not None
    assert rendered_image.size == plan.dst_region.size

    assert spy_view.call_count >= 1
    view_region_arg = spy_view.call_args_list[-1][0][1]
    assert view_region_arg == Region.from_rect(*expect_rel_rect)


def test_resize_bounds_in_parent_group_render_pipeline(mocker):
    """Testa o pipeline real de renderização quando layout.resize_bounds é executado em uma camada dentro de um GroupLayer pai transformado."""
    def fake_render_patch(image, m_render, dest_region, *args, **kwargs):
        return np.zeros((dest_region.height, dest_region.width, 4), dtype=np.uint8)

    mocker.patch("anicrop.render.render_patch", side_effect=fake_render_patch)

    group = GroupLayer()
    group.set_transform(TransformRel().translate(20, 30))

    layer = make_test_layer((30, 20), TransformRel())
    group.append(layer)

    layout = Layout()
    # Posição da camada no Canvas (Espaço Global): (20 + 30, 30 + 20) = (50, 50)
    # Redimensiona ancorando em (0.0, 0.0) -> Topo-Esquerda no Canvas em (50, 50) permanece fixo!
    layout.resize_bounds(layer, 150, 80, anchor_x=0.0, anchor_y=0.0)

    spy_view = mocker.spy(Image, "view")

    renderer = CanvasRender()
    rendered_image = renderer.render_layer(layer)

    plan = CanvasFrame(layer, Canvas(layer.global_region))
    assert plan.dst_region == Region.from_rect(50, 50, 150, 80)

    assert rendered_image is not None
    assert rendered_image.size == plan.dst_region.size

    assert spy_view.call_count >= 1
    view_region_arg = spy_view.call_args_list[-1][0][1]
    assert view_region_arg == Region.from_rect(0, 0, 100, 50)
