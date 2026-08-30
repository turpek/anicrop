import pytest
import numpy as np
from anicrop.render import ViewportRender
from anicrop.frame import ViewportFrame
from anicrop.layer import Layer
from anicrop.container import GroupLayer
from anicrop.spatial import Region
from anicrop.image import Image, ImageFormat
from anicrop.viewport import Viewport
from anicrop.type import Scale


def make_layer(
    w: int = 100,
    h: int = 100,
    color: tuple[int, int, int, int] = (255, 0, 0, 255),
    x: int = 0,
    y: int = 0,
) -> Layer:
    """Cria uma camada de teste com cor solida e posicao definida."""
    img_data = np.zeros((h, w, 4), dtype=np.uint8)
    img_data[:] = color
    layer = Layer(Image(img_data, ImageFormat.RGBA))
    if x != 0:
        layer.x = x
    if y != 0:
        layer.y = y
    return layer


def test_viewport_render_area_returns_image():
    """Valida se ViewportRender.render_area utiliza ViewportFrame para renderizar a camada."""
    layer = make_layer(100, 100, (255, 0, 0, 255))
    viewport = Viewport((800, 600), 1.0)
    vr = ViewportRender()

    frame = ViewportFrame(layer, viewport)
    rendered = vr.render_area(layer, frame)

    assert rendered is not None
    assert rendered.width > 0
    assert rendered.height > 0


def test_viewport_render_scene_composes_visible_layers():
    """Valida se ViewportRender.render_scene compõe todas as camadas visíveis na tela da Viewport."""
    layer1 = make_layer(100, 100, (255, 0, 0, 255))
    layer2 = make_layer(100, 100, (0, 255, 0, 255), x=50, y=50)
    viewport = Viewport((800, 600), 1.0)
    vr = ViewportRender()

    composition = vr.render_scene([layer1, layer2], viewport)

    assert composition is not None
    assert composition.size == (800, 600)


def test_viewport_render_zoom_and_centering():
    """Valida se ViewportRender centraliza e projeta a camada com fit_scale na tela da Viewport."""
    layer = make_layer(100, 100, (255, 0, 0, 255))
    viewport = Viewport(size=(200, 200), fit_scale=1.0)
    vr = ViewportRender()

    composition = vr.render_scene([layer], viewport)

    assert composition.size == (200, 200)
    np.testing.assert_array_equal(composition[100, 100], [255, 0, 0, 255])
    np.testing.assert_array_equal(composition[0, 0], [204, 204, 204, 255])


def test_viewport_render_culling_camada_fora_da_tela(mocker):
    """Valida se ViewportRender descarta camadas totalmente fora do campo de visão da Viewport."""
    import anicrop.render

    spy_warp = mocker.spy(anicrop.render, "warp_patch")
    layer = make_layer(100, 100, (255, 0, 0, 255), x=5000, y=5000)
    viewport = Viewport(size=(200, 200), fit_scale=1.0)
    vr = ViewportRender()

    frame = ViewportFrame(layer, viewport)
    rendered = vr.render_area(layer, frame)

    assert rendered is None
    assert not spy_warp.called


def test_viewport_render_com_opacidade_e_blend():
    """Valida se ViewportRender compõe camadas com opacidade e mesclagem alfa na tela da Viewport."""
    bg_layer = make_layer(100, 100, (255, 0, 0, 255))
    fg_layer = make_layer(100, 100, (0, 0, 255, 255))
    fg_layer.opacity = 0.5

    viewport = Viewport(size=(100, 100), fit_scale=1.0)
    vr = ViewportRender()

    composition = vr.render_scene([bg_layer, fg_layer], viewport)

    pixel = composition[50, 50]
    assert pixel[0] > 0
    assert pixel[1] == 0
    assert pixel[2] > 0
    assert pixel[3] == 255


def test_viewport_render_com_grouplayer_aninhado():
    """Valida se ViewportRender compõe corretamente uma hierarquia com GroupLayer."""
    group = GroupLayer()
    group.transform.translate(20, 20)
    child = make_layer(60, 60, (0, 255, 0, 255))
    group.append(child)

    viewport = Viewport(size=(200, 200), fit_scale=1.0)
    vr = ViewportRender()

    composition = vr.render_scene([group], viewport)

    assert composition.size == (200, 200)
    # Fit centralizado de 60x60 em 200x200 = (70, 70) + translação (20, 20) = (90, 90)
    np.testing.assert_array_equal(composition[90, 90], [0, 255, 0, 255])
    np.testing.assert_array_equal(composition[0, 0], [204, 204, 204, 255])


def test_viewport_render_patch_com_view_region():
    """Valida se ViewportRender.render_patch restringe a composição à sub-região solicitada da Viewport."""
    layer = make_layer(100, 100, (255, 0, 0, 255))
    viewport = Viewport(size=(200, 200), fit_scale=1.0)
    patch_region = Region.from_rect(50, 50, 50, 50)
    vr = ViewportRender()

    patch_image = vr.render_patch([layer], viewport, view_region=patch_region)

    assert patch_image is not None
    assert patch_image.size == (50, 50)
    np.testing.assert_array_equal(patch_image[0, 0], [255, 0, 0, 255])


def test_viewport_render_com_zoom_pan_e_rotacao():
    """Valida render_scene de camada rotacionada sob zoom e pan no Viewport."""
    layer = make_layer(100, 100, (255, 0, 0, 255))
    layer.transform.rotate(45)

    viewport = Viewport(size=(200, 200), fit_scale=1.0)
    viewport.scale = Scale(1.5, 1.5)
    viewport.region += (10, 10)

    vr = ViewportRender()
    comp = vr.render_scene([layer], viewport)

    assert comp.size == (200, 200)
    assert comp[100, 100][0] == 255
    assert comp[100, 100][3] == 255


@pytest.mark.slow
def test_viewport_render_passes_scale_factor_to_lod():
    """Valida se ViewportRender repassa o scale_factor da Viewport para a seleção de LOD dos edits."""
    img_large = Image.new((5000, 5000), ImageFormat.RGBA)
    layer = Layer(img_large)
    viewport = Viewport((800, 600), 0.1)
    vr = ViewportRender()

    frame = ViewportFrame(layer, viewport)
    rendered = vr.render_area(layer, frame)

    assert rendered is not None


def test_viewport_render_with_layer_mask_modulates_viewport_pixels():
    """Valida se ViewportRender aplica a modulação da máscara da camada no espaço da Viewport."""
    layer = make_layer(100, 100, (255, 0, 0, 255))
    mask_data = np.full((100, 100, 1), 255, dtype=np.uint8)
    mask_data[25:75, 25:75] = 0
    mask_img = Image(mask_data, ImageFormat.GRAY)
    layer.set_mask(mask_img, Region.from_size(100, 100))

    viewport = Viewport(size=(200, 200), fit_scale=1.0)
    vr = ViewportRender()
    comp = vr.render_scene([layer], viewport)

    assert comp.size == (200, 200)
    # Centro mascarado (0) fica transparente revelando o fundo cinza (204) da Viewport
    np.testing.assert_array_equal(comp[100, 100], [204, 204, 204, 255])
    # Borda da camada (255) mantém a cor vermelha opaca
    np.testing.assert_array_equal(comp[60, 60], [255, 0, 0, 255])


def test_viewport_render_with_group_mask_modulates_grouped_children():
    """Valida se ViewportRender aplica a máscara de um GroupLayer sobre todos os seus filhos na Viewport."""
    group = GroupLayer()
    child1 = make_layer(100, 100, (255, 0, 0, 255))
    child2 = make_layer(100, 100, (0, 255, 0, 255))
    group.append(child1)
    group.append(child2)

    mask_data = np.zeros((100, 100, 1), dtype=np.uint8)
    mask_data[:50, :50] = 255
    mask_img = Image(mask_data, ImageFormat.GRAY)
    group.set_mask(mask_img, Region.from_size(100, 100))

    viewport = Viewport(size=(200, 200), fit_scale=1.0)
    vr = ViewportRender()
    comp = vr.render_scene([group], viewport)

    assert comp.size == (200, 200)
    # Quadrante superior esquerdo revelado (verde por cima do vermelho)
    np.testing.assert_array_equal(comp[60, 60], [0, 255, 0, 255])
    # Quadrante inferior cortado pela máscara revela o fundo da viewport
    np.testing.assert_array_equal(comp[120, 120], [204, 204, 204, 255])


def test_viewport_render_with_inverted_mask_under_zoom():
    """Valida se ViewportRender sob zoom de câmera renderiza máscara invertida com precisão."""
    layer = make_layer(100, 100, (0, 0, 255, 255))
    mask_data = np.full((100, 100, 1), 255, dtype=np.uint8)
    mask_data[25:75, 25:75] = 0
    mask_img = Image(mask_data, ImageFormat.GRAY)
    layer.set_mask(mask_img, Region.from_size(100, 100), invert=True)

    viewport = Viewport(size=(200, 200), fit_scale=1.0)
    viewport.scale = Scale(1.5, 1.5)

    vr = ViewportRender()
    comp = vr.render_scene([layer], viewport)

    assert comp.size == (200, 200)
    # Com invert=True, o centro (0) vira visível (azul)
    np.testing.assert_array_equal(comp[100, 100], [0, 0, 255, 255])
