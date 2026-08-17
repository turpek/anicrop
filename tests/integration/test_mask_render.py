import numpy as np
import pytest

from anicrop.canvas import Canvas
from anicrop.container import GroupLayer
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.render import CanvasRender
from anicrop.spatial import Region


def make_solid_layer(w: int, h: int, color: tuple[int, int, int, int] = (255, 0, 0, 255), name: str = "Layer") -> Layer:
    data = np.full((h, w, 4), color, dtype=np.uint8)
    img = Image(data, ImageFormat.RGBA)
    return Layer(img, name=name)


def test_render_layer_with_full_white_mask_preserves_layer_pixels():
    """Valida se a renderização de uma camada com máscara 100% branca mantém todos os pixels opacos."""
    canvas = Canvas.from_size(100, 100)
    layer = make_solid_layer(100, 100, color=(255, 0, 0, 255))

    mask_data = np.full((100, 100, 1), 255, dtype=np.uint8)
    mask_img = Image(mask_data, ImageFormat.GRAY)
    layer.set_mask(mask_img, Region.from_size(100, 100))

    renderer = CanvasRender()
    result = renderer.render_scene([layer], canvas)

    np.testing.assert_array_equal(result[..., -1], 255)
    np.testing.assert_array_equal(result[..., 0], 255)


def test_render_layer_with_localized_mask_creates_transparent_cutout():
    """Valida se uma máscara com área preta central cria um recorte transparente na imagem renderizada."""
    canvas = Canvas.from_size(100, 100)
    layer = make_solid_layer(100, 100, color=(0, 255, 0, 255))

    # Cria máscara 100x100 branca com um quadrado preto 50x50 no centro
    mask_data = np.full((100, 100, 1), 255, dtype=np.uint8)
    mask_data[25:75, 25:75] = 0
    mask_img = Image(mask_data, ImageFormat.GRAY)
    layer.set_mask(mask_img, Region.from_size(100, 100))

    renderer = CanvasRender()
    result = renderer.render_scene([layer], canvas)

    # Centro recortado deve ser transparente
    np.testing.assert_array_equal(result[25:75, 25:75, -1], 0)
    # Borda externa deve permanecer opaca
    np.testing.assert_array_equal(result[:25, :, -1], 255)
    np.testing.assert_array_equal(result[75:, :, -1], 255)


def test_render_layer_with_inverted_mask_reveals_only_masked_area():
    """Valida se invert=True inverte a lógica de transparência revelando apenas a região com valor 0."""
    canvas = Canvas.from_size(100, 100)
    layer = make_solid_layer(100, 100, color=(0, 0, 255, 255))

    mask_data = np.full((100, 100, 1), 255, dtype=np.uint8)
    mask_data[25:75, 25:75] = 0
    mask_img = Image(mask_data, ImageFormat.GRAY)
    layer.set_mask(mask_img, Region.from_size(100, 100), invert=True)

    renderer = CanvasRender()
    result = renderer.render_scene([layer], canvas)

    # Com invert=True, o quadrado central (0) é o único revelado
    np.testing.assert_array_equal(result[25:75, 25:75, -1], 255)
    np.testing.assert_array_equal(result[:25, :, -1], 0)
    np.testing.assert_array_equal(result[75:, :, -1], 0)


def test_render_group_layer_with_mask_masks_all_children_together():
    """Valida se uma máscara aplicada em um GroupLayer modula simultaneamente todos os filhos do grupo."""
    canvas = Canvas.from_size(100, 100)
    group = GroupLayer()

    child1 = make_solid_layer(100, 100, color=(255, 0, 0, 255), name="Fundo")
    child2 = make_solid_layer(50, 50, color=(0, 255, 0, 255), name="Topo")
    group.append(child1)
    group.append(child2)

    # Máscara 50x50 preta aplicada no grupo
    mask_data = np.zeros((100, 100, 1), dtype=np.uint8)
    mask_data[:50, :50] = 255  # apenas o quadrante superior esquerdo visível
    mask_img = Image(mask_data, ImageFormat.GRAY)
    group.set_mask(mask_img, Region.from_size(100, 100))

    renderer = CanvasRender()
    result = renderer.render_scene([group], canvas)

    # Quadrante superior esquerdo visível
    np.testing.assert_array_equal(result[:50, :50, -1], 255)
    # Restante da cena transparente
    np.testing.assert_array_equal(result[50:, :, -1], 0)
    np.testing.assert_array_equal(result[:, 50:, -1], 0)


def test_render_layer_mask_follows_layer_transform():
    """Valida se a máscara acompanha as transformações de translação aplicadas na camada."""
    canvas = Canvas.from_size(200, 200)
    layer = make_solid_layer(100, 100, color=(200, 100, 50, 255))

    mask_data = np.full((100, 100, 1), 255, dtype=np.uint8)
    mask_img = Image(mask_data, ImageFormat.GRAY)
    layer.set_mask(mask_img, Region.from_size(100, 100))

    # Translada a camada em (50, 50)
    layer.transform.translate(50, 50)

    renderer = CanvasRender()
    result = renderer.render_scene([layer], canvas)

    # Região transladada (50:150, 50:150) deve estar opaca
    np.testing.assert_array_equal(result[50:150, 50:150, -1], 255)
    # Área externa transparente
    np.testing.assert_array_equal(result[:50, :, -1], 0)
    np.testing.assert_array_equal(result[150:, :, -1], 0)


def test_render_layer_with_mask_direct_indexing():
    """Valida se edições cumulativas na máscara via slicing modulam o canal alfa corretamente."""
    canvas = Canvas.from_size(100, 100)
    layer = make_solid_layer(100, 100, color=(255, 0, 0, 255))

    mask_data = np.full((100, 100, 1), 255, dtype=np.uint8)
    layer.set_mask(Image(mask_data, ImageFormat.GRAY), Region.from_size(100, 100))

    # Apaga metade superior e metade esquerda na mesma máscara
    layer.mask[:50, :] = 0
    layer.mask[:, :50] = 0

    renderer = CanvasRender()
    result = renderer.render_scene([layer], canvas)

    # Apenas o quadrante inferior direito (50:100, 50:100) permanece opaco
    np.testing.assert_array_equal(result[50:, 50:, -1], 255)
    np.testing.assert_array_equal(result[:50, :, -1], 0)
    np.testing.assert_array_equal(result[:, :50, -1], 0)


def test_layer_is_renderable_with_mask():
    """Valida se is_renderable avalia corretamente o overlap da máscara na camada."""
    layer = make_solid_layer(100, 100)

    # Adiciona máscara sem overlap (em 200, 200)
    mask_data = np.full((50, 50, 1), 255, dtype=np.uint8)
    layer.set_mask(Image(mask_data, ImageFormat.GRAY), Region.from_rect(200, 200, 50, 50))
    assert layer.is_renderable is False

    # Substitui por máscara com overlap (em 0, 0)
    layer.set_mask(Image(mask_data, ImageFormat.GRAY), Region.from_rect(0, 0, 50, 50))
    assert layer.is_renderable is True
